from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxClassifier(nn.Module):
    """Image-level "has object" head on the deepest encoder feature.

    Shares the encoder with the segmentation decoder, so it is both a regularizer
    and a free empty-image discriminator. smp exposes the same thing via
    ``aux_params``; SatlasSegmenter gets a manual copy.
    """

    def __init__(self, in_channels: int, classes: int, dropout: float = 0.3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout2d(dropout)
        self.fc = nn.Conv2d(in_channels, classes, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.pool(feature))).flatten(1)


class SatlasSegmenter(nn.Module):
    """Satlas pretrained backbone+FPN with a lightweight logit decoder."""

    def __init__(self, model_id: str, classes: int, decoder_channels: int = 128,
                 pretrained: bool = True, aux: bool = False, aux_dropout: float = 0.3):
        super().__init__()
        try:
            import satlaspretrain_models as satlas
            from satlaspretrain_models.utils import SatlasPretrain_weights
        except ImportError as exc:
            raise RuntimeError(
                "Satlas backend requires `pip install satlaspretrain-models==0.3.1`"
            ) from exc
        if model_id not in SatlasPretrain_weights:
            raise ValueError(f"unknown Satlas checkpoint: {model_id}")
        if pretrained:
            self.foundation = satlas.Weights().get_pretrained_model(
                model_id, fpn=True, device="cpu"
            )
        else:
            info = SatlasPretrain_weights[model_id]
            self.foundation = satlas.Model(
                info["num_channels"], info["multi_image"], info["backbone"],
                fpn=True, weights=None
            )
        # FPN and the library upsampler expose 128 channels at every scale.
        self.decoder = nn.Sequential(
            nn.Conv2d(128, decoder_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, decoder_channels),
            nn.GELU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, classes, 1),
        )
        self.aux = aux
        if aux:
            self.cls_head = AuxClassifier(128, classes, aux_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.foundation(x)
        size = features[0].shape[-2:]
        fused = features[0]
        for feature in features[1:]:
            fused = fused + F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        seg = self.decoder(fused / len(features))
        if self.aux:
            return seg, self.cls_head(fused)
        return seg

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.foundation.backbone.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        backbone_ids = {id(p) for p in self.foundation.backbone.parameters()}
        return (p for p in self.parameters() if id(p) not in backbone_ids)


class SMPModel(nn.Module):
    def __init__(self, architecture: str, encoder: str, classes: int,
                 encoder_weights: str | None = "imagenet", in_channels: int = 3,
                 aux: bool = False, aux_dropout: float = 0.3):
        super().__init__()
        import segmentation_models_pytorch as smp
        table = {
            "unet": smp.Unet,
            "unetpp": smp.UnetPlusPlus,
            "deeplabv3plus": smp.DeepLabV3Plus,
            "fpn": smp.FPN,
            "manet": smp.MAnet,
        }
        kw = dict(
            encoder_name=encoder, encoder_weights=encoder_weights,
            in_channels=in_channels, classes=classes, activation=None,
        )
        if aux:
            kw["aux_params"] = dict(classes=classes, dropout=aux_dropout, pooling="avg")
        self.net = table[architecture](**kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        return self.net.encoder.parameters()

    def head_parameters(self) -> Iterable[nn.Parameter]:
        encoder_ids = {id(p) for p in self.net.encoder.parameters()}
        return (p for p in self.parameters() if id(p) not in encoder_ids)


def build_model(config: dict, classes: int, pretrained: bool = True) -> nn.Module:
    backend = config["backend"]
    aux = bool(config.get("aux", False))
    aux_dropout = float(config.get("aux_dropout", 0.3))
    if backend == "satlas":
        return SatlasSegmenter(
            config["model_id"], classes,
            decoder_channels=int(config.get("decoder_channels", 128)),
            pretrained=pretrained, aux=aux, aux_dropout=aux_dropout,
        )
    if backend == "smp":
        return SMPModel(
            config.get("architecture", "unet"), config["encoder"], classes,
            config.get("encoder_weights", "imagenet") if pretrained else None,
            in_channels=int(config.get("in_channels", 3)),
            aux=aux, aux_dropout=aux_dropout,
        )
    raise ValueError(f"unknown backend: {backend}")


def init_first_conv(model: nn.Module, in_channels: int, n_img: int) -> None:
    """Expand pretrained RGB first-conv weights to ``n_img`` image channels and
    zero-initialize the remaining (prior) channels.

    smp/timm widen the entry conv by repeating RGB and scaling by
    in_channels/3; we undo that scale, split the image signal across the
    temporals (n_img//3 of them), and start prior channels at zero so the model
    initially behaves like the no-prior baseline.
    """
    conv = next(m for m in model.modules()
                if isinstance(m, nn.Conv2d) and m.in_channels == in_channels)
    with torch.no_grad():
        pre = conv.weight[:, :3].clone() * (conv.in_channels / 3.0)
        if n_img >= 3:
            pre = pre / max(1, n_img // 3)
        for s in range(0, n_img, 3):
            conv.weight[:, s:s + 3] = pre
        conv.weight[:, n_img:].zero_()


def center_crop(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    height, width = target.shape[-2:]
    if logits.shape[-2:] == (height, width):
        return logits
    y0 = (logits.shape[-2] - height) // 2
    x0 = (logits.shape[-1] - width) // 2
    if y0 < 0 or x0 < 0:
        raise ValueError(f"model output {logits.shape[-2:]} is smaller than target {(height, width)}")
    return logits[..., y0:y0 + height, x0:x0 + width]


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.backbone_parameters():
        parameter.requires_grad_(trainable)
