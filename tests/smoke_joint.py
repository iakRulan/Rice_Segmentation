import sys; sys.path.insert(0, '.')
import json
import numpy as np, torch

from cropseg.config import load_experiment
from cropseg.data import build_joint_records, MosaicMultiTemporalDataset, MosaicStore, tile_id
from cropseg.models import build_model, init_first_conv
from cropseg.losses import SegmentationLoss
from cropseg.engine import center_crop

root = 'data/public'
fold_file = 'data/public/folds/folds_iso5.json'

# 1) records
tr, va, train_ids = build_joint_records(root, fold_file, 0)
print('1) records train=%d val=%d train_ids=%d' % (len(tr), len(va), len(train_ids)))
assert len(tr) + len(va) == 5478, 'must cover all labeled tiles'
assert len(va) == 1162, 'fold0 held-out (1162 tiles)'

# 2) dataset 9ch
rs = MosaicStore(root, 'rice', 83, False)
ws = MosaicStore(root, 'wheat_rape', 83, False)
ds9 = MosaicMultiTemporalDataset(root, va, rs, ws, train_ids, context_size=768,
                                 normalization='imagenet', temporal='dual', prior=True)
x, y, name = ds9[0]
print('2) 9ch input:', tuple(x.shape), 'target:', tuple(y.shape), 'name:', name)
assert x.shape == (9, 768, 768) and y.shape == (3, 256, 256)
# prior: center 256 rows zero, but not all-zero overall
prior = x[6:9].numpy()
center = prior[:, 256:512, 256:512]
assert center.sum() == 0.0, 'center row must be masked'
print('   center 256 prior sum=%.0f (masked OK), full prior sum=%.0f' % (center.sum(), prior.sum()))

# 3) 3ch dataset (wheat_rape temporal, no prior)
ds3 = MosaicMultiTemporalDataset(root, va, rs, ws, train_ids, context_size=768,
                                 normalization='imagenet', temporal='wheat_rape', prior=False)
x3, y3, _ = ds3[0]
print('3) 3ch input:', tuple(x3.shape))
assert x3.shape == (3, 768, 768)

# 4) 6ch dataset (dual, no prior)
ds6 = MosaicMultiTemporalDataset(root, va, rs, ws, train_ids, context_size=768,
                                 normalization='imagenet', temporal='dual', prior=False)
x6, _, _ = ds6[0]
print('4) 6ch input:', tuple(x6.shape))
assert x6.shape == (6, 768, 768)

# 5) model build + init_first_conv for 9ch
m = build_model({'backend': 'smp', 'architecture': 'unet', 'encoder': 'resnet18',
                 'encoder_weights': None, 'in_channels': 9, 'aux': True}, 3, pretrained=False)
init_first_conv(m, 9, 6)
conv = next(c for c in m.net.encoder.modules() if isinstance(c, torch.nn.Conv2d) and c.in_channels == 9)
print('5) first conv weight shape:', tuple(conv.weight.shape))
w = conv.weight.data
assert w.shape[1] == 9
# channels 0-5 = replicated pre (two temporals), 6-8 zero
pre = w[:, 0:3]
print('   ch0-2 norm=%.3f ch3-5 norm=%.3f ch6-8 norm=%.3f' % (
    pre.norm().item(), w[:, 3:6].norm().item(), w[:, 6:9].norm().item()))
assert abs(w[:, 3:6].norm().item() - pre.norm().item()) < 1e-4, 'temporal 2 must match temporal 1'
assert w[:, 6:9].norm().item() == 0.0, 'prior channels must start at zero'
# forward with 9ch input
out = m(x.unsqueeze(0))
print('   forward seg:', out[0].shape, 'cls:', out[1].shape)
assert out[0].shape == (1, 3, 768, 768) and out[1].shape == (1, 3)
cropped = center_crop(out[0], y.unsqueeze(0))
print('   center_crop 768->256:', tuple(cropped.shape))

# 6) loss with cls over 3 classes
crit = SegmentationLoss({'bce': 1.0, 'dice': 1.0, 'lovasz': 0.5, 'cls': 0.5}).cuda()
loss = crit(cropped.cuda(), y.unsqueeze(0).cuda(), out[1].cuda())
loss.backward()
print('6) loss=%.4f backward OK' % loss.item())

print('ALL JOINT SMOKE CHECKS DONE')
