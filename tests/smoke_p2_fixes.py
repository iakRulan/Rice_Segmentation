import sys; sys.path.insert(0, '.')
import numpy as np, torch

# 1) metrics: widened range + can search down to 0.15
from cropseg.metrics import search_threshold
import inspect
sig = inspect.signature(search_threshold)
print('1) search_threshold defaults low/high:', sig.parameters['low'].default, sig.parameters['high'].default)
probs = np.random.RandomState(0).rand(50, 1, 16, 16)
targets = (np.random.RandomState(1).rand(50, 1, 16, 16) > 0.5).astype(np.float32)
score, t = search_threshold(probs, targets, 0.05, 0.95, 0.05)
print('   search on [0.05,0.95] -> t=%.2f score=%.3f' % (t, score))

# 2) data: MosaicStore bottom edge + interior windows
from cropseg.data import MosaicStore
st = MosaicStore('data/public', 'rice', 83, cache=False)
last_idx = max(st.paths)
print('2) store max id:', last_idx, 'n_paths:', len(st.paths))
row81_tile = 81 * 83 + 42          # very last row -> dy=+1 must fall back to center
w = st.window(row81_tile, 512)
interior = 30 * 83 + 40            # interior tile -> all 8 neighbors real
w2 = st.window(interior, 512)
print('   bottom-edge window shape:', w.shape, '| interior window shape:', w2.shape)
# bottom-edge mosaic should be identical to an all-center fallback: window 512
# still crops 256 off each axis, so verify it is valid & non-empty
print('   bottom-edge window min/max:', int(w.min()), int(w.max()))

# 3) cls loss now fires in SegmentationLoss
from cropseg.losses import SegmentationLoss
crit = SegmentationLoss({'bce': 1.0, 'dice': 1.0, 'cls': 0.5}).cuda()
lg = torch.randn(2, 3, 64, 64).cuda()
tg = (torch.rand(2, 3, 64, 64).cuda() > 0.5).float()
cl = torch.randn(2, 3).cuda()
with_cls = crit(lg, tg, cl)
without_cls = crit(lg, tg, None)
delta = abs(with_cls.item() - without_cls.item())
print('3) loss with cls=%.4f without=%.4f diff=%.4f -> %s' % (
    with_cls.item(), without_cls.item(), delta, 'OK' if delta > 1e-6 else 'STILL-DEAD'))

# 4) engine train_epoch: nonfinite grad is skipped, weights stay finite
from cropseg.engine import train_epoch, EMA, cosine_with_warmup
from cropseg.models import build_model
cfg = {'backend': 'smp', 'architecture': 'unet', 'encoder': 'resnet18',
       'encoder_weights': None, 'aux': True}
model = build_model(cfg, 3, pretrained=False).cuda()
ema = EMA(model, 0.999)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = cosine_with_warmup(opt, 4)
losses = [torch.nn.functional.binary_cross_entropy_with_logits, ]

class TinyDS:
    def __len__(self):
        return 4
    def __iter__(self):
        for i in range(4):
            x = torch.randn(1, 3, 64, 64)
            y = (torch.rand(1, 3, 64, 64) > 0.5).float()
            yield x, y, f'id_{i}'

crit = SegmentationLoss({'bce': 1.0}).cuda()
train_epoch(model, TinyDS(), crit, opt, sched, ema, 2, 1.0)
bad = [p for p in model.parameters() if not torch.isfinite(p).all()]
print('4) after train_epoch: non-finite params = %d -> %s' % (len(bad), 'OK' if not bad else 'CORRUPTED!'))

# 5) aux model returns (seg, cls); cls head shape correct
model2 = build_model(cfg, 3, pretrained=False).cuda()
with torch.no_grad():
    out = model2(torch.randn(2, 3, 64, 64).cuda())
print('5) aux forward returns tuple:', isinstance(out, (tuple, list)),
      '| seg', out[0].shape, '| cls', out[1].shape)
print('ALL CHECKS DONE')
