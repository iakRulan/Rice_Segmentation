import sys; sys.path.insert(0, '.')
import torch
from cropseg.losses import SegmentationLoss

# per-class pos_weight tensor path
crit = SegmentationLoss({
    'bce': 1.0, 'dice': 1.0, 'lovasz': 0.5, 'tversky': 0.2,
    'tversky_alpha': 0.35, 'tversky_beta': 0.65, 'boundary': 0.1,
    'cls': 0.5, 'pos_weight': [1.3, 1.5, 1.0], 'focal_gamma': 1.0,
})
# 3-class joint: logits (2,3,8,8), target (2,3,8,8)
logits = torch.randn(2, 3, 8, 8, requires_grad=True)
target = (torch.rand(2, 3, 8, 8) > 0.7).float()
cls_logits = torch.randn(2, 3, requires_grad=True)
loss = crit(logits, target, cls_logits)
loss.backward()
print(f"loss={loss.item():.4f} logits_grad={logits.grad.abs().mean().item():.4f} OK")
# scalar path still works
crit2 = SegmentationLoss({'bce': 1.0, 'pos_weight': 1.3})
l2 = crit2(logits.detach().requires_grad_(True), target)
l2.backward()
print("scalar pos_weight OK:", l2.item())
print("LOSS TEST DONE")
