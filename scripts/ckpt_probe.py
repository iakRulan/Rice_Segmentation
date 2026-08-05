import os, torch
wd = '/root/crop_segmentation/weights'
files = [f for f in os.listdir(wd) if f in ["best_rice.pth", "best_wheat_rape.pth", "final_rice.pth", "final_wheat_rape.pth", "v2_best_multi.pth", "v2_best_single.pth"]]
for f in files:
    p = os.path.join(wd, f)
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        cfg = ck.get('config', {})
        arch = cfg.get('arch', '?')
        enc = cfg.get('encoder', '?')
        seed = cfg.get('seed', '?')
        val = ck.get('val_iou', '?')
        sz = os.path.getsize(p) / 1e6
        print(f'{f:55s} arch={arch:15s} enc={enc:22s} seed={seed} val={val} {sz:.0f}M')
    except Exception as e:
        print(f'{f:55s} ERROR {e}')
