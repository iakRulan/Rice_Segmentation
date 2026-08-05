from pathlib import Path
import shutil, random
random.seed(0)
SRC = Path('/root/yolo_data/wheat_rape')
OUT = Path('/root/yolo_data/smoke')
for sp in ['train','val']:
    (OUT/sp/'images').mkdir(parents=True, exist_ok=True)
    (OUT/sp/'masks').mkdir(parents=True, exist_ok=True)
n = 64 if (OUT/'train'/'images').exists() else 64
for sp, k in [('train', 64), ('val', 32)]:
    imgs = sorted((SRC/sp/'images').glob('*.png'))
    chosen = random.sample(imgs, min(k, len(imgs)))
    for im in chosen:
        shutil.copy2(im, OUT/sp/'images'/im.name)
        m = SRC/sp/'masks'/im.name
        shutil.copy2(m, OUT/sp/'masks'/im.name)
    print(sp, len(chosen))
(OUT/'data.yaml').write_text(f"path: {OUT}\ntrain: images/train\nval: images/val\nmasks_dir: masks\nnc: 2\nnames: {{0: wheat, 1: rape}}\n")
print('smoke yaml written')