"""Build trainplus (train + most of val) + valhold (every-7th val) locally.
Matches the server's setup_trainval.py split exactly (sorted wheat_rape images,
every 7th -> valhold). trainplus = train + 85% val for training on all data.
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA, TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL

TP = DATA / 'trainplus' / 'train'
VH = DATA / 'valhold' / 'val'

for c in ['wheat', 'rape', 'rice']:
    (TP / 'label' / c).mkdir(parents=True, exist_ok=True)
    (VH / 'label' / c).mkdir(parents=True, exist_ok=True)
for c in ['wheat_rape', 'rice']:
    (TP / 'image' / c).mkdir(parents=True, exist_ok=True)
    (VH / 'image' / c).mkdir(parents=True, exist_ok=True)

# copy train (tr_) into trainplus
for split in ['wheat_rape', 'rice']:
    for f in os.listdir(TRAIN_IMG / split):
        if f.endswith('.png'):
            shutil.copy2(TRAIN_IMG / split / f, TP / 'image' / split / f'tr_{f}')
for c in ['wheat', 'rape', 'rice']:
    for f in os.listdir(TRAIN_LBL / c):
        if f.endswith('.png'):
            shutil.copy2(TRAIN_LBL / c / f, TP / 'label' / c / f'tr_{f}')

# split val: every 7th sorted wheat_rape image -> valhold, rest -> trainplus
val_imgs = sorted(f for f in os.listdir(VAL_IMG / 'wheat_rape') if f.endswith('.png'))
hold = {f'va_{os.path.basename(p)}' for i, p in enumerate(val_imgs) if i % 7 == 0}
print(f'val total {len(val_imgs)}, holdout {len(hold)}', flush=True)

for split in ['wheat_rape', 'rice']:
    for f in os.listdir(VAL_IMG / split):
        if not f.endswith('.png'):
            continue
        dst_tp = TP / 'image' / split / f'va_{f}'
        dst_vh = VH / 'image' / split / f'hold_{f}'
        if f'va_{f}' in hold:
            shutil.copy2(VAL_IMG / split / f, dst_vh)
        else:
            shutil.copy2(VAL_IMG / split / f, dst_tp)
for c in ['wheat', 'rape', 'rice']:
    for f in os.listdir(VAL_LBL / c):
        if not f.endswith('.png'):
            continue
        dst_tp = TP / 'label' / c / f'va_{f}'
        dst_vh = VH / 'label' / c / f'hold_{f}'
        if f'va_{f}' in hold:
            shutil.copy2(VAL_LBL / c / f, dst_vh)
        else:
            shutil.copy2(VAL_LBL / c / f, dst_tp)

for c in ['wheat', 'rape', 'rice']:
    print(f'{c}: trainplus={len(os.listdir(TP/"label"/c))} valhold={len(os.listdir(VH/"label"/c))}', flush=True)
print('trainplus setup done', flush=True)
