"""Merge train + testA pseudo-labels into data/trainpseudo (val NOT included,
so the val signal stays honest for self-trained models).
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DATA, TRAIN_IMG, TRAIN_LBL, TESTA_IMG

SRC = DATA / 'pseudo'
DST = DATA / 'trainpseudo'

for c in ['wheat', 'rape', 'rice']:
    (DST / 'label' / c).mkdir(parents=True, exist_ok=True)
for c in ['wheat_rape', 'rice']:
    (DST / 'image' / c).mkdir(parents=True, exist_ok=True)

# copy train images/labels
for split in ['wheat_rape', 'rice']:
    for f in os.listdir(TRAIN_IMG / split):
        if f.endswith('.png'):
            shutil.copy2(TRAIN_IMG / split / f, DST / 'image' / split / f)
for c in ['wheat', 'rape', 'rice']:
    for f in os.listdir(TRAIN_LBL / c):
        if f.endswith('.png'):
            shutil.copy2(TRAIN_LBL / c / f, DST / 'label' / c / f)

# copy pseudo (ta_ prefix already, no collision with train names)
for split in ['wheat_rape', 'rice']:
    src = SRC / 'image' / split
    for f in os.listdir(src):
        if f.endswith('.png'):
            shutil.copy2(src / f, DST / 'image' / split / f)
for c in ['wheat', 'rape', 'rice']:
    src = SRC / 'label' / c
    for f in os.listdir(src):
        if f.endswith('.png'):
            shutil.copy2(src / f, DST / 'label' / c / f)

for c in ['wheat', 'rape', 'rice']:
    print(f'{c}: {len(os.listdir(DST/"label"/c))} labels')
print('trainpseudo ready:', DST)
