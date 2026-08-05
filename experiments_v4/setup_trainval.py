"""Build trainplus (train + most of val) + valhold (held-out val slice) for train-on-all-data.
Holdout: every 7th val image (sorted), deterministic ~95 images. Unique tr_/va_/hold_ prefixes.
"""
import os, shutil, glob

DATA = '/root/competition_data/public'
TP = f'{DATA}/trainplus/train'
VH = f'{DATA}/valhold/val'

# init dirs
for c in ['wheat', 'rape', 'rice']:
    os.makedirs(f'{TP}/label/{c}', exist_ok=True)
    os.makedirs(f'{VH}/label/{c}', exist_ok=True)
for c in ['wheat_rape', 'rice']:
    os.makedirs(f'{TP}/image/{c}', exist_ok=True)
    os.makedirs(f'{VH}/image/{c}', exist_ok=True)

# copy train (tr_) into trainplus
for split in ['wheat_rape', 'rice']:
    for f in os.listdir(f'{DATA}/train/image/{split}'):
        if f.endswith('.png'):
            shutil.copy2(f'{DATA}/train/image/{split}/{f}', f'{TP}/image/{split}/tr_{f}')
for c in ['wheat', 'rape', 'rice']:
    for f in os.listdir(f'{DATA}/train/label/{c}'):
        if f.endswith('.png'):
            shutil.copy2(f'{DATA}/train/label/{c}/{f}', f'{TP}/label/{c}/tr_{f}')

# split val: every 7th -> holdout, rest -> trainplus
val_imgs = sorted(glob.glob(f'{DATA}/val/image/wheat_rape/*.png'))
hold = set(f'va_{os.path.basename(p)}' for i, p in enumerate(val_imgs) if i % 7 == 0)
print(f'val total {len(val_imgs)}, holdout {len(hold)}')
for split in ['wheat_rape', 'rice']:
    for f in os.listdir(f'{DATA}/val/image/{split}'):
        if not f.endswith('.png'):
            continue
        dst_tp = f'{TP}/image/{split}/va_{f}'
        dst_vh = f'{VH}/image/{split}/hold_{f}'
        (shutil.copy2(f'{DATA}/val/image/{split}/{f}', dst_vh) if f'va_{f}' in hold
         else shutil.copy2(f'{DATA}/val/image/{split}/{f}', dst_tp))
for c in ['wheat', 'rape', 'rice']:
    for f in os.listdir(f'{DATA}/val/label/{c}'):
        if not f.endswith('.png'):
            continue
        dst_tp = f'{TP}/label/{c}/va_{f}'
        dst_vh = f'{VH}/label/{c}/hold_{f}'
        (shutil.copy2(f'{DATA}/val/label/{c}/{f}', dst_vh) if f'va_{f}' in hold
         else shutil.copy2(f'{DATA}/val/label/{c}/{f}', dst_tp))

for c in ['wheat', 'rape', 'rice']:
    n_tr = len(os.listdir(f'{TP}/label/{c}'))
    n_ho = len(os.listdir(f'{VH}/label/{c}'))
    print(f'{c}: trainplus={n_tr} valhold={n_ho}')
