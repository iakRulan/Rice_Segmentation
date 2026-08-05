
import numpy as np, os
from PIL import Image
TRAIN = '/root/competition_data/public/train'
def stats(cls, img_dir, label_dir):
    imgs = sorted(f for f in os.listdir(img_dir) if f.endswith('.png'))
    areas = []
    n_empty = 0
    for f in imgs:
        lab = np.array(Image.open(os.path.join(label_dir, f)))
        a = int((lab > 0).sum())
        areas.append(a)
        if a == 0: n_empty += 1
    areas = np.array(areas)
    ne = areas[areas > 0]
    print(f'[{cls}] n={len(imgs)} empty={n_empty} ({100*n_empty/len(imgs):.1f}%)')
    print(f'  nonempty area: min={ne.min()} p10={np.percentile(ne,10):.0f} p25={np.percentile(ne,25):.0f} med={np.median(ne):.0f} p75={np.percentile(ne,75):.0f} p90={np.percentile(ne,90):.0f} max={ne.max()}')
    print(f'  tiny(<100px): {(ne<100).sum()}  small(<500): {(ne<500).sum()}  large(>10000): {(ne>10000).sum()}')
stats('wheat', TRAIN+'/image/wheat_rape', TRAIN+'/label/wheat')
stats('rape', TRAIN+'/image/wheat_rape', TRAIN+'/label/rape')
stats('rice', TRAIN+'/image/rice', TRAIN+'/label/rice')
# overlap wheat vs rape in same image
imgs = sorted(f for f in os.listdir(TRAIN+'/image/wheat_rape') if f.endswith('.png'))
both = 0
for f in imgs:
    w = np.array(Image.open(TRAIN+'/label/wheat/'+f))
    r = np.array(Image.open(TRAIN+'/label/rape/'+f))
    if (w>0).sum() and (r>0).sum(): both += 1
print(f'wheat+rape co-occurrence: {both}/{len(imgs)}')
