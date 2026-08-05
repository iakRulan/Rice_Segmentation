from PIL import Image
import numpy as np, glob, os
base = '/root/competition_data/public'
for c in ['rice', 'wheat_rape']:
    fs = sorted(glob.glob(f'{base}/train/image/{c}/*.png'))
    dims = set()
    for f in fs[:40]:
        im = Image.open(f); dims.add(im.size)
    print(c, 'sample dims', dims)
for c in ['rice', 'wheat', 'rape']:
    fs = sorted(glob.glob(f'{base}/train/label/{c}/*.png'))
    n_empty = 0; n_non = 0; areas = []; fracs = []
    for f in fs[:200]:
        m = (np.array(Image.open(f)) > 0)
        a = m.sum()
        if a == 0:
            n_empty += 1
        else:
            n_non += 1
            areas.append(a); fracs.append(a / m.size)
    med = np.median(areas) if areas else 0
    fm = np.median(fracs) if fracs else 0
    print(f'{c}: sample200 empty={n_empty} non={n_non} area_med={med:.0f} frac_med={fm:.4f}')
