from PIL import Image
import numpy as np, glob
for c in ['wheat', 'rape', 'rice']:
    fs = sorted(glob.glob(f'/root/submission_final/{c}/*.png'))
    n_nonempty = 0
    fracs = []
    for f in fs[:200]:
        a = np.array(Image.open(f))
        if (a > 0).sum() > 0:
            n_nonempty += 1
        fracs.append((a > 0).mean())
    print(f'{c}: {len(fs)} files, nonempty_frac(sample200)={n_nonempty/200:.2f}, mean_mask_frac={np.mean(fracs):.4f}')
