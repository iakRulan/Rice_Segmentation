from PIL import Image
import numpy as np, glob
base = '/root/competition_data/public/sample_submission_testA'
for c in ['rape', 'rice', 'wheat']:
    fs = sorted(glob.glob(f'{base}/{c}/*.png'))
    im = Image.open(fs[0]); a = np.array(im)
    print(c, len(fs), 'size', im.size, 'mode', im.mode, 'uniques', np.unique(a)[:5], 'frac1', round((a > 0).mean(), 4))
