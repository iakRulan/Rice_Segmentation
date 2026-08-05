import os
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
from ultralytics import YOLO
from pathlib import Path
import numpy as np
m = YOLO('/root/yolo_runs/smoke/weights/best.pt')
f = sorted(Path('/root/yolo_data/smoke/images/val').glob('*.png'))[0]
r = m.predict(str(f), imgsz=256, device=0, verbose=False)[0]
sm = r.semantic_mask
print('type:', type(sm), 'shape:', sm.shape if hasattr(sm,'shape') else None)
if hasattr(sm, 'data'):
    d = sm.data
    print('data shape:', tuple(d.shape), 'dtype:', d.dtype)
    arr = d.cpu().numpy()
    print('unique:', np.unique(arr)[:20])
    print('range:', arr.min(), arr.max())
elif isinstance(sm, np.ndarray):
    print('unique:', np.unique(sm)[:20], 'range:', sm.min(), sm.max())
else:
    print(sm)