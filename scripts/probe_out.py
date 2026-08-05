import os
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
from ultralytics import YOLO
from pathlib import Path
m = YOLO('/root/yolo_runs/smoke/weights/best.pt')
f = sorted(Path('/root/yolo_data/smoke/images/val').glob('*.png'))[0]
print('file:', f.name)
r = m.predict(str(f), imgsz=256, device=0, verbose=False)[0]
print('type:', type(r))
print('keys:', [k for k in dir(r) if not k.startswith('_')])
for attr in ['sem','masks','probs','boxes','orig_shape','names']:
    try:
        v = getattr(r, attr)
        print(attr, '->', type(v))
        if v is not None and hasattr(v, 'data'):
            print('   data shape:', tuple(v.data.shape))
        elif v is not None and hasattr(v, 'shape'):
            print('   shape:', tuple(v.shape))
    except Exception as e:
        print(attr, 'ERR', str(e)[:80])