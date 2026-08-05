import os
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
from ultralytics import YOLO
m = YOLO('yolo26s-sem.pt')
m.train(data='/root/yolo_data/smoke/data.yaml', epochs=2, imgsz=256, batch=16, device=0,
        workers=4, project='/root/yolo_runs', name='smoke', exist_ok=True, cache=True, plots=False, verbose=True)