import os
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
from ultralytics import YOLO
for n in ['yolo26n-sem.pt','yolo26s-sem.pt','yolo26m-sem.pt','yolo26l-sem.pt','yolo26x-sem.pt']:
    try:
        m = YOLO(n)
        print(n, 'OK', round(sum(p.numel() for p in m.model.parameters())/1e6,1), 'M')
    except Exception as e:
        print(n, 'ERR', str(e)[:120])