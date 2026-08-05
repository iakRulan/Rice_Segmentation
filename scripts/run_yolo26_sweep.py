import os, time
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
os.environ['YOLO_VERBOSE'] = 'False'
from ultralytics import YOLO

SIZES = [('n', 32), ('s', 32), ('m', 24), ('l', 16), ('x', 12)]
DATA = '/root/yolo_data/wheat_rape/data.yaml'

for size, bs in SIZES:
    t0 = time.time()
    tag = f'y26sem_wr_{size}'
    print(f'=== START {tag} bs={bs} {time.strftime("%H:%M:%S")} ===', flush=True)
    model = YOLO(f'yolo26{size}-sem.pt')
    model.train(data=DATA, epochs=50, imgsz=256, batch=bs, device=0, workers=8,
                project='/root/yolo_runs', name=tag, exist_ok=True, cache=True,
                plots=False, seed=0, val=True, verbose=False, patience=30)
    print(f'=== DONE {tag} in {(time.time()-t0)/60:.1f} min {time.strftime("%H:%M:%S")} ===', flush=True)
print('ALL DONE', flush=True)