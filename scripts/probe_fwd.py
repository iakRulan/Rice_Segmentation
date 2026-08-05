import os
os.environ['PATH'] = '/root/miniconda3/bin:' + os.environ.get('PATH','')
from ultralytics import YOLO
from PIL import Image
import numpy as np, torch
m = YOLO('/root/yolo_runs/smoke/weights/best.pt')
mm = m.model.to('cuda').eval()
im = np.asarray(Image.open('/root/yolo_data/smoke/images/val/clip_00333.png').convert('RGB')).astype(np.float32)/255.0
x = torch.from_numpy(im).permute(2,0,1).unsqueeze(0).cuda()
with torch.no_grad():
    out = mm(x)
print('out type:', type(out))
if isinstance(out, (tuple, list)):
    print('len:', len(out))
    for i,o in enumerate(out):
        print(i, type(o), getattr(o,'shape',None))
else:
    print('shape:', tuple(out.shape))
    print('range:', float(out.min()), float(out.max()))