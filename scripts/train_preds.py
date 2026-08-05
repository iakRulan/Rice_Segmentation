"""Single-pass inference of current best models on TRAIN set, saving prob maps (fp16) as npz."""
import os, sys
import numpy as np
from PIL import Image
import torch
from torch.cuda.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, '/root/crop_segmentation')
from src.models import MultiLabelModel, SingleLabelModel

NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
TRAIN = '/root/competition_data/public/train'
WEIGHTS = '/root/crop_segmentation/weights'
device = torch.device('cuda')


def infer_dir(model, img_dir, ch, out_npz, batch=64):
    imgs = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    tf = A.Compose([A.Normalize(**NORM), ToTensorV2()])
    preds = {}
    for i in range(0, len(imgs), batch):
        batch_imgs = imgs[i:i+batch]
        tensors = []
        for f in batch_imgs:
            image = np.array(Image.open(os.path.join(img_dir, f)).convert('RGB'))
            tensors.append(tf(image=image)['image'])
        x = torch.stack(tensors).to(device)
        with torch.no_grad(), autocast():
            out = torch.sigmoid(model(x)).float().cpu().numpy()
        for j, f in enumerate(batch_imgs):
            p = out[j][ch] if ch is not None else out[j].squeeze()
            preds[f] = p.astype(np.float16)
        if (i // batch + 1) % 20 == 0:
            print(f'  {i+batch}/{len(imgs)}', flush=True)
    np.savez(out_npz, **preds)
    print(f'saved {out_npz} ({len(preds)})', flush=True)


def main():
    wr = MultiLabelModel().to(device).eval()
    wr.load_state_dict(torch.load(os.path.join(WEIGHTS, 'best_wheat_rape.pth'), map_location=device, weights_only=False)['model_state_dict'])
    rice = SingleLabelModel().to(device).eval()
    rice.load_state_dict(torch.load(os.path.join(WEIGHTS, 'best_rice.pth'), map_location=device, weights_only=False)['model_state_dict'])

    wr_dir = os.path.join(TRAIN, 'image/wheat_rape')
    rice_dir = os.path.join(TRAIN, 'image/rice')
    print('wheat train...', flush=True)
    infer_dir(wr, wr_dir, 0, '/root/trainpred_wheat.npz')
    print('rape train...', flush=True)
    infer_dir(wr, wr_dir, 1, '/root/trainpred_rape.npz')
    print('rice train...', flush=True)
    infer_dir(rice, rice_dir, None, '/root/trainpred_rice.npz')


if __name__ == '__main__':
    main()
