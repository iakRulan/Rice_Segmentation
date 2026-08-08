"""Fine-tune a legacy tile model on native-resolution 512px mosaic windows."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opt_patch.losses_v2 import MultiTaskLoss
from scripts.infer_mosaic import MEAN, STD, MosaicStore, build, state


class TrainMosaicDataset(Dataset):
    def __init__(self, data: Path, split: str, mode: str, store: MosaicStore,
                 augment: bool):
        self.domain = 'wheat_rape' if mode in ('wheat_rape', 'wheat', 'rape') else 'rice'
        self.names = sorted(p.name for p in
                            (data/split/'image'/self.domain).glob('*.png'))
        classes = ['wheat', 'rape'] if mode == 'wheat_rape' else [mode]
        self.labels = [data/split/'label'/c for c in classes]
        self.store, self.augment = store, augment

    def __len__(self): return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        i = int(Path(name).stem.rsplit('_', 1)[1])
        x = self.store.window(i)
        y = np.stack([(np.asarray(Image.open(d/name)) > 0).astype(np.float32)
                      for d in self.labels], -1)
        if self.augment:
            k = random.randrange(4)
            x, y = np.rot90(x, k).copy(), np.rot90(y, k).copy()
            if random.random() < .5:
                x, y = x[:, ::-1].copy(), y[:, ::-1].copy()
            if random.random() < .5:
                x, y = x[::-1].copy(), y[::-1].copy()
            if random.random() < .5:
                x = np.clip(x.astype(np.float32)*random.uniform(.9, 1.1) +
                            random.uniform(-10, 10), 0, 255)
        x = x.astype(np.float32)/255
        x = (x-MEAN)/STD
        return (torch.from_numpy(x.transpose(2, 0, 1)).float(),
                torch.from_numpy(y.transpose(2, 0, 1)).float(), name)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k:v.detach().clone() for k,v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            if self.shadow[k].is_floating_point(): self.shadow[k].lerp_(v,1-self.decay)
            else: self.shadow[k].copy_(v)
    @torch.no_grad()
    def swap_in(self, model):
        old={k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow); return old


@torch.no_grad()
def evaluate(model, dl):
    model.eval(); scores=None
    for x,y,_ in dl:
        with torch.amp.autocast('cuda'):
            p=torch.sigmoid(model(x.cuda(non_blocking=True))[:,:,128:384,128:384])
        b=p.cpu().numpy()>.5; t=y.numpy()>.5
        inter=(b&t).sum((2,3)); union=(b|t).sum((2,3))
        iou=np.divide(inter,union,out=np.ones_like(inter,dtype=float),where=union>0)
        if scores is None: scores=[[] for _ in range(iou.shape[1])]
        for c in range(iou.shape[1]): scores[c].extend(iou[:,c].tolist())
    per=[float(np.mean(v)) for v in scores]
    return float(np.mean(per)),per


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['wheat_rape','wheat','rape','rice'],required=True)
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--arch',default='unet')
    ap.add_argument('--encoder',default='mit_b3')
    ap.add_argument('--epochs',type=int,default=35)
    ap.add_argument('--batch_size',type=int,default=4)
    ap.add_argument('--acc',type=int,default=1)
    ap.add_argument('--lr',type=float,default=5e-5)
    ap.add_argument('--patience',type=int,default=12)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--data_root',default='/root/competition_data/public')
    ap.add_argument('--output_dir',default='/root/crop_segmentation/weights')
    ap.add_argument('--tag',default='mosaic')
    ap.add_argument('--cache',action='store_true')
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True
    data=Path(args.data_root)
    domain='wheat_rape' if args.mode in ('wheat_rape','wheat','rape') else 'rice'
    store=MosaicStore(data,domain,args.cache)
    tr=TrainMosaicDataset(data,'train',args.mode,store,True)
    va=TrainMosaicDataset(data,'val',args.mode,store,False)
    dl=DataLoader(tr,batch_size=args.batch_size,shuffle=True,num_workers=0,
                  pin_memory=True,drop_last=False)
    vl=DataLoader(va,batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=True)
    classes=2 if args.mode=='wheat_rape' else 1
    cfg=dict(arch=args.arch,encoder=args.encoder,classes=classes)
    model=build(cfg)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(state(ck)); model.cuda()
    criterion=MultiTaskLoss(bce=1,dice=1,lovasz=.5,focal_gamma=1,pos_weight=1.2)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    steps=math.ceil(len(dl)/args.acc)*args.epochs
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=args.lr*.02)
    scaler=torch.amp.GradScaler('cuda'); ema=EMA(model,.999)
    out=Path(args.output_dir); tag=f'{args.tag}_{args.mode}_{args.encoder}_{args.seed}'
    best=-1.; stale=0; hist=[]; opt.zero_grad(set_to_none=True)
    print('[config]',tag,'train',len(tr),'val',len(va),flush=True)
    for ep in range(args.epochs):
        model.train(); total=0.; t0=time.time()
        for j,(x,y,_) in enumerate(dl):
            x,y=x.cuda(non_blocking=True),y.cuda(non_blocking=True)
            with torch.amp.autocast('cuda'):
                pred=model(x)[:,:,128:384,128:384]
                loss=criterion(pred,y)/args.acc
            scaler.scale(loss).backward(); total+=loss.item()*args.acc
            if (j+1)%args.acc==0 or j+1==len(dl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                sched.step(); ema.update(model)
        raw=ema.swap_in(model); val,per=evaluate(model,vl)
        row=dict(epoch=ep+1,loss=total/len(dl),val_iou=val,class_iou=per,
                 lr=opt.param_groups[0]['lr']); hist.append(row)
        print(f'[ep {ep+1:03d}] loss={row["loss"]:.4f} val={val:.4f} '
              f'classes={per} time={time.time()-t0:.0f}s',flush=True)
        if val>best:
            best=val; stale=0
            torch.save(dict(model_state_dict=model.state_dict(),val_iou=val,
                            class_iou=per,config=cfg),out/f'{tag}_best.pth')
            print('[best]',best,flush=True)
        else: stale+=1
        model.load_state_dict(raw)
        with open(out/f'{tag}_history.json','w') as f: json.dump(hist,f,indent=2)
        if stale>=args.patience: break

if __name__=='__main__': main()
