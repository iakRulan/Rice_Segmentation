import torch
for f in ['best_rice.pth', 'best_wheat_rape.pth', 'final_rice.pth', 'final_wheat_rape.pth']:
    ck = torch.load('/root/crop_segmentation/weights/' + f, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    if any(k.startswith('backbone.') for k in sd):
        sd = {k[9:]: v for k, v in sd.items()}
    keys = list(sd.keys())
    sig = []
    for k in keys:
        if k.split('.')[0] not in [x.split('.')[0] for x in sig]:
            sig.append(k)
    first = [k for k in keys if not any(p in k for p in ['norm', 'running', 'num_batches'])]
    heads = []
    for k in first:
        parts = k.split('.')
        if len(parts) >= 2 and parts[0] not in heads:
            heads.append(parts[0] + '.' + parts[1])
    print(f, 'val_iou=', ck.get('val_iou', '?'), 'heads=', heads[:12])
