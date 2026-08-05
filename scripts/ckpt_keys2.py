import torch
for f in ['best_wheat_rape.pth', 'best_rice.pth', 'final_wheat_rape.pth',
          's2_wheat_rape_mit_b3_42_best.pth', 's2_wheat_rape_mit_b2_42_best.pth',
          's2_wheat_rape_efficientnetb3_43_best.pth', 's2_wheat_rape_efficientnetb3_42_best.pth',
          's2_rice_mit_b2_42_best.pth', 's2_rice_mit_b3_42_best.pth']:
    ck = torch.load('/root/crop_segmentation/weights/' + f, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck)
    if any(k.startswith('backbone.') for k in sd):
        sd = {k[9:]: v for k, v in sd.items()}
    tops = []
    for k in sd:
        t = k.split('.')[0]
        if t not in tops:
            tops.append(t)
    # decoder signature
    dec_sig = ''
    for k in sd:
        if k.startswith('decoder.'):
            dec_sig = k.split('.')[1] if len(k.split('.')) > 1 else '?'
            if k.startswith('decoder.aspp'):
                dec_sig = 'aspp(deeplab)'
            break
    print(f'{f:48s} val={ck.get("val_iou",0):.4f} decoder.sig={dec_sig} tops={tops}')
