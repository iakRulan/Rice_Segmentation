import torch, sys
sys.path.insert(0, '/root/crop_segmentation')
import segmentation_models_pytorch as smp

def try_load(path, arch, enc, classes):
    try:
        m = smp.UnetPlusPlus(encoder_name=enc, encoder_weights=None, in_channels=3, classes=classes, activation=None) if arch == 'unetpp' else \
            smp.Unet(encoder_name=enc, encoder_weights=None, in_channels=3, classes=classes, activation=None) if arch == 'unet' else \
            smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=None, in_channels=3, classes=classes, activation=None)
        ck = torch.load(path, map_location='cpu', weights_only=False)
        sd = ck.get('model_state_dict', ck)
        if any(k.startswith('backbone.') for k in sd):
            sd = {k[9:]: v for k, v in sd.items()}
        m.load_state_dict(sd)
        return True
    except Exception as e:
        return False, str(e)[:60]

for f, cls in [('best_wheat_rape.pth', 2), ('best_rice.pth', 1), ('final_wheat_rape.pth', 2), ('final_rice.pth', 1)]:
    p = '/root/crop_segmentation/weights/' + f
    res = {}
    for arch in ['unet', 'unetpp']:
        ok = try_load(p, arch, 'efficientnet-b3', cls)
        res[arch] = ok if isinstance(ok, bool) else ok
    print(f, res)
