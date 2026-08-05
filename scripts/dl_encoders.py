"""Download pretrained encoder weights for several strong encoders (background)."""
import segmentation_models_pytorch as smp

ENCODERS = ['efficientnet-b5', 'mit_b2', 'timm-efficientnet-b5', 'mit_b3']
for enc in ENCODERS:
    try:
        print(f'=== downloading {enc} ===', flush=True)
        m = smp.Unet(encoder_name=enc, encoder_weights='imagenet', classes=2)
        print(f'  {enc}: OK', flush=True)
        del m
    except Exception as e:
        print(f'  {enc}: FAIL {type(e).__name__}: {str(e)[:150]}', flush=True)
print('ALL DONE', flush=True)
