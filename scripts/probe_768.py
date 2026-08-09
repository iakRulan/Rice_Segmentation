"""Probe timing + peak for mit_b3/mit_b5 Unet @768. Reports s/step (fwd+bwd).
"""
import time
import torch

def run(encoder, bs, dtype=torch.bfloat16, in_ch=9, size=768, steps=5):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name=encoder, encoder_weights=None,
                 in_channels=in_ch, classes=3, activation=None).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(bs, in_ch, size, size).cuda()
    y = (torch.rand(bs, 3, 256, 256) > 0.5).float().cuda()
    m.train()
    # warmup
    with torch.amp.autocast('cuda', dtype=dtype):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            m(x)[..., 256:512, 256:512], y)
    opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        with torch.amp.autocast('cuda', dtype=dtype):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                m(x)[..., 256:512, 256:512], y)
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1024**3
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f'{encoder} bs={bs}: {dt*1000:.0f} ms/step peak={peak:.2f}GB params={n:.1f}M')
    del m, x, y, loss, opt
    torch.cuda.empty_cache()

run('mit_b3', 2)
run('mit_b5', 1)
