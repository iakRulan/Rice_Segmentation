import os, time, json, subprocess

PY = '/root/miniconda3/bin/python'
LOG = '/root/logs/watcher_wave1.log'

def log(msg):
    line = f'[{time.strftime("%m-%d %H:%M:%S")}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')

def run(cmd, timeout=None):
    log('RUN: ' + cmd)
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or '').strip(); err = (p.stderr or '').strip()
    if out: log('OUT: ' + out[-1500:])
    if err: log('ERR: ' + err[-1500:])
    return p.returncode, out

def training_running(mode, enc):
    ps = subprocess.run("ps -eo cmd | grep 'train_strong.py' | grep '" + mode + "' | grep '" + enc + "' | grep -v grep",
                        shell=True, capture_output=True, text=True).stdout
    return bool(ps.strip())

def wait_no_training(mode, enc):
    while training_running(mode, enc):
        time.sleep(60)
    log(f'{mode} {enc} training finished')

def exists(*paths):
    return all(os.path.exists(p) for p in paths)

log('=== watcher started ===')

# Step 1: wheat_rape eff-b3 43 -> 4-model ensemble
wait_no_training('wheat_rape', 'efficientnet-b3')
if not exists('/root/ens_multi_v3.npz', '/root/empty_wheat_v3clf.npy', '/root/empty_rape_v3clf.npy'):
    cfg = json.load(open('/root/cfg_multi_avail.json'))
    cfg.append({"arch": "unet", "encoder": "efficientnet-b3",
                "weight": "/root/crop_segmentation/weights/s2_wheat_rape_efficientnetb3_43_best.pth",
                "classes": 2})
    json.dump(cfg, open('/root/cfg_multi_v3.json', 'w'), indent=1)
    log('wrote cfg_multi_v3.json')
    run(f'{PY} /root/ens_gpu.py --task multi --configs /root/cfg_multi_v3.json --out /root/ens_multi_v3.npz --scales 256,288,320 --bs 4')
    run(f'{PY} /root/clf2.py --class_name wheat --channel 0 --val_preds /root/ens_multi_v3.npz --out /root/empty_wheat_v3clf.npy')
    run(f'{PY} /root/clf2.py --class_name rape --channel 1 --val_preds /root/ens_multi_v3.npz --out /root/empty_rape_v3clf.npy')
run(f'{PY} /root/eval_final_sweep.py --class_name wheat --preds /root/ens_multi_v3.npz --channel 0 --empty_preds /root/empty_wheat_v3clf.npy')
run(f'{PY} /root/eval_final_sweep.py --class_name rape --preds /root/ens_multi_v3.npz --channel 1 --empty_preds /root/empty_rape_v3clf.npy')

# Step 2: rice eff-b3 43 -> rice 4-model ensemble
wait_no_training('rice', 'efficientnet-b3')
if not exists('/root/ens_single_v3.npz'):
    cfg = json.load(open('/root/cfg_rice_avail.json'))
    cfg.append({"arch": "unet", "encoder": "efficientnet-b3",
                "weight": "/root/crop_segmentation/weights/s2_rice_efficientnetb3_43_best.pth",
                "classes": 1})
    json.dump(cfg, open('/root/cfg_rice_v3.json', 'w'), indent=1)
    log('wrote cfg_rice_v3.json')
    run(f'{PY} /root/ens_gpu.py --task single --configs /root/cfg_rice_v3.json --out /root/ens_single_v3.npz --scales 256,288,320 --bs 4')
run(f'{PY} /root/eval_final_sweep.py --class_name rice --preds /root/ens_single_v3.npz --channel 0')

# Step 3: launch wave2 diverse models
log('launching wave2')
run('cd /root && nohup bash /root/run_campaign_wave2_new.sh > /root/logs/wave2_wrapper.log 2>&1 &')
log('=== watcher done ===')