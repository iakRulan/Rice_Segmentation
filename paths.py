"""Local path + config helpers for the crop-seg pipeline.

Everything hardcoded to /root/... on the server is centralized here so the
same scripts run on Windows.  Usage:

    from paths import load_config, DATA, VAL_IMG, ...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / 'data' / 'public'       # extracted dataset
WEIGHTS = ROOT / 'weights'            # downloaded model checkpoints
OUT = ROOT / 'outputs'                # all generated artifacts
CONFIGS = ROOT / 'configs'

TRAIN_IMG = DATA / 'train' / 'image'
TRAIN_LBL = DATA / 'train' / 'label'
VAL_IMG = DATA / 'val' / 'image'
VAL_LBL = DATA / 'val' / 'label'
TESTA_IMG = DATA / 'testA' / 'image'

PREDS = OUT / 'preds'
MASKS = OUT / 'masks'
LOGS = OUT / 'logs'
CKPT = OUT / 'ckpt'                   # locally-trained checkpoints

for d in (PREDS, MASKS, LOGS, CKPT):
    d.mkdir(parents=True, exist_ok=True)


def local_weight(path: str) -> str:
    """Translate a path to the local absolute file.
    - /root/crop_segmentation/weights/<f> -> <ROOT>/weights/<f>
    - otherwise treated as relative to ROOT (e.g. outputs/ckpt/x.pth)
    """
    if path.startswith('/root/'):
        fname = path.rsplit('/', 1)[-1]
        return str(WEIGHTS / fname)
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def load_config(name: str):
    """Load a config json and rewrite weight paths to local."""
    cfg = json.load(open(CONFIGS / name))
    for c in cfg:
        c['weight'] = local_weight(c['weight'])
    return cfg
