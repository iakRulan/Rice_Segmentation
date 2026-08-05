from pathlib import Path
from multiprocessing import Pool
import numpy as np
from PIL import Image

SRC = Path('/root/competition_data/public')
OUT = Path('/root/yolo_data')
G = None  # global context set in worker init

def _init(cfg):
    global G
    G = cfg

def _one(f):
    stem = f.stem
    mode = G['mode']; src_split = G['src_split']
    out_img = G['out_img']; out_msk = G['out_msk']
    im = np.asarray(Image.open(f).convert('RGB'))
    if mode == 'wheat_rape':
        w = np.asarray(Image.open(SRC / src_split / 'label' / 'wheat' / f.name))
        r = np.asarray(Image.open(SRC / src_split / 'label' / 'rape' / f.name))
        mask = np.zeros(im.shape[:2], dtype=np.uint8)
        mask[r > 0] = 2
        mask[w > 0] = 1
    else:
        r = np.asarray(Image.open(SRC / src_split / 'label' / 'rice' / f.name))
        mask = (r > 0).astype(np.uint8)
    Image.fromarray(im).save(out_img / (stem + '.png'))
    Image.fromarray(mask).save(out_msk / (stem + '.png'))
    return (stem, int((mask > 0).sum()), int(mask.max()))

def build_split(src_split, out_split, mode, workers=16):
    img_dir = SRC / src_split / 'image' / ('wheat_rape' if mode == 'wheat_rape' else 'rice')
    out_img = OUT / mode / out_split / 'images'
    out_msk = OUT / mode / out_split / 'masks'
    out_img.mkdir(parents=True, exist_ok=True)
    out_msk.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in img_dir.glob('*.png'))
    print(f'[{mode}/{out_split}] {len(files)} images', flush=True)
    cfg = {'mode': mode, 'src_split': src_split, 'out_img': out_img, 'out_msk': out_msk}
    with Pool(workers, initializer=_init, initargs=(cfg,)) as pool:
        stats = pool.map(_one, files)
    fg = sum(s[1] for s in stats)
    print(f'[{mode}/{out_split}] fg px={fg} avg={fg/len(files):.0f} max_class={max(s[2] for s in stats)}', flush=True)

def main():
    for mode in ['wheat_rape', 'rice']:
        build_split('train', 'train', mode)
        build_split('val', 'val', mode)
    for mode, nc, names in [('wheat_rape', 2, {0: 'wheat', 1: 'rape'}), ('rice', 1, {0: 'rice'})]:
        y = OUT / mode / 'data.yaml'
        y.write_text(f"path: {OUT / mode}\ntrain: images/train\nval: images/val\nmasks_dir: masks\nnc: {nc}\nnames: {names}\n")
        print('wrote', y, flush=True)

if __name__ == '__main__':
    main()