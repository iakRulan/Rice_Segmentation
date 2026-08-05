old = "        self.tfs = tta_transforms([256, 288, 320])"
new = ("        _sc = os.environ.get('TTA_SCALES', '256,288,320')\n"
       "        self.tfs = tta_transforms([int(s) for s in _sc.split(',')])")
p = '/root/infer_ensemble.py'
src = open(p).read()
assert old in src, 'pattern not found'
src = src.replace(old, new)
open(p, 'w').write(src)
print('patched')
