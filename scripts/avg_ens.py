"""Average two prob npz files -> out npz."""
import sys, numpy as np
a, b, w, out = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
da, db = np.load(a), np.load(b)
assert set(da.files) == set(db.files), "files mismatch"
res = {}
for k in da.files:
    res[k] = (w * da[k].astype(np.float32) + (1 - w) * db[k].astype(np.float32)).astype(np.float16)
np.savez(out, **res)
print("saved", out, len(res))
