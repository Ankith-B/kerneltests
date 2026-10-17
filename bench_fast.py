import importlib.util, time, numpy as np
spec = importlib.util.spec_from_file_location("refnp", "/home/claude/ref_mmdagg/mmdagg/np.py"); refnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refnp)
from kt.mmdagg import mmdagg
rng = np.random.default_rng(0); agree = 0; T = 40; maxrel = 0
for t in range(T):  # agreement of fast path with reference
    X = rng.standard_normal((60, 3)); Y = rng.standard_normal((60, 3)) + rng.choice([0, .3, .7])
    ro, rd = refnp.mmdagg(X, Y, B1=400, B2=400, B3=40, seed=t, return_dictionary=True)
    mo, md = mmdagg(X, Y, B1=400, B2=400, B3=40, seed=t, return_details=True)
    keys = [k for k in rd if k.startswith("Single")]
    maxrel = max(maxrel, np.max(np.abs(np.array([rd[k]["MMD"] for k in keys]) - md["mmd"]) / np.abs(md["mmd"])))
    agree += ro == mo
print(f"fast path: decisions agree {agree}/{T}; max relative diff of statistics {maxrel:.1e}")
for n in (500, 1000, 2000):
    X = rng.standard_normal((n, 10)); Y = rng.standard_normal((n, 10))
    a = time.time(); refnp.mmdagg(X, Y, B1=500, B2=500, B3=50); tr = time.time() - a
    a = time.time(); mmdagg(X, Y, B1=500, B2=500, B3=50); tf = time.time() - a
    print(f"n={n}: reference {tr:.2f}s, ours (fast) {tf:.2f}s, speedup {tr/tf:.1f}x")
