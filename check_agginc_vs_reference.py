import time, warnings, numpy as np
import importlib.util
spec = importlib.util.spec_from_file_location("refagg", "/home/claude/ref_agginc/agginc/np.py")
ref = importlib.util.module_from_spec(spec); spec.loader.exec_module(ref)
from kt.agginc import agginc, inc
from kt.mmdagg import mmdagg

rng = np.random.default_rng(0)


def data(t, n, d, alt):
    if t == "mmd":
        return rng.standard_normal((n, d)), rng.standard_normal((n + 7, d)) + 0.5 * alt
    if t == "hsic":
        X = rng.standard_normal((n, d)); Y = rng.standard_normal((n, 2))
        if alt: Y[:, 0] += 0.8 * np.sin(2 * X[:, 0])
        return X, Y
    X = rng.standard_normal((n, d)) + (0.6 if alt else 0.0)
    return X, -X  # score of N(0, I)


def ref_arrays(t, dct, nb):
    k = [f"Single test {i + 1}" for i in range(nb)]
    return (np.array([dct[s][t.upper()] for s in k]), np.array([dct[s][t.upper() + " quantile"] for s in k]),
            np.array([dct[s]["p-value"] for s in k]), np.array([dct[s]["Reject"] for s in k]))


mism = cases = 0
opts = []
for t in ("mmd", "hsic", "ksd"):
    for w in ("uniform", "decreasing", "increasing", "centred"):
        if t == "hsic" and w != "uniform": continue
        opts.append((t, dict(weights_type=w)))
    opts += [(t, dict(R=1)), (t, dict(R=500)), (t, dict(batch_size=None)), (t, dict(batch_size=37)), (t, dict(alpha=0.1, B1=300, B2=200, B3=20))]
    opts.append((t, dict(bandwidths=[np.array([0.5, 1.0]), np.array([0.3, 2.0, 4.0])] if t == "hsic" else np.array([0.3, 1.0, 3.0]))))
    if t == "mmd" or t == "ksd": opts.append((t, dict(number_bandwidths=4)))
opts += [("hsic", dict(hsic_collection_parameters=(3, -1, 1, -2, 0))), ("hsic", dict(hsic_collection_parameters=(2, 0, 3)))]

for c, (t, kw) in enumerate(opts):
    for alt in (0, 1):
        n, d = [(120, 3), (201, 1), (80, 5)][c % 3]
        X, Y = data(t, n, d, alt)
        ro, rd = ref.agginc(t, X, Y, seed=c, return_dictionary=True, **kw)
        mo, md = agginc(t, X, Y, seed=c, return_details=True, **kw)
        s, q, p, r = ref_arrays(t, rd, len(md["statistic"]))
        ok = (ro == mo and np.allclose(s, md["statistic"], rtol=1e-10, atol=0) and np.allclose(q, md["quantile"], rtol=1e-10, atol=0)
              and np.array_equal(p, md["p_value"]) and np.array_equal(r, md["reject"]))
        cases += 1; mism += not ok
        if not ok: print("MISMATCH agginc", t, kw, alt)
# inc
for c, t in enumerate(("mmd", "hsic", "ksd") * 3):
    for alt in (0, 1):
        X, Y = data(t, 100 + 13 * c, 2, alt)
        kw = {} if c < 3 else (dict(bandwidth=(0.7, 1.3) if t == "hsic" else 0.9) if c < 6 else dict(R=5, B=300, alpha=0.1))
        ro, rd = ref.inc(t, X, Y, seed=c, return_dictionary=True, **kw)
        mo, md = inc(t, X, Y, seed=c, return_details=True, **kw)
        ok = ro == mo and np.isclose(rd[t.upper()], md["statistic"], rtol=1e-10, atol=0) and np.isclose(rd[t.upper() + " quantile"], md["quantile"], rtol=1e-10, atol=0) and rd["p-value"] == md["p_value"]
        cases += 1; mism += not ok
        if not ok: print("MISMATCH inc", t, kw, alt)
print(f"{cases} cases: mismatches = {mism}")

# timing (reference needs an explicit batch_size: its "auto" OOMs here): n=2000, d=10 two-sample, B1=B2=500
X = rng.standard_normal((2000, 10)); Y = rng.standard_normal((2000, 10)) + 0.05
for name, f in (("reference agginc", lambda: ref.agginc("mmd", X, Y, B1=500, B2=500, batch_size=100)),
                ("ours agginc", lambda: agginc("mmd", X, Y, B1=500, B2=500)),
                ("kt.mmdagg (complete U)", lambda: mmdagg(X, Y, B1=500, B2=500))):
    a = time.time(); o = f(); print(f"{name}: {time.time() - a:.2f}s (output {o})")
