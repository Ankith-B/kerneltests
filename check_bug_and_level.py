import importlib.util, numpy as np
spec = importlib.util.spec_from_file_location("refnp", "/home/claude/ref_mmdagg/mmdagg/np.py"); refnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refnp)
from kt.mmdagg import mmdagg
rng = np.random.default_rng(3)
X = rng.standard_normal((80, 3)); Y = rng.standard_normal((80, 3)) + np.r_[0.5, 0, 0]
_, rd = refnp.mmdagg(X, Y, B1=500, B2=500, seed=1, return_dictionary=True)
_, md = mmdagg(X, Y, B1=500, B2=500, seed=1, return_details=True)
r_p = np.array([rd[k]["p-value"] for k in rd if k.startswith("Single")])
print("Gaussian-kernel tests (11-20): reference reports", r_p[10:13], "... correct:", md["p_value"][10:13])
print("reported p-values identical to the laplace ones:", np.array_equal(r_p[10:], r_p[:10]), "| correct ones differ:", not np.array_equal(md["p_value"][10:], md["p_value"][:10]))
R = 300; rej = sum(mmdagg(rng.standard_normal((50, 3)), rng.standard_normal((50, 3)), B1=300, B2=300, B3=30, seed=int(s)) for s in rng.integers(0, 10**9, R))
print(f"level check, {R} null trials: rejection rate {rej/R:.3f} (alpha 0.05, 2se band to {0.05+2*np.sqrt(.0475/R):.3f})")
