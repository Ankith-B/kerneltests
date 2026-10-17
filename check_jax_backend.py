"""Parity of backend="jax" vs backend="numpy" (fast path) over the 152-case sweep of check_mmdagg_full.py."""
import itertools, numpy as np
from kt.mmdagg import mmdagg
kernels = ["gaussian","laplace","imq"]+[f"matern_{i}.5_l{j}" for i in range(5) for j in (1,2)]+["laplace_gaussian","gaussian_laplace","all_matern_l1","all_matern_l2","all_matern_l1_l2","all"]
weights = ["uniform","decreasing","increasing","centred"]
rng = np.random.default_rng(5); cases = mism = 0; worst = 0.0
for k, w in itertools.product(kernels, weights):
    for nb in (10, 7):
        n2 = 30 if cases % 2 else 24
        X = rng.standard_normal((30, 3)); Y = rng.standard_normal((n2, 3)) + rng.choice([0, .5])
        kw = dict(kernel=k, weights_type=w, number_bandwidths=nb, B1=200, B2=200, B3=30, seed=cases, return_details=True)
        no, nd = mmdagg(X, Y, backend="numpy", **kw)
        jo, jd = mmdagg(X, Y, backend="jax", **kw)
        ok = (no == jo and (nd["reject"] == jd["reject"]).all()
              and np.allclose(nd["mmd"], jd["mmd"], rtol=1e-9, atol=0) and np.allclose(nd["quantile"], jd["quantile"], rtol=1e-9, atol=0)
              and np.array_equal(nd["threshold"], jd["threshold"]) and np.array_equal(nd["p_value"], jd["p_value"]))
        worst = max(worst, np.max(np.abs(nd["mmd"] - jd["mmd"]) / np.abs(nd["mmd"])))
        cases += 1; mism += not ok
        if not ok: print("MISMATCH", k, w, nb)
print(f"{cases} cases: mismatches = {mism}; max relative stat difference = {worst:.2e}")
