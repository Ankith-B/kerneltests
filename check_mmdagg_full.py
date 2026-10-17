import importlib.util, itertools, numpy as np
spec = importlib.util.spec_from_file_location("refnp", "/home/claude/ref_mmdagg/mmdagg/np.py"); refnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refnp)
from kt.mmdagg import mmdagg
kernels = ["gaussian","laplace","imq"]+[f"matern_{i}.5_l{j}" for i in range(5) for j in (1,2)]+["laplace_gaussian","gaussian_laplace","all_matern_l1","all_matern_l2","all_matern_l1_l2","all"]
weights = ["uniform","decreasing","increasing","centred"]
rng = np.random.default_rng(5); cases = mism = 0
for k, w in itertools.product(kernels, weights):
    for nb in (10, 7):                       # even and odd (centred weights differ)
        n2 = 30 if cases % 2 else 24
        X = rng.standard_normal((30, 3)); Y = rng.standard_normal((n2, 3)) + rng.choice([0, .5])
        ro, rd = refnp.mmdagg(X, Y, kernel=k, weights_type=w, number_bandwidths=nb, B1=200, B2=200, B3=30, seed=cases, return_dictionary=True)
        mo, md = mmdagg(X, Y, kernel=k, weights_type=w, number_bandwidths=nb, B1=200, B2=200, B3=30, seed=cases, return_details=True, fast=False)
        keys = [x for x in rd if x.startswith("Single")]
        ok = ro == mo and np.allclose([rd[x]["MMD"] for x in keys], md["mmd"], rtol=1e-10) and np.allclose([rd[x]["MMD quantile"] for x in keys], md["quantile"], rtol=1e-10)
        cases += 1; mism += not ok
        if not ok: print("MISMATCH", k, w, nb)
print(f"{cases} cases (19 kernel options x 4 weightings x 2 bandwidth counts, both resampling schemes): mismatches = {mism}")
