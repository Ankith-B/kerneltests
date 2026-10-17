import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location("refksd", "/home/claude/ref_ksdagg/ksdagg/np.py")
refm = importlib.util.module_from_spec(spec); spec.loader.exec_module(refm)
from kt.ksdagg import ksdagg as mine
rng = np.random.default_rng(1); mism = cases = nbug = 0
for t in range(30):
    m = [40, 80, 150][t % 3]; d = [1, 2, 5][(t // 3) % 3]
    shift, scale = [(0, 1), (0.5, 1), (0, 1.5), (0.2, 0.8), (0, 1)][t % 5]
    X = rng.standard_normal((m, d)) * scale + shift; S = -X
    wt = ["uniform", "decreasing", "increasing", "centred"][t % 4]
    nb = [10, 5, 7][t % 3]
    kw = dict(number_bandwidths=nb, weights_type=wt, B1=300 + t, B2=250, B3=50, seed=t)
    if t % 6 == 5: kw["bandwidths"] = np.linspace(0.3, 3, nb)
    if t % 7 == 6:  # parametric with precomputed null samples, shape (N, B)
        kw["approx_type"] = "parametric"; kw["bandwidths"] = np.linspace(0.5, 4, nb)
        kw["B1"] = rng.standard_normal((nb, 200)) * 0.01; kw["B2"] = rng.standard_normal((nb, 150)) * 0.01
    try:
        ro, rd = refm.ksdagg(X, S, return_dictionary=True, **kw); bug = False
    except AssertionError:  # reference bug: p<=u*w vs ksd>quantile disagree by fp rounding
        orig = np.testing.assert_array_equal; np.testing.assert_array_equal = lambda *a, **k: None
        try: ro, rd = refm.ksdagg(X, S, return_dictionary=True, **kw)
        finally: np.testing.assert_array_equal = orig
        bug = True; nbug += 1
    mo, md = mine(X, S, return_details=True, **kw)
    keys = [k for k in rd if k.startswith("Single")]
    g = lambda f: np.array([rd[k][f] for k in keys])
    ref_rej = g("KSD") > g("KSD quantile")  # the reference's quantile-based decision
    ok = (int(ref_rej.any()) == mo and (bug or ro == mo) and np.array_equal(ref_rej, md["reject"]) and np.allclose(g("KSD"), md["ksd"], rtol=1e-10, atol=0)
          and np.allclose(g("KSD quantile"), md["quantile"], rtol=1e-10, atol=0)
          and np.array_equal(g("p-value"), md["p_value"]) and np.allclose(g("p-value threshold"), md["threshold"], rtol=1e-12)
          and np.allclose(g("Bandwidth"), md["bandwidth"], rtol=1e-12))
    cases += 1; mism += not ok; print(t, ro, mo, ok, "REF-ASSERT-BUG" if bug else "")
print(f"{cases} cases: mismatches = {mism}; reference internal AssertionError in {nbug} case(s)")
