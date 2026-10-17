import sys, time, numpy as np
import importlib.util; spec = importlib.util.spec_from_file_location("refnp", "/home/claude/ref_mmdagg/mmdagg/np.py"); refnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refnp)
ref = refnp.mmdagg
from kt.mmdagg import mmdagg as mine
rng = np.random.default_rng(0); mism = 0; cases = 0; t_ref = t_mine = 0
for trial in range(24):
    m = 60; n = 60 if trial % 2 == 0 else 45          # wild bootstrap and permutation paths
    shift = [0.0, 0.3, 0.8][trial % 3]
    kernel = ["laplace_gaussian", "gaussian", "laplace"][trial % 3]
    X = rng.standard_normal((m, 4)); Y = rng.standard_normal((n, 4)) + shift
    a = time.time(); r_out, r_d = ref(X, Y, kernel=kernel, B1=500, B2=500, B3=50, seed=trial, return_dictionary=True); t_ref += time.time() - a
    a = time.time(); m_out, m_d = mine(X, Y, kernel=kernel, B1=500, B2=500, B3=50, seed=trial, return_details=True); t_mine += time.time() - a
    cases += 1
    keys = [k for k in r_d if k.startswith("Single")]
    r_mmd = np.array([r_d[k]["MMD"] for k in keys]); r_q = np.array([r_d[k]["MMD quantile"] for k in keys])
    ok = (r_out == m_out) and np.allclose(r_mmd, m_d["mmd"], rtol=1e-10) and np.allclose(r_q, m_d["quantile"], rtol=1e-10)
    mism += not ok
print(f"{cases} cases (both bootstrap types, 3 kernels, null and alternatives): mismatches = {mism}")
print(f"time: reference {t_ref:.2f}s, ours {t_mine:.2f}s")
