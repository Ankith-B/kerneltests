"""Parity check: kt.mmdfuse vs the JAX reference (github.com/antoninschrab/mmdfuse).

JAX's RNG cannot be reproduced in NumPy, so the permutation indices are drawn with the
reference's own RNG calls (same key split + random.permutation) and passed to kt via the
private ``_permutations`` hook. The reference's fused permuted statistics are captured by
wrapping its ``logsumexp`` (run with jit disabled). Run with ``--float32`` to use JAX's default
single precision (decision/p-value only compared; statistics then agree only to ~1e-5).
"""
import sys, importlib.util
import numpy as np
F32 = "--float32" in sys.argv
import jax
if not F32:
    jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import random

spec = importlib.util.spec_from_file_location("refmmdfuse", "/home/claude/ref_mmdfuse/mmdfuse/mmdfuse.py")
refm = importlib.util.module_from_spec(spec); spec.loader.exec_module(refm)
from kt.mmdfuse import mmdfuse as mine

captured = {}
_lse = refm.logsumexp
def _capture(*a, **k):
    r = _lse(*a, **k); captured["all"] = np.asarray(r); return r
refm.logsumexp = _capture

KERNELS = [("laplace", "gaussian"), ("gaussian",), ("laplace",), ("imq",), ("rq",),
           ("matern_1.5_l1", "matern_2.5_l2"), ("matern_0.5_l2", "matern_3.5_l1", "matern_4.5_l2"),
           ("gaussian", "laplace", "imq")]
rng = np.random.default_rng(7); cases = mism = 0
rtol = 1e-6
for t in range(40):
    m = [30, 60, 100, 45][t % 4]; n = [30, 45, 100, 60, 20][t % 5]; d = [1, 3, 8][t % 3]
    shift, scale = [(0, 1), (0.6, 1), (0, 1.6), (0, 1), (0.3, 1)][t % 5]
    X = rng.standard_normal((m, d)); Y = rng.standard_normal((n, d)) * scale + shift
    kern = KERNELS[t % len(KERNELS)]
    kw = dict(alpha=[0.05, 0.1, 0.01][t % 3], kernels=kern, lambda_multiplier=[1, 0.5, 2][t % 3],
              number_bandwidths=[10, 5, 7][t % 3], number_permutations=[500, 999, 300][t % 3])
    key = random.PRNGKey(t)
    with jax.disable_jit():
        ro, rp = refm.mmdfuse(jnp.asarray(X), jnp.asarray(Y), key, kw["alpha"], kern, kw["lambda_multiplier"],
                              kw["number_bandwidths"], kw["number_permutations"], True)
    ref_all = captured["all"]
    # reproduce the reference's permutation draw from its own RNG
    mm, nn = max(m, n), min(m, n); B = kw["number_permutations"]
    _, sub = random.split(key)
    idx = np.asarray(random.permutation(sub, jnp.array([[i for i in range(mm + nn)]] * (B + 1)),
                                        axis=1, independent=True))[:B]
    mo, md = mine(X, Y, return_details=True, _permutations=idx, **kw)
    ref_stat = ref_all[-1]
    k = int(np.ceil((B + 1) * (1 - kw["alpha"]))) - 1
    ref_thr = np.sort(ref_all)[k]
    # the reference's p-value is float32 even under x64 (jnp.mean of a bool array)
    ok = int(ro) == mo and np.isclose(float(rp), md["p_value"], rtol=1e-6, atol=0)
    stats_ok = (np.allclose(ref_stat, md["statistic"], rtol=rtol, atol=0)
          and np.allclose(ref_thr, md["threshold"], rtol=rtol, atol=0)
          and np.allclose(ref_all[:-1], md["permuted_statistics"], rtol=rtol, atol=1e-12))
    ok = ok and (F32 or stats_ok)
    # decision rule consistency: p <= alpha  <=>  stat > threshold
    ok = ok and (mo == int(md["statistic"] > md["threshold"]))
    cases += 1; mism += not ok
    print(t, (m, n, d), kern, int(ro), mo, float(rp), md["p_value"], ok,
          f"maxrel={np.max(np.abs(ref_all - np.r_[md['permuted_statistics'], md['statistic']]) / np.abs(ref_all)):.1e}")
print(f"{cases} cases ({'float32' if F32 else 'float64'} reference): mismatches = {mism}")
