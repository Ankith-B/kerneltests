"""Benchmark MMDAgg: reference NumPy vs kt numpy (fast) vs kt jax (CPU). n in {500,...,4000}, d=10, B1=B2=500.
Median of 3 timed runs after one warm-up; for JAX the warm-up (incl. jit compile) is reported separately."""
import importlib.util, json, statistics, subprocess, sys, time
import numpy as np
sys.path.insert(0, ".")
from kt.mmdagg import mmdagg
spec = importlib.util.spec_from_file_location("refnp", "/home/claude/ref_mmdagg/mmdagg/np.py")
refnp = importlib.util.module_from_spec(spec); spec.loader.exec_module(refnp)
import jax

def timeit(f):
    t0 = time.perf_counter(); f(); warm = time.perf_counter() - t0
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return warm, statistics.median(ts)

rows = []
for n in (500, 1000, 2000, 4000):
    rng = np.random.default_rng(n); X = rng.standard_normal((n, 10)); Y = rng.standard_normal((n, 10)) + 0.05
    kw = dict(B1=500, B2=500, seed=1)
    _, tr = timeit(lambda: refnp.mmdagg(X, Y, **kw))
    _, tn = timeit(lambda: mmdagg(X, Y, backend="numpy", **kw))
    wj, tj = timeit(lambda: mmdagg(X, Y, backend="jax", **kw))
    assert refnp.mmdagg(X, Y, **kw) == mmdagg(X, Y, backend="jax", **kw)
    rows.append(dict(n=n, ref=tr, numpy=tn, jax=tj, jax_first_call=wj))
    print(rows[-1], flush=True)
print("jax", jax.__version__, jax.devices())
print(json.dumps(rows))
print(subprocess.run(["lscpu"], capture_output=True, text=True).stdout)
