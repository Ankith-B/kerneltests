# Bug reports for the reference implementations (ready to file)

Found while porting to `kerneltests` with bit-exact parity checks. All reproduce on the current
`main` of each repository (NumPy versions, `*/np.py`); please check the JAX versions too.
None changes a correct accept/reject decision except where noted (crashes return no decision).

---

## antoninschrab/mmdagg — wrong per-test p-values reported with multiple kernels

**File:** `mmdagg/np.py`, line 352: `reject_dictionary[index]["p-value"] = p_vals[i]`
should be `p_vals[idx]` (`idx = number_bandwidths * j + i`).

**Effect:** with the default `kernel="laplace_gaussian"`, the reported p-values of the Gaussian-kernel
single tests are copies of the Laplace ones. The rejection decision is unaffected.

```python
import numpy as np
from mmdagg.np import mmdagg
rng = np.random.default_rng(3)
X = rng.standard_normal((80, 3)); Y = rng.standard_normal((80, 3)) + np.r_[0.5, 0, 0]
_, d = mmdagg(X, Y, B1=500, B2=500, seed=1, return_dictionary=True)
p = [d[k]["p-value"] for k in d if k.startswith("Single")]
print(p[:10] == p[10:])   # True: Gaussian p-values duplicate Laplace ones
```

---

## antoninschrab/ksdagg — AssertionError on ordinary input

**File:** `ksdagg/np.py`, the final `np.testing.assert_array_equal(reject_p_vals, reject_ksd_vals)`.
The two rules (`p-value <= u*w` and `KSD > quantile`) are mathematically equivalent but can disagree by
floating-point rounding, so the function raises instead of returning.

```python
import numpy as np
from ksdagg.np import ksdagg
rng = np.random.default_rng(1)
for t in range(3):
    X = rng.standard_normal(([40, 80, 150][t], 1)) * [1, 1., 1.5][t] + [0, .5, 0][t]
ksdagg(X, -X, number_bandwidths=7, weights_type="increasing", B1=302, B2=250, B3=50, seed=2)
# AssertionError (here p = 1/303 = 0.0033003300330033004 vs threshold 0.00330033003300326)
```

**Suggested fix:** use a single rejection rule (e.g. `KSD > quantile`), or compare with a tolerance.

---

## antoninschrab/agginc — three issues

**1. `warnings` is never imported.** `agginc/np.py` imports only numpy, itertools, scipy.spatial, psutil
(lines 11–14), but lines 195 and 197 call `warnings.warn(...)`. Calling `agginc("hsic", ...)` with a
non-default `number_bandwidths`, or `agginc("mmd"/"ksd", ...)` with non-default
`hsic_collection_parameters`, raises `NameError: name 'warnings' is not defined`.

**2. HSIC bandwidth check tests the same list twice.** Line 215:
`assert len(bandwidths) == 2 and len(bandwidths[0]) > 0 and len(bandwidths[0]) > 0` — the last term
should be `len(bandwidths[1]) > 0`, so an empty Y-bandwidth list is not caught.

**3. `batch_size="auto"` allocates all memory at once.** Lines 694–695 size the batch from available RAM
divided by the size of a small array, which always yields a single batch. At n=2000, d=10 (B1=B2=500)
this tried to allocate about 9 GB and the process was killed by the out-of-memory handler; passing
`batch_size=100` works. Capping the per-batch memory (we use about 256 MB) fixes it.
