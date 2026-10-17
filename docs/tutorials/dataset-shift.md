# Detecting dataset shift

A model trained on one feature distribution can degrade silently when the deployment data drifts. A two-sample test asks directly: are the training features and the deployment features drawn from the same distribution?

!!! note "Speed"
    These examples use reduced resampling counts (`B1=500, B2=500` or `number_permutations=500`) so they run in a few seconds. The library defaults are 2000; use those for real analyses. All output below was produced by running the code exactly as shown (kerneltests 0.1.0.dev0, NumPy 2.x, scikit-learn 1.9).

We simulate a classic covariate shift: deployment rows are *selected* with probability increasing in feature 0 (selection bias), while no feature value is modified.

```python
import numpy as np
from sklearn.datasets import make_classification
import kt

X, y = make_classification(n_samples=2000, n_features=5, n_informative=3,
                           n_redundant=0, random_state=0)
rng = np.random.default_rng(0)
idx = rng.permutation(len(X))
train, pool = X[idx[:300]], X[idx[300:]]

# Deployment A: same distribution as training (no shift).
deploy_same = pool[:300]
# Deployment B: covariate shift by selection bias -- rows with a large
# feature 0 are kept with higher probability (no feature values are altered).
p = 1 / (1 + np.exp(-1.5 * pool[:, 0]))
keep = rng.random(len(pool)) < p
deploy_shift = pool[keep][:300]
print("mean of feature 0: train %.2f, shifted deployment %.2f"
      % (train[:, 0].mean(), deploy_shift[:, 0].mean()))

for name, D in [("no shift", deploy_same), ("shifted", deploy_shift)]:
    r1 = kt.two_sample(train, D, method="mmdagg", B1=500, B2=500)
    r2 = kt.two_sample(train, D, method="mmdfuse", number_permutations=500)
    print(f"{name:9s} mmdagg reject={r1.reject}  mmdfuse reject={r2.reject} "
          f"(p={r2.details['p_value']:.3f})")
```

Output:

```text
mean of feature 0: train 0.41, shifted deployment 1.06
no shift  mmdagg reject=False  mmdfuse reject=False (p=0.473)
shifted   mmdagg reject=True  mmdfuse reject=True (p=0.002)
```

Both tests stay silent on the unshifted deployment set and reject on the shifted one. MMD-FUSE also reports a p-value; MMDAgg reports a decision plus per-kernel diagnostics.

## Which bandwidth detected it?

`res.details` lists every kernel/bandwidth MMDAgg aggregated over, with its p-value and individual decision (made at the multiplicity-corrected level, so a small p-value alone is not a rejection):

```python
res = kt.two_sample(train, deploy_shift, method="mmdagg", B1=500, B2=500)
d = res.details
for k, bw, pv, rej in zip(d["kernel"], d["bandwidth"], d["p_value"], d["reject"]):
    print(f"{k:9s} bandwidth={bw:7.3f}  p={pv:.3f}  reject={bool(rej)}")
```

```text
laplace   bandwidth=  0.277  p=0.092  reject=False
laplace   bandwidth=  0.481  p=0.006  reject=True
laplace   bandwidth=  0.837  p=0.002  reject=True
laplace   bandwidth=  1.456  p=0.002  reject=True
laplace   bandwidth=  2.534  p=0.002  reject=True
laplace   bandwidth=  4.407  p=0.002  reject=True
laplace   bandwidth=  7.667  p=0.002  reject=True
laplace   bandwidth= 13.337  p=0.002  reject=True
laplace   bandwidth= 23.201  p=0.002  reject=True
laplace   bandwidth= 40.360  p=0.002  reject=True
gaussian  bandwidth=  0.135  p=0.172  reject=False
gaussian  bandwidth=  0.234  p=0.291  reject=False
gaussian  bandwidth=  0.408  p=0.150  reject=False
gaussian  bandwidth=  0.711  p=0.004  reject=True
gaussian  bandwidth=  1.239  p=0.002  reject=True
gaussian  bandwidth=  2.157  p=0.002  reject=True
gaussian  bandwidth=  3.757  p=0.002  reject=True
gaussian  bandwidth=  6.544  p=0.002  reject=True
gaussian  bandwidth= 11.397  p=0.002  reject=True
gaussian  bandwidth= 19.849  p=0.002  reject=True
```

The smallest bandwidths do not reject; medium and large ones do. That fits the cause: the shift is a smooth change in the location of feature 0, a coarse-scale difference, not a fine local one. Treat this as a qualitative hint, not a formal localisation of the shift.
