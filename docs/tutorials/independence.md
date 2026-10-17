# Testing independence of features

Pearson correlation measures only *linear* association. For \(y = x^2 + \varepsilon\) with \(x\) symmetric around 0, \(y\) is a function of \(x\) plus noise, yet the correlation is close to zero. A kernel independence test (HSICAggInc) can detect any kind of dependence.

!!! note "Speed"
    These examples use reduced resampling counts (`B1=500, B2=500` or `number_permutations=500`) so they run in a few seconds. The library defaults are 2000; use those for real analyses. All output below was produced by running the code exactly as shown (kerneltests 0.1.0.dev0, NumPy 2.x, scikit-learn 1.9).

```python
import numpy as np
from scipy.stats import pearsonr
import kt

rng = np.random.default_rng(0)
n = 500
x = rng.uniform(-1, 1, n)
y_dep = x**2 + 0.1 * rng.standard_normal(n)     # nonlinear dependence
y_ind = rng.uniform(-1, 1, n) ** 2 + 0.1 * rng.standard_normal(n)  # independent of x

for name, y in [("y = x^2 + noise", y_dep), ("independent y", y_ind)]:
    r, p = pearsonr(x, y)
    res = kt.independence(x[:, None], y[:, None], B1=500, B2=500)
    print(f"{name:16s} Pearson r={r:+.3f} (p={p:.3f})  HSICAggInc reject={res.reject}")
```

Output:

```text
y = x^2 + noise  Pearson r=+0.079 (p=0.077)  HSICAggInc reject=True
independent y    Pearson r=+0.051 (p=0.256)  HSICAggInc reject=False
```

Pearson's test does not reject in either case (sample \(r \approx 0.08\), p = 0.077 for the dependent pair). HSICAggInc rejects for the quadratic dependence and not for the independent pair. Inputs to `kt.independence` are 2-D arrays of shape `(n, d)`, hence `x[:, None]`.
