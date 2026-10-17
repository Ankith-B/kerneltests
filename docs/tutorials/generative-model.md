# Checking a generative model

Suppose a generative model is meant to sample from a reference distribution, here the 2-D standard normal. Our *slightly wrong* model has the right mean and nearly the right variance, but its samples are heavier-tailed (Student-t, 5 degrees of freedom, rescaled to unit variance). A per-coordinate std check barely tells them apart.

!!! note "Speed"
    These examples use reduced resampling counts (`B1=500, B2=500` or `number_permutations=500`) so they run in a few seconds. The library defaults are 2000; use those for real analyses. All output below was produced by running the code exactly as shown (kerneltests 0.1.0.dev0, NumPy 2.x, scikit-learn 1.9).

- **Score known**: if you can evaluate \(\nabla_x \log p(x)\) for the reference (for \(N(0, I)\) it is \(-x\)), use `kt.goodness_of_fit` with KSDAgg. No reference samples are needed.
- **Only samples**: if the reference is available only as data, use `kt.two_sample` between model samples and reference samples.

```python
import numpy as np
import kt

# Reference distribution: 2-D standard normal, whose score is grad log p(x) = -x.
# "Generative model": a slightly wrong model -- correct mean, but its samples
# are heavier-tailed (Student-t with 5 degrees of freedom, rescaled to unit variance).
rng = np.random.default_rng(0)
n, d, df = 500, 2, 5
good = rng.standard_normal((n, d))
bad = rng.standard_t(df, size=(n, d)) / np.sqrt(df / (df - 2))
print("sample std  good: %s  bad: %s" % (good.std(0).round(2), bad.std(0).round(2)))

# 1) Score is known: goodness-of-fit with KSDAgg.
for name, S in [("good model", good), ("bad model", bad)]:
    r = kt.goodness_of_fit(S, score_X=-S, method="ksdagg", B1=500, B2=500)
    print(f"ksdagg    {name:10s} reject={r.reject}  "
          f"bandwidths rejecting: {int(r.details['reject'].sum())}/{len(r.details['reject'])}")

# 2) Only samples exist: two-sample test against reference draws.
ref = np.random.default_rng(1).standard_normal((n, d))
for name, S in [("good model", good), ("bad model", bad)]:
    r = kt.two_sample(S, ref, method="mmdagg", B1=500, B2=500)
    print(f"mmdagg    {name:10s} reject={r.reject}")
```

Output:

```text
sample std  good: [0.98 0.97]  bad: [0.99 0.92]
ksdagg    good model reject=False  bandwidths rejecting: 0/10
ksdagg    bad model  reject=True  bandwidths rejecting: 7/10
mmdagg    good model reject=False
mmdagg    bad model  reject=True
```

Both routes accept the correct model and reject the heavy-tailed one at \(\alpha = 0.05\). A single run is one draw. Non-rejection of the good model is what level-\(\alpha\) control predicts most of the time, not a proof that it is correct.
