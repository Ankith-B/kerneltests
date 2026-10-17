# Quickstart

```bash
pip install -e .
```

## Two-sample test

```python
import numpy as np
import kt

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 2))
Y = rng.standard_normal((200, 2)) + 0.5
res = kt.two_sample(X, Y)
print(res)   # TestResult(reject=True, method='mmdagg', alpha=0.05)
if res:      # a TestResult is truthy exactly when it rejects
    print("distributions differ")
```

## Goodness-of-fit test

KSDAgg needs only the model's **score** \(\nabla_x \log p(x)\) evaluated at the sample. You never need the normalising constant. For \(N(0, I)\), the score is \(-x\).

```python
rng = np.random.default_rng(1)
X = rng.standard_normal((300, 2)) + 0.3
res = kt.goodness_of_fit(X, score_X=-X)
print(res.reject, int(res.details["reject"].sum()), "of", len(res.details["reject"]))
# True 10 of 10
```

## Low-level functions

`kt.mmdagg.mmdagg` and `kt.ksdagg.ksdagg` return `0` or `1`. With `return_details=True`, they return `(decision, details)` instead.
