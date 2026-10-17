# kerneltests

`kerneltests` (import name `kt`) gives you modern kernel hypothesis tests behind one small API: two-sample testing (**MMDAgg**, **MMDAggInc**, **MMD-FUSE**), goodness-of-fit testing (**KSDAgg**, **KSDAggInc**) and independence testing (**HSICAggInc**). Every port is checked against the original reference implementations by Schrab et al. and gives **bit-exact** agreement for the same seed (MMD-FUSE, whose reference is JAX, agrees to rtol 1e-6 given the same permutations) (identical decisions, statistics, quantiles, p-values and thresholds) when run with `fast=False`. The default `fast=True` path for equal sample sizes is 1.7–2.3× faster than the reference MMDAgg and agrees with it up to floating-point rounding (identical decisions in all tested cases; statistics within ~1e-12 relative). The ports also fix five confirmed bugs in the references (all listed in CHANGELOG.md):

1. **MMDAgg p-values.** With `kernel="laplace_gaussian"`, the reference reports Gaussian-kernel p-values that duplicate the Laplace ones, because `mmdagg/np.py` line 352 indexes `p_vals[i]` where it should use `p_vals[idx]`.
2. **KSDAgg internal assertion.** The reference can raise an `AssertionError` when its two equivalent rejection rules (`p-value <= threshold` and `KSD > quantile`) disagree because of floating-point rounding.
3. **AggInc missing import.** Non-default configuration can call `warnings.warn` without importing `warnings`, raising `NameError`.
4. **AggInc HSIC bandwidth validation.** The reference checks the X bandwidth list twice instead of also checking the Y bandwidth list, so an empty Y list is not rejected.
5. **AggInc automatic batch size.** The reference can size a batch using nearly all available RAM, causing excessive allocations or out-of-memory failures; the port caps automatic batch memory use.

## Install

```bash
pip install git+https://github.com/Ankith-B/kerneltests.git  # or, from a clone:
pip install -e .
```

You need Python 3.9 or newer, NumPy 1.22 or newer, and SciPy 1.8 or newer.

Optional JAX backend for MMDAgg (`mmdagg(X, Y, backend="jax")`, also accepted by `kt.two_sample`):

```bash
pip install -e .[jax]
```

It uses the same random draws as the NumPy path and gives identical decisions; see the *Backends* section of the user guide and `experiments/jax_backend/results.md` for timings.

## Quickstart

Two-sample test (are `X` and `Y` drawn from the same distribution?):

```python
import numpy as np
import kt

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 2))
Y = rng.standard_normal((200, 2)) + 0.5
res = kt.two_sample(X, Y)
print(res)          # TestResult(reject=True, method='mmdagg', alpha=0.05)
```

Goodness-of-fit test (does `X` come from N(0, I), whose score is `-x`?):

```python
rng = np.random.default_rng(1)
X = rng.standard_normal((300, 2)) + 0.3
res = kt.goodness_of_fit(X, score_X=-X)
print(res.reject, int(res.details["reject"].sum()), "of", len(res.details["reject"]))
# True 10 of 10
```

`res.details` holds diagnostics for each kernel and bandwidth (statistic, quantile, p-value, threshold, reject). See the [documentation](docs/index.md).

## Citing

If you use these tests, please cite the original papers:

```bibtex
@article{schrab2023mmdagg,
  author  = {Antonin Schrab and Ilmun Kim and M{\'e}lisande Albert and B{\'e}atrice Laurent and Benjamin Guedj and Arthur Gretton},
  title   = {{MMD} Aggregated Two-Sample Test},
  journal = {Journal of Machine Learning Research},
  year    = {2023}, volume = {24}, number = {194}, pages = {1--81},
  url     = {http://jmlr.org/papers/v24/21-1289.html}
}
@inproceedings{schrab2022ksdagg,
  author    = {Antonin Schrab and Benjamin Guedj and Arthur Gretton},
  title     = {{KSD} Aggregated Goodness-of-fit Test},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2022}
}
```

The ports are adapted from the MIT-licensed reference packages [antoninschrab/mmdagg](https://github.com/antoninschrab/mmdagg), [antoninschrab/ksdagg](https://github.com/antoninschrab/ksdagg) and [antoninschrab/mmdfuse](https://github.com/antoninschrab/mmdfuse).

## Testing and verification

```bash
pip install -e .[test]
pytest --cov=kt --cov-config=.coveragerc
```

The unit tests check power against clear alternatives, Type I error control under the null (Monte Carlo, within 3 standard errors), determinism for a fixed seed, input validation, and a regression test for the p-value bug. Parity with the references is checked separately against every MMDAgg kernel option, all four weightings, both resampling schemes, and KSDAgg wild-bootstrap and parametric modes. See [docs/verification.md](docs/verification.md). CI runs on Linux, macOS and Windows with Python 3.9 to 3.13.

## Roadmap

- GPU benchmarks for the JAX backend (implemented, untested on GPU)
- Fast path for the permutation (unequal sample size) MMDAgg calibration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every new port has to match its reference implementation exactly.

## License

MIT. See [LICENSE](LICENSE).
