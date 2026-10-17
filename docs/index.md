# kerneltests

`kerneltests` (import name `kt`) gives you modern kernel hypothesis tests behind one API:

| Problem | Entry point | Method |
|---|---|---|
| Two-sample: is \(P_X = P_Y\)? | `kt.two_sample(X, Y)` | MMDAgg (Schrab et al., JMLR 2023); MMDAggInc (linear time) |
| Goodness of fit: is \(X \sim p\)? | `kt.goodness_of_fit(X, score_X)` | KSDAgg (Schrab, Guedj, Gretton, NeurIPS 2022); KSDAggInc |
| Independence: is \(X \perp Y\)? | `kt.independence(X, Y)` | HSICAggInc (Schrab, Kim, Guedj, Gretton, NeurIPS 2022) |

All entry points return a [`TestResult`](api.md). For the same seed, every port agrees bit-exactly with the original reference implementation (for MMDAgg, with `fast=False`; the default fast path agrees up to floating-point rounding), apart from five documented bug fixes (see [Verification](verification.md)).

Next: [Quickstart](quickstart.md) · [User guide](user-guide.md) · [API reference](api.md)

## Roadmap

- MMD-FUSE two-sample test
- GPU backend (JAX or PyTorch)
- Fast path for the permutation (unequal sample size) MMDAgg calibration
