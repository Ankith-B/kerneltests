# Verification

## Exact-parity methodology

Each port reproduces its reference's random-number protocol (`numpy.random.RandomState(seed)`, with the same draw order and shapes) and its floating-point evaluation order. The reference weights code is reused verbatim in `kt/_common.py`. Vectorised steps (bisection, quantile lookup, shared Stein-kernel terms) were checked to leave the results unchanged.

Parity scripts run the reference packages (`mmdagg/np.py`, `ksdagg/np.py`) and `kt` on the same inputs and compare decisions, statistics, quantiles, p-values and thresholds (relative tolerance 1e-10; in practice the values are identical):

- **MMDAgg**: all 19 kernel options × 4 weightings × 2 bandwidth counts (even and odd), using both the wild-bootstrap (`m == n`) and permutation (`m != n`) paths, under the null and under alternatives.
- **KSDAgg**: sample sizes 40/80/150, dimensions 1/2/5, all weightings, automatic and user-supplied bandwidths, and wild-bootstrap and parametric modes.

The unit test suite (`tests/`) also checks power, Type I error control under the null, determinism, and validation.

## Reference bugs found

1. **MMDAgg reported duplicated p-values.** With `kernel="laplace_gaussian"`, the loop that writes the per-kernel results in `mmdagg/np.py` (line 352) reads `p_vals[i]` (the bandwidth index) instead of `p_vals[idx]` (the global index). As a result, the Gaussian entries report the Laplace p-values. The rejection decision itself is unaffected. `kt` reports the correct p-values, and `tests/test_mmdagg.py::test_pvalues_reported_per_kernel` guards the fix.
2. **KSDAgg internal AssertionError.** The reference computes the rejection both as `p-value <= threshold` and as `KSD > quantile`, then asserts the two agree. Mathematically they are equivalent, but floating-point rounding can make them disagree, and the reference then raises `AssertionError`. `kt` uses only the quantile rule. On the cases where the reference fails, `kt` matches the decision the reference's quantile rule gives once the assertion is bypassed.

The ports also replace the references' `assert` checks with `ValueError`. `ksdagg` accepts any 1-D array-like for `bandwidths`; the reference silently ignores inputs that are not ndarrays.
