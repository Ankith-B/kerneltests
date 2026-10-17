# User guide

## What each test tests

**MMDAgg** (`kt.two_sample`, `method="mmdagg"`) tests \(H_0: P = Q\) from samples `X` (shape `(m, d)`) and `Y` (shape `(n, d)`). It computes the MMD U-statistic for a collection of kernels and bandwidths, then rejects if any single test rejects at its adjusted level. The adjustment is calibrated so the aggregated test still has level `alpha`. Calibration uses a wild bootstrap when `m == n` (unless `permutations_same_sample_size=True`) and permutations otherwise.

**KSDAgg** (`kt.goodness_of_fit`, `method="ksdagg"`) tests \(H_0: X \sim p\) using the kernel Stein discrepancy with the IMQ Stein kernel. `score_X` must have the same shape as `X` and contain \(\nabla \log p\) evaluated at each row of `X`. Calibration uses a wild bootstrap by default. With `approx_type="parametric"`, you pass arrays `B1` and `B2` of shape `(number_bandwidths, B)` holding KSD values simulated under the null.

**Incomplete U-statistic variants** (`method="mmdagginc"`, `"ksdagginc"`, and `kt.independence`, which uses HSICAggInc) estimate each statistic from `R` diagonals of the kernel matrix instead of all pairs, so cost grows linearly in the sample size. They trade some power for speed and are the right choice for large samples (thousands of points or more). They are calibrated with a wild bootstrap. HSICAggInc tests \(H_0: X \perp Y\) from paired samples (row `i` of `X` pairs with row `i` of `Y`).

**Caution:** HSICAggInc aggregates 25 kernel/bandwidth pairs by default. With small `B1`, each single test's adjusted level can fall below `1/(B1+1)`, and the test then cannot reject at all. Keep `B1` at its default (500) or higher.

## Choosing parameters

- **`alpha`**: the significance level, in (0, 1). The default is 0.05.
- **`B1`**: the number of bootstrap/permutation samples used to estimate quantiles. **`B2`**: the number used to estimate the level-correction probability. **`B3`**: the number of bisection steps for the correction. The defaults (2000, 2000, 50) follow the reference. Lowering `B1` and `B2` to a few hundred speeds the test up at the cost of a less precise level correction.
- **`seed`**: fixes the resampling, which makes results reproducible. It defaults to 42, as in the reference.
- **`number_bandwidths`** (default 10) and **`weights_type`** (`"uniform"`, `"decreasing"`, `"increasing"` or `"centred"`): the weights spread the level across bandwidths. `"decreasing"` favours small bandwidths and `"increasing"` favours large ones.
- **MMDAgg `kernel`**: `"gaussian"`, `"laplace"`, `"imq"`, `"matern_{0..4}.5_l1"`, `"matern_{0..4}.5_l2"`, or a collection: `"laplace_gaussian"` (the default) / `"gaussian_laplace"`, `"all_matern_l1"`, `"all_matern_l2"`, `"all_matern_l1_l2"`, `"all"`. Bandwidths are chosen from inter-sample distances (using the first 500 points of each sample). L1 kernels use cityblock distances and L2 kernels use Euclidean distances.
- **KSDAgg**: `kernel="imq"` is the only kernel. You can pass explicit positive `bandwidths` to override the automatic grid.

## Interpreting `details`

`TestResult.details` is a dictionary of arrays with one entry for each (kernel, bandwidth) pair:

| Key | Meaning |
|---|---|
| `kernel`, `bandwidth` | which single test the entry belongs to |
| `mmd` / `ksd` | observed statistic |
| `quantile` | adjusted rejection threshold for the statistic |
| `p_value` | single-test p-value (estimated from the `B1` samples) |
| `threshold` | adjusted level for that single test (`u * weight`) |
| `reject` | whether this single test rejects (`statistic > quantile`) |

The overall decision is `details["reject"].any()`. Compare `p_value` against `threshold`, not against `alpha`. A `p_value` below `alpha` but above `threshold` does not lead to a rejection. The entries that reject show which kernels and scales detect the difference.
