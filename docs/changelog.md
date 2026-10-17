# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `kt.two_sample` (MMDAgg) and `kt.goodness_of_fit` (KSDAgg) entry points, which return `kt.TestResult`.
- `kt.mmdagg.mmdagg`, supporting every reference kernel option, all four weightings, and wild-bootstrap and permutation calibration.
- `kt.ksdagg.ksdagg`, supporting the IMQ Stein kernel with wild-bootstrap and parametric calibration and user-supplied bandwidths.
- `kt.mmdfuse.mmdfuse` and `kt.two_sample(..., method="mmdfuse")`: MMD-FUSE (Biggs, Schrab, Gretton, NeurIPS 2023), NumPy port of the JAX reference with all 14 kernels; parity checked with shared permutations (`check_mmdfuse_vs_reference.py`).
- Documentation, CI (Linux, macOS and Windows; Python 3.9 to 3.13) and coverage gates.

### Fixed (relative to the reference implementations)
- MMDAgg: Gaussian-kernel p-values duplicated the Laplace ones for `kernel="laplace_gaussian"` (`p_vals[i]` was used instead of `p_vals[idx]`).
- KSDAgg: removed a spurious internal `AssertionError` caused by floating-point disagreement between two equivalent rejection rules.
