# kt vs existing Python kernel-testing software

Reproduce (from this directory): `pip install hyppo==0.5.2 --break-system-packages && python run_comparison.py timing && for s in H0_two meanshift mixture_fine pertunif_d1 H0_indep indep_sin indep_linear; do python run_comparison.py setting $s; done`
(timing.csv, power.csv; power.csv appends, delete it first.) CPU: Intel Xeon @ 2.80GHz, 2 vCPUs (lscpu). hyppo 0.5.2.

## Feature table (verified from PyPI JSON, cloned sources, kt test run)
| | kt 0.0.1 | hyppo 0.5.2 | PyRKHSstats 2.1.0 | torch-two-sample | mmdagg/ksdagg/agginc/mmdfuse (ref.) |
|---|---|---|---|---|---|
| Two-sample / indep. / GoF | Y/Y/Y | Y/Y/Y (kgof.FSSD) | Y/Y(+cond.)/N | Y/N/N | one problem each (agginc: all 3) |
| Aggregated/adaptive bandwidth | Y | N (median heuristic) | N | N | Y |
| Incomplete-U linear-time | Y | N | N | N | agginc only |
| Unified API | Y | Y (class-based) | N | N | N |
| Test suite + coverage | 101 tests, 98.7% | tests + codecov | partial tests | tests | none |
| CI | GitHub Actions 3 OS x 5 Py | CircleCI | none found | Travis (legacy) | none |
| Input validation | ValueError | ValueError | partial | none checked | `assert` |
| Docs site | mkdocs | hyppo.neurodata.io | README | sphinx dir | README |
| Last release / commit | unreleased | PyPI 2025-05-24 | PyPI 2022-01-05 | not on PyPI; commit 2025-11-12 | not on PyPI; commits 2024-05-31 |
| License | MIT | MIT | GPL-3 | BSD-3 | MIT |

## Wall time (s, median of 3, 1 warm-up excluded, d=10, H0 data)
| test | n=200 | 1000 | 2000 |
|---|---|---|---|
| kt.mmdagg | 0.37 | 4.79 | 15.9 |
| kt.mmdagginc | 0.43 | 28.4* | 4.58 |
| hyppo.MMD | 0.08 | 1.06 | 4.41 |
| hyppo.Energy | 0.04 | 0.70 | 3.22 |
| kt.hsicagginc | 0.15 | 9.98 | 30.8 |
| hyppo.Hsic | 0.009 | 0.21 | 0.58 |
| hyppo.Dcorr | 0.007 | 0.12 | 0.40 |
*non-monotone; not investigated (possible batch_size="auto" artifact) - needs rerun.

## Level (R=200) and power (R=100), n=200, alpha=0.05
| setting | kt.mmdagg | kt.mmdagginc | hyppo.MMD | hyppo.Energy |
|---|---|---|---|---|
| H0 N(0,I10) | 0.030 | 0.030 | 0.015 | 0.015 |
| mean shift 0.4 in x1 | 0.58 | 0.61 | **0.73** | **0.73** |
| bimodal vs unimodal x1 (d=10) | 0.05 | 0.04 | 0.01 | 0.01 |
| perturbed uniform d=1, 3 bumps | 1.0 | 1.00 | 0.20 | 0.29 |

| setting | kt.hsicagginc | hyppo.Hsic | hyppo.Dcorr |
|---|---|---|---|
| H0 indep. | 0.045 | 0.015 | 0.015 |
| y1=sin(6x1)+noise | 0.04 | 0.02 | 0.02 |
| y1=0.25x1+noise | 0.08 | 0.11 | 0.11 |

Defaults: kt mmdagg (Laplace+Gaussian, 10 bandwidths, B1=B2=2000), mmdagginc/hsicagginc (R=200, B=500); kt seed varied per replicate. hyppo: Gaussian kernel, median-heuristic gamma, `auto=True` => chi-square approximation for n>20 (no permutations), so its timings are not permutation-test timings.

Caveats: hyppo chi2 approx is conservative here; mixture/sin/linear alternatives too weak to discriminate (all near level); MC SE <=0.05; 2-vCPU VM; torch-two-sample not installed (source only); hyppo CI read from clone (.circleci), no GitHub API access.
