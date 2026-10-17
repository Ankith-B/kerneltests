# kt level and power study

Reproduce (from repo root): `python experiments/level_power/run.py && python experiments/level_power/run.py --power-only --B 500 --out results_power_B500.csv && python experiments/level_power/summarize.py`

Environment: `{"python": "3.13.16", "numpy": "2.5.3", "scipy": "1.18.1", "platform": "Linux-6.18.44-fc-v139-x86_64-with-glibc2.39", "cpus": 2, "procs": 2, "B1_B2_n500": 100, "B1_B2_n100": 500, "B3": 50, "alpha": 0.05, "wall_seconds": 2739.9235248565674, "kt_version": "0.1.0.dev0", "quick": false}`

alpha=0.05. B1=B2=500 for n=100, B1=B2=100 for n=500 (reduced from library defaults: mmdagg/ksdagg 2000, agginc 500) to fit the time budget; B3=50, R(inc)=200 default, 10 bandwidths, uniform weights, all other args library defaults. Test seed = data seed + 1.

## Level (H0), rejection rate with 95% Clopper-Pearson CI

| method | d | n | R | rate | CI | flag |
|---|---|---|---|---|---|---|
| hsicagginc | 1 | 100 | 1000 | 0.031 | [0.021, 0.044] |  |
| hsicagginc | 1 | 500 | 1000 | 0.002 | [0.000, 0.007] |  |
| hsicagginc | 5 | 100 | 1000 | 0.027 | [0.018, 0.039] |  |
| hsicagginc | 5 | 500 | 1000 | 0.003 | [0.001, 0.009] |  |
| hsicagginc | 10 | 100 | 1000 | 0.033 | [0.023, 0.046] |  |
| hsicagginc | 10 | 500 | 1000 | 0.002 | [0.000, 0.007] |  |
| ksdagg | 1 | 100 | 1000 | 0.054 | [0.041, 0.070] |  |
| ksdagg | 1 | 500 | 1000 | 0.040 | [0.029, 0.054] |  |
| ksdagg | 5 | 100 | 1000 | 0.031 | [0.021, 0.044] |  |
| ksdagg | 5 | 500 | 1000 | 0.042 | [0.030, 0.056] |  |
| ksdagg | 10 | 100 | 1000 | 0.054 | [0.041, 0.070] |  |
| ksdagg | 10 | 500 | 1000 | 0.056 | [0.043, 0.072] |  |
| ksdagginc | 1 | 100 | 1000 | 0.060 | [0.046, 0.077] |  |
| ksdagginc | 1 | 500 | 1000 | 0.043 | [0.031, 0.057] |  |
| ksdagginc | 5 | 100 | 1000 | 0.029 | [0.020, 0.041] |  |
| ksdagginc | 5 | 500 | 1000 | 0.047 | [0.035, 0.062] |  |
| ksdagginc | 10 | 100 | 1000 | 0.052 | [0.039, 0.068] |  |
| ksdagginc | 10 | 500 | 1000 | 0.058 | [0.044, 0.074] |  |
| mmdagg | 1 | 100 | 1000 | 0.047 | [0.035, 0.062] |  |
| mmdagg | 1 | 500 | 1000 | 0.018 | [0.011, 0.028] |  |
| mmdagg | 5 | 100 | 1000 | 0.035 | [0.024, 0.048] |  |
| mmdagg | 5 | 500 | 1000 | 0.004 | [0.001, 0.010] |  |
| mmdagg | 10 | 100 | 1000 | 0.056 | [0.043, 0.072] |  |
| mmdagg | 10 | 500 | 1000 | 0.015 | [0.008, 0.025] |  |
| mmdagginc | 1 | 100 | 1000 | 0.048 | [0.036, 0.063] |  |
| mmdagginc | 1 | 500 | 1000 | 0.021 | [0.013, 0.032] |  |
| mmdagginc | 5 | 100 | 1000 | 0.034 | [0.024, 0.047] |  |
| mmdagginc | 5 | 500 | 1000 | 0.012 | [0.006, 0.021] |  |
| mmdagginc | 10 | 100 | 1000 | 0.056 | [0.043, 0.072] |  |
| mmdagginc | 10 | 500 | 1000 | 0.013 | [0.007, 0.022] |  |

## Power (n=500, B1=B2=500, R=200), rejection rate [95% CP CI]


### mean_shift (d=5)

| method | eff=0.05 | eff=0.1 | eff=0.15 | eff=0.2 | eff=0.3 |
|---|---|---|---|---|---|
| mmdagg | 0.06 [0.03,0.10] | 0.05 [0.02,0.09] | 0.20 [0.15,0.26] | 0.45 [0.37,0.52] | 0.89 [0.84,0.93] |
| mmdagginc | 0.06 [0.03,0.10] | 0.05 [0.02,0.09] | 0.17 [0.12,0.22] | 0.36 [0.30,0.44] | 0.83 [0.78,0.88] |

### var_change (d=5)

| method | eff=0.1 | eff=0.2 | eff=0.3 | eff=0.4 | eff=0.6 |
|---|---|---|---|---|---|
| mmdagg | 0.08 [0.05,0.13] | 0.56 [0.48,0.63] | 0.96 [0.93,0.99] | 0.98 [0.96,1.00] | 0.99 [0.97,1.00] |
| mmdagginc | 0.07 [0.04,0.11] | 0.49 [0.42,0.57] | 0.94 [0.90,0.97] | 0.99 [0.97,1.00] | 1.00 [0.98,1.00] |

### ksd_shift (d=5)

| method | eff=0.05 | eff=0.1 | eff=0.15 | eff=0.2 | eff=0.3 |
|---|---|---|---|---|---|
| ksdagg | 0.08 [0.05,0.13] | 0.30 [0.24,0.37] | 0.65 [0.58,0.72] | 0.93 [0.89,0.96] | 1.00 [0.98,1.00] |
| ksdagginc | 0.07 [0.04,0.11] | 0.20 [0.15,0.27] | 0.56 [0.49,0.63] | 0.90 [0.85,0.94] | 1.00 [0.98,1.00] |

### ksd_scale (d=5)

| method | eff=0.05 | eff=0.1 | eff=0.2 | eff=0.3 | eff=0.4 |
|---|---|---|---|---|---|
| ksdagg | 0.04 [0.02,0.08] | 0.08 [0.05,0.13] | 0.40 [0.33,0.47] | 0.88 [0.83,0.92] | 1.00 [0.98,1.00] |
| ksdagginc | 0.04 [0.01,0.07] | 0.07 [0.04,0.11] | 0.28 [0.22,0.35] | 0.76 [0.69,0.82] | 0.99 [0.96,1.00] |

### hsic_dep (d=1)

| method | eff=0.05 | eff=0.1 | eff=0.15 | eff=0.2 | eff=0.3 |
|---|---|---|---|---|---|
| hsicagginc | 0.04 [0.01,0.07] | 0.14 [0.10,0.20] | 0.32 [0.26,0.39] | 0.61 [0.54,0.68] | 0.96 [0.92,0.98] |

Level-run mean seconds per test (contaminated, indicative only):

|                     |   mean |   median |
|:--------------------|-------:|---------:|
| ('hsicagginc', 100) |  0.012 |    0.011 |
| ('hsicagginc', 500) |  0.077 |    0.077 |
| ('ksdagg', 100)     |  0.024 |    0.022 |
| ('ksdagg', 500)     |  0.191 |    0.173 |
| ('ksdagginc', 100)  |  0.039 |    0.038 |
| ('ksdagginc', 500)  |  0.174 |    0.173 |
| ('mmdagg', 100)     |  0.042 |    0.047 |
| ('mmdagg', 500)     |  0.521 |    0.538 |
| ('mmdagginc', 100)  |  0.137 |    0.145 |
| ('mmdagginc', 500)  |  0.736 |    0.741 |

## Timing
Per-test times are NOT reported as benchmarks: both the level run and the B=500 power run overlapped other CPU jobs on this 2-core machine (pytest, a JAX backend check), so wall times are contaminated. The raw `seconds` column is kept in the CSVs for reference only. The level-run table below is indicative only.


## Caveats
- Data seeds: `(base*100003 + rep) mod (2^31-1)` with a hand-rolled per-setting `base`. Exhaustively checked: no seed is shared by two settings in this design. It was kept rather than switched to SeedSequence, because switching would change the already-run level data. Adding new settings could produce collisions; re-check if the design changes.
- B=100 power (appendix) is superseded; the n=500 level numbers in the table (also B=100) are valid but conservative, especially MMD/HSIC.
- B1/B2 reduced from defaults (see above); this coarsens quantile estimation but does not invalidate the test level.
- Same data seeds are shared across mmdagg/mmdagginc (and ksdagg/ksdagginc) so comparisons are paired.
- Run was interrupted once by a container restart and resumed (complete 50-rep chunks kept, partial chunk rerun with identical seeds); timings measured with 2 worker processes on a 2-core machine; absolute numbers are machine-specific.
- HSIC alternative is linear Gaussian dependence only; d for power is 5 (1 for HSIC).

## Appendix: SUPERSEDED power at B1=B2=100
These numbers are superseded by the B=500 tables above. B=100 makes the aggregated tests very conservative (the per-kernel quantile is near the max of 101), so power was badly understated. Kept for transparency only.


### mean_shift (d=5)

| method | eff=0.05 | eff=0.1 | eff=0.15 | eff=0.2 | eff=0.3 |
|---|---|---|---|---|---|
| mmdagg | 0.01 [0.00,0.03] | 0.01 [0.00,0.04] | 0.06 [0.03,0.10] | 0.05 [0.02,0.09] | 0.16 [0.11,0.22] |
| mmdagginc | 0.03 [0.01,0.06] | 0.01 [0.00,0.04] | 0.06 [0.03,0.10] | 0.10 [0.07,0.16] | 0.27 [0.21,0.33] |

### var_change (d=5)

| method | eff=0.1 | eff=0.2 | eff=0.3 | eff=0.4 | eff=0.6 |
|---|---|---|---|---|---|
| mmdagg | 0.01 [0.00,0.04] | 0.12 [0.08,0.17] | 0.18 [0.13,0.25] | 0.25 [0.19,0.32] | 0.26 [0.20,0.33] |
| mmdagginc | 0.03 [0.01,0.06] | 0.10 [0.06,0.15] | 0.27 [0.21,0.33] | 0.34 [0.27,0.41] | 0.32 [0.25,0.38] |

### ksd_shift (d=5)

| method | eff=0.05 | eff=0.1 | eff=0.15 | eff=0.2 | eff=0.3 |
|---|---|---|---|---|---|
| ksdagg | 0.07 [0.04,0.11] | 0.24 [0.19,0.31] | 0.58 [0.51,0.65] | 0.86 [0.81,0.91] | 0.97 [0.94,0.99] |
| ksdagginc | 0.07 [0.04,0.11] | 0.19 [0.14,0.25] | 0.50 [0.43,0.57] | 0.84 [0.78,0.89] | 0.93 [0.88,0.96] |

### ksd_scale (d=5)

| method | eff=0.05 | eff=0.1 | eff=0.2 | eff=0.3 | eff=0.4 |
|---|---|---|---|---|---|
| ksdagg | 0.04 [0.01,0.07] | 0.07 [0.04,0.11] | 0.33 [0.27,0.40] | 0.80 [0.73,0.85] | 0.97 [0.94,0.99] |
| ksdagginc | 0.04 [0.01,0.07] | 0.07 [0.04,0.11] | 0.21 [0.16,0.27] | 0.67 [0.60,0.73] | 0.92 [0.87,0.95] |

### hsic_dep (d=1)

| method | eff=0.05 | eff=0.1 | eff=0.15 | eff=0.2 | eff=0.3 |
|---|---|---|---|---|---|
| hsicagginc | 0.01 [0.00,0.03] | 0.02 [0.01,0.05] | 0.03 [0.01,0.06] | 0.04 [0.02,0.08] | 0.16 [0.11,0.22] |
