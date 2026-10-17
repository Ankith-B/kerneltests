# JAX backend benchmark (MMDAgg)

Script: `experiments/jax_backend/bench.py` (raw output incl. full `lscpu`: `bench_output.txt`).
Setup: d=10, B1=B2=500, B3=50, kernel="laplace_gaussian", 10 bandwidths, equal n (wild bootstrap), mean shift 0.05.
Timings are the median of 3 runs after one warm-up run. "JAX first call" is the warm-up, which includes jit compilation.

Hardware: `lscpu` reports 2 vCPUs (1 thread per core), Intel(R) Xeon(R) Processor @ 2.10GHz, x86_64, ~7 GB RAM.
Software: JAX 0.11.2 (CPU, `jax_enable_x64`), NumPy/SciPy as installed.

| n | reference np.py (s) | kt numpy fast (s) | kt jax CPU (s) | JAX first call (s) | jax vs ref | jax vs kt numpy |
|---|---|---|---|---|---|---|
| 500  | 3.52  | 2.28  | 0.82  | 2.28  | 4.3x | 2.8x |
| 1000 | 10.28 | 5.90  | 3.71  | 4.40  | 2.8x | 1.6x |
| 2000 | 40.38 | 16.51 | 9.78  | 11.22 | 4.1x | 1.7x |
| 4000 | 100.02 | 33.08 | 30.63 | 33.34 | 3.3x | 1.08x |

Decisions of all three agreed at every n (asserted in the script).

**No GPU was available.** The JAX path is device-agnostic and would run on a GPU with a CUDA jaxlib, but GPU
correctness and speed are untested; no GPU numbers are claimed.

Caveats: a 2-core shared VM, so timings are noisy; at n=4000 the run is dominated by the (B1+B2+1)-column
matrix products, which NumPy's BLAS and XLA's CPU dot perform similarly, so the gain shrinks to ~8%.
Compile time is ~0.7-2.7 s per new shape and is paid once per (n, kernel, bandwidth-count) combination.
