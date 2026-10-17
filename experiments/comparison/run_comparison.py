"""Head-to-head kt vs hyppo 0.5.2. All libraries at defaults. Usage: python run_comparison.py [timing|power|all]"""
import sys, time, csv, warnings, numpy as np
import kt
from hyppo.ksample import MMD, Energy
from hyppo.independence import Hsic, Dcorr
warnings.filterwarnings("ignore")
OUT = __file__.rsplit("/", 1)[0]

TS = {"kt.mmdagg": lambda X, Y: kt.two_sample(X, Y, method="mmdagg").reject,
      "kt.mmdagginc": lambda X, Y: kt.two_sample(X, Y, method="mmdagginc").reject,
      "hyppo.MMD": lambda X, Y: MMD().test(X, Y)[1] < .05,
      "hyppo.Energy": lambda X, Y: Energy().test(X, Y)[1] < .05}
IN = {"kt.hsicagginc": lambda X, Y: kt.independence(X, Y, method="hsicagginc").reject,
      "hyppo.Hsic": lambda X, Y: Hsic().test(X, Y)[1] < .05,
      "hyppo.Dcorr": lambda X, Y: Dcorr().test(X, Y)[1] < .05}

def pert_unif(n, d, rng, amp, P=2):
    # Schrab et al. perturbed uniform (d dims, P bumps per dim, amplitude amp); rejection sampling
    out = []
    while len(out) < n:
        x = rng.uniform(size=(4 * n, d)); u = rng.uniform(size=4 * n)
        def g(t):  # bump with zero integral on each cell
            s = t * P - np.floor(t * P); return np.where(s < .5, 1., -1.) * np.sin(2 * np.pi * s) ** 2
        dens = 1 + amp * np.prod(g(x) * rng.choice([1], size=(1, d)), axis=1)
        out.extend(x[u * 2 < dens]); 
    return np.array(out[:n])

def gen(name, n, rng):
    d = 10
    if name == "H0_two": return rng.normal(size=(n, d)), rng.normal(size=(n, d))
    if name == "meanshift": return rng.normal(size=(n, d)), rng.normal(size=(n, d)) + np.r_[0.4, np.zeros(d - 1)]
    if name == "pertunif_d1": return rng.uniform(size=(n, 1)), pert_unif(n, 1, rng, 1.0, P=3)
    if name == "mixture_fine":  # Y: x1 bimodal +-0.25 with sd .1 vs X: matching mean/var unimodal
        X = rng.normal(size=(n, d)); Y = rng.normal(size=(n, d))
        s = np.sqrt(0.25**2 + 0.1**2); X[:, 0] *= s
        Y[:, 0] = rng.choice([-.25, .25], n) + .1 * rng.normal(size=n); return X, Y
    if name == "H0_indep": return rng.normal(size=(n, d)), rng.normal(size=(n, d))
    if name == "indep_sin":  # fine-scale dependence: y1 = sin(6 x1) + noise
        X = rng.normal(size=(n, d)); Y = rng.normal(size=(n, d)); Y[:, 0] = np.sin(6 * X[:, 0]) + .5 * rng.normal(size=n); return X, Y
    if name == "indep_linear":
        X = rng.normal(size=(n, d)); Y = rng.normal(size=(n, d)); Y[:, 0] = .25 * X[:, 0] + rng.normal(size=n); return X, Y

def timing():
    rows = []
    for n in (200, 1000, 2000):
        rng = np.random.default_rng(0)
        for tests, setting in ((TS, "H0_two"), (IN, "H0_indep")):
            X, Y = gen(setting, n, rng)
            for k, f in tests.items():
                f(X, Y)  # warm-up (numba JIT in hyppo / imports), excluded
                ts = []
                for _ in range(3):
                    t = time.perf_counter(); f(X, Y); ts.append(time.perf_counter() - t)
                rows.append(dict(test=k, n=n, d=10, median_s=round(float(np.median(ts)), 3))); print(rows[-1], flush=True)
    write("timing.csv", rows)

def power():
    rows = []
    plan = [(TS, "H0_two", 200), (TS, "meanshift", 100), (TS, "mixture_fine", 100), (TS, "pertunif_d1", 100),
            (IN, "H0_indep", 200), (IN, "indep_sin", 100), (IN, "indep_linear", 100)]
    for tests, setting, R in plan:
        rej = {k: 0 for k in tests}
        for r in range(R):
            X, Y = gen(setting, 200, np.random.default_rng(1000 + r))
            for k, f in tests.items():
                rej[k] += bool(f(X, Y)) if k.startswith("hyppo") else bool(kt_seeded(k, X, Y, r))
        for k in tests:
            p = rej[k] / R
            rows.append(dict(setting=setting, test=k, n=200, R=R, rate=p, mc_se=round(np.sqrt(p * (1 - p) / R), 3))); print(rows[-1], flush=True)
    write("power.csv", rows)

def power_one(setting):  # resumable: run one setting, append to power.csv
    import os
    plan = {"H0_two": (TS, 200), "meanshift": (TS, 100), "mixture_fine": (TS, 100), "pertunif_d1": (TS, 100),
            "H0_indep": (IN, 200), "indep_sin": (IN, 100), "indep_linear": (IN, 100)}
    tests, R = plan[setting]; rej = {k: 0 for k in tests}
    for r in range(R):
        X, Y = gen(setting, 200, np.random.default_rng(1000 + r))
        for k, f in tests.items():
            rej[k] += bool(f(X, Y)) if k.startswith("hyppo") else bool(kt_seeded(k, X, Y, r))
    fn = f"{OUT}/power.csv"; new = not os.path.exists(fn)
    with open(fn, "a", newline="") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["setting", "test", "n", "R", "rate", "mc_se"])
        for k in tests:
            p = rej[k] / R; w.writerow([setting, k, 200, R, p, round(float(np.sqrt(p * (1 - p) / R)), 3)])

def kt_seeded(k, X, Y, r):  # kt's permutation/wild-bootstrap seed defaults to 42; vary it per replicate
    if k == "kt.mmdagg": return kt.two_sample(X, Y, method="mmdagg", seed=r).reject
    if k == "kt.mmdagginc": return kt.two_sample(X, Y, method="mmdagginc", seed=r).reject
    return kt.independence(X, Y, method="hsicagginc", seed=r).reject

def write(fn, rows):
    with open(f"{OUT}/{fn}", "w", newline="") as fh:
        w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "all"
    if a in ("timing", "all"): timing()
    if a in ("power", "all"): power()
    if a == "setting": power_one(sys.argv[2])
