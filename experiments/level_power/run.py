"""Level/power study for kt. Usage: python experiments/level_power/run.py [--B 200] [--quick]"""
import os
os.environ.setdefault("OMP_NUM_THREADS","1"); os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("MKL_NUM_THREADS","1")
import sys, csv, time, json, platform, argparse, multiprocessing as mp
import numpy as np, scipy
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import kt

HERE = os.path.dirname(os.path.abspath(__file__))
ALPHA = 0.05

def gen(study, eff, n, d, rs):
    X = rs.randn(n, d)
    if study in ("level_mmd", "mean_shift"):
        Y = rs.randn(n, d); Y[:, 0] += eff; return X, Y
    if study == "var_change":
        return X, rs.randn(n, d) * np.sqrt(1 + eff)
    if study in ("level_ksd", "ksd_shift"):  # model N(0,I); data N(eff e1, I)
        X[:, 0] += eff; return X, -X
    if study == "ksd_scale":                  # data N(0,(1+eff)I)
        X = X * np.sqrt(1 + eff); return X, -X
    if study in ("level_hsic", "hsic_dep"):
        Y = eff * X + rs.randn(n, d); return X, Y
    raise ValueError(study)

def run_test(method, X, Y, seed, B):
    kw = dict(B1=B, B2=B, seed=seed)
    if method.startswith("mmd"): return kt.two_sample(X, Y, method=method, **kw)
    if method.startswith("ksd"): return kt.goodness_of_fit(X, Y, method=method, **kw)
    return kt.independence(X, Y, method=method, **kw)

def plan(quick, B):
    Rl, Rp = (20, 10) if quick else (1000, 200)
    jobs = []
    lv = {"mmdagg": "level_mmd", "mmdagginc": "level_mmd", "ksdagg": "level_ksd", "ksdagginc": "level_ksd", "hsicagginc": "level_hsic"}
    for m, s in lv.items():
        for d in (1, 5, 10):
            for n in (100, 500):
                jobs.append((s, m, 0.0, n, d, Rl))
    pw = [("mean_shift", ["mmdagg", "mmdagginc"], [0.05, 0.1, 0.15, 0.2, 0.3], 5),
          ("var_change", ["mmdagg", "mmdagginc"], [0.1, 0.2, 0.3, 0.4, 0.6], 5),
          ("ksd_shift", ["ksdagg", "ksdagginc"], [0.05, 0.1, 0.15, 0.2, 0.3], 5),
          ("ksd_scale", ["ksdagg", "ksdagginc"], [0.05, 0.1, 0.2, 0.3, 0.4], 5),
          ("hsic_dep", ["hsicagginc"], [0.05, 0.1, 0.15, 0.2, 0.3], 1)]
    for s, ms, effs, d in pw:
        for m in ms:
            for e in effs:
                jobs.append((s, m, e, 500, d, Rp))
    # split into chunks of 50 reps for load balance; key unique per (setting) so data identical across methods
    out = []
    for (s, m, e, n, d, R) in jobs:
        base = (sum(map(ord, s)) * 7919 + int(e * 1000) * 131 + n * 17 + d) % 10**6
        for c0 in range(0, R, 50):
            out.append((s, m, e, n, d, c0, min(50, R - c0), (B if n >= 500 else 500), base))
    return out

def chunk2(a):
    s, m, e, n, d, c0, R, B, base = a
    out = []
    for r in range(c0, c0 + R):
        seed = (base * 100003 + r) % (2**31 - 1)
        X, Y = gen(s, e, n, d, np.random.RandomState(seed))
        t = time.perf_counter(); res = run_test(m, X, Y, seed + 1, B); dt = time.perf_counter() - t
        out.append((s, m, e, n, d, r, seed, int(res.reject), dt))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=100, help="B1=B2 for n=500 settings; n=100 settings always use B1=B2=500"); ap.add_argument("--quick", action="store_true"); ap.add_argument("--power-only", action="store_true"); ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=os.cpu_count()); a = ap.parse_args()
    tasks = plan(a.quick, a.B); t0 = time.time()
    if a.power_only: tasks = [t for t in tasks if not t[0].startswith("level")]
    fn = os.path.join(HERE, a.out or ("results_quick.csv" if a.quick else "results.csv"))
    hdr = ["study", "method", "effect", "n", "d", "rep", "data_seed", "reject", "seconds"]
    # resume: keep only complete 50-rep chunks already on disk, rerun the rest
    done, keep = set(), []
    if os.path.exists(fn):
        import collections
        rows = [r for r in csv.reader(open(fn)) if len(r) == 9 and r[0] != "study"]
        by = collections.defaultdict(list)
        for r in rows: by[(r[0], r[1], float(r[2]), int(r[3]), int(r[4]), int(r[5]) // 50 * 50)].append(r)
        want = {(t[0], t[1], float(t[2]), t[3], t[4], t[5]): t[6] for t in tasks}
        for k, v in by.items():
            if k in want and len(v) == want[k]: done.add(k); keep += v
    tasks = [t for t in tasks if (t[0], t[1], float(t[2]), t[3], t[4], t[5]) not in done]
    print(f"resuming: {len(done)} chunks kept, {len(tasks)} to run", flush=True)
    with open(fn, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(keep)
    with mp.Pool(a.procs) as pool, open(fn, "a", newline="") as f:
        w = csv.writer(f)
        for i, rows in enumerate(pool.imap_unordered(chunk2, tasks)):
            w.writerows(rows); f.flush()
            if i % 20 == 0: print(f"{i+1}/{len(tasks)} {time.time()-t0:.0f}s", flush=True)
    meta = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                cpus=os.cpu_count(), procs=a.procs, B1_B2_n500=a.B, B1_B2_n100=500, B3=50, alpha=ALPHA, wall_seconds=time.time() - t0,
                kt_version=getattr(kt, "__version__", "n/a"), quick=a.quick)
    json.dump(meta, open(os.path.join(HERE, (os.path.splitext(a.out)[0] + "_meta.json") if a.out else ("meta_quick.json" if a.quick else "meta.json")), "w"), indent=1)
    print(meta)
