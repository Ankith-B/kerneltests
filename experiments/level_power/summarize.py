"""Build results.md from results.csv + meta.json."""
import os, json, pandas as pd
from scipy.stats import beta
H = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(H, "results.csv")); meta = json.load(open(os.path.join(H, "meta.json")))
def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a/2, k, n-k+1); hi = 1.0 if k == n else beta.ppf(1-a/2, k+1, n-k); return lo, hi
L = []
L.append("# kt level and power study\n")
L.append("Reproduce (from repo root): `python experiments/level_power/run.py && python experiments/level_power/run.py --power-only --B 500 --out results_power_B500.csv && python experiments/level_power/summarize.py`\n")
L.append(f"Environment: `{json.dumps(meta)}`\n")
L.append("alpha=0.05. B1=B2=500 for n=100, B1=B2=%s for n=500 (reduced from library defaults: mmdagg/ksdagg 2000, agginc 500) to fit the time budget; B3=50, R(inc)=200 default, 10 bandwidths, uniform weights, all other args library defaults. Test seed = data seed + 1.\n" % meta["B1_B2_n500"])
L.append("## Level (H0), rejection rate with 95% Clopper-Pearson CI\n")
L.append("| method | d | n | R | rate | CI | flag |\n|---|---|---|---|---|---|---|")
lv = d[d.study.str.startswith("level")]
for (m, dd, n), g in lv.groupby(["method", "d", "n"]):
    k, R = g.reject.sum(), len(g); lo, hi = cp(k, R)
    L.append(f"| {m} | {dd} | {n} | {R} | {k/R:.3f} | [{lo:.3f}, {hi:.3f}] | {'VIOLATION' if lo > 0.05 else ''} |")
def power_tables(pw, title):
    L.append(title)
    for s_, gs in pw.groupby("study", sort=False):
        L.append(f"\n### {s_} (d={gs.d.iloc[0]})\n")
        effs = sorted(gs.effect.unique())
        L.append("| method | " + " | ".join(f"eff={e:g}" for e in effs) + " |\n|---" + "|---" * len(effs) + "|")
        for m in gs.method.unique():
            row = []
            for e in effs:
                g = gs[(gs.method == m) & (gs.effect == e)]; k_, R = g.reject.sum(), len(g); lo, hi = cp(k_, R)
                row.append(f"{k_/R:.2f} [{lo:.2f},{hi:.2f}]")
            L.append(f"| {m} | " + " | ".join(row) + " |")
pB = os.path.join(H, "results_power_B500.csv")
if os.path.exists(pB):
    p5 = pd.read_csv(pB)
    power_tables(p5, "\n## Power (n=500, B1=B2=500, R=200), rejection rate [95% CP CI]\n")
L.append("\nLevel-run mean seconds per test (contaminated, indicative only):\n")
L.append(lv.groupby(["method", "n"]).seconds.agg(["mean", "median"]).round(3).to_markdown())
L.append("\n## Timing\nPer-test times are NOT reported as benchmarks: both the level run and the B=500 power run overlapped other CPU jobs on this 2-core machine (pytest, a JAX backend check), so wall times are contaminated. The raw `seconds` column is kept in the CSVs for reference only. The level-run table below is indicative only.\n")
L.append("\n## Caveats\n- Data seeds: `(base*100003 + rep) mod (2^31-1)` with a hand-rolled per-setting `base`. Exhaustively checked: no seed is shared by two settings in this design. It was kept rather than switched to SeedSequence, because switching would change the already-run level data. Adding new settings could produce collisions; re-check if the design changes.\n- B=100 power (appendix) is superseded; the n=500 level numbers in the table (also B=100) are valid but conservative, especially MMD/HSIC.\n- B1/B2 reduced from defaults (see above); this coarsens quantile estimation but does not invalidate the test level.\n- Same data seeds are shared across mmdagg/mmdagginc (and ksdagg/ksdagginc) so comparisons are paired.\n- Run was interrupted once by a container restart and resumed (complete 50-rep chunks kept, partial chunk rerun with identical seeds); timings measured with 2 worker processes on a 2-core machine; absolute numbers are machine-specific.\n- HSIC alternative is linear Gaussian dependence only; d for power is 5 (1 for HSIC).")
power_tables(d[~d.study.str.startswith("level")], "\n## Appendix: SUPERSEDED power at B1=B2=100\nThese numbers are superseded by the B=500 tables above. B=100 makes the aggregated tests very conservative (the per-kernel quantile is near the max of 101), so power was badly understated. Kept for transparency only.\n")
open(os.path.join(H, "results.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))
