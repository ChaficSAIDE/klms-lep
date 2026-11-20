"""Multi-seed synthetic experiments (partial drift + global inversion).

Usage: python run_synthetic.py [n_seeds]
Writes results/synthetic_raw.csv and results/synthetic_curves.npz
"""
import sys, os, time
import numpy as np
import pandas as pd
from multiprocessing import Pool
import kaf, streams

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

WIN = 100        # window for recovery-time measurement
TOL = 0.03       # within 3 points of the phase's steady-state accuracy


def make_models(dim, cfg):
    D, s, mu, dl = cfg["budget"], cfg["sigma"], cfg["mu"], cfg["delta"]
    base = dict(sigma=s, mu=mu, delta=dl)
    models = {
        "KLMS-Dict (FIFO)": kaf.KLMSDict(dim, budget=D, **base),
        "KLMS-Dict (4x)": kaf.KLMSDict(dim, budget=4 * D, **base),
        "FB-QKLMS": kaf.QKLMS_FB(dim, sigma=s, mu=mu, eps=cfg["eps_q"], budget=D),
        "KRLS-T": kaf.KRLST(dim, sigma=s, lam=cfg["lam_krlst"], sn2=cfg["sn2"], budget=D),
        "SW-KRLS": kaf.SWKRLS(dim, sigma=s, budget=D, reg=cfg["sn2"]),
        "ADWIN+Restart": kaf.ADWINRestart(dim, budget=D, **base),
        "KLMS-DriftDict": kaf.DriftDict(dim, budget=D, **base),
        "KLMS-LEP (proposed)": kaf.LocalPrune(dim, budget=D, lam=cfg["lam"], z=cfg["z"], **base),
    }
    if cfg.get("ablation"):
        models = {k: v for k, v in models.items() if k in ("KLMS-Dict (FIFO)", "KLMS-LEP (proposed)")}
        models["LEP: prune only (FIFO evict)"] = kaf.LocalPrune(dim, budget=D, lam=cfg["lam"], z=cfg["z"], evict="fifo", **base)
        models["LEP: evict only (no prune)"] = kaf.LocalPrune(dim, budget=D, lam=cfg["lam"], z=np.inf, **base)
    return models


def recovery_times(correct, taus, n):
    ends = list(taus[1:]) + [n]
    out = []
    for tau, end in zip(taus, ends):
        seg = correct[tau:end]
        L = len(seg)
        steady = seg[int(0.7 * L):].mean()
        cs = np.concatenate([[0], np.cumsum(seg)])
        rt = L
        for t in range(0, L - WIN):
            if (cs[t + WIN] - cs[t]) / WIN >= steady - TOL:
                rt = t
                break
        out.append(rt)
    return out


def run_one(args):
    bench, rho, seed, cfg = args
    if bench == "partial":
        X, y, taus, drifted = streams.partial_drift_stream(rho, seed=seed, **cfg.get("stream", {}))
    else:
        X, y, taus, drifted = streams.inversion_stream(seed=seed)
    n, dim = X.shape
    rows, curves = [], {}
    for name, m in make_models(dim, cfg).items():
        t0 = time.time()
        correct = np.zeros(n)
        sizes = np.zeros(n)
        for t in range(n):
            yhat = m.predict(X[t])
            correct[t] = float(np.sign(yhat if yhat != 0 else 1e-9) == y[t])
            m.update(X[t], y[t])
            sizes[t] = m.size
        post = slice(taus[0], n)
        dr = drifted[taus[0]:]
        c = correct[post]
        rts = recovery_times(correct, taus, n)
        rows.append(dict(bench=bench, rho=rho, seed=seed, method=name,
                         overall=correct.mean(), post=c.mean(),
                         post_unchanged=c[~dr].mean() if (~dr).any() else np.nan,
                         post_drifted=c[dr].mean() if dr.any() else np.nan,
                         first500=np.mean([correct[t:t + 500].mean() for t in taus]),
                         recovery=np.mean(rts), dict_size=sizes.mean(),
                         events=m.n_events, sec=time.time() - t0))
        curves[name] = correct
    return rows, (bench, rho, seed, curves, taus)


CFG = dict(budget=40, sigma=1.5, mu=0.3, delta=0.2, eps_q=0.9, lam_krlst=0.995,
           sn2=0.1, lam=0.98, z=1.645)

if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    jobs = [("partial", rho, s, CFG) for rho in (0.25, 0.5, 1.0) for s in range(n_seeds)]
    jobs += [("inversion", 1.0, s, dict(CFG, sigma=4.0)) for s in range(n_seeds)]
    with Pool(4) as p:
        res = p.map(run_one, jobs)
    rows = [r for rr, _ in res for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "synthetic_raw.csv"), index=False)
    curves = {}
    for _, (b, rho, s, cv, taus) in res:
        for k, v in cv.items():
            curves[f"{b}|{rho}|{s}|{k}"] = v
    np.savez_compressed(os.path.join(OUT, "synthetic_curves.npz"), **curves)
    g = df.groupby(["bench", "rho", "method"])[["overall", "post", "post_unchanged",
                                                   "post_drifted", "first500", "recovery",
                                                   "dict_size"]]
    pd.set_option("display.width", 250)
    print(g.mean().round(3))
    print(g.std().round(3))
