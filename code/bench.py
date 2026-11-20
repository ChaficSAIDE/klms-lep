"""Unified multi-seed benchmark runner.

python bench.py <benchmark> [n_seeds]
benchmarks: partial25, partial50, partial100, inversion, emg, cwru, tune_<name>

Hyper-parameters shared by all KLMS variants (sigma, mu, delta) and the
baselines' own parameters are selected on the PRE-DRIFT segment only
(seed 0) by `tune`, and stored in results/params_<bench>.json.
The proposed method's parameters (lam=0.98, z=1.645) are fixed everywhere.
"""
import sys, os, json, time, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool
import kaf, streams, data_real

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

WIN, TOL = 100, 0.03
LEP = dict(lam=0.98, z=1.645)
BUDGET = dict(partial25=40, partial50=40, partial100=40, inversion=30, emg=40, cwru=25, cwru4=25)


def get_stream(bench, seed):
    if bench.startswith("partial"):
        rho = int(bench[7:]) / 100
        X, y, taus, dr = streams.partial_drift_stream(rho, seed=seed)
        return X, y, taus, dr
    if bench == "inversion":
        X, y, taus = data_real.original_synthetic(seed=seed)
    elif bench == "emg":
        X, y, taus = data_real.emg_stream(seed=seed)
    elif bench == "cwru":
        X, y, taus = data_real.cwru_stream(seed=seed)
    elif bench == "cwru4":
        return data_real.cwru_fault_stream(seed=seed)
    return X, y, taus, None


def make_models(dim, P, D, only=None):
    base = dict(sigma=P["sigma"], mu=P["mu"], delta=P["delta"])
    m = {
        "KLMS-Dict (FIFO)": lambda: kaf.KLMSDict(dim, budget=D, **base),
        "KLMS-Dict (4x)": lambda: kaf.KLMSDict(dim, budget=4 * D, **base),
        "FB-QKLMS": lambda: kaf.QKLMS_FB(dim, sigma=P["sigma"], mu=P["mu"], eps=P["eps_q"], budget=D),
        "KRLS-T": lambda: kaf.KRLST(dim, sigma=P["sigma"], lam=P["lam_t"], sn2=P["sn2"], budget=D),
        "SW-KRLS": lambda: kaf.SWKRLS(dim, sigma=P["sigma"], budget=D, reg=P["sn2"]),
        "ADWIN+Restart": lambda: kaf.ADWINRestart(dim, budget=D, **base),
        "KLMS-DriftDict": lambda: kaf.DriftDict(dim, budget=D, **base),
        "KLMS-LEP (proposed)": lambda: kaf.LocalPrune(dim, budget=D, **LEP, **base),
        # ablations of the proposed method
        "LEP w/o pruning": lambda: kaf.LocalPrune(dim, budget=D, lam=LEP["lam"], z=np.inf, **base),
        "LEP w/ FIFO eviction": lambda: kaf.LocalPrune(dim, budget=D, evict="fifo", **LEP, **base),
    }
    if only is not None:
        m = {k: v for k, v in m.items() if k in only}
    return m


def recovery_times(correct, taus, n):
    """Steps after each drift until the WIN-sample accuracy is back within TOL of
    the method's own pre-drift steady-state accuracy (last 30% of the previous
    segment). Capped at the segment length (never recovered)."""
    bounds = [0] + list(taus) + [n]
    out = []
    for j, tau in enumerate(taus):
        prev = correct[bounds[j]:tau]
        ref = prev[int(0.7 * len(prev)):].mean()
        seg = correct[tau:bounds[j + 2]]
        L = len(seg)
        cs = np.concatenate([[0], np.cumsum(seg)])
        rt = L
        for t in range(0, L - WIN):
            if (cs[t + WIN] - cs[t]) / WIN >= ref - TOL:
                rt = t
                break
        out.append(rt)
    return out


class OVR:
    """One-vs-rest wrapper for multi-class streams (labels 0..K-1)."""

    def __init__(self, ctor, K):
        self.ms = [ctor() for _ in range(K)]

    def predict_label(self, x):
        return int(np.argmax([m.predict(x) for m in self.ms]))

    def update(self, x, lab):
        for c, m in enumerate(self.ms):
            m.update(x, 1.0 if c == lab else -1.0)

    @property
    def size(self):
        return sum(m.size for m in self.ms)

    @property
    def n_events(self):
        return sum(m.n_events for m in self.ms)


def prequential(model, X, y):
    n = len(y)
    correct = np.zeros(n)
    sizes = np.zeros(n)
    multi = isinstance(model, OVR)
    for t in range(n):
        if multi:
            correct[t] = float(model.predict_label(X[t]) == y[t])
        else:
            yh = model.predict(X[t])
            correct[t] = float((1.0 if yh >= 0 else -1.0) == y[t])
        model.update(X[t], y[t])
        sizes[t] = model.size
    return correct, sizes


def build(ctor, y):
    K = len(np.unique(y))
    return OVR(ctor, K) if (K > 2 or y.min() >= 0) else ctor()


def run_one(args):
    bench, seed, P, only = args
    X, y, taus, dr = get_stream(bench, seed)
    n, dim = X.shape
    rows, curves = [], {}
    for name, ctor in make_models(dim, P, BUDGET[bench], only).items():
        t0 = time.time()
        m = build(ctor, y)
        correct, sizes = prequential(m, X, y)
        c = correct[taus[0]:]
        row = dict(bench=bench, seed=seed, method=name, overall=correct.mean(), post=c.mean(),
                   first300=np.mean([correct[t:t + 300].mean() for t in taus]),
                   recovery=np.mean(recovery_times(correct, taus, n)),
                   dict_size=sizes.mean(), events=m.n_events,
                   us_per_step=1e6 * (time.time() - t0) / n)
        if dr is not None:
            d = dr[taus[0]:]
            row["post_unchanged"] = c[~d].mean() if (~d).any() else np.nan
            row["post_drifted"] = c[d].mean()
        rows.append(row)
        curves[name] = correct
        curves[name + "|size"] = sizes
    return rows, curves, taus


# ---------------------------------------------------------------- tuning
def _score(args):
    bench, P, which = args
    X, y, taus, _ = get_stream(bench, 0)
    X, y = X[:taus[0]], y[:taus[0]]
    m = build(list(make_models(X.shape[1], P, BUDGET[bench], [which]).values())[0], y)
    return prequential(m, X, y)[0].mean()


def tune(bench):
    X, y, taus, _ = get_stream(bench, 0)
    d = X.shape[1]
    s0 = np.sqrt(d)
    grid = [dict(sigma=s, mu=mu, delta=dl) for s in (0.25 * s0, 0.5 * s0, s0, 2 * s0)
            for mu in (0.05, 0.2, 0.5) for dl in (0.2, 0.5, 0.8)]
    filler = dict(eps_q=1.0, lam_t=0.999, sn2=0.1)
    with Pool(4) as p:
        sc = p.map(_score, [(bench, dict(g, **filler), "KLMS-Dict (FIFO)") for g in grid])
    best = dict(grid[int(np.argmax(sc))], **filler)
    print("KLMS-Dict grid best", best, max(sc))
    g2 = [dict(best, eps_q=e * best["sigma"]) for e in (0.25, 0.5, 1.0)]
    with Pool(4) as p:
        sc2 = p.map(_score, [(bench, g, "FB-QKLMS") for g in g2])
    best["eps_q"] = g2[int(np.argmax(sc2))]["eps_q"]
    g3 = [dict(best, lam_t=l, sn2=s) for l in (0.99, 0.995, 0.999) for s in (0.01, 0.1, 1.0)]
    with Pool(4) as p:
        sc3 = p.map(_score, [(bench, g, "KRLS-T") for g in g3])
    k = int(np.argmax(sc3))
    best["lam_t"], best["sn2"] = g3[k]["lam_t"], g3[k]["sn2"]
    print("final", best, "FBQ", max(sc2), "KRLST", max(sc3))
    json.dump(best, open(os.path.join(OUT, f"params_{bench}.json"), "w"), indent=1)
    return best


if __name__ == "__main__":
    bench = sys.argv[1]
    if bench.startswith("tune_"):
        tune(bench[5:])
        sys.exit()
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    P = json.load(open(os.path.join(OUT, f"params_{bench}.json")))
    with Pool(4) as p:
        res = p.map(run_one, [(bench, s, P, None) for s in range(n_seeds)])
    df = pd.DataFrame([r for rr, _, _ in res for r in rr])
    df.to_csv(os.path.join(OUT, f"raw_{bench}.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, f"curves_{bench}.npz"),
                        taus=np.array(res[0][2]),
                        **{f"{s}|{k}": v for s, (_, cv, _) in enumerate(res) for k, v in cv.items()})
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    print(df.groupby("method").mean(numeric_only=True).drop(columns="seed").round(3))
