"""Budget sweep (partial drift, rho=0.5) and sensitivity of the proposed
method to its two parameters (lam, z).

python sweep.py budget [n_seeds]
python sweep.py sens [n_seeds]
"""
import sys, os, json
import numpy as np
import pandas as pd
from multiprocessing import Pool
import bench, kaf

OUT = bench.OUT
METHODS = ["KLMS-Dict (FIFO)", "ADWIN+Restart", "KRLS-T", "KLMS-LEP (proposed)",
           "LEP w/o pruning", "LEP w/ FIFO eviction"]


def run_budget(args):
    D, seed, P, b = args
    X, y, taus, dr = bench.get_stream(b, seed)
    rows = []
    for name, ctor in bench.make_models(X.shape[1], P, D, METHODS).items():
        m = bench.build(ctor, y)
        c, s = bench.prequential(m, X, y)
        rows.append(dict(bench=b, D=D, seed=seed, method=name, post=c[taus[0]:].mean(),
                         first300=np.mean([c[t:t + 300].mean() for t in taus]),
                         dict_size=s.mean(), events=m.n_events))
    return rows


def run_sens(args):
    lam, z, seed, P, b = args
    X, y, taus, dr = bench.get_stream(b, seed)
    base = dict(sigma=P["sigma"], mu=P["mu"], delta=P["delta"])
    m = kaf.LocalPrune(X.shape[1], budget=bench.BUDGET[b], lam=lam, z=z, **base)
    c, s = bench.prequential(m, X, y)
    return dict(bench=b, lam=lam, z=z, seed=seed, post=c[taus[0]:].mean(),
                first300=np.mean([c[t:t + 300].mean() for t in taus]), events=m.n_events)


if __name__ == "__main__":
    mode = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    benches = sys.argv[3].split(",") if len(sys.argv) > 3 else ["partial50"]
    jobs, rows = [], []
    for b in benches:
        P = json.load(open(os.path.join(OUT, f"params_{b}.json")))
        if mode == "budget":
            Ds = [int(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else (10, 20, 40, 80, 160)
            jobs += [(D, s, P, b) for D in Ds for s in range(n)]
        else:
            jobs += [(l, z, s, P, b) for l in (0.95, 0.98, 0.99) for z in (1.28, 1.645, 2.33)
                     for s in range(n)]
    with Pool(4) as p:
        res = p.map(run_budget if mode == "budget" else run_sens, jobs)
    rows = [r for rr in res for r in (rr if isinstance(rr, list) else [rr])]
    df = pd.DataFrame(rows)
    tag = ("_D" + sys.argv[4].replace(",", "-")) if len(sys.argv) > 4 else ""
    df.to_csv(os.path.join(OUT, f"{mode}_{'_'.join(benches)}{tag}.csv"), index=False)
    pd.set_option("display.width", 250)
    key = ["bench", "D", "method"] if mode == "budget" else ["bench", "lam", "z"]
    print(df.groupby(key).mean(numeric_only=True).drop(columns="seed").round(3))
