"""Figures for the revised paper (writes PDFs to ../figures/)."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Pool
import bench, kaf, streams

R = bench.OUT
P_OUT = os.path.join(bench.HERE, "..", "figures")
os.makedirs(P_OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
                     "pdf.fonttype": 42})
STY = {"KLMS-Dict (FIFO)": ("#7f7f7f", "--"), "ADWIN+Restart": ("#9467bd", ":"),
       "KRLS-T": ("#ff7f0e", "-."), "KLMS-LEP (proposed)": ("#1f77b4", "-"),
       "FB-QKLMS": ("#2ca02c", (0, (3, 1, 1, 1))), "KLMS-Dict (4x)": ("#d62728", (0, (5, 2)))}
LBL = {"KLMS-LEP (proposed)": "KLMS-LEP (proposed)", "KLMS-Dict (FIFO)": "KLMS-Dict (FIFO)",
       "KLMS-Dict (4x)": "KLMS-Dict (4$\\times$ budget)"}
W1 = 3.45  # IEEE column width (in)


def smooth(c, w=100):
    k = np.ones(w) / w
    return np.convolve(c, k, mode="same")


def curves_from_npz(bench_name, methods):
    z = np.load(os.path.join(R, f"curves_{bench_name}.npz"))
    taus = z["taus"]
    out = {}
    for m in methods:
        arr = np.array([z[k] for k in z.files if k.endswith("|" + m)])
        out[m] = arr
    return out, taus


def _run_curve(args):
    b, D, seed, m = args
    P = json.load(open(os.path.join(R, f"params_{b}.json")))
    X, y, taus, _ = bench.get_stream(b, seed)
    ctor = bench.make_models(X.shape[1], P, D, [m])[m]
    c, s = bench.prequential(bench.build(ctor, y), X, y)
    return b, D, seed, m, c, s


def fig_partial(n=10):
    meth = ["KLMS-Dict (FIFO)", "ADWIN+Restart", "KRLS-T", "KLMS-LEP (proposed)"]
    jobs = [("partial50", 160, s, m) for s in range(n) for m in meth]
    with Pool(4) as p:
        res = p.map(_run_curve, jobs)
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.3), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1]))
    for m in meth:
        C = np.array([r[4] for r in res if r[3] == m])
        S = np.array([r[5] for r in res if r[3] == m])
        col, ls = STY[m]
        mu = smooth(C.mean(0))
        ax[0].plot(mu, color=col, ls=ls, label=LBL.get(m, m))
        ax[1].plot(S.mean(0), color=col, ls=ls)
    for a in ax:
        for t in (1500, 3000):
            a.axvline(t, color="k", lw=0.6, alpha=0.4)
    ax[0].set_ylabel("Prequential accuracy")
    ax[0].set_ylim(0.45, 1.0)
    ax[0].legend(loc="lower left", ncol=2, frameon=False)
    ax[1].set_ylabel(r"$|\mathcal{D}_t|$")
    ax[1].set_xlabel("Time step")
    ax[1].set_xlim(0, 4500)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(P_OUT, "fig_partial_curves.pdf"))
    plt.close(fig)


def fig_budget():
    df = pd.read_csv(os.path.join(R, "budget_partial50_partial100.csv"))
    meth = ["KLMS-Dict (FIFO)", "ADWIN+Restart", "KRLS-T", "KLMS-LEP (proposed)",
            "LEP w/o pruning", "LEP w/ FIFO eviction"]
    sty = dict(STY)
    sty["LEP w/o pruning"] = ("#17becf", (0, (1, 1)))
    sty["LEP w/ FIFO eviction"] = ("#8c564b", (0, (4, 1, 1, 1, 1, 1)))
    fig, ax = plt.subplots(1, 2, figsize=(W1 * 2, 2.7))
    for j, b in enumerate(["partial50", "partial100"]):
        d = df[df.bench == b]
        for m in meth:
            g = d[d.method == m].groupby("D")["first300"]
            mu, sd = g.mean(), g.std()
            col, ls = sty[m]
            ax[j].errorbar(mu.index, mu.values, yerr=sd.values, color=col, ls=ls, marker="o",
                           ms=2.5, capsize=2, label=LBL.get(m, m))
        ax[j].set_xscale("log", base=2)
        ax[j].set_xticks([10, 20, 40, 80, 160]); ax[j].set_xticklabels([10, 20, 40, 80, 160])
        ax[j].set_xlabel(r"Budget $D_{\max}$")
        ax[j].set_title([r"Partial drift, $\rho=0.5$", r"Global drift, $\rho=1$"][j], fontsize=8)
    ax[0].set_ylabel("Accuracy, first 300 steps after drift")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(pad=0.3, rect=(0, 0.15, 1, 1))
    fig.savefig(os.path.join(P_OUT, "fig_budget.pdf"))
    plt.close(fig)


def fig_localization(seeds=10, lag=60):
    """Standardised removal gain of every atom `lag` steps after the first drift,
    split by whether the atom's region drifted (rho=0.5, D=160)."""
    P = json.load(open(os.path.join(R, "params_partial50.json")))
    zs = {True: [], False: []}
    for seed in range(seeds):
        X, y, taus, dr, reg = streams.partial_drift_stream(0.5, seed=seed, return_regions=True)
        m = kaf.LocalPrune(X.shape[1], budget=160, z=np.inf, sigma=P["sigma"], mu=P["mu"],
                           delta=P["delta"])  # z=inf: observe, never prune
        for t in range(taus[0] + lag):
            m.update(X[t], y[t])
        ph = slice(taus[0], taus[1])
        drifted_regions = set(reg[ph][dr[ph]])
        for a, D, w, C in zip(m.A, m.removal_gain(), m.w, m.Cst):
            t0 = int(np.where((X[:taus[0] + lag] == a).all(1))[0][0])
            if t0 >= taus[0]:          # only atoms that existed before the drift
                continue
            zstat = -D / (2 * abs(w) * np.sqrt(m.s2 * C) + 1e-12)
            zs[reg[t0] in drifted_regions].append(zstat)
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    bins = np.linspace(-5, 7, 49)
    ax.hist(np.clip(zs[False], -5, 7), bins=bins, alpha=0.65, color="#2ca02c",
            label=f"atoms in unchanged regions (n={len(zs[False])})", density=True)
    ax.hist(np.clip(zs[True], -5, 7), bins=bins, alpha=0.65, color="#d62728",
            label=f"atoms in drifted regions (n={len(zs[True])})", density=True)
    ax.axvline(1.645, color="k", ls="--", lw=0.8)
    ax.text(1.8, ax.get_ylim()[1] * 0.5, r"prune ($z=1.645$) $\rightarrow$", fontsize=7)
    ax.set_xlabel(r"Standardised removal gain $-\Delta_i/(2|w_i|\sqrt{s^2C_i})$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(P_OUT, "fig_localization.pdf"))
    plt.close(fig)
    zt, zf = np.array(zs[True]), np.array(zs[False])
    json.dump(dict(n_drift=len(zt), n_stable=len(zf), frac_drift_flag=float((zt > 1.645).mean()),
                   frac_stable_flag=float((zf > 1.645).mean())),
              open(os.path.join(R, "localization_stats.json"), "w"), indent=1)


def fig_real():
    fig, ax = plt.subplots(2, 1, figsize=(W1, 3.4))
    meth = ["KLMS-Dict (FIFO)", "ADWIN+Restart", "KRLS-T", "KLMS-LEP (proposed)"]
    for j, (b, w) in enumerate([("emg", 300), ("cwru4", 100)]):
        cv, taus = curves_from_npz(b, meth)
        for m in meth:
            col, ls = STY[m]
            ax[j].plot(smooth(cv[m].mean(0), w), color=col, ls=ls, label=LBL.get(m, m))
        for t in taus:
            ax[j].axvline(t, color="k", lw=0.6, alpha=0.4)
        ax[j].set_ylabel("Prequential accuracy")
        n = cv[meth[0]].shape[1]
        ax[j].set_xlim(w // 2, n - w // 2)
    ax[0].set_title("(a) EMG Physical Action, cross-subject drift", fontsize=8)
    ax[1].set_title("(b) CWRU 4-class, inner-race severity drift", fontsize=8)
    ax[1].set_xlabel("Time step")
    ax[0].legend(loc="lower right", ncol=2, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(P_OUT, "fig_real_curves.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["localization", "budget", "real", "partial"]
    for w in which:
        globals()["fig_" + w]()
        print("done", w)
