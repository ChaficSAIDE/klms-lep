"""LaTeX tables from results/*.csv (writes ../tables/tab_*.tex)."""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import bench

R = bench.OUT
P = os.path.join(bench.HERE, "..", "tables")
os.makedirs(P, exist_ok=True)
PROP = "KLMS-LEP (proposed)"
ORDER = ["KLMS-Dict (FIFO)", "KLMS-Dict (4x)", "FB-QKLMS", "SW-KRLS", "KRLS-T",
         "ADWIN+Restart", "KLMS-DriftDict", PROP]
NAME = {"KLMS-Dict (FIFO)": "KLMS-Dict (FIFO)", "KLMS-Dict (4x)": "KLMS-Dict ($4\\times$)$^\\ast$",
        "FB-QKLMS": "FB-QKLMS~\\cite{zhao2013fbqklms}", "SW-KRLS": "SW-KRLS~\\cite{vanvaerenbergh2006swkrls}",
        "KRLS-T": "KRLS-T~\\cite{vanvaerenbergh2012krlst}", "ADWIN+Restart": "ADWIN+Restart",
        "KLMS-DriftDict": "ADWIN+RR-Prune", PROP: "\\textbf{KLMS-LEP}",
        "LEP w/o pruning": "LEP w/o pruning (eviction only)",
        "LEP w/ FIFO eviction": "LEP w/ FIFO eviction (pruning only)"}


SHORT = {"LEP w/ FIFO eviction": "Pruning only (FIFO eviction)",
         "LEP w/o pruning": "Eviction only (no pruning)"}


def load(b):
    return pd.read_csv(os.path.join(R, f"raw_{b}.csv"))


def cell(df, m, col, best, mark=True, fmt="{:.3f}", sd=True):
    v = df[df.method == m][col].values
    s = fmt.format(v.mean())
    if sd:
        s += "{\\scriptsize$\\pm$" + f"{v.std(ddof=1):.3f}".lstrip("0") + "}"
    if np.isclose(v.mean(), best, atol=5e-4) and m != "KLMS-Dict (4x)":
        s = "\\textbf{" + s + "}"
    if mark and m != PROP:
        p = df[df.method == PROP].sort_values("seed")[col].values
        q = df[df.method == m].sort_values("seed")[col].values
        if np.any(p != q):
            st = wilcoxon(p, q)
            if st.pvalue < 0.01:
                s += "$^{+}$" if p.mean() > q.mean() else "$^{-}$"
    return s


def best_of(df, col, methods, higher=True):
    g = df[df.method.isin([m for m in methods if m != "KLMS-Dict (4x)"])].groupby("method")[col].mean()
    return g.max() if higher else g.min()


def tab_synthetic():
    benches = ["partial25", "partial50", "partial100"]
    dfs = {b: load(b) for b in benches}
    lines = []
    for m in ORDER:
        row = [NAME[m]]
        for b in benches:
            d = dfs[b]
            row.append(cell(d, m, "post", best_of(d, "post", ORDER)))
            row.append(cell(d, m, "first300", best_of(d, "first300", ORDER), sd=False))
        lines.append(" & ".join(row) + " \\\\")
        if m == "KLMS-DriftDict":
            lines.append("\\midrule")
    open(os.path.join(P, "tab_synthetic.tex"), "w").write("\n".join(lines) + "\n")


def tab_real():
    out = []
    for b in ["emg", "cwru4"]:
        d = load(b)
        out.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{'EMG Physical Action ($D_{\\max}=40$)' if b == 'emg' else 'CWRU 4-class ($D_{\\max}=25$ per class)'}}}}} \\\\")
        for m in ORDER:
            row = [NAME[m],
                   cell(d, m, "overall", best_of(d, "overall", ORDER)),
                   cell(d, m, "post", best_of(d, "post", ORDER)),
                   cell(d, m, "first300", best_of(d, "first300", ORDER), sd=False),
                   cell(d, m, "recovery", best_of(d, "recovery", ORDER, False), mark=False,
                        fmt="{:.0f}", sd=False),
                   f"{d[d.method == m]['us_per_step'].mean():.0f}"]
            out.append(" & ".join(row) + " \\\\")
        out.append("\\midrule")
    open(os.path.join(P, "tab_real.tex"), "w").write("\n".join(out[:-1]) + "\n")


def tab_ablation():
    rows = []
    sets = [("partial50", "Partial $\\rho{=}0.5$"), ("emg", "EMG"), ("cwru4", "CWRU-4")]
    ms = ["KLMS-Dict (FIFO)", "LEP w/ FIFO eviction", "LEP w/o pruning", PROP]
    dfs = {b: load(b) for b, _ in sets}
    for m in ms:
        r = [NAME[m]]
        for b, _ in sets:
            d = dfs[b]
            r.append(cell(d, m, "post", best_of(d, "post", ms), mark=False, sd=False))
            r.append(cell(d, m, "first300", best_of(d, "first300", ms), mark=False, sd=False))
        r[0] = SHORT.get(m, r[0])
        rows.append(" & ".join(r) + " \\\\")
    open(os.path.join(P, "tab_ablation.tex"), "w").write("\n".join(rows) + "\n")


def tab_sens():
    d = pd.read_csv(os.path.join(R, "sens_partial50_emg.csv"))
    g = d.groupby(["bench", "lam", "z"])["post"].mean().unstack("z")
    rows = []
    for lam in (0.95, 0.98, 0.99):
        r = [f"{lam:.2f}"]
        for b in ("partial50", "emg"):
            for z in (1.28, 1.645, 2.33):
                v = g.loc[(b, lam), z]
                s = f"{v:.3f}"
                if lam == 0.98 and z == 1.645:
                    s = "\\textbf{" + s + "}"
                r.append(s)
        rows.append(" & ".join(r) + " \\\\")
    open(os.path.join(P, "tab_sens.tex"), "w").write("\n".join(rows) + "\n")


def tab_largebudget():
    rows = []
    ms = ["KLMS-Dict (FIFO)", "ADWIN+Restart", "KRLS-T", PROP, "LEP w/o pruning", "LEP w/ FIFO eviction"]
    sets = [("budget_partial50_partial100.csv", "partial50", 160),
            ("budget_partial50_partial100.csv", "partial100", 160),
            ("budget_emg_D160.csv", "emg", 160), ("budget_cwru4_D100.csv", "cwru4", 100)]
    dfs = []
    for f, b, D in sets:
        p = os.path.join(R, f)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        dfs.append((b, D, d[(d.bench == b) & (d.D == D)]))
    for m in ms:
        r = [NAME[m]]
        for b, D, d in dfs:
            g = d.groupby("method")
            for col in ("post", "first300"):
                v = d[d.method == m][col]
                best = g[col].mean().max()
                s = f"{v.mean():.3f}"
                if np.isclose(v.mean(), best, atol=5e-4):
                    s = "\\textbf{" + s + "}"
                r.append(s)
            r.append(f"{d[d.method == m]['dict_size'].mean():.0f}")
        rows.append(" & ".join(r) + " \\\\")
    open(os.path.join(P, "tab_large.tex"), "w").write("\n".join(rows) + "\n")


def inline(tex="main_access.tex"):
    """Refresh the tables inlined in the paper between % BEGIN/% END markers."""
    import re
    f = os.path.join(P, tex)
    if not os.path.exists(f):
        return  # the paper source is not part of this repository
    s = open(f).read()
    def rep(m):
        return "% BEGIN " + m.group(1) + "\n" + open(os.path.join(P, m.group(1))).read().rstrip("\n") + "\n% END " + m.group(1)
    s = re.sub(r"% BEGIN (tab_[a-z]+\.tex)\n.*?\n% END \1", rep, s, flags=re.S)
    open(f, "w").write(s)


if __name__ == "__main__":
    for f in (tab_synthetic, tab_real, tab_ablation, tab_sens, tab_largebudget):
        f()
        print("ok", f.__name__)
    inline()
