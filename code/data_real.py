"""Real-data streams (EMG Physical Action, CWRU), using the feature extraction of
the original submission (original_experiment.py) with cached features.
Seeds change the within-segment sample order (the drift structure is fixed)."""
import os, numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import original_experiment as OE

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
CACHE = os.path.join(HERE, "..", "results", "cache")
os.makedirs(CACHE, exist_ok=True)


def _emg_segments():
    f = os.path.join(CACHE, "emg_feats.npz")
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True)
        return list(z["X"]), list(z["y"])
    # original loader concatenates subjects; we keep the per-subject blocks
    X, y, taus = OE.load_emg(os.path.join(DATA, "emg", "EMG Physical Action Data Set"))
    b = [0] + list(taus) + [len(X)]
    Xs = [X[b[i]:b[i + 1]] for i in range(4)]
    ys = [y[b[i]:b[i + 1]] for i in range(4)]
    np.savez(f, X=np.array(Xs, dtype=object), y=np.array(ys, dtype=object))
    return Xs, ys


def _cwru_feats(fname):
    W, S = 256, 128
    mat = sio.loadmat(os.path.join(DATA, "cwru", f"{fname}.mat"))
    key = next(k for k in mat if "DE" in k and not k.startswith("_"))
    sig = mat[key].flatten().astype(float)
    # same 11 features as the original loader
    def feats(seg):
        fm = np.abs(np.fft.rfft(seg))
        return np.array([np.mean(seg), np.std(seg), np.ptp(seg), np.sqrt(np.mean(seg ** 2)),
                         np.mean(np.abs(seg)), fm[:40].max(), fm[:40].mean(), fm[40:80].sum(),
                         float(np.argmax(fm[:80])), np.percentile(np.abs(seg), 90),
                         np.diff(seg).std()])
    return np.array([feats(sig[i:i + W]) for i in range(0, len(sig) - W, S)])


def _cwru_segments(files):
    f = os.path.join(CACHE, "cwru_" + "_".join(files) + ".npz")
    if not os.path.exists(f):
        np.savez(f, normal=_cwru_feats("97"), **{k: _cwru_feats(k) for k in files})
    return np.load(f)


def preprocess(X, fit_end, n_comp):
    pca = PCA(n_components=min(n_comp, X.shape[1])).fit(X[:fit_end])
    Z = pca.transform(X)
    return StandardScaler().fit(Z[:fit_end]).transform(Z)


def emg_stream(seed=0, n_comp=32):
    Xs, ys = _emg_segments()
    rng = np.random.default_rng(seed)
    X, y, taus, n = [], [], [], 0
    for Xi, yi in zip(Xs, ys):
        p = rng.permutation(len(Xi))
        X.append(np.asarray(Xi, float)[p]); y.append(np.asarray(yi, float)[p])
        if n: taus.append(n)
        n += len(Xi)
    X, y = np.vstack(X), np.concatenate(y)
    return preprocess(X, taus[0], n_comp), 2 * y - 1, taus


def cwru_stream(seed=0, files=("105", "169", "209"), n_per_stage=350, n_comp=11):
    """Normal (-1) vs inner-race fault (+1); fault severity grows 0.007->0.014->0.021 in."""
    z = _cwru_segments(list(files))
    rng = np.random.default_rng(seed)
    norm = z["normal"]
    X, y, taus, n = [], [], [], 0
    for f in files:
        flt = z[f]
        m = min(n_per_stage, len(norm), len(flt))
        A = np.vstack([norm[rng.choice(len(norm), m, replace=False)],
                       flt[rng.choice(len(flt), m, replace=False)]])
        lab = np.r_[-np.ones(m), np.ones(m)]
        p = rng.permutation(2 * m)
        X.append(A[p]); y.append(lab[p])
        if n: taus.append(n)
        n += 2 * m
    X, y = np.vstack(X), np.concatenate(y)
    return preprocess(X, taus[0], n_comp), y, taus


def original_synthetic(seed=0):
    """Global-inversion stream of the original submission (with its preprocessing)."""
    X, y, d = OE.make_synthetic(n=3000, d=20, seed=seed)
    return OE.preprocess(X, d[0], 20), 2.0 * y - 1, list(d)


def cwru_fault_stream(seed=0, n_per_class=300, n_comp=11):
    """4-class fault diagnosis (normal / inner race / ball / outer race, 0.007 in)
    in which ONLY the inner-race class drifts: its severity grows
    0.007 -> 0.014 -> 0.021 in (files 105 -> 169 -> 209), while the other
    classes keep their signature. A real-data instance of partial drift.
    Labels are 0..3; 'drifted' marks inner-race samples."""
    z = _cwru_segments(["105", "169", "209", "118", "130"])
    rng = np.random.default_rng(seed)
    pools = {"N": z["normal"], "B": z["118"], "O": z["130"]}
    used = {k: rng.permutation(len(v)) for k, v in pools.items()}
    ptr = {k: 0 for k in pools}
    X, y, dr, taus, n = [], [], [], [], 0
    for ir in ("105", "169", "209"):
        parts, labs = [], []
        for lab, k in ((0, "N"), (2, "B"), (3, "O")):
            idx = used[k][ptr[k]:ptr[k] + n_per_class]
            ptr[k] += n_per_class
            if len(idx) < n_per_class:          # re-use if a pool runs short
                idx = rng.choice(len(pools[k]), n_per_class)
            parts.append(pools[k][idx]); labs.append(np.full(n_per_class, lab))
        f = z[ir]
        parts.append(f[rng.choice(len(f), n_per_class, replace=False)])
        labs.append(np.full(n_per_class, 1))
        A, L = np.vstack(parts), np.concatenate(labs)
        p = rng.permutation(len(L))
        X.append(A[p]); y.append(L[p]); dr.append(L[p] == 1)
        if n: taus.append(n)
        n += len(L)
    X = np.vstack(X)
    return preprocess(X, taus[0], n_comp), np.concatenate(y), taus, np.concatenate(dr)
