"""
KLMS-DriftDict Experiment (Vectorized)
=======================================
Paper: Online Kernel Adaptive Filtering with Drift-Triggered Dictionary Pruning
       for Non-Stationary Pattern Recognition
Target: Pattern Recognition Letters (Elsevier)
Author: Chafic Saide

Comparison (honest framing):
  - KLMS-Dict (large budget): accumulates stale atoms, high memory
  - SSW-KRLS: fixed sliding-window forgetting
  - ADWIN+Restart: detect drift, reset dictionary (too aggressive)
  - KLMS-DriftDict: bounded budget + smart drift-triggered pruning (proposed)
"""

import os
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from river.drift import ADWIN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# VECTORIZED KERNEL FUNCTIONS
# ─────────────────────────────────────────────

def rbf_vec(D, x, sigma):
    """Vectorized RBF: D is (n,d) array, x is (d,). Returns (n,)."""
    diff = D - x
    return np.exp(-np.sum(diff**2, axis=1) / (2 * sigma**2))

# ─────────────────────────────────────────────
# BASE KLMS CLASS (NumPy arrays throughout)
# ─────────────────────────────────────────────

class KLMSBase:
    def __init__(self, sigma, mu, delta, D_max, d):
        self.sigma = sigma
        self.mu = mu
        self.delta = delta
        self.D_max = D_max
        self.d = d
        self.D = np.empty((0, d))    # dictionary atoms
        self.W = np.empty(0)          # weights
        self.ages = np.empty(0, int)  # age (steps since added)
        self.n = 0                    # step counter

    def _novelty(self, x):
        if len(self.D) == 0:
            return True
        return rbf_vec(self.D, x, self.sigma).max() < self.delta

    def predict(self, x):
        if len(self.D) == 0:
            return 0.0
        return float(self.W @ rbf_vec(self.D, x, self.sigma))

    def _admit(self, x, w_init=0.0):
        """Admit new atom; enforce budget by removing oldest if needed."""
        if len(self.D) >= self.D_max:
            oldest = np.argmax(self.ages)
            self.D    = np.delete(self.D, oldest, axis=0)
            self.W    = np.delete(self.W, oldest)
            self.ages = np.delete(self.ages, oldest)
        self.D    = np.vstack([self.D, x])
        self.W    = np.append(self.W, w_init)
        self.ages = np.append(self.ages, 0)

    def _update_weights(self, x, e):
        if len(self.D) == 0:
            return
        k = rbf_vec(self.D, x, self.sigma)
        self.W += self.mu * e * k

    def step(self, x, y):
        """One prequential step. Returns (y_hat, error)."""
        self.n += 1
        self.ages += 1
        y_hat = self.predict(x)
        e = y - y_hat
        if self._novelty(x):
            self._admit(x)
        self._update_weights(x, e)
        return y_hat, e

    def dict_size(self):
        return len(self.D)


# ─────────────────────────────────────────────
# 1. KLMS-Dict (large budget) — Baseline
# ─────────────────────────────────────────────

class KLMSDict(KLMSBase):
    """Standard KLMS-Dict. D_max set large to accumulate stale atoms."""
    pass


# ─────────────────────────────────────────────
# 2. SSW-KRLS — Sliding-Window Baseline
# ─────────────────────────────────────────────

class SSWKRLSDict(KLMSBase):
    """Sliding-window KRLS: always evict oldest, apply forgetting factor."""
    def __init__(self, sigma, mu, delta, D_max, d, forgetting=0.97):
        super().__init__(sigma, mu, delta, D_max, d)
        self.forgetting = forgetting

    def step(self, x, y):
        self.n += 1
        self.ages += 1
        self.W *= self.forgetting          # forgetting on all weights
        y_hat = self.predict(x)
        e = y - y_hat
        # Always slide: add x, enforce window size
        self.D    = np.vstack([self.D, x]) if len(self.D) else x[None]
        self.W    = np.append(self.W, 0.0)
        self.ages = np.append(self.ages, 0)
        if len(self.D) > self.D_max:
            self.D    = self.D[1:]
            self.W    = self.W[1:]
            self.ages = self.ages[1:]
        self._update_weights(x, e)
        return y_hat, e


# ─────────────────────────────────────────────
# 3. ADWIN + Restart — Naive Baseline
# ─────────────────────────────────────────────

class ADWINRestart(KLMSBase):
    def __init__(self, sigma, mu, delta, D_max, d, adwin_delta=0.002):
        super().__init__(sigma, mu, delta, D_max, d)
        self.adwin_delta = adwin_delta
        self.detector = ADWIN(delta=adwin_delta)
        self.n_drifts = 0

    def step(self, x, y):
        self.n += 1
        self.ages += 1
        y_hat = self.predict(x)
        e = y - y_hat
        self.detector.update(abs(e))
        if self.detector.drift_detected:
            # Full reset
            d = self.d
            self.D    = np.empty((0, d))
            self.W    = np.empty(0)
            self.ages = np.empty(0, int)
            self.detector = ADWIN(delta=self.adwin_delta)
            self.n_drifts += 1
        if self._novelty(x):
            self._admit(x)
        self._update_weights(x, e)
        return y_hat, e


# ─────────────────────────────────────────────
# 4. KLMS-DriftDict — PROPOSED
# ─────────────────────────────────────────────

class KLMSDriftDict(KLMSBase):
    def __init__(self, sigma, mu, delta, D_max, d,
                 alpha=0.5, prune_ratio=0.4, adwin_delta=0.002,
                 relevance_window=50):
        super().__init__(sigma, mu, delta, D_max, d)
        self.alpha = alpha
        self.prune_ratio = prune_ratio
        self.adwin_delta = adwin_delta
        self.relevance_window = relevance_window
        self.detector = ADWIN(delta=adwin_delta)
        self.recent_X = np.empty((0, d))
        self.n_drifts = 0

    def _prune(self, x):
        n = len(self.D)
        if n == 0:
            return
        # Recency score: newer = higher
        rec = 1.0 - self.ages / (self.ages.max() + 1e-9)

        # Relevance: mean |w_i * k(x_recent, a_i)| over recent window
        if len(self.recent_X) > 0:
            # K: (n_recent, n_dict)
            K = np.array([rbf_vec(self.D, xr, self.sigma)
                          for xr in self.recent_X])
            rel = np.mean(np.abs(self.W) * np.abs(K), axis=0)
            rel /= rel.max() + 1e-9
        else:
            rel = np.zeros(n)

        scores = self.alpha * rec + (1 - self.alpha) * rel
        n_prune = max(1, int(n * self.prune_ratio))
        keep = np.argsort(scores)[n_prune:]   # keep top scorers
        self.D    = self.D[keep]
        self.W    = self.W[keep]
        self.ages = self.ages[keep]

    def step(self, x, y):
        self.n += 1
        self.ages += 1

        # Track recent samples for relevance scoring
        self.recent_X = np.vstack([self.recent_X, x]) \
            if len(self.recent_X) else x[None]
        if len(self.recent_X) > self.relevance_window:
            self.recent_X = self.recent_X[-self.relevance_window:]

        y_hat = self.predict(x)
        e = y - y_hat

        self.detector.update(abs(e))
        if self.detector.drift_detected:
            self._prune(x)
            self.detector = ADWIN(delta=self.adwin_delta)
            self.n_drifts += 1

        if self._novelty(x):
            self._admit(x)
        self._update_weights(x, e)
        return y_hat, e


# ─────────────────────────────────────────────
# OVR CLASSIFIER
# ─────────────────────────────────────────────

def make_models(model_cls, n_classes, d, kwargs):
    return [model_cls(d=d, **kwargs) for _ in range(n_classes)]

def ovr_predict(models, x):
    return int(np.argmax([m.predict(x) for m in models]))

def ovr_update(models, x, y_true):
    y_pred = ovr_predict(models, x)
    for c, m in enumerate(models):
        m.step(x, 1.0 if c == y_true else -1.0)
    return y_pred

def total_dict_size(models):
    return sum(m.dict_size() for m in models)


# ─────────────────────────────────────────────
# PREQUENTIAL EVALUATION
# ─────────────────────────────────────────────

def run_experiment(X, y, n_classes, model_cls, kwargs, drift_point=None):
    """Test-then-train prequential evaluation."""
    n, d = X.shape
    models = make_models(model_cls, n_classes, d, kwargs)
    correct = np.zeros(n)
    dsizes  = np.zeros(n)

    for t in range(n):
        x = X[t]
        y_pred = ovr_update(models, x, int(y[t]))
        correct[t] = int(y_pred == int(y[t]))
        dsizes[t]  = total_dict_size(models)

        if (t + 1) % 500 == 0:
            w = correct[max(0, t-499):t+1]
            print(f"    t={t+1:5d} | acc={w.mean():.3f} | |D|={dsizes[t]:.0f}")

    acc_series = np.cumsum(correct) / (np.arange(n) + 1)
    return acc_series, dsizes, correct


# ─────────────────────────────────────────────
# SYNTHETIC DATASET
# ─────────────────────────────────────────────

def make_synthetic(n=3000, d=20, drift_pts=None, seed=42):
    """
    Binary stream with two abrupt drifts.
    Phase 0 [0,1000)   : N(-1,I) vs N(+1,I)
    Phase 1 [1000,2000) : N(+2,0.5I) vs N(-2,0.5I)   [inverted+tighter]
    Phase 2 [2000,3000) : N(-1.5,I) vs N(+1.5,I)      [partial return]
    """
    if drift_pts is None:
        drift_pts = [1000, 2000]
    np.random.seed(seed)
    params = [(-1., 1., 1.0), (2., -2., 0.5), (-1.5, 1.5, 1.0)]
    bounds = [(0, drift_pts[0]), (drift_pts[0], drift_pts[1]), (drift_pts[1], n)]
    X, y = [], []
    for (t0, t1), (m0, m1, s) in zip(bounds, params):
        nn = t1 - t0
        labels = np.random.randint(0, 2, nn)
        for lbl in labels:
            mu = m0 if lbl == 0 else m1
            X.append(np.random.randn(d) * s + mu)
            y.append(lbl)
    return np.array(X, dtype=float), np.array(y, int), drift_pts


# ─────────────────────────────────────────────
# EMG PHYSICAL ACTION DATASET LOADER
# ─────────────────────────────────────────────

def load_emg(emg_dir, window=200, step=50):
    """
    EMG Physical Action Data Set (UCI, Theodoridis 2011).
    4 subjects x (10 Normal + 10 Aggressive) actions, 8 EMG channels.
    
    Concept drift scenario: data concatenated subject-by-subject.
    Each subject transition = genuine distribution shift
    (different physiology, electrode placement, muscle strength).
    
    Task: binary classification — Normal (0) vs Aggressive (1).
    Returns: X (windows x features), y (binary), drift_indices (subject boundaries)
    """
    NORMAL_ACTIONS = ['Bowing','Clapping','Handshaking','Hugging',
                      'Jumping','Running','Seating','Standing','Walking','Waving']
    AGGR_ACTIONS   = ['Elbowing','Frontkicking','Hamering','Headering',
                      'Kneeing','Pulling','Punching','Pushing','Sidekicking','Slapping']

    def extract_features(seg):
        """16 features per channel x 8 channels = 128 features."""
        feats = []
        for ch in range(seg.shape[1]):
            s = seg[:, ch].astype(float)
            f = np.fft.rfft(s)
            fm = np.abs(f)
            feats.extend([
                np.mean(np.abs(s)),          # MAV
                np.sqrt(np.mean(s**2)),      # RMS
                np.std(s),                   # STD
                np.max(np.abs(s)),           # peak
                float(fm[:10].mean()),       # low-freq energy
                float(fm[10:30].mean()),     # mid-freq energy
                float(fm[30:].mean()),       # high-freq energy
                float(np.argmax(fm)),        # dominant freq
            ])
        return np.array(feats, dtype=float)

    def load_action_files(sub_dir, actions, label):
        """Load all windows from a list of action files."""
        segs, labs = [], []
        txt_dir = os.path.join(sub_dir, 'txt')
        for action in actions:
            fpath = os.path.join(txt_dir, f'{action}.txt')
            if not os.path.exists(fpath):
                continue
            data = np.loadtxt(fpath)
            for start in range(0, len(data) - window, step):
                seg = data[start:start + window]
                segs.append(extract_features(seg))
                labs.append(label)
        return segs, labs

    all_X, all_y, drift_indices = [], [], []
    n_total = 0

    for sub in ['sub1', 'sub2', 'sub3', 'sub4']:
        sub_dir = os.path.join(emg_dir, sub)
        norm_dir = os.path.join(sub_dir, 'Normal')
        aggr_dir = os.path.join(sub_dir, 'Aggressive')

        seg_n, lab_n = load_action_files(norm_dir, NORMAL_ACTIONS, 0)
        seg_a, lab_a = load_action_files(aggr_dir, AGGR_ACTIONS,   1)

        sub_X = np.array(seg_n + seg_a, dtype=float)
        sub_y = np.array(lab_n + lab_a, dtype=int)

        # Shuffle within subject
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(sub_X))
        sub_X, sub_y = sub_X[perm], sub_y[perm]

        if n_total > 0:
            drift_indices.append(n_total)

        all_X.append(sub_X)
        all_y.append(sub_y)
        n_total += len(sub_X)

    return np.vstack(all_X), np.concatenate(all_y), drift_indices


# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────

def load_uci_har(data_dir):
    X_tr = np.loadtxt(os.path.join(data_dir, 'train', 'X_train.txt'))
    y_tr = np.loadtxt(os.path.join(data_dir, 'train', 'y_train.txt')).astype(int) - 1
    X_te = np.loadtxt(os.path.join(data_dir, 'test',  'X_test.txt'))
    y_te = np.loadtxt(os.path.join(data_dir, 'test',  'y_test.txt')).astype(int) - 1
    return X_tr, y_tr, X_te, y_te

def make_har_stream(X_tr, y_tr, X_te, y_te, drift_pt=2000, n_total=4000, seed=42):
    rng = np.random.default_rng(seed)
    i_tr = rng.permutation(len(X_tr))[:drift_pt]
    i_te = rng.permutation(len(X_te))[:n_total - drift_pt]
    X_post = X_te[i_te] * 1.4   # 40% feature-scale shift
    X = np.vstack([X_tr[i_tr], X_post])
    y = np.concatenate([y_tr[i_tr], y_te[i_te]])
    return X, y, drift_pt

def load_cwru(cwru_dir, n_per_stage=400, seed=42):
    """Binary: normal(0) vs fault(1). 3 severity stages = 2 drift points."""
    W, S = 256, 128
    rng = np.random.default_rng(seed)

    def feats(seg):
        f = np.fft.rfft(seg)
        fm = np.abs(f)
        return np.array([
            np.mean(seg), np.std(seg),
            np.max(seg)-np.min(seg),
            np.sqrt(np.mean(seg**2)),
            np.mean(np.abs(seg)),
            float(fm[:40].max()),
            float(fm[:40].mean()),
            float(fm[40:80].sum()),
            float(np.argmax(fm[:80])),
            float(np.percentile(np.abs(seg), 90)),
            float(np.diff(seg).std()),
        ], dtype=float)

    def load_mat(fname):
        mat = sio.loadmat(os.path.join(cwru_dir, f'{fname}.mat'))
        key = next(k for k in mat if 'DE' in k and not k.startswith('_'))
        return mat[key].flatten().astype(float)

    normal = load_mat('97')
    # extract normal windows
    norm_segs = np.array([feats(normal[i:i+W])
                          for i in range(0, len(normal)-W, S)])

    all_X, all_y, drifts = [], [], []
    n_samples = 0
    for fname in ['105', '118', '130']:
        data = load_mat(fname)
        fault_segs = np.array([feats(data[i:i+W])
                               for i in range(0, len(data)-W, S)])
        nn = min(n_per_stage, len(norm_segs), len(fault_segs))
        i_n = rng.choice(len(norm_segs),  nn, replace=False)
        i_f = rng.choice(len(fault_segs), nn, replace=False)
        stage_X = np.vstack([norm_segs[i_n], fault_segs[i_f]])
        stage_y = np.array([0]*nn + [1]*nn)
        perm = rng.permutation(len(stage_X))
        if n_samples > 0:
            drifts.append(n_samples)
        all_X.append(stage_X[perm])
        all_y.append(stage_y[perm])
        n_samples += len(stage_X)

    return np.vstack(all_X), np.concatenate(all_y), drifts


# ─────────────────────────────────────────────
# PREPROCESSING (fit on pre-drift only)
# ─────────────────────────────────────────────

def preprocess(X, drift_pt, n_components):
    fit_end = drift_pt if drift_pt else len(X)
    n_comp = min(n_components, X.shape[1], fit_end - 1)
    pca = PCA(n_components=n_comp).fit(X[:fit_end])
    Xr  = pca.transform(X)
    sc  = StandardScaler().fit(Xr[:fit_end])
    return sc.transform(Xr)


# ─────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────

COLORS = {'KLMS-Dict (large)': '#e74c3c', 'SSW-KRLS': '#f39c12',
          'ADWIN+Restart': '#9b59b6', 'KLMS-DriftDict': '#27ae60'}
LINES  = {'KLMS-Dict (large)': '--', 'SSW-KRLS': '-.',
          'ADWIN+Restart': ':', 'KLMS-DriftDict': '-'}

def smooth(x, w=80):
    return np.convolve(x, np.ones(w)/w, mode='same')

def plot_acc(results, drifts, title, fname):
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, (acc, _, _) in results.items():
        ax.plot(smooth(acc), label=name,
                color=COLORS[name], ls=LINES[name], lw=2)
    for d in drifts:
        ax.axvline(d, color='gray', ls='--', alpha=0.6, lw=1)
        ax.text(d+10, 0.22, 'Drift', fontsize=8, color='gray', rotation=90)
    ax.set(xlabel='Time step', ylabel='Prequential Accuracy', title=title,
           ylim=[0.15, 1.05])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close(); print(f"  Saved: {fname}")

def plot_dsize(results, drifts, title, fname):
    fig, ax = plt.subplots(figsize=(9, 3))
    for name, (_, ds, _) in results.items():
        ax.plot(smooth(ds, 30), label=name,
                color=COLORS[name], ls=LINES[name], lw=2)
    for d in drifts:
        ax.axvline(d, color='gray', ls='--', alpha=0.6, lw=1)
    ax.set(xlabel='Time step', ylabel='|D_t|', title=title)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close(); print(f"  Saved: {fname}")

def plot_recovery(results, drift_pt, window, fname):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, (_, _, corr) in results.items():
        seg = corr[drift_pt:drift_pt+window]
        cum = np.cumsum(seg) / (np.arange(len(seg))+1)
        ax.plot(cum, label=name, color=COLORS[name], ls=LINES[name], lw=2)
    ax.set(xlabel='Steps after drift', ylabel='Recovery accuracy',
           title='Post-Drift Recovery Speed')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=150)
    plt.close(); print(f"  Saved: {fname}")

def print_table(results, drift_pt, label):
    print(f"\n{'='*62}")
    print(f"  RESULTS -- {label}")
    print(f"{'='*62}")
    print(f"  {'Method':<22} {'Overall':>9} {'Post-Drift':>11} {'Mean|D|':>9}")
    print(f"  {'-'*55}")
    for name, (_, ds, corr) in results.items():
        ov = corr.mean()
        pd = corr[drift_pt:].mean() if drift_pt < len(corr) else ov
        md = ds.mean()
        tag = " <<" if name == 'KLMS-DriftDict' else ""
        print(f"  {name:<22} {ov:>9.4f} {pd:>11.4f} {md:>9.1f}{tag}")
    print(f"{'='*62}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    SIG, MU, DELTA = 1.0, 0.08, 0.65
    D_SMALL = 30    # tight budget for bounded methods
    D_LARGE = 120   # 4x larger for static KLMS-Dict baseline

    BOUNDED_KW  = dict(sigma=SIG, mu=MU, delta=DELTA, D_max=D_SMALL)
    LARGE_KW    = dict(sigma=SIG, mu=MU, delta=DELTA, D_max=D_LARGE)
    SSW_KW      = dict(sigma=SIG, mu=MU, delta=DELTA, D_max=D_SMALL, forgetting=0.96)
    DRIFT_KW    = dict(sigma=SIG, mu=MU, delta=DELTA, D_max=D_SMALL,
                       alpha=0.5, prune_ratio=0.4)

    METHODS = {
        'KLMS-Dict (large)': (KLMSDict,      LARGE_KW),
        'SSW-KRLS':          (SSWKRLSDict,   SSW_KW),
        'ADWIN+Restart':     (ADWINRestart,  BOUNDED_KW),
        'KLMS-DriftDict':    (KLMSDriftDict, DRIFT_KW),
    }

    # ── EXP 0: SYNTHETIC ──────────────────────────────────────
    print("\n" + "="*62)
    print("  EXP 0: SYNTHETIC -- Controlled Abrupt Drift (binary)")
    print("="*62)

    X_s, y_s, dpts = make_synthetic(n=3000, d=20)
    # Preprocess: fit on pre-drift only
    Xs = preprocess(X_s, dpts[0], n_components=20)

    syn_res = {}
    for name, (cls, kw) in METHODS.items():
        print(f"\n  [{name}]")
        acc, ds, corr = run_experiment(Xs, y_s, 2, cls, kw, dpts[0])
        syn_res[name] = (acc, ds, corr)

    print_table(syn_res, dpts[0], "SYNTHETIC")
    plot_acc(syn_res, dpts, "Synthetic -- Prequential Accuracy (2 Abrupt Drifts)",
             "fig1_syn_accuracy.png")
    plot_dsize(syn_res, dpts, "Synthetic -- Dictionary Size", "fig2_syn_dictsize.png")
    plot_recovery(syn_res, dpts[0], 300, "fig3_syn_recovery.png")

    # ── EXP 1: UCI HAR ────────────────────────────────────────
    print("\n" + "="*62)
    print("  EXP 1: UCI HAR -- Subject-Swap + Feature-Scale Drift")
    print("="*62)

    HAR_DIR = os.path.join(os.path.dirname(RESULTS_DIR),
                           'paper2_experiment', 'uci_har', 'UCI HAR Dataset')
    try:
        X_tr, y_tr, X_te, y_te = load_uci_har(HAR_DIR)
        X_h, y_h, dp_h = make_har_stream(X_tr, y_tr, X_te, y_te,
                                          drift_pt=2000, n_total=4000)
        Xh = preprocess(X_h, dp_h, n_components=25)

        har_res = {}
        for name, (cls, kw) in METHODS.items():
            print(f"\n  [{name}]")
            acc, ds, corr = run_experiment(Xh, y_h, 6, cls, kw, dp_h)
            har_res[name] = (acc, ds, corr)

        print_table(har_res, dp_h, "UCI HAR")
        plot_acc(har_res, [dp_h], "UCI HAR -- Prequential Accuracy", "fig4_har_accuracy.png")
        plot_dsize(har_res, [dp_h], "UCI HAR -- Dictionary Size", "fig5_har_dictsize.png")
        plot_recovery(har_res, dp_h, 500, "fig6_har_recovery.png")

    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    # ── EXP 2: CWRU ───────────────────────────────────────────
    print("\n" + "="*62)
    print("  EXP 2: CWRU -- Fault Severity Drift (binary)")
    print("="*62)

    CWRU_DIR = os.path.join(os.path.dirname(RESULTS_DIR),
                            'paper_experiment', 'cwru_data')
    try:
        X_c, y_c, dp_c = load_cwru(CWRU_DIR, n_per_stage=350)
        Xc = preprocess(X_c, dp_c[0], n_components=11)

        # Binary task -> tighter budget
        C_BOUNDED = dict(sigma=0.8, mu=MU, delta=0.6, D_max=25)
        C_LARGE   = dict(sigma=0.8, mu=MU, delta=0.6, D_max=100)
        C_SSW     = dict(sigma=0.8, mu=MU, delta=0.6, D_max=25, forgetting=0.96)
        C_DRIFT   = dict(sigma=0.8, mu=MU, delta=0.6, D_max=25, alpha=0.5, prune_ratio=0.4)

        CWRU_METHODS = {
            'KLMS-Dict (large)': (KLMSDict,      C_LARGE),
            'SSW-KRLS':          (SSWKRLSDict,   C_SSW),
            'ADWIN+Restart':     (ADWINRestart,  C_BOUNDED),
            'KLMS-DriftDict':    (KLMSDriftDict, C_DRIFT),
        }

        cwru_res = {}
        for name, (cls, kw) in CWRU_METHODS.items():
            print(f"\n  [{name}]")
            acc, ds, corr = run_experiment(Xc, y_c, 2, cls, kw, dp_c[0])
            cwru_res[name] = (acc, ds, corr)

        print_table(cwru_res, dp_c[0], "CWRU")
        plot_acc(cwru_res, dp_c, "CWRU -- Prequential Accuracy (Fault Severity Drift)",
                 "fig7_cwru_accuracy.png")
        plot_dsize(cwru_res, dp_c, "CWRU -- Dictionary Size", "fig8_cwru_dictsize.png")
        plot_recovery(cwru_res, dp_c[0], 300, "fig9_cwru_recovery.png")

    except FileNotFoundError as e:
        print(f"  WARNING: {e}")

    # -- EXP 3: EMG Physical Action (direct link to Chef's SPL paper) ------
    print("\n" + "="*62)
    print("  EXP 3: EMG Physical Action -- Cross-Subject Drift (binary)")
    print("="*62)

    EMG_DIR = os.path.join(RESULTS_DIR, 'emg_data',
                           'EMG Physical Action Data Set')
    try:
        X_e, y_e, dp_e = load_emg(EMG_DIR, window=200, step=50)
        print(f"  EMG loaded: {len(X_e)} windows, {X_e.shape[1]} features")
        print(f"  Drift points (subject boundaries): {dp_e}")
        print(f"  Class balance: normal={int((y_e==0).sum())}, aggressive={int((y_e==1).sum())}")

        Xe = preprocess(X_e, dp_e[0], n_components=32)

        E_BOUNDED = dict(sigma=1.0, mu=0.05, delta=0.65, D_max=40)
        E_LARGE   = dict(sigma=1.0, mu=0.05, delta=0.65, D_max=160)
        E_SSW     = dict(sigma=1.0, mu=0.05, delta=0.65, D_max=40, forgetting=0.97)
        E_DRIFT   = dict(sigma=1.0, mu=0.05, delta=0.65, D_max=40,
                         alpha=0.5, prune_ratio=0.4)

        EMG_METHODS = {
            'KLMS-Dict (large)': (KLMSDict,      E_LARGE),
            'SSW-KRLS':          (SSWKRLSDict,   E_SSW),
            'ADWIN+Restart':     (ADWINRestart,  E_BOUNDED),
            'KLMS-DriftDict':    (KLMSDriftDict, E_DRIFT),
        }

        emg_res = {}
        for name, (cls, kw) in EMG_METHODS.items():
            print(f"\n  [{name}]")
            acc, ds, corr = run_experiment(Xe, y_e, 2, cls, kw, dp_e[0])
            emg_res[name] = (acc, ds, corr)

        print_table(emg_res, dp_e[0], "EMG Physical Action (Cross-Subject)")
        plot_acc(emg_res, dp_e,
                 'EMG Physical Action -- Prequential Accuracy (Cross-Subject Drift)',
                 'fig10_emg_accuracy.png')
        plot_dsize(emg_res, dp_e,
                   'EMG Physical Action -- Dictionary Size',
                   'fig11_emg_dictsize.png')
        plot_recovery(emg_res, dp_e[0], 500,
                      'fig12_emg_recovery.png')

    except Exception as e:
        print(f"  WARNING: {e}")
        import traceback; traceback.print_exc()

    print("\nDone. Figures saved to paper6_experiment/")


if __name__ == '__main__':
    main()
