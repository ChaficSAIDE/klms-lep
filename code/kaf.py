"""Kernel adaptive filters with budget / drift handling, all with a common interface.

Every model exposes:
    predict(x) -> float        (a priori output)
    update(x, y) -> float      (test-then-train step; returns a priori error)
    size                       (current dictionary size)
    n_events                   (number of prune / reset events, for reporting)

Binary classification is done by regressing on y in {-1,+1}; the label is sign(output).
"""
import numpy as np
from river.drift import ADWIN


def gauss(A, x, sigma):
    """Gaussian kernel between rows of A (n,d) and x (d,)."""
    if A.shape[0] == 0:
        return np.zeros(0)
    d2 = np.sum((A - x) ** 2, axis=1)
    return np.exp(-d2 / (2.0 * sigma ** 2))


class _DictKLMS:
    """Coherence-sparsified KLMS (KLMS-Dict), shared backbone for several variants."""

    def __init__(self, dim, sigma=1.0, mu=0.2, delta=0.5, budget=40):
        self.sigma, self.mu, self.delta, self.budget = sigma, mu, delta, budget
        self.A = np.zeros((0, dim))
        self.w = np.zeros(0)
        self.age = np.zeros(0, dtype=int)
        self.n_events = 0
        self._k = None

    @property
    def size(self):
        return self.A.shape[0]

    def predict(self, x):
        self._k = gauss(self.A, x, self.sigma)
        return float(self._k @ self.w) if self.size else 0.0

    # hooks -------------------------------------------------------------
    def _after_error(self, x, e, k):
        """Called after the a priori error is known, before insertion/update."""

    def _evict_index(self):
        return int(np.argmax(self.age))  # FIFO: oldest atom

    def _on_insert(self):
        pass

    def _on_remove(self, keep):
        pass

    # -------------------------------------------------------------------
    def _remove(self, keep):
        self.A, self.w, self.age = self.A[keep], self.w[keep], self.age[keep]
        self._on_remove(keep)

    def update(self, x, y):
        yhat = self.predict(x)
        e = y - yhat
        k = self._k
        self.age += 1
        self._after_error(x, e, k)
        k = gauss(self.A, x, self.sigma)  # dictionary may have changed
        if self.size == 0 or np.max(k) < self.delta:
            if self.size >= self.budget:
                j = self._evict_index()
                keep = np.ones(self.size, bool)
                keep[j] = False
                self._remove(keep)
                k = k[keep]
            self.A = np.vstack([self.A, x])
            self.w = np.append(self.w, 0.0)
            self.age = np.append(self.age, 0)
            k = np.append(k, 1.0)
            self._on_insert()
        self.w += self.mu * e * k
        return e


class KLMSDict(_DictKLMS):
    """KLMS-Dict with FIFO eviction at a fixed budget."""


class ADWINRestart(_DictKLMS):
    """KLMS-Dict; the whole dictionary is cleared at every ADWIN alarm."""

    def __init__(self, dim, adwin_delta=0.002, **kw):
        super().__init__(dim, **kw)
        self.adwin_delta = adwin_delta
        self.det = ADWIN(delta=adwin_delta)

    def _after_error(self, x, e, k):
        self.det.update(float(abs(e)))
        if self.det.drift_detected:
            self._remove(np.zeros(self.size, bool))
            self.det = ADWIN(delta=self.adwin_delta)
            self.n_events += 1


class DriftDict(_DictKLMS):
    """KLMS-DriftDict baseline: ADWIN alarm -> prune the
    fraction p of atoms with lowest recency-relevance score."""

    def __init__(self, dim, adwin_delta=0.002, alpha=0.5, p=0.4, L=50, **kw):
        super().__init__(dim, **kw)
        self.adwin_delta, self.alpha, self.p, self.L = adwin_delta, alpha, p, L
        self.det = ADWIN(delta=adwin_delta)
        self.buf = []

    def _after_error(self, x, e, k):
        self.buf.append(x)
        if len(self.buf) > self.L:
            self.buf.pop(0)
        self.det.update(float(abs(e)))
        if self.det.drift_detected and self.size > 0:
            r = 1.0 - self.age / (self.age.max() + 1e-12)
            W = np.array(self.buf)
            Kw = np.exp(-((W[:, None, :] - self.A[None]) ** 2).sum(-1) / (2 * self.sigma ** 2))
            v = (np.abs(self.w)[None] * Kw).mean(0)
            v = (v - v.min()) / (v.max() - v.min() + 1e-12)
            s = self.alpha * r + (1 - self.alpha) * v
            n_p = max(1, int(np.floor(self.p * self.size)))
            keep = np.ones(self.size, bool)
            keep[np.argsort(s)[:n_p]] = False
            self._remove(keep)
            self.det = ADWIN(delta=self.adwin_delta)
            self.n_events += 1


class LocalPrune(_DictKLMS):
    """Proposed: error-driven localized pruning (KLMS-LEP).

    For every atom i we keep two exponentially weighted statistics of the
    a priori residual r_t and the atom activation k_{t,i}=kappa(x_t, a_i):
        A_i = sum_t lam^{T-t} r_t k_{t,i},     B_i = sum_t lam^{T-t} k_{t,i}^2.
    Removing atom i changes every recent residual from r_t to r_t + w_i k_{t,i};
    the change in recent squared error is therefore
        Delta_i = sum_t (r_t + w_i k_{t,i})^2 - r_t^2 = w_i (2 A_i + w_i B_i).
    Delta_i < 0 means that, on the data the atom has recently seen, the zero
    weight explains the targets better than the atom's current weight: the
    atom is locally obsolete. We prune it once this is significant:
        -Delta_i > z * s_i,  s_i = 2 |w_i| sqrt(sigma_r^2 * C_i)
    where C_i = sum lam^{2(T-t)} k_{t,i}^2 is the variance proxy of A_i under
    a zero-mean residual of variance sigma_r^2 (tracked online).
    Cost: O(D) per step (k is already computed for the prediction).

    Budget eviction (dictionary full): the atom with the smallest Delta_i
    among atoms older than one effective window is evicted (least useful),
    falling back to FIFO if none is old enough.
    """

    def __init__(self, dim, lam=0.98, z=1.645, evict="delta", s2_rate=0.001,
                 var="global", **kw):
        super().__init__(dim, **kw)
        self.lam, self.z, self.evict, self.s2_rate = lam, z, evict, s2_rate
        self.var = var
        self.Vst = np.zeros(0)
        self.Ast = np.zeros(0)
        self.Bst = np.zeros(0)
        self.Cst = np.zeros(0)
        self.s2 = 1.0  # running residual variance
        self.win = int(round(1.0 / (1.0 - lam)))
        self.pruned_log = []  # (t, n_pruned)
        self.t = 0

    def _on_insert(self):
        self.Ast = np.append(self.Ast, 0.0)
        self.Bst = np.append(self.Bst, 0.0)
        self.Cst = np.append(self.Cst, 0.0)
        self.Vst = np.append(self.Vst, 0.0)

    def _on_remove(self, keep):
        self.Ast, self.Bst, self.Cst = self.Ast[keep], self.Bst[keep], self.Cst[keep]
        self.Vst = self.Vst[keep]

    def removal_gain(self):
        return self.w * (2 * self.Ast + self.w * self.Bst)

    def _after_error(self, x, e, k):
        self.t += 1
        if self.size == 0:
            return
        lam = self.lam
        self.Ast = lam * self.Ast + e * k
        self.Bst = lam * self.Bst + k * k
        self.Cst = lam * lam * self.Cst + k * k
        self.Vst = lam * lam * self.Vst + (e * k) ** 2
        D = self.removal_gain()
        if self.var == "self":   # self-normalised: Var(A_i) ~ sum lam^2(T-t) r_t^2 k_t^2
            sd = 2 * np.abs(self.w) * np.sqrt(self.Vst) + 1e-12
        else:
            sd = 2 * np.abs(self.w) * np.sqrt(self.s2 * self.Cst) + 1e-12
        bad = (-D > self.z * sd) & (self.age >= self.win // 2)
        if bad.any():
            self._remove(~bad)
            self.n_events += 1
            self.pruned_log.append((self.t, int(bad.sum())))
        # residual variance under the stationary hypothesis (slow EWMA)
        self.s2 = (1 - self.s2_rate) * self.s2 + self.s2_rate * e * e

    def _evict_index(self):
        if self.evict == "fifo":
            return int(np.argmax(self.age))
        old = self.age >= self.win
        if not old.any():
            return int(np.argmax(self.age))
        D = np.where(old, self.removal_gain(), np.inf)
        return int(np.argmin(D))


class QKLMS_FB:
    """Fixed-budget quantized KLMS (FB-QKLMS, Zhao et al., Signal Processing 2013).

    Quantization: if the nearest centre is within eps, its coefficient is
    updated (a_j += mu e); otherwise a new centre with coefficient mu e is
    added. When the budget is exceeded, the centre with the smallest
    significance is removed. Significance is approximated by
    |a_i| * n_i where n_i is an exponentially-weighted count of samples
    quantized to centre i (its empirical input mass), following the
    'significance' idea of FB-QKLMS.
    """

    def __init__(self, dim, sigma=1.0, mu=0.2, eps=0.5, budget=40, gamma=0.999):
        self.sigma, self.mu, self.eps, self.budget, self.gamma = sigma, mu, eps, budget, gamma
        self.C = np.zeros((0, dim))
        self.a = np.zeros(0)
        self.n = np.zeros(0)
        self.n_events = 0

    @property
    def size(self):
        return self.C.shape[0]

    def predict(self, x):
        return float(gauss(self.C, x, self.sigma) @ self.a) if self.size else 0.0

    def update(self, x, y):
        e = y - self.predict(x)
        self.n *= self.gamma
        if self.size:
            d = np.sqrt(((self.C - x) ** 2).sum(1))
            j = int(np.argmin(d))
            if d[j] <= self.eps:
                self.a[j] += self.mu * e
                self.n[j] += 1
                return e
        self.C = np.vstack([self.C, x])
        self.a = np.append(self.a, self.mu * e)
        self.n = np.append(self.n, 1.0)
        if self.size > self.budget:
            sig = np.abs(self.a[:-1]) * self.n[:-1]
            j = int(np.argmin(sig))
            keep = np.ones(self.size, bool)
            keep[j] = False
            self.C, self.a, self.n = self.C[keep], self.a[keep], self.n[keep]
        return e


class KRLST:
    """Kernel recursive least-squares tracker (Van Vaerenbergh et al., TNNLS 2012).

    Gaussian-process view of KRLS with back-to-prior forgetting (lam) and
    a fixed budget M; when the budget is exceeded, the basis element whose
    removal causes the smallest posterior-mean error is discarded.
    """

    def __init__(self, dim, sigma=1.0, lam=0.999, sn2=0.1, budget=40, jitter=1e-6):
        self.sigma, self.lam, self.sn2, self.M, self.jit = sigma, lam, sn2, budget, jitter
        self.B = np.zeros((0, dim))
        self.Q = np.zeros((0, 0))
        self.m = np.zeros(0)
        self.S = np.zeros((0, 0))
        self.n_events = 0

    @property
    def size(self):
        return self.B.shape[0]

    def predict(self, x):
        if self.size == 0:
            return 0.0
        k = gauss(self.B, x, self.sigma)
        return float(k @ (self.Q @ self.m))

    def update(self, x, y):
        yp = self.predict(x)
        e = y - yp
        kss = 1.0 + self.jit
        if self.size == 0:
            self.B = x[None].copy()
            self.Q = np.array([[1.0 / kss]])
            self.m = np.array([y * kss / (kss + self.sn2)])
            self.S = np.array([[kss - kss ** 2 / (kss + self.sn2)]])
            return e
        # forgetting (back-to-prior)
        K = np.exp(-((self.B[:, None] - self.B[None]) ** 2).sum(-1) / (2 * self.sigma ** 2))
        K += self.jit * np.eye(self.size)
        self.S = self.lam * self.S + (1 - self.lam) * K
        self.m = np.sqrt(self.lam) * self.m
        kbs = gauss(self.B, x, self.sigma)
        q = self.Q @ kbs
        ymean = q @ self.m
        gamma2 = max(kss - kbs @ q, 0.0)
        h = self.S @ q
        sf2 = max(gamma2 + q @ h, 0.0)
        sy2 = self.sn2 + sf2
        if gamma2 < 1e-4:  # reduced update, no new basis element
            p = h
            self.m = self.m + (y - ymean) / sy2 * p
            self.S = self.S - np.outer(p, p) / sy2
            return e
        p = np.append(h, sf2)
        self.m = np.append(self.m, ymean) + (y - ymean) / sy2 * p
        S = np.zeros((self.size + 1, self.size + 1))
        S[:-1, :-1] = self.S
        S[:-1, -1] = h
        S[-1, :-1] = h
        S[-1, -1] = sf2
        self.S = S - np.outer(p, p) / sy2
        Q = np.zeros((self.size + 1, self.size + 1))
        Q[:-1, :-1] = self.Q + np.outer(q, q) / gamma2
        Q[:-1, -1] = -q / gamma2
        Q[-1, :-1] = -q / gamma2
        Q[-1, -1] = 1.0 / gamma2
        self.Q = Q
        self.B = np.vstack([self.B, x])
        if self.size > self.M:
            err = np.abs(self.Q @ self.m) / np.diag(self.Q)
            r = int(np.argmin(err))
            keep = np.ones(self.size, bool)
            keep[r] = False
            Qs = self.Q[r, keep]
            qs = self.Q[r, r]
            self.Q = self.Q[np.ix_(keep, keep)] - np.outer(Qs, Qs) / qs
            self.m = self.m[keep]
            self.S = self.S[np.ix_(keep, keep)]
            self.B = self.B[keep]
        return e


class SWKRLS:
    """Sliding-window KRLS (fixed window = budget, Van Vaerenbergh 2006), regularised."""

    def __init__(self, dim, sigma=1.0, budget=40, reg=0.1):
        self.sigma, self.N, self.reg = sigma, budget, reg
        self.X = np.zeros((0, dim))
        self.Y = np.zeros(0)
        self.alpha = np.zeros(0)
        self.n_events = 0
        self._cnt = 0

    @property
    def size(self):
        return self.X.shape[0]

    def predict(self, x):
        return float(gauss(self.X, x, self.sigma) @ self.alpha) if self.size else 0.0

    def update(self, x, y):
        e = y - self.predict(x)
        self.X = np.vstack([self.X, x])[-self.N:]
        self.Y = np.append(self.Y, y)[-self.N:]
        K = np.exp(-((self.X[:, None] - self.X[None]) ** 2).sum(-1) / (2 * self.sigma ** 2))
        self.alpha = np.linalg.solve(K + self.reg * np.eye(self.size), self.Y)
        return e
