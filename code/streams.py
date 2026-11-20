"""Synthetic streams for the revised experiments."""
import numpy as np


def partial_drift_stream(rho, n_phase=1500, n_phases=3, d=4, K=8, sep=6.0,
                         noise=0.05, seed=0, off=1.5, spread=0.7, return_regions=False):
    """Localized (partial) concept drift.

    K regions (centres on a sphere of radius sep). Region c holds two Gaussian
    blobs at m_c +/- off*v_c (isotropic std `spread`) carrying opposite labels.
    At every phase change a random subset of round(rho*K) regions has its
    local concept replaced (the two labels swap inside that region; the input
    distribution is unchanged); the other regions keep their concept.
    rho = 1 is global (real) drift.

    Returns X (n,d), y (n,) in {-1,+1}, drift times, and the per-sample
    'drifted region' indicator for the phase the sample belongs to.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((K, d))
    M = sep * M / np.linalg.norm(M, axis=1, keepdims=True)
    V = rng.standard_normal((K, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    V0 = V.copy()  # geometry is fixed; only the concept (label map) drifts
    nd = int(round(rho * K))
    X, y, drifted, regions = [], [], [], []
    changed = np.zeros(K, bool)
    taus = []
    for ph in range(n_phases):
        if ph > 0:
            idx = rng.choice(K, nd, replace=False)
            V[idx] = -V[idx]
            changed = np.zeros(K, bool)
            changed[idx] = True
            taus.append(ph * n_phase)
        c = rng.integers(0, K, n_phase)
        s = rng.choice([-1.0, 1.0], n_phase)          # side of the local boundary
        x = M[c] + off * s[:, None] * V0[c] + spread * rng.standard_normal((n_phase, d))
        # label of each side is set by the (possibly flipped) local concept V
        lab = s * np.sign(np.einsum("nd,nd->n", V[c], V0[c]))
        flip = rng.random(n_phase) < noise
        lab[flip] *= -1
        X.append(x)
        y.append(lab)
        drifted.append(changed[c])
        regions.append(c)
    if return_regions:
        return np.vstack(X), np.concatenate(y), taus, np.concatenate(drifted), np.concatenate(regions)
    return np.vstack(X), np.concatenate(y), taus, np.concatenate(drifted)


def inversion_stream(seed=0, n_phase=1000, d=20):
    """The 3-phase global-inversion stream of the original submission."""
    rng = np.random.default_rng(seed)
    def phase(m0, m1, s, n):
        yy = rng.choice([-1.0, 1.0], n)
        mu = np.where(yy[:, None] < 0, m0, m1)
        return mu + s * rng.standard_normal((n, d)), yy
    one = np.ones(d)
    X0, y0 = phase(-one, one, 1.0, n_phase)
    X1, y1 = phase(2 * one, -2 * one, np.sqrt(0.5), n_phase)
    X2, y2 = phase(-0.8 * one, 0.8 * one, 1.0, n_phase)
    return (np.vstack([X0, X1, X2]), np.concatenate([y0, y1, y2]),
            [n_phase, 2 * n_phase], np.ones(3 * n_phase, bool))
