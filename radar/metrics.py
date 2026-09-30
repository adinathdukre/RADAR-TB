from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def f1_at(p, y, t) -> float:
    yp = (np.asarray(p) >= t).astype(int)
    y = np.asarray(y).astype(int)
    tp = int(((yp == 1) & (y == 1)).sum())
    fp = int(((yp == 1) & (y == 0)).sum())
    fn = int(((yp == 0) & (y == 1)).sum())
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def auc(p, y) -> float:
    p, y = np.asarray(p), np.asarray(y)
    a, b = p[y == 1], p[y == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    r = rankdata(np.r_[a, b])
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def cross_site_worst_f1(pa, ya, pb, yb, grid=None) -> float:
    grid = np.linspace(0.02, 0.98, 481) if grid is None else grid
    ta = max(grid, key=lambda t: f1_at(pa, ya, t))
    tb = max(grid, key=lambda t: f1_at(pb, yb, t))
    return min(f1_at(pb, yb, ta), f1_at(pa, ya, tb))


def bootstrap_worst_site_f1(site_data: dict, n_boot: int = 100, seed: int = 0,
                            plateau_tol: float = 0.01, grid=None) -> float:
    grid = np.linspace(0.05, 0.95, 181) if grid is None else grid
    rng = np.random.default_rng(seed)
    agg = np.zeros(len(grid))
    for _ in range(n_boot):
        res = [(np.asarray(p), np.asarray(l), rng.integers(0, len(l), len(l)))
               for p, l in site_data.values()]
        for ti, t in enumerate(grid):
            agg[ti] += min(f1_at(p[i], l[i], t) for p, l, i in res)
    agg /= n_boot
    best, lo = float(agg.max()), int(np.argmax(agg))
    hi = lo
    while lo - 1 >= 0 and agg[lo - 1] >= best - plateau_tol:
        lo -= 1
    while hi + 1 < len(agg) and agg[hi + 1] >= best - plateau_tol:
        hi += 1
    t = float(grid[(lo + hi) // 2])
    return float(min(f1_at(p, l, t) for p, l in site_data.values()))
