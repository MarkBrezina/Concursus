"""Insurance: claim processes, aggregate claims, premium principles, surplus and ruin.

The classic collective-risk model.  Claims *arrive* according to a counting
process and each claim has a *size* drawn from a severity distribution::

    from stochastics.finance import insurance as ins

    claims = ins.ClaimProcess(ins.Poisson(rate=120),                 # 120 claims a year
                              ins.weibull(shape=1.4, scale=900))     # Weibull claim sizes
    claims.expected_total(T=1)        # E[N(1)] * E[X]: the expected total claims for the year
    A = claims.aggregate(T=1)         # whole distribution of the year's total claims S
    ins.premium_table(A)              # every premium principle side by side

    firm = ins.Company.priced(claims, capital=50_000, principle='std_dev', beta=0.5)
    firm.ruin_probability(horizon=10) # Monte-Carlo, exact at claim epochs
    firm.ruin_exact([0, 20e3, 50e3])  # Pollaczek-Khinchine bracket (Poisson arrivals)
    firm.plot()                       # surplus paths, survival curve, ruin vs capital, premium

Arrival processes: :class:`Poisson` (interarrival times Exp(rate)),
:class:`Renewal` (any interarrival law, e.g. Weibull gaps) and :class:`Hawkes`
(self-exciting: one claim makes the next more likely, e.g. storms, epidemics).
Severities are any frozen ``scipy.stats`` distribution; ``weibull``,
``lognormal``, ``gamma``, ``pareto`` and ``exponential`` are shortcuts.

Premium principles (see :data:`PRINCIPLES`): expected value, variance, standard
deviation, percentile (VaR), TVaR, exponential (zero-utility), Esscher,
proportional hazard and Wang transform.  Exponential and Esscher premiums need
the moment generating function of the claims, which does not exist for
lognormal, Pareto or Weibull with shape < 1 claims: those return ``inf``
rather than a number that depends on where the sample happened to stop.

Conventions: time is in years; surplus is ``U(t) = u + c t - S(t)`` with a
constant premium *rate* ``c``; ruin can only happen at a claim, so it is
detected exactly (no time grid).  Not modelled: interest on the surplus,
expenses, reinsurance, dependence between claim sizes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy import integrate, optimize, special, stats

from .. import _viz as V

__all__ = [
    'weibull', 'weibull_with_mean', 'lognormal', 'gamma', 'exponential', 'pareto',
    'Arrivals', 'Poisson', 'Renewal', 'Hawkes',
    'ClaimProcess', 'ClaimSample', 'AggregateClaims',
    'PRINCIPLES', 'premium', 'premium_table', 'compare_principles',
    'Company', 'SurplusResult',
]


# ====================================================================== severities
def weibull(shape: float, scale: float = 1.0):
    """Weibull claim sizes: ``P(X > x) = exp(-(x/scale)^shape)``; mean ``scale * Gamma(1 + 1/shape)``."""
    return stats.weibull_min(shape, scale=scale)


def weibull_with_mean(shape: float, mean: float):
    """Weibull with the given shape and mean claim size."""
    return weibull(shape, mean / math.gamma(1.0 + 1.0 / shape))


def lognormal(mu: float, sigma: float):
    """Lognormal: ``log X ~ N(mu, sigma^2)``."""
    return stats.lognorm(sigma, scale=math.exp(mu))


def gamma(shape: float, scale: float = 1.0):
    return stats.gamma(shape, scale=scale)


def exponential(mean: float = 1.0):
    return stats.expon(scale=mean)


def pareto(alpha: float, xm: float = 1.0):
    """Pareto (type I) with tail index ``alpha`` on ``[xm, inf)``; variance is infinite for alpha <= 2."""
    return stats.pareto(alpha, scale=xm)


def _label(dist) -> str:
    """Readable name of a frozen scipy distribution: ``weibull_min(c=1.4, scale=1097)``."""
    names = list(dist.dist.shapes.replace(',', ' ').split()) if dist.dist.shapes else []
    parts = [f'{n}={a:.4g}' for n, a in zip(names, dist.args)]
    parts += [f'{k}={v:.4g}' for k, v in dist.kwds.items()]
    return f"{dist.dist.name}({', '.join(parts)})"


def _scale(dist) -> float:
    if 'scale' in dist.kwds:
        return float(dist.kwds['scale'])
    k = dist.dist.numargs
    return float(dist.args[k + 1]) if len(dist.args) > k + 1 else 1.0


_NO_MGF = {'lognorm', 'pareto', 'lomax', 'invgamma', 'fisk', 'burr', 'burr12', 'cauchy', 't', 'levy', 'loggamma'}


def _mgf_ok(dist, r: float) -> bool:
    """Is E[exp(r X)] finite?  Known families are decided analytically; others by tail comparison."""
    if r <= 0:
        return True
    name = dist.dist.name
    if name in _NO_MGF:
        return False
    if name in ('expon', 'gamma'):
        return r < 1.0 / _scale(dist)
    if name == 'weibull_min':
        c = float(dist.args[0]) if dist.args else float(dist.kwds['c'])
        return True if c > 1.0 else (r < 1.0 / _scale(dist) if c == 1.0 else False)
    a, b = dist.isf(1e-6), dist.isf(1e-15)
    if not (np.isfinite(a) and np.isfinite(b)) or b <= a:
        return bool(np.isfinite(b))
    return (r * b + dist.logsf(b)) < (r * a + dist.logsf(a)) - 1e-9


def _mgf(dist, r: float, power: int = 0) -> float:
    """``E[X^power exp(r X)]`` by quadrature (``inf`` when it does not exist); ``power=1`` is M'(r)."""
    if not _mgf_ok(dist, r):
        return math.inf
    hi = dist.isf(1e-15)
    pts = [dist.ppf(q) for q in (0.5, 0.9, 0.99, 0.999)]
    val, _ = integrate.quad(lambda x: x ** power * math.exp(min(r * x, 700.0)) * dist.pdf(x), 0.0, hi,
                            points=[p for p in pts if 0 < p < hi], limit=300)
    return float(val)


# ======================================================================== arrivals
class Arrivals:
    """Claim arrival times on ``[0, T]``."""
    name = 'arrivals'

    def sample(self, T: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """``(path, time)`` of every claim in ``n`` independent years, sorted by path then time."""
        raise NotImplementedError

    def mean_count(self, T: float) -> float:
        raise NotImplementedError

    def var_count(self, T: float) -> tuple[float, bool]:
        """``(Var N(T), exact?)``."""
        raise NotImplementedError

    @property
    def rate(self) -> float:
        """Long-run claims per unit time."""
        raise NotImplementedError


@dataclass
class Poisson(Arrivals):
    """Homogeneous Poisson: ``N(T) ~ Poisson(rate T)``, interarrival times ``Exp(rate)``."""
    rate_: float

    def __init__(self, rate: float):
        if rate <= 0:
            raise ValueError('rate must be positive')
        self.rate_ = float(rate)

    name = 'Poisson'

    @property
    def rate(self) -> float:
        return self.rate_

    @property
    def interarrival(self):
        return stats.expon(scale=1.0 / self.rate_)

    def sample(self, T, n, rng):
        counts = rng.poisson(self.rate_ * T, n)
        path = np.repeat(np.arange(n), counts)
        times = rng.uniform(0.0, T, path.size)
        order = np.lexsort((times, path))
        return path, times[order]

    def mean_count(self, T):
        return self.rate_ * T

    def var_count(self, T):
        return self.rate_ * T, True

    def __repr__(self):
        return f'Poisson(rate={self.rate_:g})'


@dataclass
class Renewal(Arrivals):
    """Renewal process: i.i.d. gaps from any distribution, e.g. ``weibull(1.5, 0.01)``."""
    gap: object

    name = 'Renewal'

    @property
    def interarrival(self):
        return self.gap

    @property
    def rate(self) -> float:
        return 1.0 / float(self.gap.mean())

    def sample(self, T, n, rng):
        mu = float(self.gap.mean())
        k = int(T / mu + 6.0 * math.sqrt(max(T / mu, 1.0)) + 10)
        t = np.cumsum(self.gap.rvs(size=(n, k), random_state=rng), axis=1)
        while np.any(t[:, -1] <= T):
            more = np.cumsum(self.gap.rvs(size=(n, k), random_state=rng), axis=1)
            t = np.concatenate([t, t[:, -1:] + more], axis=1)
        mask = t <= T
        return np.nonzero(mask)[0], t[mask]

    def mean_count(self, T):
        mu, var = float(self.gap.mean()), float(self.gap.var())
        return T / mu + (var - mu * mu) / (2.0 * mu * mu)      # renewal function, large-T expansion

    def var_count(self, T):
        mu, var = float(self.gap.mean()), float(self.gap.var())
        return T * var / mu ** 3, False                        # asymptotic

    def __repr__(self):
        return f'Renewal(gap={self.gap.dist.name})'


@dataclass
class Hawkes(Arrivals):
    """Self-exciting arrivals: ``lambda(t) = mu + sum_i alpha exp(-beta (t - t_i))``.

    Each claim raises the claim intensity by ``alpha``, which then decays at rate
    ``beta``; the branching ratio ``alpha / beta`` (< 1) is the average number of
    follow-on claims each claim triggers.  Simulated exactly through its cluster
    (immigrant + offspring) representation.
    """
    mu: float
    alpha: float
    beta: float

    name = 'Hawkes'

    def __post_init__(self):
        if self.mu <= 0 or self.alpha < 0 or self.beta <= 0:
            raise ValueError('need mu > 0, alpha >= 0, beta > 0')
        if self.alpha >= self.beta:
            raise ValueError('need alpha < beta (branching ratio < 1)')

    @property
    def branching(self) -> float:
        return self.alpha / self.beta

    @property
    def rate(self) -> float:
        return self.mu / (1.0 - self.branching)

    def sample(self, T, n, rng):
        k0 = rng.poisson(self.mu * T, n)
        path = np.repeat(np.arange(n), k0)
        times = rng.uniform(0.0, T, path.size)
        parts_p, parts_t = [path], [times]
        pp, pt = path, times
        while pt.size:
            kids = rng.poisson(self.branching, pt.size)
            kp = np.repeat(pp, kids)
            kt = np.repeat(pt, kids) + rng.exponential(1.0 / self.beta, kp.size)
            keep = kt <= T
            pp, pt = kp[keep], kt[keep]
            parts_p.append(pp)
            parts_t.append(pt)
        path, times = np.concatenate(parts_p), np.concatenate(parts_t)
        order = np.lexsort((times, path))
        return path[order], times[order]

    def mean_count(self, T):
        lam, r = self.rate, self.beta - self.alpha
        return lam * T + (self.mu - lam) * (1.0 - math.exp(-r * T)) / r

    def var_count(self, T):
        return self.rate * T / (1.0 - self.branching) ** 2, False   # asymptotic

    def __repr__(self):
        return f'Hawkes(mu={self.mu:g}, alpha={self.alpha:g}, beta={self.beta:g})'


# ================================================================= sampled claims
@dataclass
class ClaimSample:
    """Claims from ``n`` independent periods ``[0, T]`` (flat arrays sorted by path, then time)."""
    path: np.ndarray
    time: np.ndarray
    size: np.ndarray
    n: int
    T: float

    def counts(self) -> np.ndarray:
        """Number of claims per path."""
        return np.bincount(self.path, minlength=self.n)

    def totals(self, t: float | None = None) -> np.ndarray:
        """Aggregate claims ``S(t)`` per path (default ``S(T)``)."""
        if t is None or t >= self.T:
            return np.bincount(self.path, weights=self.size, minlength=self.n)
        m = self.time <= t
        return np.bincount(self.path[m], weights=self.size[m], minlength=self.n)

    def interarrival(self) -> np.ndarray:
        """Gaps between consecutive claims (the first gap runs from time 0)."""
        if not self.time.size:
            return np.empty(0)
        first = np.r_[True, self.path[1:] != self.path[:-1]]
        prev = np.where(first, 0.0, np.r_[0.0, self.time[:-1]])
        return self.time - prev

    def _running(self) -> np.ndarray:
        """Cumulative claims within each path, evaluated at every claim."""
        if not self.size.size:
            return np.empty(0)
        cs = np.cumsum(self.size)
        first = np.r_[True, self.path[1:] != self.path[:-1]]
        base = np.maximum.accumulate(np.where(first, cs - self.size, 0.0))
        return cs - base

    def ruin_times(self, capital: float, premium_rate: float) -> np.ndarray:
        """First time ``u + c t - S(t) < 0`` per path (``inf`` if the path survives to ``T``)."""
        out = np.full(self.n, np.inf)
        if not self.size.size:
            return out
        bad = np.flatnonzero(capital + premium_rate * self.time - self._running() < 0.0)
        if bad.size:
            p, first = np.unique(self.path[bad], return_index=True)
            out[p] = self.time[bad[first]]
        return out

    def aggregate_on(self, grid: np.ndarray) -> np.ndarray:
        """``S(t)`` for every path on a time grid, shape ``(n, len(grid))``."""
        grid = np.asarray(grid, float)
        G = grid.size
        idx = np.minimum(np.searchsorted(grid, self.time, side='left'), G - 1)
        flat = np.bincount(self.path * G + idx, weights=self.size, minlength=self.n * G)
        return np.cumsum(flat.reshape(self.n, G), axis=1)

    def surplus_on(self, grid: np.ndarray, capital: float, premium_rate: float) -> np.ndarray:
        """``U(t) = u + c t - S(t)`` on a grid (ignores ruin: the process just keeps going)."""
        grid = np.asarray(grid, float)
        return capital + premium_rate * grid[None, :] - self.aggregate_on(grid)


# ================================================================ aggregate claims
class AggregateClaims:
    """Distribution of the total claims ``S`` over one period, as a discrete law ``(values, probs)``.

    Built either from Monte-Carlo totals (``from_sample``) or from the exact
    compound-Poisson pmf on a grid (``ClaimProcess.aggregate``); every method
    below works the same on both.
    """

    def __init__(self, values, probs=None, label: str = '', mgf_ok: Callable[[float], bool] | None = None,
                 source: str = 'sample', log_mgf_fn: Callable[[float], float] | None = None,
                 esscher_fn: Callable[[float], float] | None = None):
        values = np.asarray(values, float)
        probs = np.full(values.size, 1.0 / values.size) if probs is None else np.clip(np.asarray(probs, float), 0.0, None)
        order = np.argsort(values, kind='stable')
        self.values, self.probs = values[order], probs[order] / probs.sum()
        self.cdf_ = np.cumsum(self.probs)
        self.label, self.source = label, source
        self._mgf_ok = mgf_ok or (lambda a: True)
        self._log_mgf_fn, self._esscher_fn = log_mgf_fn, esscher_fn     # exact transforms when the law is known

    @classmethod
    def from_sample(cls, totals, **kw):
        return cls(np.asarray(totals, float), None, source='sample', **kw)

    # -- moments
    @property
    def mean(self) -> float:
        return float(np.dot(self.values, self.probs))

    @property
    def var(self) -> float:
        return float(np.dot((self.values - self.mean) ** 2, self.probs))

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    @property
    def skew(self) -> float:
        s = self.std
        return float(np.dot((self.values - self.mean) ** 3, self.probs) / s ** 3) if s > 0 else 0.0

    # -- distribution
    def cdf(self, x) -> np.ndarray:
        i = np.searchsorted(self.values, np.asarray(x, float), side='right')
        return np.where(i > 0, self.cdf_[np.clip(i - 1, 0, None)], 0.0)

    def quantile(self, q: float) -> float:
        """Smallest value with ``P(S <= x) >= q`` (the Value-at-Risk)."""
        return float(self.values[min(np.searchsorted(self.cdf_, q - 1e-12, side='left'), self.values.size - 1)])

    def tvar(self, q: float) -> float:
        """Tail value at risk (expected shortfall): ``VaR_q + E[(S - VaR_q)+] / (1 - q)``."""
        v = self.quantile(q)
        return v + float(np.dot(np.clip(self.values - v, 0.0, None), self.probs)) / (1.0 - q)

    # -- transforms
    def log_mgf(self, a: float) -> float:
        """``log E[exp(a S)]`` (``inf`` if the claims have no moment generating function at ``a``)."""
        if a == 0:
            return 0.0
        if not self._mgf_ok(a):
            return math.inf
        if self._log_mgf_fn is not None:
            return float(self._log_mgf_fn(a))
        keep = self.probs > 0
        return float(special.logsumexp(a * self.values[keep] + np.log(self.probs[keep])))

    def esscher(self, h: float) -> float:
        """``E[S e^{hS}] / E[e^{hS}]`` - the mean after tilting the law by ``e^{hS}``."""
        if h == 0:
            return self.mean
        if not self._mgf_ok(h):
            return math.inf
        if self._esscher_fn is not None:
            return float(self._esscher_fn(h))
        keep = self.probs > 0
        w = h * self.values[keep] + np.log(self.probs[keep])
        w = np.exp(w - w.max())
        return float(np.dot(self.values[keep], w) / w.sum())

    def distortion(self, g: Callable[[np.ndarray], np.ndarray]) -> float:
        """Distortion premium ``int_0^inf g(P(S > x)) dx`` for an increasing ``g`` with g(0)=0, g(1)=1."""
        x = self.values
        surv = np.clip(1.0 - self.cdf_[:-1], 0.0, 1.0)
        return float(x[0] * float(g(np.array(1.0))) + np.dot(g(surv), np.diff(x)))

    def summary(self) -> pd.Series:
        return pd.Series({'mean': self.mean, 'std': self.std, 'skew': self.skew,
                          'VaR 95%': self.quantile(0.95), 'VaR 99%': self.quantile(0.99),
                          'TVaR 99%': self.tvar(0.99), 'VaR 99.5%': self.quantile(0.995)},
                         name=self.label or 'aggregate claims')

    def plot(self, ax=None, premiums: dict | None = None, bins: int = 80):
        """Density of the aggregate claims, optionally with premium markers."""
        import matplotlib.pyplot as plt
        with V.themed():
            own = ax is None
            if own:
                fig, ax = plt.subplots(figsize=(7.5, 3.8))
            lo, hi = self.quantile(0.0005), self.quantile(0.9995)
            dens, edges = np.histogram(self.values, bins=bins, range=(lo, hi), weights=self.probs, density=True)
            ax.fill_between(0.5 * (edges[1:] + edges[:-1]), dens, step=None, color=V.BLUE, alpha=0.25, lw=0)
            ax.plot(0.5 * (edges[1:] + edges[:-1]), dens, color=V.BLUE)
            ax.axvline(self.mean, color=V.TEXT, lw=1.0, ls=':')
            ax.text(self.mean, ax.get_ylim()[1] * 0.97, ' E[S]', color=V.TEXT_2, fontsize=8, va='top')
            for k, (name, val) in enumerate((premiums or {}).items()):
                if np.isfinite(val) and lo < val < hi:
                    ax.axvline(val, color=V.SERIES[(k + 1) % 8], lw=1.1)
                    ax.text(val, ax.get_ylim()[1] * (0.9 - 0.07 * (k % 8)), f' {name}', color=V.SERIES[(k + 1) % 8],
                            fontsize=7.5, va='top')
            ax.set_xlabel('total claims in the period, S')
            ax.set_ylabel('density')
            ax.set_title(self.label or 'Aggregate claims')
            return ax


# =========================================================================== claim process
@dataclass
class ClaimProcess:
    """Claim arrivals x claim sizes, independent of each other."""
    arrivals: Arrivals
    severity: object

    # -- theory ------------------------------------------------------------
    def mean_claims(self, T: float = 1.0) -> float:
        """``E[N(T)]``."""
        return self.arrivals.mean_count(T)

    def expected_total(self, T: float = 1.0) -> float:
        """``E[S(T)] = E[N(T)] E[X]``: expected number of claims times the expected claim size."""
        return self.mean_claims(T) * float(self.severity.mean())

    def variance_total(self, T: float = 1.0) -> tuple[float, bool]:
        """``Var S = E[N] Var X + Var N (E X)^2``; the flag says whether ``Var N`` is exact."""
        vn, exact = self.arrivals.var_count(T)
        m, v = float(self.severity.mean()), float(self.severity.var())
        return self.mean_claims(T) * v + vn * m * m, exact

    def loss_ratio_table(self, T: float = 1.0) -> pd.Series:
        vt, exact = self.variance_total(T)
        return pd.Series({'E[N]': self.mean_claims(T), 'E[X]': float(self.severity.mean()),
                          'E[S]': self.expected_total(T), 'sd[S]': math.sqrt(vt),
                          'cv[S]': math.sqrt(vt) / self.expected_total(T),
                          'variance exact': exact}, name=f'{self.arrivals!r}, {self.severity.dist.name}')

    # -- simulation ----------------------------------------------------------
    def sample(self, T: float = 1.0, n: int = 10_000, seed: int | None = None) -> ClaimSample:
        rng = np.random.default_rng(seed)
        path, time = self.arrivals.sample(T, n, rng)
        size = self.severity.rvs(size=path.size, random_state=rng)
        return ClaimSample(path, time, np.asarray(size, float), n, float(T))

    def mgf_ok(self, a: float) -> bool:
        return _mgf_ok(self.severity, a)

    def aggregate(self, T: float = 1.0, n: int = 100_000, seed: int | None = None, method: str = 'auto',
                  grid: int = 2 ** 16) -> AggregateClaims:
        """Distribution of the total claims in ``[0, T]``.

        ``method='pmf'`` evaluates the compound-Poisson law exactly on a grid by
        FFT (Poisson arrivals with finite-variance claims); ``'mc'`` simulates
        ``n`` periods.  ``'auto'`` picks the pmf when it applies.
        """
        poisson = isinstance(self.arrivals, Poisson)
        finite = bool(np.isfinite(self.severity.var()))
        if method == 'auto':
            method = 'pmf' if poisson and finite else 'mc'
        label = f'{self.arrivals!r} x {self.severity.dist.name}, T={T:g}'
        exact = {}
        if poisson:        # compound Poisson: log E e^{aS} = lam T (M(a) - 1) and the Esscher mean is lam T M'(h)
            lamT = self.arrivals.mean_count(T)
            exact = dict(log_mgf_fn=lambda a: lamT * (_mgf(self.severity, a) - 1.0),
                         esscher_fn=lambda h: lamT * _mgf(self.severity, h, 1))
        if method == 'mc':
            return AggregateClaims.from_sample(self.sample(T, n, seed).totals(), label=label + ' (MC)',
                                               mgf_ok=self.mgf_ok, **exact)
        if not poisson or not finite:
            raise ValueError("the exact pmf needs Poisson arrivals and finite-variance claims; use method='mc'")
        lam = self.arrivals.mean_count(T)
        mean, sd = self.expected_total(T), math.sqrt(self.variance_total(T)[0])
        upper = mean + 14.0 * sd + 4.0 * float(self.severity.isf(1e-9))
        h = upper / grid
        edges = (np.arange(grid + 1) - 0.5) * h
        edges[0] = 0.0
        sev = np.diff(self.severity.cdf(edges))                  # P(X rounds to j h)
        phi = np.fft.rfft(sev)
        pmf = np.fft.irfft(np.exp(lam * (phi - 1.0)), grid)
        return AggregateClaims(np.arange(grid) * h, pmf, label=label + ' (exact pmf)', source='pmf',
                               mgf_ok=self.mgf_ok, **exact)

    def __repr__(self):
        return f'ClaimProcess({self.arrivals!r}, {_label(self.severity)})'


# ================================================================== premium principles
def _defaults(A: AggregateClaims, base: float = 0.5) -> dict:
    """Parameters that give every principle the same first-order loading ``base * sd[S]``.

    For roughly normal claims: expected value ``theta = base sd/mean``; variance
    ``alpha = base/sd``; exponential ``a = 2 base/sd`` (loading ``a sd^2/2``);
    Esscher ``h = base/sd``; Wang ``lam = base``.  Differences between them then
    come from skewness and tails - which is the point of comparing them.
    """
    sd = max(A.std, 1e-300)
    return {'expected_value': {'theta': base * sd / A.mean},
            'variance': {'alpha': base / sd},
            'std_dev': {'beta': base},
            'percentile': {'q': 0.99},
            'tvar': {'q': 0.99},
            'exponential': {'a': 2.0 * base / sd},
            'esscher': {'h': base / sd},
            'proportional_hazard': {'rho': 1.0 + base},
            'wang': {'lam': base}}


PRINCIPLES: dict[str, Callable[..., float]] = {
    'expected_value': lambda A, theta: (1.0 + theta) * A.mean,
    'variance': lambda A, alpha: A.mean + alpha * A.var,
    'std_dev': lambda A, beta: A.mean + beta * A.std,
    'percentile': lambda A, q: A.quantile(q),
    'tvar': lambda A, q: A.tvar(q),
    'exponential': lambda A, a: A.log_mgf(a) / a,
    'esscher': lambda A, h: A.esscher(h),
    'proportional_hazard': lambda A, rho: A.distortion(lambda p: np.power(p, 1.0 / rho)),
    'wang': lambda A, lam: A.distortion(lambda p: stats.norm.cdf(stats.norm.ppf(np.clip(p, 0.0, 1.0)) + lam)),
}


def premium(A: AggregateClaims, principle: str = 'expected_value', **params) -> float:
    """Premium for the period under one principle.

    ============== =============================== =========================================
    principle      formula                         parameter
    ============== =============================== =========================================
    expected_value ``(1 + theta) E[S]``            ``theta`` (relative loading)
    variance       ``E[S] + alpha Var[S]``         ``alpha``
    std_dev        ``E[S] + beta sd[S]``           ``beta``
    percentile     ``VaR_q(S)``                    ``q``
    tvar           ``TVaR_q(S)``                   ``q``
    exponential    ``log E[e^{aS}] / a``           ``a`` (risk aversion; needs the MGF)
    esscher        ``E[S e^{hS}] / E[e^{hS}]``     ``h`` (needs the MGF)
    proportional_hazard ``int P(S>x)^(1/rho) dx``  ``rho >= 1``
    wang           ``int Phi(Phi^-1(P(S>x)) + lam) dx``  ``lam >= 0``
    ============== =============================== =========================================

    Parameters left out take the loading-matched defaults of :func:`_defaults`.
    """
    if principle not in PRINCIPLES:
        raise KeyError(f'unknown principle {principle!r}; choose from {sorted(PRINCIPLES)}')
    p = {**_defaults(A)[principle], **params}
    return float(PRINCIPLES[principle](A, **p))


def premium_table(A: AggregateClaims, base: float = 0.5, **overrides) -> pd.DataFrame:
    """All principles side by side, with defaults matched to a first-order loading of ``base`` sd.

    ``overrides`` maps a principle to its parameters, e.g. ``std_dev={'beta': 1.0}``.
    """
    d = _defaults(A, base)
    rows = []
    for name in PRINCIPLES:
        params = {**d[name], **overrides.get(name, {})}
        val = float(PRINCIPLES[name](A, **params))
        rows.append({'principle': name, 'parameters': ', '.join(f'{k}={v:.4g}' for k, v in params.items()),
                     'premium': val, 'loading': val / A.mean - 1.0 if np.isfinite(val) else np.nan,
                     'loading in sd': (val - A.mean) / A.std if np.isfinite(val) else np.nan})
    return pd.DataFrame(rows).set_index('principle')


# ======================================================================== company, ruin
@dataclass
class SurplusResult:
    """Simulated surplus processes of a company: ruin times and the claims behind them."""
    claims: ClaimSample
    capital: float
    premium_rate: float
    horizon: float
    ruin_time: np.ndarray

    @property
    def n(self) -> int:
        return self.claims.n

    def ruin_probability(self, t: float | None = None) -> float:
        """Fraction of paths ruined by time ``t`` (default: the whole horizon)."""
        return float(np.mean(self.ruin_time <= (self.horizon if t is None else t)))

    def survival(self, grid) -> np.ndarray:
        """``P(no ruin by t)`` on a time grid."""
        rt = np.sort(self.ruin_time)
        return 1.0 - np.searchsorted(rt, np.asarray(grid, float), side='right') / self.n

    def interval(self, p: float, z: float = 1.96) -> tuple[float, float]:
        """Wilson interval for a Monte-Carlo probability."""
        n = self.n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return max(0.0, c - w), min(1.0, c + w)

    def summary(self) -> pd.Series:
        p = self.ruin_probability()
        lo, hi = self.interval(p)
        ruined = self.ruin_time[np.isfinite(self.ruin_time)]
        return pd.Series({'paths': self.n, 'horizon': self.horizon, 'ruin probability': p,
                          'ci low': lo, 'ci high': hi,
                          'median ruin time': float(np.median(ruined)) if ruined.size else np.nan,
                          'claims per year': self.claims.counts().mean() / self.claims.T})


@dataclass
class Company:
    """An insurer: a claim process, initial capital ``u`` and a premium rate ``c`` per year."""
    claims: ClaimProcess
    capital: float
    premium_rate: float

    def __post_init__(self):
        if self.capital < 0 or self.premium_rate <= 0:
            raise ValueError('need capital >= 0 and premium_rate > 0')

    @classmethod
    def priced(cls, claims: ClaimProcess, capital: float, principle: str = 'expected_value',
               period: float = 1.0, **params) -> 'Company':
        """Charge the premium that ``principle`` gives for one ``period`` of claims, spread evenly over time."""
        A = claims.aggregate(period)
        return cls(claims, capital, premium(A, principle, **params) / period)

    # -- theory ---------------------------------------------------------------
    @property
    def claim_rate(self) -> float:
        return self.claims.arrivals.rate

    @property
    def safety_loading(self) -> float:
        """Relative safety loading ``c / (rate E[X]) - 1``; ruin is certain if it is <= 0."""
        return self.premium_rate / (self.claim_rate * float(self.claims.severity.mean())) - 1.0

    def adjustment_coefficient(self) -> float:
        """Lundberg exponent ``R > 0`` with ``lambda (M_X(R) - 1) = c R`` (Poisson arrivals).

        ``nan`` when it does not exist (non-positive loading or claims with a heavy tail).
        """
        if not isinstance(self.claims.arrivals, Poisson):
            raise ValueError('the adjustment coefficient is defined here for Poisson arrivals')
        lam, c, X = self.claim_rate, self.premium_rate, self.claims.severity
        if self.safety_loading <= 0 or not _mgf_ok(X, 1e-6 / float(X.mean())):
            return math.nan
        f = lambda r: lam * (_mgf(X, r) - 1.0) - c * r  # noqa: E731
        lo = 1e-6 / float(X.mean())
        hi = 10 * lo
        while f(hi) < 0.0 and _mgf_ok(X, 2 * hi) and hi < 1e6 / float(X.mean()):
            lo, hi = hi, 2 * hi
        if f(hi) < 0.0:                     # the root sits just below the abscissa of convergence
            a, b = hi, 2 * hi
            for _ in range(60):
                mid = 0.5 * (a + b)
                a, b = (mid, b) if _mgf_ok(X, mid) else (a, mid)
            hi = a * (1 - 1e-9)
            if f(hi) < 0.0:
                return math.nan
        return float(optimize.brentq(f, lo, hi, xtol=1e-12))

    def lundberg_bound(self, u) -> np.ndarray:
        """Upper bound ``psi(u) <= exp(-R u)`` on the infinite-horizon ruin probability."""
        return np.exp(-self.adjustment_coefficient() * np.asarray(u, float))

    def ruin_exact(self, u, n_grid: int = 4000, u_max: float | None = None) -> pd.DataFrame:
        """Infinite-horizon ruin probability ``psi(u)`` with guaranteed bounds (Poisson arrivals).

        Pollaczek-Khinchine: ``psi(u) = P(L_1 + ... + L_N > u)`` with ``N`` geometric
        (parameter ``rho = lambda E[X] / c``) and ladder heights ``L`` of density
        ``P(X > x) / E[X]``.  The ladder-height law is discretised once rounding
        down and once rounding up, and the geometric sum is evaluated by Panjer's
        recursion: the two results bracket the true value.
        """
        if not isinstance(self.claims.arrivals, Poisson):
            raise ValueError('Pollaczek-Khinchine applies to Poisson arrivals')
        u = np.atleast_1d(np.asarray(u, float))
        lam, c, X = self.claim_rate, self.premium_rate, self.claims.severity
        m = float(X.mean())
        rho = lam * m / c
        if rho >= 1.0:
            return pd.DataFrame({'u': u, 'lower': 1.0, 'upper': 1.0, 'lundberg': 1.0}).set_index('u')
        top = float(u_max if u_max is not None else max(u.max(), 1e-9) * 1.05 + 1e-9)
        K = int(n_grid)
        h = top / K
        # integral of the survival function over each grid cell: quad on the first cell (cusp), Gauss-Legendre after
        xi, wi = np.polynomial.legendre.leggauss(8)
        nodes = (np.arange(K)[:, None] + 0.5 * (1.0 + xi)[None, :]) * h
        cell = 0.5 * h * (X.sf(nodes) @ wi)
        cell[0] = integrate.quad(X.sf, 0.0, h, limit=200)[0]
        FI = np.r_[0.0, np.cumsum(cell) / m]                      # F_I(k h), k = 0..K

        def panjer(f: np.ndarray) -> np.ndarray:
            f0 = f[0]
            g = np.zeros(K + 1)
            g[0] = (1.0 - rho) / (1.0 - rho * f0)
            scale = rho / (1.0 - rho * f0)
            for k in range(1, K + 1):
                g[k] = scale * np.dot(f[1:k + 1], g[k - 1::-1][:k])
            return g

        f_up = np.r_[0.0, np.diff(FI)]                            # round ladder heights up: pessimistic
        f_dn = np.r_[np.diff(FI), 0.0]                            # round down: optimistic
        psi_up = 1.0 - np.cumsum(panjer(f_up))
        psi_dn = 1.0 - np.cumsum(panjer(f_dn))
        grid = np.arange(K + 1) * h
        out = pd.DataFrame({'u': u, 'lower': np.clip(np.interp(u, grid, psi_dn), 0, 1),
                            'upper': np.clip(np.interp(u, grid, psi_up), 0, 1)})
        R = self.adjustment_coefficient()
        out['lundberg'] = np.exp(-R * u) if np.isfinite(R) else np.nan
        return out.set_index('u')

    # -- simulation -------------------------------------------------------------------
    def simulate(self, horizon: float = 10.0, n: int = 10_000, seed: int | None = None) -> SurplusResult:
        """Simulate ``n`` surplus processes over ``horizon`` years (ruin detected exactly at claims)."""
        sample = self.claims.sample(horizon, n, seed)
        return SurplusResult(sample, self.capital, self.premium_rate, float(horizon),
                             sample.ruin_times(self.capital, self.premium_rate))

    def ruin_probability(self, horizon: float = 10.0, n: int = 10_000, seed: int | None = None) -> float:
        """Finite-horizon ruin probability by Monte Carlo."""
        return self.simulate(horizon, n, seed).ruin_probability()

    def plot(self, horizon: float = 10.0, n: int = 4000, seed: int | None = 0, show: bool = True):
        """Four panels: surplus paths, survival curve, ruin versus capital, one year of claims."""
        import matplotlib.pyplot as plt
        res = self.simulate(horizon, n, seed)
        grid = np.linspace(0.0, horizon, 241)
        U = res.claims.surplus_on(grid, self.capital, self.premium_rate)
        with V.themed():
            fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
            ax = axes[0, 0]
            lo, med, hi = np.percentile(U, [5, 50, 95], axis=0)
            ax.fill_between(grid, lo, hi, color=V.BLUE, alpha=0.12, lw=0, label='5-95% of paths')
            ruined = np.flatnonzero(np.isfinite(res.ruin_time))
            alive = np.flatnonzero(~np.isfinite(res.ruin_time))
            for p in alive[:14]:
                ax.plot(grid, U[p], color=V.BLUE, lw=0.7, alpha=0.55)
            for p in ruined[:6]:
                k = np.searchsorted(grid, res.ruin_time[p])
                ax.plot(grid[:k + 1], U[p, :k + 1], color=V.RED, lw=0.9, alpha=0.8)
                ax.plot(res.ruin_time[p], 0, 'o', color=V.RED, ms=3.5)
            ax.plot(grid, med, color=V.TEXT, lw=1.4, label='median')
            ax.axhline(0, color=V.TEXT_2, lw=0.9)
            ax.set_ylim(*V.robust_limits(U[:, ::8], 1, 99))
            ax.set_title('Surplus  U(t) = u + c t - S(t)')
            ax.set_xlabel('years')
            ax.legend(loc='upper left')
            ax = axes[0, 1]
            surv = res.survival(grid)
            lo_ci = np.array([1 - res.interval(1 - s)[1] for s in surv])
            hi_ci = np.array([1 - res.interval(1 - s)[0] for s in surv])
            ax.fill_between(grid, lo_ci, hi_ci, color=V.BLUE, alpha=0.2, lw=0)
            ax.plot(grid, surv, color=V.BLUE, label='simulated survival')
            if isinstance(self.claims.arrivals, Poisson) and self.safety_loading > 0:
                try:
                    psi = float(self.ruin_exact([self.capital])['upper'].iloc[0])
                    ax.axhline(1 - psi, color=V.ORANGE, ls='--', label='infinite horizon (P-K)')
                except Exception:      # noqa: BLE001 - the panel is still useful without the reference line
                    pass
            ax.set_ylim(max(0.0, min(surv.min(), 1.0) - 0.05), 1.01)
            ax.set_title('Probability the company is still alive')
            ax.set_xlabel('years')
            ax.legend(loc='upper right')
            ax = axes[1, 0]
            if isinstance(self.claims.arrivals, Poisson) and self.safety_loading > 0:
                top = max(self.capital * 2, 8 * self.claims.severity.mean() * 4)
                us = np.linspace(0.0, top, 60)
                ex = self.ruin_exact(us, n_grid=1500, u_max=top)
                ax.fill_between(us, ex['lower'], ex['upper'], color=V.BLUE, alpha=0.35, lw=0, label='exact bracket (P-K)')
                if np.isfinite(ex['lundberg']).all():
                    ax.plot(us, ex['lundberg'], color=V.ORANGE, ls='--', label='Lundberg bound')
                mc = self.claims.sample(horizon, n, seed)
                pts = np.linspace(0, top, 9)
                ax.plot(pts, [np.mean(mc.ruin_times(u, self.premium_rate) <= horizon) for u in pts], 'o',
                        color=V.TEXT, ms=4, label=f'Monte Carlo, {horizon:g} years')
                ax.axvline(self.capital, color=V.TEXT_2, lw=0.8, ls=':')
                ax.set_yscale('log')
                ax.set_ylim(1e-4, 1.3)
                ax.legend(loc='lower left')
            else:
                ax.text(0.5, 0.5, 'exact ruin formulas need Poisson arrivals\nand a positive safety loading',
                        ha='center', va='center', color=V.TEXT_2, transform=ax.transAxes)
            ax.set_title('Ruin probability against starting capital')
            ax.set_xlabel('initial capital u')
            ax = axes[1, 1]
            A = self.claims.aggregate(1.0, n=40_000, seed=seed)
            A.plot(ax=ax, premiums={'premium': self.premium_rate, 'VaR 99%': A.quantile(0.99)})
            ax.set_title('One year of claims against one year of premium')
            V.suptitle(fig, f'{self.claims!r}\ncapital {self.capital:,.0f}, premium {self.premium_rate:,.0f}/yr '
                            f'(safety loading {self.safety_loading:.1%}), ruin within {horizon:g}y: '
                            f'{res.ruin_probability():.2%}', y=1.0)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            return V.finish(fig, show)


def compare_principles(claims: ClaimProcess, capital: float, horizon: float = 10.0, period: float = 1.0,
                       n: int = 20_000, seed: int | None = 0, base: float = 0.5, **overrides) -> pd.DataFrame:
    """Price one book under every principle and see which premiums keep the company alive.

    All principles are evaluated on the *same* simulated claims, so differences in
    ruin probability come from the premium alone.  Principles that do not exist for
    these claims (``inf`` premium) are skipped.
    """
    A = claims.aggregate(period)
    table = premium_table(A, base, **overrides)
    sample = claims.sample(horizon, n, seed)
    rows = []
    for name, row in table.iterrows():
        if not np.isfinite(row['premium']):
            rows.append({'principle': name, 'parameters': row['parameters'], 'premium': np.inf,
                         'loading': np.nan, 'ruin probability': np.nan, 'ci low': np.nan, 'ci high': np.nan})
            continue
        c = row['premium'] / period
        res = SurplusResult(sample, capital, c, float(horizon), sample.ruin_times(capital, c))
        p = res.ruin_probability()
        lo, hi = res.interval(p)
        rows.append({'principle': name, 'parameters': row['parameters'], 'premium': row['premium'],
                     'loading': row['loading'], 'ruin probability': p, 'ci low': lo, 'ci high': hi})
    return pd.DataFrame(rows).set_index('principle')
