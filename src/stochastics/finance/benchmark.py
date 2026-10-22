"""How closely do the models behave like real markets?  And what does each extra bit of fidelity cost?

    from stochastics.finance import realdata, benchmark as bm

    spx = realdata.sample('sp500')
    board = bm.compare(spx, [bm.Gaussian(), bm.GARCH(), bm.Heston(), bm.Traders(), bm.News()], calibrate_budget=70)
    board.table; board.plot()                         # ranking, and the per-fact errors as a heatmap

    fit = bm.calibrate(bm.Heston(), spx, budget=60)   # tune one model's free parameters to the series
    bm.assess(spx, fit.candidate).table()             # fact by fact: real, model, error
    bm.sweep({'S&P': spx, 'EUR': realdata.sample('eur_d')}, models, n_jobs=2)      # many series at once
    bm.holdout(spx, models)                           # tune on the first half, score on the second

How the comparison works
------------------------
*Fingerprint.*  Twelve scale-free statistics of a return series (:data:`FACT_NAMES`): skewness, kurtosis and the
Hill tail index (log), tail-to-body quantile ratio, ``acf(r)`` at lag 1, ``acf(|r|)`` at lags 1, 5 and 20
(volatility clustering), the leverage correlation, the dispersion of log realised volatility, and variance ratios
at 5 and 20 bars (momentum against mean reversion).  Returns are standardised first, so the *level* of volatility
never enters; sessions (overnight, weekend gaps) are respected: no autocorrelation pair spans a gap.

*Error.*  For each fact, the model's mean over ``replicates`` simulated series of the same size as the real one,
minus the real value, in units of the real value's own sampling noise (a bootstrap of the real series).  Below
about 2 a model cannot be told apart from the data on that fact.  ``rms_error`` is the root mean square of those
errors (each capped at 5), the same yardstick for every model, so wide-but-vague models are not rewarded.

*p-value.*  Is the real fingerprint a plausible draw from the model?  The sum of squared z-scores (distance from
the model's mean in units of the model's own spread) is compared with the same quantity for each simulated
replicate (leave one out).  The smallest possible value is ``1 / (replicates + 1)``.

*Calibration.*  :func:`calibrate` tunes a model's few free parameters by simulated method of moments (minimise a
Huber-type rms with common random numbers; random starts, then Nelder-Mead).  The fitted model is then re-scored
on fresh random numbers.  That removes the search's own simulation noise but not the fact that the real series is the
one it was tuned on; :func:`holdout` removes that too (tune on the first half of a series, score on the second).
Parameters that end on the edge of their allowed range are reported (``Fit.at_bounds``).

*Speed against accuracy.*  :func:`sweep` scores the same candidates on several series; ``Sweep.cost_accuracy``
and :func:`plot_frontier` show error against seconds per replicate with the Pareto-optimal points and the knee
(the fastest setting within a tolerance of the best error, or within one standard error of it).  :func:`frontier`
does the same for fidelity settings of one model (traders, population size, time step), :func:`budget_curve` for
the amount of fitting, :func:`convergence` for the number of replicates.

Limits: every statistic is estimated from one real sample, and a few of them (kurtosis, tail index) are noisy for
heavy-tailed data, which is why the bootstrap yardstick matters.  The block bootstrap (blocks of ``sqrt(n)`` bars) still
understates the sampling noise of the slowest-decaying statistics, so an error of 2-3 noise units for a long-memory
fact is weaker evidence than it looks.  Intraday series have a time-of-day pattern of volatility that the fingerprint
does not remove.  Models with more free parameters can fit more: the scoreboard does not penalise them (use
:func:`holdout`).  The agent-based :class:`Exchange` is assessed as designed, not tuned, and its replicates are
resampled from a pool of simulated days, which understates the model's own spread a little.
"""
from __future__ import annotations

import math
import time
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize

from .. import _viz as V
from .realdata import RealSeries

__all__ = ['FACT_NAMES', 'fingerprint', 'Param', 'Candidate', 'Gaussian', 'StudentT', 'GARCH', 'Heston', 'Ensemble',
           'Traders', 'TradersSV', 'News', 'Exchange', 'Assessment', 'Scoreboard', 'Sweep', 'Fit', 'assess', 'compare', 'sweep',
           'calibrate', 'frontier', 'knee', 'pareto', 'convergence', 'plot_frontier', 'budget_curve', 'plot_budget', 'holdout', 'ERROR_CAP']

FACT_NAMES = ('skew', 'log kurtosis', 'log tail index', 'tail/body ratio', 'acf(r) lag 1', 'acf(|r|) lag 1',
              'acf(|r|) lag 5', 'acf(|r|) lag 20', 'leverage', 'vol of vol', 'variance ratio 5', 'variance ratio 20')
ERROR_CAP = 5.0


# ===================================================================== fingerprint
def _segments(x) -> list[np.ndarray]:
    if isinstance(x, RealSeries):
        return x.segments()
    if isinstance(x, np.ndarray) and x.ndim == 1:
        return [x.astype(float)]
    return [np.asarray(s, float) for s in x if len(s)]


def fingerprint(segments, vol_window: int = 20) -> pd.Series:
    """The twelve scale-free statistics of :data:`FACT_NAMES` for returns given as one array or a list of sessions."""
    segs = _segments(segments)
    pool = np.concatenate(segs) if segs else np.empty(0)
    n = pool.size
    if n < 100:
        raise ValueError(f'need at least 100 returns, got {n}')
    sd = pool.std()
    if not sd > 0:
        raise ValueError('returns have zero variance')
    z = [(s - pool.mean()) / sd for s in segs]
    zp = np.concatenate(z)
    kurt = float(np.mean(zp ** 4))
    k = max(20, int(0.05 * n))
    a = np.sort(np.abs(zp))[::-1]
    hill = k / float(np.sum(np.log(a[:k] / a[k]))) if a[k] > 0 else np.nan
    q = np.quantile(zp, [0.01, 0.25, 0.75, 0.99])
    habs = [np.abs(x) for x in z]
    mabs = float(np.concatenate(habs).mean())
    habs = [x - mabs for x in habs]

    def acf(xs, lag):
        xs = [x for x in xs if len(x) > lag]          # sessions shorter than the lag have no pairs: leave them out
        num = sum(float(np.dot(x[:-lag], x[lag:])) for x in xs)
        den = sum(float(np.dot(x, x)) for x in xs)
        return num / den if den > 0 else np.nan

    x0 = np.concatenate([x[:-1] for x in z if len(x) > 1])
    y1 = np.concatenate([x[1:] ** 2 for x in z if len(x) > 1])
    lev = float(np.corrcoef(x0, y1)[0, 1])
    logrv = []
    for x in z:
        m = len(x) // vol_window
        if m:
            logrv.append(np.log(np.mean(x[:m * vol_window].reshape(m, vol_window) ** 2, axis=1) + 1e-12))
    vov = float(np.std(np.concatenate(logrv))) if logrv else np.nan

    def vr(lag):
        sums = []
        for x in z:
            if len(x) > lag:
                c = np.r_[0.0, np.cumsum(x)]
                sums.append(c[lag:] - c[:-lag])
        if not sums:
            return np.nan
        s = np.concatenate(sums)
        return float(np.mean(s ** 2) / lag)

    return pd.Series({
        'skew': float(np.mean(zp ** 3)), 'log kurtosis': math.log(kurt), 'log tail index': math.log(hill) if hill == hill else np.nan,
        'tail/body ratio': float((q[3] - q[0]) / (q[2] - q[1])), 'acf(r) lag 1': acf(z, 1),
        'acf(|r|) lag 1': acf(habs, 1), 'acf(|r|) lag 5': acf(habs, 5), 'acf(|r|) lag 20': acf(habs, 20),
        'leverage': lev, 'vol of vol': vov, 'variance ratio 5': vr(5), 'variance ratio 20': vr(20),
    })[list(FACT_NAMES)]


def _bootstrap_sd(segs: list[np.ndarray], boot: int, rng: np.random.Generator, block: int | None = None) -> np.ndarray:
    """Sampling noise of each fact for the real series: sessions are resampled whole when there are enough of
    them, otherwise a moving block bootstrap with blocks of ``sqrt(n)`` returns (at least 20) is used."""
    out = []
    if len(segs) >= 8:
        for _ in range(boot):
            pick = rng.integers(0, len(segs), len(segs))
            try:
                out.append(fingerprint([segs[i] for i in pick]).to_numpy())
            except ValueError:
                continue
    else:
        x = np.concatenate(segs)
        n = len(x)
        block = block or max(20, int(math.sqrt(n)))
        nb = int(math.ceil(n / block))
        for _ in range(boot):
            starts = rng.integers(0, n - block, nb)
            y = np.concatenate([x[s:s + block] for s in starts])[:n]
            try:
                out.append(fingerprint(y).to_numpy())
            except ValueError:
                continue
    return np.nanstd(np.array(out), axis=0, ddof=1)


# ===================================================================== candidates
Param = namedtuple('Param', 'low high default log')


class Candidate:
    """A model that can simulate returns at the resolution of a real series.

    ``space`` lists the free parameters (``Param(low, high, default, log)``); ``simulate`` returns sessions of
    log returns whose total length is at least ``sum(lengths)``.
    """
    label = 'candidate'
    version = ''
    space: dict = {}
    fixed: dict = {}

    def __init__(self, label: str | None = None, **params):
        self.params = {k: p.default for k, p in self.space.items()}
        unknown = set(params) - set(self.space) - set(self.fixed)
        if unknown:
            raise TypeError(f'{type(self).__name__}: unknown parameters {sorted(unknown)}')
        self.options = {**self.fixed, **{k: v for k, v in params.items() if k in self.fixed}}
        self.params.update({k: v for k, v in params.items() if k in self.space})
        if label:
            self.label = label

    def __repr__(self):
        p = ', '.join(f'{k}={v:.3g}' for k, v in self.params.items())
        o = ', '.join(f'{k}={v}' for k, v in self.options.items())
        return f'{self.label}({", ".join(x for x in (p, o) if x)})'

    def with_params(self, **p) -> 'Candidate':
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.params = {**self.params, **p}
        return new

    def variant(self, label: str | None = None, **options) -> 'Candidate':
        """A copy with different fidelity options (e.g. ``streams=30``) and the same fitted parameters."""
        unknown = set(options) - set(self.fixed)
        if unknown:
            raise TypeError(f'{type(self).__name__}: unknown options {sorted(unknown)}')
        new = self.with_params()
        new.options = {**self.options, **options}
        if label:
            new.label = label
        return new

    def simulate(self, lengths, seed: int, dt: float) -> list[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def prepare(self, n_jobs: int = 1) -> None:
        """Do any one-off work before replicates are drawn (a no-op for the quick models)."""

    def cost(self, lengths, measured: float) -> float:
        """Seconds to produce one replicate of this size; ``measured`` is the mean wall time of a replicate."""
        return measured

    @staticmethod
    def _split(r: np.ndarray, lengths) -> list[np.ndarray]:
        out, i = [], 0
        for n in lengths:
            out.append(r[i:i + n])
            i += n
        return [s for s in out if len(s) >= 20]


class Gaussian(Candidate):
    """Independent normal returns: the null model (geometric Brownian motion)."""
    label = 'Gaussian'

    def simulate(self, lengths, seed, dt):
        return self._split(np.random.default_rng(seed).standard_normal(int(sum(lengths))), lengths)


class StudentT(Candidate):
    """Independent Student-t returns: fat tails, no clustering."""
    label = 'Student-t'
    space = {'nu': Param(2.2, 60.0, 5.0, True)}

    def simulate(self, lengths, seed, dt):
        nu = self.params['nu']
        r = np.random.default_rng(seed).standard_t(nu, int(sum(lengths)))
        return self._split(r, lengths)


class GARCH(Candidate):
    """GJR-GARCH(1,1) with Student-t innovations: the practitioner's benchmark for tails plus clustering plus leverage."""
    label = 'GARCH-t'
    space = {'alpha': Param(0.005, 0.35, 0.08, False), 'beta': Param(0.5, 0.998, 0.9, False),
             'gamma': Param(0.0, 0.4, 0.05, False), 'nu': Param(2.5, 60.0, 8.0, True)}

    def simulate(self, lengths, seed, dt):
        a, b, g, nu = (self.params[k] for k in ('alpha', 'beta', 'gamma', 'nu'))
        pers = a + b + 0.5 * g
        if pers >= 0.9995:
            s = 0.9995 / pers
            a, b, g = a * s, b * s, g * s
            pers = 0.9995
        w = 1.0 - pers
        n = int(sum(lengths))
        rng = np.random.default_rng(seed)
        burn = 300
        z = rng.standard_t(nu, n + burn) * math.sqrt((nu - 2.0) / nu)
        h, out = 1.0, np.empty(n + burn)
        for t in range(n + burn):
            e = math.sqrt(h) * z[t]
            out[t] = e
            h = w + (a + (g if e < 0 else 0.0)) * e * e + b * h
        return self._split(out[burn:], lengths)


def _market(version: int, n: int, dt: float, seed: int, **kw):
    from .market import Market
    return Market(version=version, steps=int(n), T=n * dt, sigma=0.2, mu=0.0, seed=seed, **kw).run(seed)


def _heston_dict(halflife: float, cv: float, rho: float, dt: float, sigma: float = 0.2) -> dict:
    kappa = math.log(2.0) / (halflife * dt)
    return {'kappa': kappa, 'theta': sigma ** 2, 'xi': cv * sigma * math.sqrt(2.0 * kappa), 'rho': rho, 'v0': sigma ** 2}


class Heston(Candidate):
    """Version 1: one Heston stochastic-volatility path.  ``halflife`` is the variance half-life in bars,
    ``cv`` the stationary coefficient of variation of the variance."""
    label = 'v1 Heston'
    version = '1'
    space = {'halflife': Param(2.0, 3000.0, 60.0, True), 'cv': Param(0.05, 3.0, 0.7, True),
             'rho': Param(-0.95, 0.5, -0.5, False)}

    def simulate(self, lengths, seed, dt):
        p = self.params
        n = int(sum(lengths))
        res = _market(1, n, dt, seed, heston=_heston_dict(p['halflife'], p['cv'], p['rho'], dt))
        return self._split(res.returns, lengths)


class Ensemble(Candidate):
    """Version 2: the average of ``streams`` ABM / GBM / bridge / Heston / Merton streams with a common factor."""
    label = 'v2 ensemble'
    version = '2'
    space = {'correlation': Param(0.0, 0.95, 0.25, False), 'w_heston': Param(0.0, 1.0, 0.2, False),
             'w_merton': Param(0.0, 1.0, 0.2, False), 'cv': Param(0.05, 3.0, 0.7, True),
             'jump_size': Param(1.0, 30.0, 5.0, True)}
    fixed = {'streams': 100, 'halflife': 60.0}

    def simulate(self, lengths, seed, dt):
        p, o = self.params, self.options
        n = int(sum(lengths))
        rest = max(1e-6, 1.0 - 0.5 * (p['w_heston'] + p['w_merton']))
        mix = (rest, rest, rest, p['w_heston'] + 1e-6, p['w_merton'] + 1e-6)       # abm gbm bridge heston merton
        sig = 0.2
        merton = {'rate': 1.0 / (100 * dt), 'mean': 0.0, 'std': p['jump_size'] * sig * math.sqrt(dt)}
        res = _market(2, n, dt, seed, streams=int(o['streams']), correlation=p['correlation'], mix=mix,
                      heston=_heston_dict(o['halflife'], p['cv'], -0.5, dt), merton=merton)
        return self._split(res.returns, lengths)


class Traders(Candidate):
    """Version 3: one internal value (GBM or Heston) with ``streams`` traders whose views fluctuate around it.
    ``crowd`` is the standard deviation of the *average* trader's log deviation from the value in units of one
    bar's volatility (each trader's own deviation is ``crowd * sqrt(streams)``, and prices are the geometric
    mean of the traders), so ``streams`` changes only how finely the crowd is sampled, not what it does.
    ``halflife`` is how fast a deviation closes, in bars."""
    label = 'v3 traders'
    version = '3'
    space = {'crowd': Param(0.2, 10.0, 2.0, True), 'halflife': Param(1.5, 500.0, 20.0, True)}
    fixed = {'streams': 100, 'value': 'gbm', 'value_halflife': 60.0, 'value_cv': 0.7, 'value_rho': -0.5}

    def simulate(self, lengths, seed, dt):
        p, o = self.params, self.options
        n = int(sum(lengths))
        sig = 0.2
        kw = {}
        if o['value'] == 'heston':
            kw['heston'] = _heston_dict(p.get('value_halflife', o.get('value_halflife')),
                                        p.get('value_cv', o.get('value_cv')), p.get('value_rho', o.get('value_rho')), dt)
        res = _market(3, n, dt, seed, streams=int(o['streams']), value=o['value'], aggregate='geometric',
                      deviation=p['crowd'] * math.sqrt(o['streams']) * sig * math.sqrt(dt),
                      reversion=math.log(2.0) / (p['halflife'] * dt), bias=0.0, **kw)
        return self._split(res.returns, lengths)


class TradersSV(Traders):
    """Version 3 whose internal value has stochastic volatility (Heston), with the value's own variance half-life,
    vol-of-vol (``value_cv``) and leverage (``value_rho``) tuned too.  Asks whether version 3 fails because of its
    traders or because of its Gaussian value."""
    label = 'v3 + SV value'
    space = {**Traders.space, 'value_halflife': Param(2.0, 3000.0, 60.0, True), 'value_cv': Param(0.05, 3.0, 0.7, True),
             'value_rho': Param(-0.95, 0.5, -0.5, False)}
    fixed = {'streams': 100, 'value': 'heston'}


class News(Candidate):
    """Version 4: version 3 plus news.  Jumps of lognormal size and random sign arrive by a Hawkes clock
    (``branching`` = 0 gives Poisson; ``events`` is the mean number per 100 bars), and traders under-react to each
    jump.  ``jump_scale`` is the median jump in units of one bar's volatility; ``crowd`` as in :class:`Traders`."""
    label = 'v4 news'
    version = '4'
    space = {'events': Param(0.05, 10.0, 1.0, True), 'jump_scale': Param(0.5, 40.0, 4.0, True),
             'branching': Param(0.0, 0.95, 0.5, False), 'crowd': Param(0.2, 10.0, 2.0, True),
             'underreaction': Param(0.0, 0.9, 0.4, False)}
    fixed = {'streams': 100, 'value': 'gbm', 'halflife': 20.0, 'decay_halflife': 10.0, 'value_halflife': 60.0,
             'value_cv': 0.7, 'value_rho': -0.5}

    def simulate(self, lengths, seed, dt):
        p, o = self.params, self.options
        n = int(sum(lengths))
        sig = 0.2
        step = sig * math.sqrt(dt)
        rate = p['events'] / (100 * dt)                      # events per year (long-run mean)
        b = float(np.clip(p['branching'], 0.0, 0.95))
        beta = math.log(2.0) / (o['decay_halflife'] * dt)
        kw = dict(event_rate=rate * (1 - b) if b > 0 else rate, event_process='hawkes' if b > 0 else 'poisson',
                  event_alpha=b * beta, event_beta=beta)
        if o['value'] == 'heston':
            kw['heston'] = _heston_dict(o['value_halflife'], o['value_cv'], o['value_rho'], dt)
        res = _market(4, n, dt, seed, streams=int(o['streams']), value=o['value'], aggregate='geometric',
                      deviation=p['crowd'] * math.sqrt(o['streams']) * step,
                      reversion=math.log(2.0) / (o['halflife'] * dt), bias=0.0,
                      jump_median=min(0.45, p['jump_scale'] * step), jump_sigma=0.6, jump_cap=0.5, panic=1.0,
                      panic_decay=10 * dt, underreaction=p['underreaction'], **kw)
        return self._split(res.returns, lengths)


_POOLS: dict = {}


def _exchange_day(args):
    mix, tick_s, bar_ticks, schedule, seed = args
    from . import exchange
    t0 = time.perf_counter()
    res = exchange.simulate(mix, seed=seed, schedule=schedule, record_trades=False, sim_secs_per_tick=tick_s)
    r = np.diff(np.log(res.price[::bar_ticks]))
    return r[np.isfinite(r)], time.perf_counter() - t0


class Exchange(Candidate):
    """Versions 5-7: a population of trading agents over simulated days (see ``exchange.simulate``).

    A trading day costs seconds, so the model is simulated once into a *pool* of ``pool_days`` independent days
    and every replicate is drawn from the pool by resampling days (a bootstrap of the model).  The cost of a
    replicate is then the cost of the days it needs, ``seconds_per_day`` times the number of days in the real series.

    ``population`` is a preset name or a ``{type: count}`` dict, ``scale`` multiplies every count,
    ``bar_minutes`` is the bar the days are aggregated to (it must match the real data),
    ``sim_secs_per_tick`` is the engine's time step.  There are no free parameters: the agent populations are
    assessed as designed, not fitted.
    """
    label = 'exchange'
    fixed = {'population': 'brief', 'scale': 1.0, 'bar_minutes': 5, 'sim_secs_per_tick': 15, 'pool_days': 24,
             'pool_seed': 0, 'schedule': True}

    def _mix(self):
        from . import exchange
        pop = self.options['population']
        base = dict(exchange.POPULATIONS[pop]) if isinstance(pop, str) else dict(pop)
        s = float(self.options['scale'])
        return {k: max(1, int(round(v * s))) for k, v in base.items()}

    def _key(self):
        o = self.options
        pop = o['population'] if isinstance(o['population'], str) else tuple(sorted(o['population'].items()))
        return (pop, float(o['scale']), float(o['bar_minutes']), float(o['sim_secs_per_tick']),
                int(o['pool_seed']), repr(o['schedule']))

    def prepare(self, n_jobs: int = 1) -> None:
        """Simulate the pool's days (once; a larger request only adds the missing days)."""
        key, need = self._key(), int(self.options['pool_days'])
        have = _POOLS.get(key, {'days': [], 'secs': [], 'wall': 0.0, 'tried': 0})
        if have['tried'] >= need:
            return
        o = self.options
        tick_s = float(o['sim_secs_per_tick'])
        bar_ticks = int(round(o['bar_minutes'] * 60 / tick_s))
        if abs(bar_ticks * tick_s - o['bar_minutes'] * 60) > 1e-6:
            raise ValueError('bar_minutes must be a whole number of simulation ticks')
        mix = self._mix()
        jobs = [(mix, tick_s, bar_ticks, o['schedule'], int(o['pool_seed']) * 10_000 + d)
                for d in range(have['tried'], need)]
        t0 = time.perf_counter()
        if n_jobs and n_jobs > 1 and len(jobs) > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                res = list(ex.map(_exchange_day, jobs))
        else:
            res = [_exchange_day(j) for j in jobs]
        wall = time.perf_counter() - t0
        res = [(r, t) for r, t in res if len(r) >= 20]
        if not res and not have['days']:
            raise RuntimeError('the exchange produced no usable days')
        _POOLS[key] = {'days': have['days'] + [r for r, _ in res], 'secs': have['secs'] + [t for _, t in res],
                       'wall': have['wall'] + wall, 'tried': need}

    def pool(self, n_jobs: int = 1):
        """``(days, seconds_per_day, wall_seconds)`` for the first ``pool_days`` days of the pool."""
        self.prepare(n_jobs)
        p, k = _POOLS[self._key()], int(self.options['pool_days'])
        return p['days'][:k], float(np.mean(p['secs'][:k])), p['wall']

    @property
    def seconds_per_day(self) -> float:
        """CPU seconds to simulate one day (measured inside the days, so independent of the number of workers)."""
        return self.pool()[1]

    def _days_needed(self, lengths) -> int:
        days = self.pool()[0]
        return int(math.ceil(sum(lengths) / np.mean([len(d) for d in days])))

    def simulate(self, lengths, seed, dt):
        days = self.pool()[0]
        idx = np.random.default_rng(seed).integers(0, len(days), self._days_needed(lengths))
        return [days[i] for i in idx]

    def cost(self, lengths, measured):
        return self.seconds_per_day * self._days_needed(lengths)


# ===================================================================== assessment
@dataclass
class Assessment:
    """The result of :func:`assess`: how one candidate compares with one real series."""
    candidate: Candidate
    real_name: str
    real: pd.Series
    sims: np.ndarray                  # replicates x facts
    boot_sd: pd.Series
    seconds: float                    # mean seconds per replicate
    n_returns: int
    p_value: float
    extra: dict = field(default_factory=dict)

    @property
    def _use(self) -> np.ndarray:
        """Facts that exist for the real series (a lag longer than every session leaves a fact undefined)."""
        return np.isfinite(self.real.to_numpy())

    @property
    def _ok(self) -> np.ndarray:
        return self.sims[np.all(np.isfinite(self.sims[:, self._use]), axis=1)]

    @property
    def model_mean(self) -> pd.Series:
        ok = self._ok
        return pd.Series(ok.mean(axis=0) if len(ok) else np.full(len(FACT_NAMES), np.nan), index=FACT_NAMES)

    @property
    def model_sd(self) -> pd.Series:
        ok = self._ok
        return pd.Series(ok.std(axis=0, ddof=1) if len(ok) > 1 else np.full(len(FACT_NAMES), np.nan), index=FACT_NAMES)

    @property
    def errors(self) -> pd.Series:
        """Model mean minus real, in units of the real value's sampling noise (positive: model has more)."""
        return ((self.model_mean - self.real) / self.boot_sd).rename('error')

    @property
    def rms_error(self) -> float:
        e = np.minimum(np.abs(self.errors.to_numpy()), ERROR_CAP)
        return float(np.sqrt(np.nanmean(e ** 2)))

    def rms_se(self, draws: int = 300, seed: int = 0) -> float:
        """Standard error of ``rms_error`` caused by having only a few replicates (resamples the replicates)."""
        F = self._ok
        if len(F) < 3:
            return float('nan')
        rng = np.random.default_rng(seed)
        real, sd = self.real.to_numpy(), self.boot_sd.to_numpy()
        vals = [_rms(F[rng.integers(0, len(F), len(F))].mean(0), real, sd) for _ in range(draws)]
        return float(np.std(vals))

    @property
    def valid(self) -> int:
        """Replicates that produced a usable fingerprint (a wild parameter set can overflow)."""
        return len(self._ok)

    @property
    def within(self) -> int:
        """How many of the twelve facts the model matches to within two noise units."""
        return int(np.sum(np.abs(self.errors.to_numpy()) <= 2.0))

    def table(self) -> pd.DataFrame:
        """Fact by fact: the real value, the model's mean and spread, the real sampling noise and the error."""
        z = (self.real - self.model_mean) / self.model_sd.clip(lower=1e-9)
        return pd.DataFrame({'real': self.real, 'model mean': self.model_mean, 'model sd': self.model_sd,
                             'real noise (bootstrap sd)': self.boot_sd, 'error (noise units)': self.errors,
                             'z (real vs model)': z})

    def summary(self) -> pd.Series:
        e = self.errors.abs().sort_values(ascending=False)
        return pd.Series({'model': self.candidate.label, 'series': self.real_name, 'rms error': self.rms_error,
                          'p-value': self.p_value, 'facts within 2': self.within,
                          'seconds / replicate': self.seconds, 'worst fact': e.index[0],
                          'replicates': len(self.sims), 'valid': self.valid, 'returns': self.n_returns})

    def plot(self, show: bool = True):
        import matplotlib.pyplot as plt
        err = (self._ok - self.real.to_numpy()) / self.boot_sd.to_numpy()
        with V.themed():
            fig, ax = plt.subplots(figsize=(9, 5))
            y = np.arange(len(FACT_NAMES))[::-1]
            ax.axvspan(-1, 1, color=V.OTHER, alpha=0.25, lw=0)
            ax.axvline(0, color=V.TEXT, lw=1)
            for i, yy in enumerate(y):
                ax.plot(np.clip(err[:, i], -8, 8), np.full(len(err), yy), '.', color=V.BLUE, alpha=0.35, ms=6)
            ax.plot(np.clip(self.errors.to_numpy(), -8, 8), y, 'D', color=V.ORANGE, ms=7, label='model mean')
            ax.set_yticks(y)
            ax.set_yticklabels(FACT_NAMES)
            ax.set_xlim(-8.5, 8.5)
            ax.set_xlabel('model minus real, in units of the real series\' sampling noise (0 = matches)')
            ax.set_title(f'{self.candidate.label} vs {self.real_name}: rms error {self.rms_error:.2f}, '
                         f'p = {self.p_value:.2f}')
            ax.legend(loc='lower right')
            fig.tight_layout()
            return V.finish(fig, show)


def _rms(model_mean, real, sd) -> float:
    """Root mean square of the capped errors over the facts that exist."""
    e = np.minimum(np.abs((model_mean - real) / sd), ERROR_CAP)
    return float(np.sqrt(np.nanmean(e ** 2)))


def _p_value(real: np.ndarray, sims: np.ndarray) -> float:
    """Leave-one-out rank of the real fingerprint's squared z-distance among the replicates'."""
    use = np.isfinite(real)
    real, sims = real[use], sims[:, use]
    ok = np.all(np.isfinite(sims), axis=1)
    F = sims[ok]
    k = len(F)
    if k < 5 or real.size == 0:
        return float('nan')
    mu, sd = F.mean(0), F.std(0, ddof=1).clip(1e-9)
    d_real = float(np.sum(((real - mu) / sd) ** 2))
    d = np.empty(k)
    for i in range(k):
        rest = np.delete(F, i, axis=0)
        m, s = rest.mean(0), rest.std(0, ddof=1).clip(1e-9)
        d[i] = np.sum(((F[i] - m) / s) ** 2)
    return float((1 + np.sum(d >= d_real)) / (k + 1))


def _replicate(args):
    cand, lengths, seed, dt = args
    t0 = time.perf_counter()
    try:
        with np.errstate(all='ignore'):            # a wild parameter set may overflow: it is counted as a miss below
            segs = cand.simulate(lengths, seed, dt)
            f = fingerprint(segs).to_numpy()
    except Exception:  # noqa: BLE001
        f = np.full(len(FACT_NAMES), np.nan)
    return f, time.perf_counter() - t0


def _simulate_many(cand, lengths, dt, seeds, n_jobs):
    jobs = [(cand, lengths, int(s), dt) for s in seeds]
    if n_jobs and n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            res = list(ex.map(_replicate, jobs))
    else:
        res = [_replicate(j) for j in jobs]
    return np.array([r[0] for r in res]), float(np.mean([r[1] for r in res]))


def _real_parts(real, boot, seed, dt=None):
    segs = _segments(real)
    name = real.name if isinstance(real, RealSeries) else 'series'
    if dt is None:
        dt = real.dt if isinstance(real, RealSeries) else 1.0 / 252
    f = fingerprint(segs)
    sd = pd.Series(_bootstrap_sd(segs, boot, np.random.default_rng(seed + 7)), index=FACT_NAMES)
    return segs, name, dt, f, sd


def assess(real, candidate: Candidate, replicates: int = 20, seed: int = 0, boot: int = 200, n_jobs: int = 1,
           dt: float | None = None, _cache: dict | None = None) -> Assessment:
    """Compare one candidate with one real series (a :class:`RealSeries`, or returns as an array or list of sessions)."""
    segs, name, dt, f, sd = _cache['parts'] if _cache else _real_parts(real, boot, seed, dt)
    lengths = [len(s) for s in segs]
    candidate.prepare(n_jobs)
    heavy = type(candidate).prepare is not Candidate.prepare          # pooled models: replicates are cheap draws
    sims, secs = _simulate_many(candidate, lengths, dt, [seed + 1000 + i for i in range(replicates)],
                                1 if heavy else n_jobs)
    return Assessment(candidate, name, f, sims, sd, candidate.cost(lengths, secs), int(sum(lengths)),
                      _p_value(f.to_numpy(), sims))


# ===================================================================== comparing many
@dataclass
class Scoreboard:
    """Candidates ranked against one real series (``table``) with their per-fact errors (``errors``)."""
    real_name: str
    table: pd.DataFrame
    errors: pd.DataFrame
    assessments: dict

    def plot(self, show: bool = True):
        import matplotlib.pyplot as plt
        e = self.errors
        with V.themed():
            fig, ax = plt.subplots(figsize=(1.0 * len(FACT_NAMES) + 3, 0.55 * len(e) + 2.2))
            im = ax.imshow(np.clip(e.to_numpy(), -ERROR_CAP, ERROR_CAP), cmap='RdBu_r', vmin=-ERROR_CAP,
                           vmax=ERROR_CAP, aspect='auto')
            ax.set_xticks(range(len(e.columns)))
            ax.set_xticklabels(e.columns, rotation=40, ha='right')
            ax.set_yticks(range(len(e)))
            ax.set_yticklabels(e.index)
            for i in range(e.shape[0]):
                for j in range(e.shape[1]):
                    ax.text(j, i, f'{e.iat[i, j]:.1f}', ha='center', va='center', fontsize=8,
                            color='white' if abs(e.iat[i, j]) > 3 else V.TEXT)
            ax.set_title(f'{self.real_name}: model minus real, in noise units (red: model has more, blue: less)')
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            return V.finish(fig, show)


def compare(real, candidates, replicates: int = 20, seed: int = 0, boot: int = 200, n_jobs: int = 1,
            calibrate_budget: int = 0, cal_replicates: int = 6, dt: float | None = None, verbose: bool = False) -> Scoreboard:
    """Assess many candidates against one real series; with ``calibrate_budget > 0`` each candidate that has free
    parameters is first fitted to the series (the reported score uses fresh random numbers)."""
    parts = _real_parts(real, boot, seed, dt)
    rows, errs, ass = [], {}, {}
    for cand in candidates:
        t0 = time.perf_counter()
        fit_s, fit_rms, fit_bounds = 0.0, float('nan'), ''
        if calibrate_budget and cand.space:
            fit = calibrate(cand, real, budget=calibrate_budget, replicates=cal_replicates, seed=seed + 31,
                            n_jobs=n_jobs, _parts=parts)
            cand, fit_s, fit_rms, fit_bounds = fit.candidate, fit.seconds, fit.objective, ', '.join(fit.at_bounds)
        a = assess(real, cand, replicates, seed, boot, n_jobs, _cache={'parts': parts})
        ass[cand.label] = a
        rows.append({'model': cand.label, 'rms error': a.rms_error, 'facts within 2': a.within, 'p-value': a.p_value,
                     'rms se': a.rms_se(), 'seconds / replicate': a.seconds, 'failed': len(a.sims) - a.valid, 'fit seconds': fit_s, 'fit rms (search)': fit_rms,
                     'at bounds': fit_bounds, 'worst fact': a.errors.abs().idxmax()})
        errs[cand.label] = a.errors
        if verbose:
            print(f'{cand.label:<14s} rms {a.rms_error:5.2f}  p {a.p_value:5.2f}  {time.perf_counter() - t0:6.1f}s')
    table = pd.DataFrame(rows).set_index('model').sort_values('rms error')
    return Scoreboard(parts[1], table, pd.DataFrame(errs).T.loc[table.index], ass)


# ===================================================================== many series
def _compare_one(args):
    real, candidates, kw = args
    return compare(real, candidates, **kw)


@dataclass
class Sweep:
    """The same candidates scored against several real series (:func:`sweep`)."""
    boards: dict                      # series name -> Scoreboard
    assets: dict                      # series name -> asset class

    @property
    def models(self) -> list:
        return list(dict.fromkeys(m for b in self.boards.values() for m in b.assessments))

    def grid(self, column: str) -> pd.DataFrame:
        """Any column of the scoreboards (``'rms error'``, ``'facts within 2'``, ``'failed'``, ...), models by series."""
        return self._grid(column)

    def _grid(self, column: str) -> pd.DataFrame:
        return pd.DataFrame({n: b.table[column] for n, b in self.boards.items()}).reindex(self.models)

    @property
    def rms(self) -> pd.DataFrame:
        """rms error, models by series."""
        return self._grid('rms error')

    @property
    def seconds(self) -> pd.DataFrame:
        return self._grid('seconds / replicate')

    @property
    def p_values(self) -> pd.DataFrame:
        return self._grid('p-value')

    def by_asset(self) -> pd.DataFrame:
        """Mean rms error of each model within each asset class, and over all series."""
        out = self.rms.T.groupby(pd.Series(self.assets)).mean().T
        out['all series'] = self.rms.mean(axis=1)
        return out

    def parameters(self) -> pd.DataFrame:
        """The fitted parameters, one row per (model, series)."""
        rows = []
        for n, b in self.boards.items():
            for m, a in b.assessments.items():
                if a.candidate.params:
                    rows.append({'series': n, 'model': m, **a.candidate.params})
        return pd.DataFrame(rows).set_index(['model', 'series'])

    def cost_accuracy(self, tol: float = 0.15) -> pd.DataFrame:
        """Mean error and mean cost of each model over the series, with Pareto and knee flags."""
        df = pd.DataFrame({'rms error': self.rms.mean(axis=1), 'seconds / replicate': self.seconds.mean(axis=1)})
        df['rms se'] = pd.DataFrame({n: b.table['rms se'] for n, b in self.boards.items()}).reindex(self.models).pow(2).mean(axis=1).pow(0.5) / np.sqrt(len(self.boards))
        df['pareto'] = pareto(df)
        df['knee'] = df.index == knee(df, tol)
        return df

    def fact_errors(self, signed: bool = False) -> pd.DataFrame:
        """Error of each model on each fact, averaged over the series (facts by models).  ``signed=True`` keeps the
        sign (positive: the model has more of it than the data)."""
        def one(m):
            es = [b.errors.loc[m] for b in self.boards.values() if m in b.errors.index]
            e = pd.concat(es, axis=1)
            return (e if signed else e.abs()).mean(axis=1)
        return pd.DataFrame({m: one(m) for m in self.models}).reindex(FACT_NAMES)

    def plot_facts(self, show: bool = True, signed: bool = True):
        """Which facts each model gets wrong, averaged over the series."""
        import matplotlib.pyplot as plt
        g = self.fact_errors(signed)
        lim = ERROR_CAP if signed else ERROR_CAP
        with V.themed():
            fig, ax = plt.subplots(figsize=(0.95 * g.shape[1] + 3.5, 0.42 * len(g) + 2.0))
            im = ax.imshow(np.clip(g.to_numpy(), -lim if signed else 0, lim), cmap=V.DIVERGING if signed else V.SEQUENTIAL,
                           vmin=-lim if signed else 0, vmax=lim, aspect='auto')
            ax.set_xticks(range(g.shape[1]))
            ax.set_xticklabels(g.columns, rotation=30, ha='right')
            ax.set_yticks(range(len(g)))
            ax.set_yticklabels(g.index)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    v = g.iat[i, j]
                    if v == v:
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=8,
                                color='white' if abs(v) > 3.2 else V.TEXT)
            ax.set_title('model minus real, in noise units, averaged over the series' if signed else
                         'absolute error in noise units, averaged over the series')
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            return V.finish(fig, show)

    def only(self, names) -> 'Sweep':
        """The sweep restricted to some of its series."""
        return Sweep({n: self.boards[n] for n in names}, {n: self.assets[n] for n in names})

    def __add__(self, other: 'Sweep') -> 'Sweep':
        """Put the candidates of two sweeps side by side (series present in both are merged, others kept)."""
        boards = dict(self.boards)
        for n, b in other.boards.items():
            if n in boards:
                a = boards[n]
                tab = pd.concat([a.table, b.table]).sort_values('rms error')
                boards[n] = Scoreboard(a.real_name, tab, pd.concat([a.errors, b.errors]).loc[tab.index],
                                       {**a.assessments, **b.assessments})
            else:
                boards[n] = b
        return Sweep(boards, {**self.assets, **other.assets})

    def plot(self, show: bool = True):
        import matplotlib.pyplot as plt
        g = self.rms
        order = sorted(self.boards, key=lambda n: (self.assets[n], n))
        g = g[order]
        with V.themed():
            fig, ax = plt.subplots(figsize=(1.15 * len(order) + 3, 0.5 * len(g) + 2.2))
            im = ax.imshow(g.to_numpy(), cmap=V.SEQUENTIAL, vmin=0, vmax=ERROR_CAP, aspect='auto')
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([f'{n}\n({self.assets[n]})' for n in order])
            ax.set_yticks(range(len(g)))
            ax.set_yticklabels(g.index)
            for i in range(g.shape[0]):
                for j in range(g.shape[1]):
                    v = g.iat[i, j]
                    if v == v:
                        ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=9, color='white' if v > 2.8 else V.TEXT)
            ax.set_title('rms error against each real series, in sampling-noise units (darker: further from the data)')
            ax.grid(False)
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            return V.finish(fig, show)


def sweep(reals, candidates, n_jobs: int = 1, **compare_kw) -> Sweep:
    """Run :func:`compare` for each real series (a dict of name -> RealSeries, or a list of RealSeries).

    ``n_jobs > 1`` runs different series in different processes, unless a candidate needs a one-off preparation
    (the exchange pools), in which case the series are scored one after the other.
    """
    items = list(reals.items()) if isinstance(reals, dict) else [(r.name, r) for r in reals]
    kw = {k: v for k, v in compare_kw.items() if k != 'n_jobs'}
    heavy = any(type(c).prepare is not Candidate.prepare for c in candidates)
    jobs = [(r, candidates, kw) for _, r in items]
    if n_jobs and n_jobs > 1 and not heavy and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            boards = list(ex.map(_compare_one, jobs))
    else:
        boards = [compare(r, candidates, n_jobs=n_jobs, **kw) for r, _, kw in jobs] if heavy else [_compare_one(j) for j in jobs]
    assets = {n: (r.asset if isinstance(r, RealSeries) else 'other') for n, r in items}
    return Sweep(dict(zip([n for n, _ in items], boards)), assets)


# ===================================================================== calibration
@dataclass
class Fit:
    """Result of :func:`calibrate`."""
    candidate: Candidate
    objective: float
    history: pd.DataFrame
    seconds: float
    evaluations: int
    at_bounds: list = field(default_factory=list)       # parameters that ended within 1% of their allowed range's edge

    def curve(self) -> pd.DataFrame:
        """Best search rms found so far against evaluations used and seconds elapsed."""
        h = self.history
        best = h.loc[h['objective'].cummin().index, 'objective'].cummin()
        idx = h['objective'].expanding().apply(lambda x: x.argmin(), raw=True).astype(int)
        return pd.DataFrame({'evaluations': np.arange(1, len(h) + 1), 'seconds': h['elapsed'].to_numpy(),
                             'best rms so far': h['rms'].to_numpy()[idx.to_numpy()]})


def _to_unit(space: dict, params: dict) -> np.ndarray:
    u = []
    for k, p in space.items():
        v, lo, hi = params[k], p.low, p.high
        if p.log:
            u.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        else:
            u.append((v - lo) / (hi - lo))
    return np.array(u)


def _from_unit(space: dict, u: np.ndarray) -> dict:
    out = {}
    for x, (k, p) in zip(np.clip(u, 0.0, 1.0), space.items()):
        out[k] = math.exp(math.log(p.low) + x * (math.log(p.high) - math.log(p.low))) if p.log else p.low + x * (p.high - p.low)
    return out


def calibrate(candidate: Candidate, real, budget: int = 60, replicates: int = 6, seed: int = 0, n_jobs: int = 1,
              boot: int = 120, dt: float | None = None, verbose: bool = False, _parts=None) -> Fit:
    """Fit the free parameters of ``candidate`` to ``real`` by simulated method of moments.

    The objective is the model's error over ``replicates`` simulated series that reuse the same random seeds at
    every evaluation (common random numbers), so it is a smooth function of the parameters.  It is a Huber-type
    rms (squared up to 3 noise units, linear beyond), so a fact that is far off still pulls the search.  The
    search starts from the defaults plus ``max(6, budget // 6)`` random points and then runs Nelder-Mead from
    the best.  ``Fit.objective`` is the capped rms the search saw (comparable with ``Assessment.rms_error``, but
    optimistic: re-assess the fitted model on fresh seeds) and ``Fit.at_bounds`` names parameters that ended
    at the edge of their allowed range, a sign the model wants more than it is allowed.
    """
    t0 = time.perf_counter()
    if not candidate.space:
        return Fit(candidate, float('nan'), pd.DataFrame(), 0.0, 0, [])
    segs, name, dt, f, sd = _parts or _real_parts(real, boot, seed, dt)
    lengths = [len(s) for s in segs]
    seeds = [seed + 5000 + i for i in range(replicates)]
    space = candidate.space
    f_arr, sd_arr = f.to_numpy(), sd.to_numpy()
    use = np.isfinite(f_arr)
    rows: list[dict] = []

    def objective(u):
        u = np.clip(u, 0.0, 1.0)
        params = _from_unit(space, u)
        sims, _ = _simulate_many(candidate.with_params(**params), lengths, dt, seeds, n_jobs)
        ok = np.all(np.isfinite(sims[:, use]), axis=1)
        if ok.sum() < max(2, replicates // 2):
            val, rms = 10.0, ERROR_CAP
        else:
            e = np.abs((sims[ok].mean(0) - f_arr) / sd_arr)[use]
            e = np.where(np.isfinite(e), e, ERROR_CAP * 2)
            val = float(np.sqrt(np.mean(np.where(e <= 3.0, e ** 2, 3.0 * (2.0 * e - 3.0))))) + 0.05 * (replicates - ok.sum())
            rms = float(np.sqrt(np.mean(np.minimum(e, ERROR_CAP) ** 2)))
        rows.append({**params, 'objective': val, 'rms': rms, 'elapsed': time.perf_counter() - t0})
        if verbose:
            print(f'{len(rows):3d} {val:6.3f}', {k: round(v, 3) for k, v in params.items()})
        return val

    rng = np.random.default_rng(seed + 99)
    d = len(space)
    start = [_to_unit(space, candidate.params)] + [rng.random(d) for _ in range(max(6, budget // 6))]
    scores = [objective(u) for u in start]
    best = start[int(np.argmin(scores))]
    left = max(10, budget - len(start))
    res = optimize.minimize(objective, best, method='Nelder-Mead',
                            options={'maxfev': left, 'xatol': 1e-3, 'fatol': 1e-3, 'initial_simplex': _simplex(best, 0.15)})
    hist = pd.DataFrame(rows)
    top = hist.loc[hist['objective'].idxmin()]
    fitted = candidate.with_params(**{k: float(top[k]) for k in space})
    u = _to_unit(space, fitted.params)
    edge = [k for k, x in zip(space, u) if x < 0.01 or x > 0.99]
    return Fit(fitted, float(top['rms']), hist, time.perf_counter() - t0, len(rows), edge)


def _simplex(x0: np.ndarray, step: float) -> np.ndarray:
    d = len(x0)
    s = np.tile(x0, (d + 1, 1))
    for i in range(d):
        s[i + 1, i] = s[i + 1, i] + step if s[i + 1, i] + step <= 1.0 else s[i + 1, i] - step
    return np.clip(s, 0.0, 1.0)


# ===================================================================== speed against accuracy
def pareto(df: pd.DataFrame, cost: str = 'seconds / replicate', error: str = 'rms error') -> pd.Series:
    """True for the rows that no other row beats on both cost and error."""
    c, e = df[cost].to_numpy(), df[error].to_numpy()
    keep = np.ones(len(df), bool)
    for i in range(len(df)):
        keep[i] = not np.any((c <= c[i]) & (e <= e[i]) & ((c < c[i]) | (e < e[i])))
    return pd.Series(keep, index=df.index, name='pareto')


def knee(df: pd.DataFrame, tol: float = 0.15, cost: str = 'seconds / replicate', error: str = 'rms error') -> str:
    """The label of the fastest row whose error is within ``tol`` (relative) of the best error, or within one
    standard error of it when the table has an ``'rms se'`` column (differences smaller than the simulation
    noise are not worth paying for)."""
    i = df[error].idxmin()
    best = df.loc[i, error]
    limit = best * (1.0 + tol)
    if 'rms se' in df:
        limit = max(limit, best + float(df.loc[i, 'rms se']))
    return str(df[df[error] <= limit][cost].idxmin())


def frontier(real, candidates, replicates: int = 12, seed: int = 0, boot: int = 200, n_jobs: int = 1,
             tol: float = 0.15, dt: float | None = None) -> pd.DataFrame:
    """Score several fidelity settings of a model against one real series: runtime, error, p-value, Pareto flag
    and the knee (the fastest setting within ``tol`` of the best error).  Give each candidate its own ``label``."""
    sb = compare(real, candidates, replicates, seed, boot, n_jobs, dt=dt)
    df = sb.table[['rms error', 'rms se', 'p-value', 'seconds / replicate']].copy()
    df['pareto'] = pareto(df)
    df['knee'] = df.index == knee(df, tol)
    return df.sort_values('seconds / replicate')


def convergence(a: Assessment, sizes=(3, 5, 8, 12, 20, 40), draws: int = 200, seed: int = 0) -> pd.DataFrame:
    """How many replicates are enough?  The spread of ``rms_error`` when only ``k`` of an assessment's
    replicates are used (resampled with replacement)."""
    rng = np.random.default_rng(seed)
    F = a._ok
    real, sd = a.real.to_numpy(), a.boot_sd.to_numpy()
    rows = []
    for k in sizes:
        vals = [_rms(F[rng.integers(0, len(F), k)].mean(0), real, sd) for _ in range(draws)]
        rows.append({'replicates': k, 'mean rms error': np.mean(vals), 'sd of rms error': np.std(vals)})
    return pd.DataFrame(rows).set_index('replicates')


def plot_frontier(df: pd.DataFrame, show: bool = True, title: str | None = None, cost: str = 'seconds / replicate',
                  error: str = 'rms error'):
    """Error against cost (log axis): the Pareto-optimal points joined by a line, the knee ringed, and the
    simulation noise (``'rms se'``) as error bars when the table has it."""
    import matplotlib.pyplot as plt
    d = df.copy()
    if 'pareto' not in d:
        d['pareto'] = pareto(d, cost, error)
    with V.themed():
        fig, ax = plt.subplots(figsize=(8.5, 5.6))
        x, y = d[cost].clip(lower=1e-4), d[error]
        if 'rms se' in d:
            ax.errorbar(x, y, yerr=d['rms se'], fmt='none', ecolor=V.OTHER, lw=1, capsize=2, zorder=1)
        ax.scatter(x[~d['pareto']], y[~d['pareto']], s=46, color=V.OTHER, zorder=2, label='beaten on both')
        front = d[d['pareto']].sort_values(cost)
        ax.step(front[cost].clip(lower=1e-4), front[error], where='post', color=V.BLUE, lw=1.3, zorder=2)
        ax.scatter(front[cost].clip(lower=1e-4), front[error], s=60, color=V.BLUE, zorder=3, label='Pareto-optimal')
        if 'knee' in d and d['knee'].any():
            k = d[d['knee']]
            ax.scatter(k[cost].clip(lower=1e-4), k[error], s=260, facecolors='none', edgecolors=V.ORANGE, lw=2, zorder=4,
                       label='knee')
        for name, row in d.iterrows():
            ax.annotate(str(name), (max(row[cost], 1e-4), row[error]), xytext=(6, 5), textcoords='offset points',
                        fontsize=8, color=V.TEXT_2)
        ax.set_xscale('log')
        ax.set_xlabel('seconds per replicate (log scale)')
        ax.set_ylabel('rms error (noise units; lower is closer to the data)')
        se = d['rms se'] if 'rms se' in d else 0.0
        lo, hi = float((y - se).min()), float((y + se).max())
        pad = 0.3 * max(hi - lo, 0.4)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
        ax.set_title(title or 'Accuracy against cost')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3)
        fig.tight_layout()
        return V.finish(fig, show)


def budget_curve(fit: Fit, real, checkpoints=(1, 5, 10, 20, 40, 80, 160), replicates: int = 16, seed: int = 777,
                 boot: int = 120) -> pd.DataFrame:
    """Is more fitting worth it?  For several budgets, the best parameters found within the first ``k`` evaluations
    of ``fit`` re-assessed on fresh random numbers.  ``search rms`` is what the search saw, ``fresh-seed rms`` the score on new
    simulations of the same model (the real series is still the same one: for a test on unseen data see
    :func:`holdout`); the gap between them is the search fitting its own simulation noise."""
    base = fit.candidate
    h = fit.history
    parts = _real_parts(real, boot, seed)
    keys = [k for k in base.space]
    rows = []
    for k in checkpoints:
        k = min(int(k), len(h))
        top = h.iloc[:k].loc[h.iloc[:k]['objective'].idxmin()]
        cand = base.with_params(**{p: float(top[p]) for p in keys})
        a = assess(real, cand, replicates, seed, boot, _cache={'parts': parts})
        rows.append({'evaluations': k, 'seconds': float(h['elapsed'].iloc[k - 1]), 'search rms': float(top['rms']),
                     'fresh-seed rms': a.rms_error, 'rms se': a.rms_se()})
    return pd.DataFrame(rows).drop_duplicates('evaluations').set_index('evaluations')


def plot_budget(curves: dict, show: bool = True, title: str = 'Is more fitting worth it?'):
    """Fresh-seed error (solid, with its noise as a band) and the error the search saw (dashed) against the seconds spent
    fitting, one line per model; ``curves`` maps a label to a :func:`budget_curve` table."""
    import matplotlib.pyplot as plt
    with V.themed():
        fig, ax = plt.subplots(figsize=(9, 5.6))
        for i, (name, c) in enumerate(curves.items()):
            col = V.SERIES[i % len(V.SERIES)]
            x = c['seconds'].clip(lower=1e-2)
            ax.plot(x, c['fresh-seed rms'], '-o', color=col, ms=4, label=name)
            ax.fill_between(x, c['fresh-seed rms'] - c['rms se'], c['fresh-seed rms'] + c['rms se'], color=col, alpha=0.12, lw=0)
            ax.plot(x, c['search rms'], '--', color=col, lw=1, alpha=0.8)
        ax.set_xscale('log')
        ax.set_xlabel('seconds spent fitting (log scale)')
        ax.set_ylabel('rms error (noise units)')
        ax.set_ylim(bottom=0)
        ax.set_title(title + '  (solid: fresh random numbers, dashed: what the search saw)')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=min(6, len(curves)))
        fig.tight_layout()
        return V.finish(fig, show)


def _split(real, fraction: float):
    if isinstance(real, RealSeries):
        k = int(len(real) * fraction)
        return real.head(k), real.window(last=len(real) - k)
    x = np.concatenate(_segments(real))
    k = int(len(x) * fraction)
    return x[:k], x[k:]


def _holdout_one(args):
    real, candidates, fraction, kw = args
    first, second = _split(real, fraction)
    reps, budget, cal_reps = kw['replicates'], kw['calibrate_budget'], kw['cal_replicates']
    seed, boot, n_jobs = kw['seed'], kw['boot'], kw['n_jobs']
    parts2 = _real_parts(second, boot, seed)
    rows = []
    for cand in candidates:
        if cand.space:
            fit1 = calibrate(cand, first, budget, cal_reps, seed + 31, n_jobs, boot)
            fit2 = calibrate(cand, second, budget, cal_reps, seed + 31, n_jobs, boot, _parts=parts2)
            moved, home = fit1.candidate, fit2.candidate
        else:
            moved = home = cand
        a_moved = assess(second, moved, reps, seed, boot, n_jobs, _cache={'parts': parts2})
        a_home = a_moved if moved is home else assess(second, home, reps, seed, boot, n_jobs, _cache={'parts': parts2})
        rows.append({'model': cand.label, 'rms: fitted on 1st half': a_moved.rms_error,
                     'rms: fitted on 2nd half': a_home.rms_error, 'loss from transfer': a_moved.rms_error - a_home.rms_error,
                     'facts within 2 (1st-half fit)': a_moved.within, 'p-value (1st-half fit)': a_moved.p_value})
    return pd.DataFrame(rows).set_index('model')


def holdout(reals, candidates, fraction: float = 0.5, replicates: int = 24, calibrate_budget: int = 70,
            cal_replicates: int = 6, seed: int = 0, boot: int = 200, n_jobs: int = 1) -> pd.DataFrame:
    """A test the fit cannot flatter: tune each model on the first ``fraction`` of a series, then score it on the rest.

    ``rms: fitted on 1st half`` is that held-out score; ``rms: fitted on 2nd half`` is the usual in-period score on
    the same second half, so ``loss from transfer`` is the price of having to extrapolate in time (small: the
    model captured something stable; large: it fitted the period).  ``reals`` is a series or a dict of series
    (``n_jobs > 1`` then runs different series in different processes).  Needs enough data for two halves.
    """
    kw = dict(replicates=replicates, calibrate_budget=calibrate_budget, cal_replicates=cal_replicates, seed=seed,
              boot=boot, n_jobs=1)
    if not isinstance(reals, dict):
        return _holdout_one((reals, candidates, fraction, {**kw, 'n_jobs': n_jobs}))
    jobs = [(r, candidates, fraction, kw) for r in reals.values()]
    if n_jobs and n_jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(jobs))) as ex:
            out = list(ex.map(_holdout_one, jobs))
    else:
        out = [_holdout_one(j) for j in jobs]
    return pd.concat(dict(zip(reals, out)), names=['series', 'model'])
