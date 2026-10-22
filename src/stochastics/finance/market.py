"""Markets built from baseline stochastic processes: versions 1 to 4.

    from stochastics.finance import market

    m = market.Market(version=3, streams=100, steps=2000, seed=1)
    r = m.run()                       # MarketResult
    r.price, r.streams, r.value       # the market, its constituent streams, the internal value
    r.facts()                         # stylised facts of the price (tails, volatility clustering...)
    r.plot()

    cmp = market.compare_versions(streams=100, steps=2000, seed=1)    # all four side by side
    cmp.table; cmp.plot()

=========  ===========================================================================
version    what the market price is
=========  ===========================================================================
1          one Heston stochastic-volatility path
2          the average of ``streams`` independent-ish streams (ABM, GBM, bridge,
           Heston, Merton) that share a common factor
3          the average of ``streams`` *traders* who each hold a view that fluctuates
           around one hidden internal value (a GBM, bridge, ABM or Heston path)
4          as 3, but the internal value also receives news: jumps arriving by a
           Poisson or Hawkes process, each of lognormal size and random sign; traders
           follow the value (with a lag, if you set one) and get jumpier after news
=========  ===========================================================================

Two modelling points worth knowing.

* Averaging N independent noise sources divides the variance by N and, by the
  central limit theorem, removes fat tails: the average of 1000 independent
  streams is nearly a smooth curve.  Version 2 therefore gives the streams a
  common factor (``correlation``), which is what keeps the average moving.
* A jump size "lognormal, plus or minus" is applied to the log of the value and
  capped at ``jump_cap``: a lognormal has no exponential moments, so an uncapped
  ``exp(+Y)`` would have infinite mean.  Jumps are not drift-compensated.

Units: time in years, volatility annualised, ``steps`` points over ``T`` years.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .. import _viz as V
from . import _streams as K

__all__ = ['Market', 'MarketResult', 'VersionComparison', 'compare_versions', 'stylized_facts', 'KINDS']

KINDS = ('abm', 'gbm', 'bridge', 'heston', 'merton')


# ================================================================ stylised facts
def _acf(x: np.ndarray, lags) -> np.ndarray:
    x = x - x.mean()
    den = float(np.dot(x, x))
    return np.array([float(np.dot(x[:-k], x[k:])) / den if den > 0 else np.nan for k in lags])


def stylized_facts(price, dt: float | None = None) -> pd.Series:
    """Empirical regularities that real markets show and a "market-like" path should too.

    Returns of equity-like data typically have: excess kurtosis > 0 and tail index 3-5
    (fat tails), ``acf(r)`` near 0 (no linear predictability), ``acf(|r|)`` clearly
    positive and slowly decaying (volatility clustering) and a negative leverage
    correlation (down moves raise future volatility).
    """
    p = np.asarray(price, float)
    r = np.diff(np.log(p))
    n = r.size
    k = max(20, int(0.05 * n))
    a = np.sort(np.abs(r))[::-1]
    hill = k / float(np.sum(np.log(a[:k] / a[k]))) if a[k] > 0 else np.nan
    lev = float(np.corrcoef(r[:-1], r[1:] ** 2)[0, 1]) if n > 3 else np.nan
    peak = np.maximum.accumulate(p)
    return pd.Series({
        'annualised vol': r.std(ddof=1) * (math.sqrt(1.0 / dt) if dt else 1.0),
        'skew': float(stats.skew(r)),
        'excess kurtosis': float(stats.kurtosis(r)),
        'tail index (Hill, top 5%)': hill,
        'acf(r), lag 1': float(_acf(r, [1])[0]),
        'acf(|r|), lag 1': float(_acf(np.abs(r), [1])[0]),
        'acf(|r|), lag 10': float(_acf(np.abs(r), [10])[0]),
        'leverage corr(r, next r^2)': lev,
        'max drawdown': float(np.max(1.0 - p / peak)),
    })


# ================================================================== the result
@dataclass
class MarketResult:
    """One simulated market: the price, its streams, and (versions 3-4) the value and news."""
    version: int
    times: np.ndarray
    price: np.ndarray
    streams: np.ndarray
    kinds: np.ndarray
    value: np.ndarray | None = None
    events: pd.DataFrame | None = None
    extras: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0])

    @property
    def returns(self) -> np.ndarray:
        """Log returns of the market price."""
        return np.diff(np.log(self.price))

    def facts(self) -> pd.Series:
        return stylized_facts(self.price, self.dt).rename(f'version {self.version}')

    def tracking(self) -> pd.Series:
        """How closely the price follows the internal value (versions 3-4)."""
        if self.value is None:
            raise ValueError('only versions 3 and 4 have an internal value')
        rp, rv = np.diff(np.log(self.price)), np.diff(np.log(self.value))
        cc = [float(np.corrcoef(rp[k:], rv[:len(rv) - k])[0, 1]) for k in range(25)]
        gap = np.log(self.price / self.value)
        return pd.Series({'mean log gap': gap.mean(), 'sd of log gap': gap.std(),
                          'corr of returns with value returns': cc[0],
                          'best lag (steps)': int(np.argmax(cc)), 'corr at best lag': max(cc)})

    def plot(self, show: bool = True, max_streams: int = 80):
        import matplotlib.pyplot as plt
        t, r = self.times, self.returns
        with V.themed():
            fig, axes = plt.subplots(2, 2, figsize=(12, 7.4))
            ax = axes[0, 0]
            idx = np.linspace(0, self.streams.shape[0] - 1, min(max_streams, self.streams.shape[0])).astype(int)
            for i in idx:
                ax.plot(t, self.streams[i], color=V.OTHER, lw=0.5, alpha=0.35)
            ax.plot(t, self.price, color=V.BLUE, lw=1.8, label='market price')
            if self.value is not None:
                ax.plot(t, self.value, color=V.ORANGE, lw=1.3, ls='--', label='internal value')
            if self.events is not None and len(self.events):
                ev = self.events
                ax.plot(ev['time'], np.interp(ev['time'], t, self.price), 'v', ms=4, color=V.RED, alpha=0.7,
                        label=f'news ({len(ev)})', ls='none')
            lo, hi = V.robust_limits(np.r_[self.price, self.streams[idx].ravel()], 0.5, 99.5, 0.08)
            ax.set_ylim(lo, hi)
            ax.set_xlabel('years')
            ax.set_title(f'Version {self.version}: {self.streams.shape[0]} stream(s) and the market')
            ax.legend(loc='upper left')
            ax = axes[0, 1]
            ax.plot(t[1:], r, color=V.BLUE, lw=0.6)
            w = max(10, len(r) // 50)
            vol = pd.Series(r).rolling(w).std().to_numpy()
            ax.plot(t[1:], 3 * vol, color=V.ORANGE, lw=1.3, label=f'+/- 3 x rolling sd ({w} steps)')
            ax.plot(t[1:], -3 * vol, color=V.ORANGE, lw=1.3)
            ax.set_title('Returns and volatility')
            ax.set_xlabel('years')
            ax.legend(loc='upper right')
            ax = axes[1, 0]
            sd = r.std(ddof=1)
            z = (r - r.mean()) / sd
            bins = np.linspace(-6, 6, 61)
            ax.hist(z, bins=bins, density=True, color=V.BLUE, alpha=0.45, label='market returns')
            xs = np.linspace(-6, 6, 400)
            ax.plot(xs, stats.norm.pdf(xs), color=V.TEXT, lw=1.2, label='normal')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
            ax.set_xlabel('standardised return')
            ax.set_title(f'Tails: excess kurtosis {stats.kurtosis(r):.2f}')
            ax.legend()
            ax = axes[1, 1]
            lags = np.arange(1, 41)
            band = 1.96 / math.sqrt(len(r))
            ax.axhspan(-band, band, color=V.OTHER, alpha=0.25, lw=0)
            ax.plot(lags, _acf(r, lags), 'o-', ms=3, color=V.BLUE, label='returns')
            ax.plot(lags, _acf(np.abs(r), lags), 'o-', ms=3, color=V.ORANGE, label='absolute returns')
            ax.set_xlabel('lag (steps)')
            ax.set_title('Autocorrelation: volatility clustering shows in |r|')
            ax.legend()
            V.suptitle(fig, self.params.get('title', f'Market, version {self.version}'))
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            return V.finish(fig, show)


# ====================================================================== the market
@dataclass
class Market:
    """Specification of a stochastic market; ``run()`` simulates it.

    Common: ``version`` (1-4), ``steps``, ``T`` (years), ``s0``, ``streams``, ``mu``, ``sigma``
    (annual drift and volatility), ``seed``.

    Heston (version 1, and the Heston streams elsewhere): ``heston=dict(kappa=3, theta=sigma**2,
    xi=0.45, rho=-0.7, v0=theta)``.

    Version 2: ``kinds`` (subset of ``KINDS``), ``mix`` (weights, default equal), ``dispersion`` (spread of
    each stream's parameters), ``correlation`` (share of each stream's shocks that is common),
    ``merton=dict(rate=3, mean=-0.02, std=0.05)``, ``aggregate`` ('mean', 'median', 'geometric').

    Versions 3-4: ``value`` ('gbm', 'bridge', 'abm' or 'heston'; ``value_target`` is the bridge end),
    ``tether`` ('ou': deviations mean-revert to the value; 'follow': they random-walk and merely
    share its increments), ``deviation`` (stationary sd of a trader's log deviation), ``reversion``
    (OU speed per year), ``bias`` (sd of each trader's constant mis-estimate of value).

    Version 4: ``event_process`` ('poisson' or 'hawkes'), ``event_rate`` (per year; the baseline rate for
    Hawkes with ``event_alpha`` / ``event_beta``), jump sizes ``jump_median``, ``jump_sigma`` (lognormal),
    ``p_up`` (probability a jump is positive), ``jump_cap`` (largest log jump), and the traders' reaction:
    ``panic`` (extra noise right after news, in units of the median jump) with decay ``panic_decay`` (years),
    ``underreaction`` (average share of a jump a trader fails to price in at once; the gap then closes at
    speed ``reversion``, which produces post-news drift).
    """
    version: int = 1
    steps: int = 1000
    T: float = 1.0
    s0: float = 100.0
    streams: int = 100
    mu: float = 0.05
    sigma: float = 0.2
    seed: int | None = None
    heston: dict = field(default_factory=dict)
    kinds: tuple = KINDS
    mix: tuple | None = None
    dispersion: float = 0.25
    correlation: float = 0.25
    merton: dict = field(default_factory=dict)
    aggregate: str = 'mean'
    value: str = 'gbm'
    value_target: float | None = None
    tether: str = 'ou'
    deviation: float = 0.015
    reversion: float = 50.0
    bias: float = 0.004
    event_process: str = 'poisson'
    event_rate: float = 12.0
    event_alpha: float = 50.0
    event_beta: float = 125.0
    jump_median: float = 0.02
    jump_sigma: float = 0.6
    p_up: float = 0.5
    jump_cap: float = 0.5
    panic: float = 1.0
    panic_decay: float = 0.01
    underreaction: float = 0.4

    def __post_init__(self):
        if self.version not in (1, 2, 3, 4):
            raise ValueError('version must be 1, 2, 3 or 4')
        if self.steps < 10 or self.streams < 1:
            raise ValueError('need steps >= 10 and streams >= 1')
        bad = set(self.kinds) - set(KINDS)
        if bad:
            raise ValueError(f'unknown stream kinds {sorted(bad)}; choose from {KINDS}')
        if self.value not in ('gbm', 'bridge', 'abm', 'heston'):
            raise ValueError("value must be 'gbm', 'bridge', 'abm' or 'heston'")
        if self.tether not in ('ou', 'follow'):
            raise ValueError("tether must be 'ou' or 'follow'")
        if self.event_process not in ('poisson', 'hawkes'):
            raise ValueError("event_process must be 'poisson' or 'hawkes'")
        if self.aggregate not in ('mean', 'median', 'geometric'):
            raise ValueError("aggregate must be 'mean', 'median' or 'geometric'")
        if self.event_process == 'hawkes' and self.event_alpha >= self.event_beta:
            raise ValueError('Hawkes needs event_alpha < event_beta')

    # --------------------------------------------------------------- helpers
    @property
    def dt(self) -> float:
        return self.T / self.steps

    def _heston(self, n: int, rng, hetero: float = 0.0) -> dict:
        h = {'kappa': 3.0, 'theta': self.sigma ** 2, 'xi': 0.45, 'rho': -0.7, 'v0': None, **self.heston}
        h['v0'] = h['theta'] if h['v0'] is None else h['v0']
        out = {k: np.full(n, float(v)) for k, v in h.items()}
        if hetero > 0:
            for k in ('theta', 'xi', 'kappa'):
                out[k] = out[k] * np.exp(hetero * rng.standard_normal(n) - 0.5 * hetero ** 2)
            out['v0'] = out['theta'].copy()
            out['rho'] = np.clip(out['rho'] + 0.1 * hetero * rng.standard_normal(n), -0.95, 0.0)
        return out

    def _aggregate(self, S: np.ndarray) -> np.ndarray:
        if self.aggregate == 'mean':
            return S.mean(axis=0)
        if self.aggregate == 'median':
            return np.median(S, axis=0)
        return np.exp(np.log(S).mean(axis=0))

    def _title(self, extra: str = '') -> str:
        names = {1: 'single Heston', 2: 'ensemble average', 3: 'internal value + traders',
                 4: 'internal value with news + traders'}
        return f'Version {self.version}: {names[self.version]}{extra}'

    # ------------------------------------------------------------------ run
    def run(self, seed: int | None = None) -> MarketResult:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        times = np.arange(self.steps + 1) * self.dt
        return {1: self._v1, 2: self._v2, 3: self._v3, 4: self._v4}[self.version](rng, times)

    def _v1(self, rng, times) -> MarketResult:
        dt = self.dt
        h = self._heston(1, rng)
        dW = rng.normal(0, math.sqrt(dt), (1, self.steps))
        dZ = rng.normal(0, math.sqrt(dt), (1, self.steps))
        S, var = K.heston(self.s0, self.mu, h['kappa'], h['theta'], h['xi'], h['rho'], h['v0'], dW, dZ, dt)
        return MarketResult(1, times, S[0], S, np.array(['heston']), extras={'variance': var[0]},
                            params={'title': self._title(f' (kappa={h["kappa"][0]:g}, rho={h["rho"][0]:g})')})

    def _v2(self, rng, times) -> MarketResult:
        n, steps, dt, T = self.streams, self.steps, self.dt, self.T
        kinds = np.array(self.kinds)
        w = np.ones(len(kinds)) if self.mix is None else np.asarray(self.mix, float)
        which = rng.choice(len(kinds), size=n, p=w / w.sum())
        disp = self.dispersion
        sig = self.sigma * np.exp(disp * rng.standard_normal(n) - 0.5 * disp ** 2)
        mu = self.mu + 0.1 * disp * rng.standard_normal(n)
        rho = float(np.clip(self.correlation, 0.0, 1.0))
        common = rng.normal(0, math.sqrt(dt), steps)
        dW = math.sqrt(rho) * common[None, :] + math.sqrt(1 - rho) * rng.normal(0, math.sqrt(dt), (n, steps))
        S = np.empty((n, steps + 1))
        mj = {'rate': 3.0, 'mean': -0.02, 'std': 0.05, **self.merton}
        for ki, kind in enumerate(kinds):
            sel = np.flatnonzero(which == ki)
            if not sel.size:
                continue
            if kind == 'gbm':
                S[sel] = K.gbm(self.s0, mu[sel], sig[sel], dW[sel], dt)
            elif kind == 'abm':
                S[sel] = K.abm(self.s0, mu[sel], sig[sel], dW[sel], dt)
            elif kind == 'bridge':
                target = self.s0 * np.exp((mu[sel] - 0.5 * sig[sel] ** 2) * T + sig[sel] * math.sqrt(T) * rng.standard_normal(sel.size))
                S[sel] = K.bridge(self.s0, target, sig[sel], dW[sel], dt)
            elif kind == 'heston':
                h = self._heston(sel.size, rng, hetero=disp)
                dZ = rng.normal(0, math.sqrt(dt), (sel.size, steps))
                S[sel] = K.heston(self.s0, mu[sel], h['kappa'], h['theta'], h['xi'], h['rho'], h['v0'], dW[sel], dZ, dt)[0]
            else:       # merton
                jumps = K.merton_jumps(np.full(sel.size, mj['rate']), mj['mean'], mj['std'], steps, dt, rng)
                kbar = math.exp(mj['mean'] + 0.5 * mj['std'] ** 2) - 1
                inc = (mu[sel] - 0.5 * sig[sel] ** 2 - mj['rate'] * kbar)[:, None] * dt + sig[sel][:, None] * dW[sel] + jumps
                S[sel] = self.s0 * np.exp(K.cum0(inc))
        return MarketResult(2, times, self._aggregate(S), S, kinds[which],
                            params={'title': self._title(f' ({n} streams, common factor {rho:.2f}, {self.aggregate})')})

    # ----- internal value + traders (3 and 4)
    def _value_path(self, rng, dt) -> np.ndarray:
        steps, T = self.steps, self.T
        dW = rng.normal(0, math.sqrt(dt), (1, steps))
        if self.value == 'gbm':
            return K.gbm(self.s0, self.mu, self.sigma, dW, dt)[0]
        if self.value == 'abm':
            return K.abm(self.s0, self.mu, self.sigma, dW, dt)[0]
        if self.value == 'bridge':
            target = self.s0 * math.exp(self.mu * T) if self.value_target is None else self.value_target
            return K.bridge(self.s0, target, self.sigma, dW, dt)[0]
        h = self._heston(1, rng)
        dZ = rng.normal(0, math.sqrt(dt), (1, steps))
        return K.heston(self.s0, self.mu, h['kappa'], h['theta'], h['xi'], h['rho'], h['v0'], dW, dZ, dt)[0][0]

    def _news(self, rng, dt):
        """Event arrivals and jumps: per-step log-jump totals, an event table and the arrival intensity."""
        steps = self.steps
        intensity = np.full(steps, float(self.event_rate))
        if self.event_process == 'poisson':
            counts = rng.poisson(self.event_rate * dt, steps)
        else:
            mu, a, b = self.event_rate, self.event_alpha, self.event_beta
            counts = np.zeros(steps, dtype=int)
            lam, decay = mu, math.exp(-b * dt)
            for k in range(steps):
                intensity[k] = lam
                counts[k] = rng.poisson(lam * dt)
                lam = mu + (lam - mu) * decay + a * counts[k]
        total = int(counts.sum())
        sign = np.where(rng.random(total) < self.p_up, 1.0, -1.0)
        size = np.minimum(rng.lognormal(math.log(self.jump_median), self.jump_sigma, total), self.jump_cap)
        jump = sign * size
        step = np.repeat(np.arange(steps), counts)
        inc = np.bincount(step, weights=jump, minlength=steps)
        events = pd.DataFrame({'step': step + 1, 'time': (step + 1) * dt, 'jump': jump})
        return inc, events, intensity

    def _value_traders(self, rng, times, news: bool) -> MarketResult:
        n, steps, dt = self.streams, self.steps, self.dt
        V0 = self._value_path(rng, dt)
        extras = {}
        events = None
        scale = None
        value = V0
        if news:
            inc, events, intensity = self._news(rng, dt)
            value = V0 * np.exp(np.r_[0.0, np.cumsum(inc)])
            heat = np.empty(steps)
            h, decay = 0.0, math.exp(-dt / self.panic_decay)
            for k in range(steps):
                h = h * decay + abs(inc[k])
                heat[k] = h
            scale = 1.0 + self.panic * heat / self.jump_median
            extras.update(intensity=intensity, heat=heat, noise_scale=scale)
        theta = self.reversion * np.exp(0.5 * rng.standard_normal(n) - 0.125)
        dev = self.deviation * np.exp(0.4 * rng.standard_normal(n) - 0.08)
        eta = dev * np.sqrt(2.0 * theta)
        bias = self.bias * rng.standard_normal(n)
        z = rng.standard_normal((n, steps))
        kappa = np.clip(rng.uniform(0.0, 2.0 * self.underreaction, n), 0.0, 1.0)
        if self.tether == 'ou':
            d = K.ou_deviation(theta, eta, bias, rng.standard_normal(n), z, dt, scale,
                               kick=inc if news else None, kappa=kappa)
        else:
            m = np.ones(steps) if scale is None else scale
            d = bias[:, None] + K.cum0(eta[:, None] * math.sqrt(dt) * m[None, :] * z)
        S = value[None, :] * np.exp(d)
        extras.update(theta=theta, deviation=dev, bias=bias, underreaction=kappa)
        return MarketResult(4 if news else 3, times, self._aggregate(S), S, np.array(['trader'] * n), value=value,
                            events=events, extras=extras,
                            params={'title': self._title(f' ({n} traders, value={self.value}, tether={self.tether}'
                                                         + (f', {self.event_process} news)' if news else ')'))})

    def _v3(self, rng, times) -> MarketResult:
        return self._value_traders(rng, times, news=False)

    def _v4(self, rng, times) -> MarketResult:
        return self._value_traders(rng, times, news=True)


# ================================================================== comparison
@dataclass
class VersionComparison:
    """The four versions run with the same settings."""
    results: dict[int, MarketResult]
    table: pd.DataFrame

    def plot(self, show: bool = True):
        import matplotlib.pyplot as plt
        with V.themed():
            fig, axes = plt.subplots(2, 4, figsize=(15, 6.2), gridspec_kw={'height_ratios': [1.4, 1]})
            for k, (v, r) in enumerate(self.results.items()):
                ax = axes[0, k]
                for row in r.streams[:: max(1, r.streams.shape[0] // 40)]:
                    ax.plot(r.times, row, color=V.OTHER, lw=0.4, alpha=0.3)
                ax.plot(r.times, r.price, color=V.BLUE, lw=1.5)
                if r.value is not None:
                    ax.plot(r.times, r.value, color=V.ORANGE, lw=1.1, ls='--')
                ax.set_ylim(*V.robust_limits(np.r_[r.price, r.streams.ravel()], 1, 99, 0.1))
                ax.set_title(f'Version {v}')
                ax2 = axes[1, k]
                ret = r.returns
                z = (ret - ret.mean()) / ret.std(ddof=1)
                ax2.hist(z, bins=np.linspace(-6, 6, 49), density=True, color=V.BLUE, alpha=0.45)
                xs = np.linspace(-6, 6, 300)
                ax2.plot(xs, stats.norm.pdf(xs), color=V.TEXT, lw=1)
                ax2.set_yscale('log')
                ax2.set_ylim(1e-4, 1)
                ax2.set_title(f'kurtosis {stats.kurtosis(ret):.1f}, acf|r| {_acf(np.abs(ret), [1])[0]:.2f}', fontsize=9)
            V.suptitle(fig, 'Four versions of a stochastic market')
            fig.tight_layout(rect=(0, 0, 1, 0.95))
            return V.finish(fig, show)


def compare_versions(versions=(1, 2, 3, 4), seed: int | None = 0, **kwargs) -> VersionComparison:
    """Run several versions with shared settings and tabulate their stylised facts."""
    results = {v: Market(version=v, seed=seed, **kwargs).run() for v in versions}
    table = pd.concat([r.facts() for r in results.values()], axis=1)
    table.columns = [f'version {v}' for v in results]
    return VersionComparison(results, table)
