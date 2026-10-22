"""Simulated paths: statistics, theory overlays, arithmetic and plots.

``Paths`` holds an ``(n_runs, n_steps + 1)`` array on a shared time grid.
Statistics across runs come back as pandas Series indexed by time::

    X.mean, X.std, X.var, X.median, X.quantile(0.95)
    X.final                       # terminal values (one per run)
    X.describe()                  # terminal moments: simulated vs theory

Paths combine pathwise: ``0.5*X + 0.2*Y + 0.3*Z`` is a new Paths. When the
components were driven by *independent* noise, the theory follows exactly:
means add, variances add (scaled by w^2), and the terminal density is the
**convolution** of the scaled component densities.
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import fftconvolve

from .. import _viz as V

__all__ = ['Theory', 'Paths', 'terminal_density', 'convolved_density', 'plot_terminal',
           'plot_many']

_group_counter = itertools.count(1)


def new_group() -> int:
    """A fresh id for an independent noise source."""
    return next(_group_counter)


@dataclass
class Theory:
    """Known properties of a process. Any field may be None.

    mean(t), var(t)  -> arrays for an array of times
    pdf(x, t)        -> density of X_t (continuous processes)
    pmf(k, t)        -> probabilities of X_t on its lattice (discrete processes)
    """
    mean: Callable | None = None
    var: Callable | None = None
    pdf: Callable | None = None
    pmf: Callable | None = None
    cdf: Callable | None = None                  # used for exact binning in convolutions
    support: tuple[float, float] | None = None   # hard bounds of X_t, e.g. (0, inf)
    notes: str = ''


@dataclass
class Paths:
    values: np.ndarray                       # (n_runs, n_steps + 1)
    times: np.ndarray                        # (n_steps + 1,)
    name: str = 'X'
    params: dict = field(default_factory=dict)
    theory: Theory = field(default_factory=Theory)
    extras: dict = field(default_factory=dict)   # companion series: variance, intensity, regime...
    groups: frozenset = field(default_factory=frozenset)  # noise sources used
    discrete: bool = False
    terms: tuple | None = None               # (offset, ((w, base Paths), ...)) for linear combos

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2 or self.values.shape[1] != len(self.times):
            raise ValueError('values must be (n_runs, len(times))')

    # ------------------------------------------------------------- shape
    @property
    def n_runs(self) -> int:
        return self.values.shape[0]

    @property
    def n_steps(self) -> int:
        return self.values.shape[1] - 1

    @property
    def dt(self) -> float:
        return float(self.times[1] - self.times[0]) if len(self.times) > 1 else 0.0

    @property
    def T(self) -> float:
        return float(self.times[-1])

    def __len__(self) -> int:
        return self.n_runs

    def __getitem__(self, index):
        """``X[i]`` -> one path (array); ``X[a:b]`` -> Paths with those runs."""
        if isinstance(index, (int, np.integer)):
            return self.values[index]
        return Paths(self.values[index], self.times, self.name, self.params, self.theory,
                     {}, self.groups, self.discrete)

    # --------------------------------------------------------- statistics
    def _series(self, data, name):
        return pd.Series(data, index=pd.Index(self.times, name='time'), name=name)

    @property
    def mean(self) -> pd.Series:
        return self._series(self.values.mean(axis=0), 'mean')

    @property
    def var(self) -> pd.Series:
        return self._series(self.values.var(axis=0, ddof=1), 'var')

    @property
    def std(self) -> pd.Series:
        return self._series(self.values.std(axis=0, ddof=1), 'std')

    @property
    def median(self) -> pd.Series:
        return self._series(np.median(self.values, axis=0), 'median')

    def quantile(self, q):
        qs = np.atleast_1d(q)
        data = np.quantile(self.values, qs, axis=0)
        if np.ndim(q) == 0:
            return self._series(data[0], f'q{q:g}')
        return pd.DataFrame(data.T, index=pd.Index(self.times, name='time'),
                            columns=[f'q{x:g}' for x in qs])

    @property
    def final(self) -> np.ndarray:
        return self.values[:, -1]

    def at(self, t: float) -> np.ndarray:
        """Values of every run at the grid time closest to ``t``."""
        return self.values[:, int(np.argmin(np.abs(self.times - t)))]

    @property
    def increments(self) -> np.ndarray:
        return np.diff(self.values, axis=1)

    @property
    def log_returns(self) -> np.ndarray:
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.diff(np.log(self.values), axis=1)

    def theory_mean(self, t=None):
        t = self.times if t is None else np.asarray(t, float)
        return None if self.theory.mean is None else np.asarray(self.theory.mean(t), float)

    def theory_var(self, t=None):
        t = self.times if t is None else np.asarray(t, float)
        return None if self.theory.var is None else np.asarray(self.theory.var(t), float)

    def describe(self) -> pd.DataFrame:
        """Terminal statistics, simulated vs theoretical (where known)."""
        x = self.final
        se_mean = x.std(ddof=1) / np.sqrt(len(x))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            skew, kurt = (stats.skew(x), stats.kurtosis(x)) if np.ptp(x) > 0 else (0.0, 0.0)
        rows = {
            'mean': (x.mean(), self._t_last(self.theory_mean)),
            'std': (x.std(ddof=1), _sqrt_or_none(self._t_last(self.theory_var))),
            'var': (x.var(ddof=1), self._t_last(self.theory_var)),
            'skew': (skew, None), 'excess kurtosis': (kurt, None),
            'min': (x.min(), None), 'median': (np.median(x), None), 'max': (x.max(), None),
        }
        df = pd.DataFrame(rows, index=['simulated', 'theory']).T
        df['z (mean)'] = [(df.loc['mean', 'simulated'] - df.loc['mean', 'theory']) / se_mean
                          if df.loc['mean', 'theory'] is not None and se_mean > 0 else None] + [None] * 7
        df.index.name = f'{self.name} at t={self.T:g}  ({self.n_runs} runs)'
        return df

    def _t_last(self, fn):
        v = fn(np.array([self.T]))
        return None if v is None else float(v[0])

    def hurst(self, max_lag: int | None = None) -> float:
        """Hurst exponent from variance scaling: Var[X(t+k) - X(t)] ~ k^(2H)."""
        max_lag = max_lag or max(4, self.n_steps // 10)
        lags = np.unique(np.geomspace(1, max_lag, 12).astype(int))
        v = [np.var(self.values[:, k:] - self.values[:, :-k]) for k in lags]
        slope = np.polyfit(np.log(lags), np.log(v), 1)[0]
        return float(slope / 2.0)

    def acf(self, max_lag: int = 20) -> pd.Series:
        """Autocorrelation of increments, averaged over runs."""
        d = self.increments - self.increments.mean(axis=1, keepdims=True)
        denom = (d * d).sum(axis=1)
        out = [1.0] + [float(np.mean((d[:, k:] * d[:, :-k]).sum(axis=1) / denom))
                       for k in range(1, max_lag + 1)]
        return pd.Series(out, index=pd.Index(range(max_lag + 1), name='lag'), name='acf')

    def first_passage(self, level: float, direction: str = 'up') -> np.ndarray:
        """First grid time each run reaches ``level`` (NaN if never)."""
        hit = self.values >= level if direction == 'up' else self.values <= level
        idx = np.argmax(hit, axis=1).astype(float)
        idx[~hit.any(axis=1)] = np.nan
        return np.where(np.isnan(idx), np.nan, self.times[np.nan_to_num(idx).astype(int)])

    def max_drawdown(self) -> np.ndarray:
        """Largest peak-to-trough fall of each run (relative for positive paths)."""
        peak = np.maximum.accumulate(self.values, axis=1)
        if np.all(self.values > 0):
            return np.max(1.0 - self.values / peak, axis=1)
        return np.max(peak - self.values, axis=1)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.values.T, index=pd.Index(self.times, name='time'),
                            columns=[f'run_{i}' for i in range(self.n_runs)])

    # --------------------------------------------------------- arithmetic
    def _linear(self):
        return self.terms if self.terms is not None else (0.0, ((1.0, self),))

    def _check(self, other: 'Paths'):
        if other.values.shape != self.values.shape or not np.allclose(other.times, self.times):
            raise ValueError('paths must share the same runs and time grid (same Process)')

    def _combine(self, other, sign: float = 1.0) -> 'Paths':
        if isinstance(other, Process_like):
            other = other.paths
        if isinstance(other, Paths):
            self._check(other)
            o1, t1 = self._linear()
            o2, t2 = other._linear()
            terms = (o1 + sign * o2, t1 + tuple((sign * w, p) for w, p in t2))
            values = self.values + sign * other.values
            name = f'{self.name} {"+" if sign > 0 else "−"} {other.name}'
            return Paths(values, self.times, name, {}, _linear_theory(*terms),
                         groups=self.groups | other.groups,
                         discrete=self.discrete and other.discrete, terms=terms)
        c = float(other)
        offset, terms = self._linear()
        terms_new = (offset + sign * c, terms)
        return Paths(self.values + sign * c, self.times, f'{self.name} {"+" if sign > 0 else "−"} {c:g}',
                     self.params, _linear_theory(*terms_new), {}, self.groups, False, terms_new)

    def __add__(self, other):
        return self._combine(other, 1.0)

    def __radd__(self, other):
        return self._combine(other, 1.0) if not (isinstance(other, (int, float)) and other == 0) else self

    def __sub__(self, other):
        return self._combine(other, -1.0)

    def __rsub__(self, other):
        return (-self)._combine(other, 1.0)

    def __mul__(self, other):
        if isinstance(other, Paths):
            self._check(other)
            return Paths(self.values * other.values, self.times, f'({self.name})·({other.name})',
                         groups=self.groups | other.groups)
        a = float(other)
        offset, terms = self._linear()
        terms_new = (a * offset, tuple((a * w, p) for w, p in terms))
        name = f'{a:g}·{self.name}' if self.terms is None else f'{a:g}·({self.name})'
        return Paths(a * self.values, self.times, name, self.params,
                     _linear_theory(*terms_new), {}, self.groups, False, terms_new)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Paths):
            self._check(other)
            return Paths(self.values / other.values, self.times, f'({self.name})/({other.name})',
                         groups=self.groups | other.groups)
        return self * (1.0 / float(other))

    def __neg__(self):
        return self * -1.0

    def apply(self, func: Callable[[np.ndarray], np.ndarray], name: str | None = None) -> 'Paths':
        """Pathwise transform (theory is dropped)."""
        return Paths(func(self.values), self.times, name or f'f({self.name})', groups=self.groups)

    def exp(self) -> 'Paths':
        return self.apply(np.exp, f'exp({self.name})')

    def log(self) -> 'Paths':
        return self.apply(np.log, f'log({self.name})')

    @property
    def independent(self) -> bool:
        """True when every linear component was driven by its own noise."""
        if self.terms is None:
            return True
        seen: set = set()
        for _, p in self.terms[1]:
            if seen & p.groups:
                return False
            seen |= p.groups
        return True

    # ---------------------------------------------------------- densities
    def terminal_density(self, x: np.ndarray) -> np.ndarray | None:
        """Density of X_T on ``x`` from theory (convolution for independent sums)."""
        return terminal_density(self, x)

    # --------------------------------------------------------------- plot
    def plot(self, max_paths: int = 150, *, bands: bool = True, theory: bool = True,
             logy: bool | str = 'auto', title: str | None = None, show: bool = True):
        """Left: sample paths, mean and 5-95% band; right: terminal distribution."""
        return _plot_paths(self, max_paths, bands, theory, logy, title, show)

    def __repr__(self) -> str:
        return (f'Paths({self.name}: {self.n_runs} runs x {self.n_steps} steps, '
                f'T={self.T:g}, dt={self.dt:g})')


class Process_like:  # marker base so Paths arithmetic accepts a Process (see processes.py)
    paths: Paths


def _sqrt_or_none(v):
    return None if v is None else float(np.sqrt(max(v, 0.0)))


def _linear_theory(offset: float, terms: tuple) -> Theory:
    comps = [p for _, p in terms]
    mean = var = None
    if all(p.theory.mean is not None for p in comps):
        def mean(t, _o=offset, _terms=terms):
            return _o + sum(w * np.asarray(p.theory.mean(t), float) for w, p in _terms)
    seen: set = set()
    independent = True
    for p in comps:
        if seen & p.groups:
            independent = False
        seen |= p.groups
    if independent and all(p.theory.var is not None for p in comps):
        def var(t, _terms=terms):
            return sum(w * w * np.asarray(p.theory.var(t), float) for w, p in _terms)
    return Theory(mean=mean, var=var,
                  notes='' if independent else 'components share noise: variance not additive')


# ------------------------------------------------------------- densities
def terminal_density(paths: Paths, x: np.ndarray) -> np.ndarray | None:
    """Theoretical density of the terminal value, or None if unknown."""
    x = np.asarray(x, float)
    T = paths.T
    if paths.terms is None:
        if paths.theory.pdf is None:
            return None
        return np.asarray(paths.theory.pdf(x, T), float)
    offset, terms = paths.terms
    if not paths.independent or any(p.theory.pdf is None for _, p in terms):
        return None
    grid, dens = convolved_density([w for w, _ in terms], [p for _, p in terms])
    return np.interp(x - offset, grid, dens, left=0.0, right=0.0)


def _range(p: Paths, w: float):
    """Grid range for w * X_T: empirical 0.05-99.95% band, widened, clipped to the support."""
    lo, hi = np.percentile(p.final, [0.05, 99.95])
    span = max(hi - lo, 1e-9)
    lo, hi = lo - 0.25 * span, hi + 0.25 * span
    if p.theory.support is not None:
        s_lo, s_hi = p.theory.support(p.T) if callable(p.theory.support) else p.theory.support
        lo, hi = max(lo, s_lo), min(hi, s_hi)
    y = sorted((w * lo, w * hi))
    return y[0], y[1]


def _component_masses(p: Paths, w: float, h: float):
    """Probability mass of w * X_T in cells of width h (exact via the cdf when known)."""
    ylo, yhi = _range(p, w)
    centres = np.arange(np.floor(ylo / h) * h, yhi + h, h)
    T = p.T
    if p.theory.cdf is not None:
        edges = np.append(centres - h / 2, centres[-1] + h / 2) / w
        F = np.asarray(p.theory.cdf(edges, T), float)
        mass = np.abs(np.diff(F))
    else:
        mass = np.asarray(p.theory.pdf(centres / w, T), float) * h / abs(w)
    return centres, np.nan_to_num(mass, nan=0.0, posinf=0.0)


def convolved_density(weights: Sequence[float], components: Sequence[Paths], points: int = 2 ** 16):
    """Density of ``sum_k w_k X_k(T)`` for independent components, by numerical convolution.

    Each scaled component w_k X_k(T) is binned into exact probability masses
    on a common grid (via its cdf), the mass vectors are convolved (FFT) and
    divided by the cell width. Returns ``(grid, density)``.
    """
    spans = [np.diff(_range(p, w))[0] for w, p in zip(weights, components)]
    h = max(sum(spans), 1e-12) / points
    grid = mass = None
    for w, p in zip(weights, components):
        g, m = _component_masses(p, w, h)
        if grid is None:
            grid, mass = g, m
        else:
            mass = fftconvolve(mass, m)
            grid = grid[0] + g[0] + h * np.arange(len(mass))
    return grid, np.clip(mass, 0.0, None) / h


# ------------------------------------------------------------------ plots
def _plot_paths(p: Paths, max_paths, bands, theory, logy, title, show):
    vals = p.values
    if logy == 'auto':
        fin = vals[np.isfinite(vals)]
        pos = fin[fin > 0]
        logy = bool(len(pos) > 0.999 * len(fin) and
                    np.percentile(pos, 99.5) / max(np.percentile(pos, 50), 1e-300) > 10)
    with V.themed():
        fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True,
                                      gridspec_kw={'width_ratios': [3.2, 1], 'wspace': 0.04})
        k = min(max_paths // 3 if p.discrete else max_paths, p.n_runs)   # lattice paths overlap
        idx = np.linspace(0, p.n_runs - 1, k).astype(int)
        drawstyle = 'steps-post' if p.discrete else 'default'
        for i in idx:
            ax.plot(p.times, vals[i], color=V.BLUE, alpha=0.18 if k > 30 else 0.5, linewidth=0.8,
                    drawstyle=drawstyle)
        if bands:
            q = np.quantile(vals, [0.05, 0.95], axis=0)
            ax.fill_between(p.times, q[0], q[1], color=V.BLUE, alpha=0.08, linewidth=0,
                            label='5–95% of runs')
        ax.plot(p.times, vals.mean(axis=0), color=V.TEXT, linewidth=2.0, label='sample mean')
        m, v = p.theory_mean(), p.theory_var()
        if theory and m is not None:
            ax.plot(p.times, m, color=V.ORANGE, linewidth=1.6, label='theory mean')
            if v is not None:
                sd = np.sqrt(np.maximum(v, 0))
                ax.plot(p.times, m + sd, color=V.ORANGE, linewidth=0.9, alpha=0.8, label='theory ± 1 sd')
                ax.plot(p.times, m - sd, color=V.ORANGE, linewidth=0.9, alpha=0.8)
        ax.legend(loc='upper left', ncols=2)
        ax.set(xlabel='time', ylabel=p.name.split(' ')[0] if len(p.name) < 12 else 'value')
        if logy:
            ax.set_yscale('log')
            lo, hi = np.percentile(vals[vals > 0], [0.3, 99.7])
            ax.set_ylim(lo / 1.5, hi * 1.5)
        else:
            ax.set_ylim(*V.robust_limits(vals, 0.2, 99.8))
        ax.set_title(title or f'{p.name}   ·   {p.n_runs} runs × {p.n_steps} steps, T = {p.T:g}')

        x = p.final
        ylo, yhi = ax.get_ylim()
        inside = x[(x >= ylo) & (x <= yhi)]
        if p.discrete:
            lo_i, hi_i = np.floor(inside.min()), np.ceil(inside.max())
            step = _lattice_step(x)
            edges = np.arange(lo_i - step / 2, hi_i + step, step)
        if logy:
            # density per log10 unit, so mass near zero does not swamp the picture
            e = np.linspace(np.log10(ylo), np.log10(yhi), 46)
            counts, _ = np.histogram(np.log10(x[x > 0]), bins=e)
            dens_hist = counts / (len(x) * np.diff(e))
            axh.stairs(dens_hist, 10 ** e, orientation='horizontal', fill=True, color=V.BLUE,
                       alpha=0.15)
            axh.stairs(dens_hist, 10 ** e, orientation='horizontal', color=V.BLUE, linewidth=1.4)
        else:
            if not p.discrete:
                edges = np.linspace(inside.min(), inside.max(), 45)
            axh.hist(x, bins=edges, orientation='horizontal', density=True, color=V.BLUE, alpha=0.15)
            axh.hist(x, bins=edges, orientation='horizontal', density=True, histtype='step',
                     color=V.BLUE, linewidth=1.4)
        if theory:
            if p.discrete and p.theory.pmf is not None:
                ks = np.arange(lo_i, hi_i + step, step)
                pm = np.asarray(p.theory.pmf(ks, p.T), float) / step
                axh.plot(pm, ks, 'o', color=V.ORANGE, markersize=3.5, label='theory')
            else:
                grid = (np.geomspace(max(ylo, 1e-300), yhi, 600) if logy
                        else np.linspace(ylo, yhi, 600))
                dens = terminal_density(p, grid)
                if dens is not None:
                    if logy:
                        dens = dens * grid * np.log(10)
                    label = 'theory' if p.terms is None else 'convolution'
                    axh.plot(dens, grid, color=V.ORANGE, linewidth=1.6, label=label)
            if axh.get_legend_handles_labels()[0]:
                axh.legend(loc='upper right')
        axh.set(xlabel='density per log10 unit' if logy else 'density', title=f'X at t = {p.T:g}')
        axh.tick_params(labelleft=False)
        axh.grid(axis='y', visible=False)
    return V.finish(fig, show)


def _lattice_step(x: np.ndarray) -> float:
    d = np.diff(np.unique(np.round(x, 9)))
    return float(d.min()) if len(d) else 1.0


def plot_terminal(*paths: Paths, labels: Sequence[str] | None = None, bins: int = 60,
                  theory: bool = True, title: str = 'Terminal distributions', show: bool = True):
    """Overlay the terminal distributions of several Paths (same axis)."""
    labels = labels or [p.name for p in paths]
    allv = np.concatenate([p.final for p in paths])
    logx = bool(allv.min() > 0 and np.percentile(allv, 99) / np.percentile(allv, 50) > 10)
    if logx:  # plot the density of log10(value)
        lo, hi = np.percentile(allv, [0.5, 99.5])
        edges = np.geomspace(lo, hi, bins)
        grid = np.geomspace(lo, hi, 800)
    else:
        lo, hi = V.robust_limits(allv, 0.5, 99.5, pad=0.02)
        edges = np.linspace(lo, hi, bins)
        grid = np.linspace(lo, hi, 800)
    with V.themed():
        fig, ax = plt.subplots(figsize=(9, 4))
        for k, (p, lab) in enumerate(zip(paths, labels)):
            color = V.SERIES[k % len(V.SERIES)]
            if logx:
                counts, _ = np.histogram(np.log10(p.final[p.final > 0]), bins=np.log10(edges))
                ax.stairs(counts / (len(p.final) * np.diff(np.log10(edges))), edges, color=color,
                          linewidth=1.6, label=lab)
            else:
                ax.hist(p.final, bins=edges, density=True, histtype='step', color=color,
                        linewidth=1.6, label=lab)
            if theory:
                dens = terminal_density(p, grid)
                if dens is not None:
                    ax.plot(grid, dens * (grid * np.log(10) if logx else 1.0), color=color,
                            linewidth=1.0, alpha=0.9)
        if logx:
            ax.set_xscale('log')
        ax.legend(loc='upper right')
        ax.set(title=title, xlabel='value at T' + ('  (log scale)' if logx else ''),
               ylabel=('density per log10 unit' if logx else 'density')
               + ('   (thin lines: theory)' if theory else ''))
    return V.finish(fig, show)


def plot_many(paths: Sequence[Paths], max_paths: int = 60, ncols: int = 3, show: bool = True):
    """Small multiples: a few paths + mean for each process."""
    n = len(paths)
    nrows = int(np.ceil(n / ncols))
    with V.themed():
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows), squeeze=False)
        for ax, p in zip(axs.flat, paths):
            idx = np.linspace(0, p.n_runs - 1, min(max_paths, p.n_runs)).astype(int)
            for i in idx:
                ax.plot(p.times, p.values[i], color=V.BLUE, alpha=0.2, linewidth=0.7,
                        drawstyle='steps-post' if p.discrete else 'default')
            ax.plot(p.times, p.values.mean(axis=0), color=V.TEXT, linewidth=1.8)
            m = p.theory_mean()
            if m is not None:
                ax.plot(p.times, m, color=V.ORANGE, linewidth=1.3)
            ax.set_ylim(*V.robust_limits(p.values, 0.5, 99.5))
            ax.set_title(p.name, fontsize=9.5)
        for ax in list(axs.flat)[n:]:
            ax.set_visible(False)
        fig.tight_layout()
    return V.finish(fig, show)
