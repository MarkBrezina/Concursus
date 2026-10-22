"""Shared pieces for population models: traits, inequality measures, analysis.

Traits
------
``Trait('talent', 'normal', 0.6, 0.1, clip=(0, 1))`` describes how a trait is
distributed across a population; ``sample_traits`` draws a whole population.
Distributions: 'normal' (mean, sd), 'lognormal' (mean, sd of the variable
itself), 'uniform' (low, high), 'beta' (a, b), 'gamma' (mean, cv),
'constant' (value).

Analysis
--------
``trait_effects`` asks "which traits matter?": Spearman rank correlation of
each trait with an outcome (lifespan, wealth, infections caused) plus how the
top decile differs from everyone else in standard-deviation units.
"""
from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ['Trait', 'sample_traits', 'mutate', 'gini', 'lorenz', 'top_share', 'trait_effects',
           'leaders', 'Result', 'Timer']


@dataclass(frozen=True)
class Trait:
    name: str
    dist: str = 'normal'
    a: float = 0.0
    b: float = 1.0
    clip: tuple[float, float] | None = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d, a, b = self.dist, self.a, self.b
        if d == 'normal':
            x = rng.normal(a, b, n)
        elif d == 'lognormal':
            if a <= 0:
                raise ValueError('lognormal mean must be positive')
            s2 = np.log1p((b / a) ** 2)
            x = rng.lognormal(np.log(a) - s2 / 2, np.sqrt(s2), n)
        elif d == 'uniform':
            x = rng.uniform(a, b, n)
        elif d == 'beta':
            x = rng.beta(a, b, n)
        elif d == 'gamma':          # mean a, coefficient of variation b
            k = 1.0 / b ** 2
            x = rng.gamma(k, a / k, n)
        elif d == 'constant':
            x = np.full(n, float(a))
        else:
            raise ValueError(f'unknown distribution {d!r}')
        if self.clip is not None:
            x = np.clip(x, *self.clip)
        return x


def sample_traits(specs: Sequence[Trait], n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {t.name: t.sample(n, rng) for t in specs}


def mutate(values: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative (lognormal) mutation: keeps positive traits positive."""
    if rate <= 0:
        return values.copy()
    return values * np.exp(rate * rng.standard_normal(np.shape(values)))


# ------------------------------------------------------------- inequality
def gini(x: np.ndarray) -> float:
    """Gini coefficient (0 = perfect equality, 1 = one holder owns everything)."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if n == 0 or x.sum() <= 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def lorenz(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(population share, wealth share) points of the Lorenz curve."""
    x = np.sort(np.asarray(x, float))
    cum = np.concatenate([[0.0], np.cumsum(x)])
    return np.linspace(0, 1, len(cum)), cum / cum[-1] if cum[-1] > 0 else cum


def top_share(x: np.ndarray, q: float) -> float:
    """Share of the total held by the top fraction q (e.g. 0.01 = top 1%)."""
    x = np.sort(np.asarray(x, float))[::-1]
    k = max(1, int(round(q * len(x))))
    total = x.sum()
    return float(x[:k].sum() / total) if total > 0 else 0.0


# --------------------------------------------------------------- analysis
def trait_effects(table: pd.DataFrame, traits: Sequence[str], target: str,
                  top: float = 0.1) -> pd.DataFrame:
    """How each trait relates to ``target``.

    spearman_rho / p_value : monotone association across everyone
    top_vs_all_sd          : (mean trait in the top ``top`` by target - mean trait overall) / sd
    """
    df = table.dropna(subset=[target])
    cut = df[target].quantile(1 - top)
    best = df[df[target] >= cut]
    rows = []
    for t in traits:
        rho, p = stats.spearmanr(df[t], df[target])
        sd = df[t].std() or 1.0
        rows.append({'trait': t, 'spearman_rho': rho, 'p_value': p,
                     'top_vs_all_sd': (best[t].mean() - df[t].mean()) / sd,
                     'mean_all': df[t].mean(), f'mean_top_{int(top * 100)}%': best[t].mean()})
    return pd.DataFrame(rows).set_index('trait').sort_values('spearman_rho', key=np.abs,
                                                             ascending=False)


def leaders(table: pd.DataFrame, by: str, k: int = 10, ascending: bool = False,
            columns: Sequence[str] | None = None) -> pd.DataFrame:
    out = table.sort_values(by, ascending=ascending).head(k)
    return out[list(columns)] if columns else out


# ------------------------------------------------------------------ result
@dataclass
class Result:
    """Common container: time series, ledger, per-agent table, animation frames."""
    model: str
    counts: pd.DataFrame
    ledger: pd.DataFrame
    agents: pd.DataFrame
    frames: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)
    runtime: float = 0.0

    def leaders(self, by: str, k: int = 10, ascending: bool = False, **filters) -> pd.DataFrame:
        df = self.agents
        for col, val in filters.items():
            df = df[df[col] == val]
        return leaders(df, by, k, ascending)

    def __repr__(self) -> str:
        t = self.counts.index
        return (f'{type(self).__name__}({self.model}: t={t[0]:g}->{t[-1]:g}, '
                f'{len(self.agents)} agents recorded, runtime {self.runtime:.1f}s)')


class Timer:
    def __enter__(self):
        self.t0 = _time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = _time.perf_counter() - self.t0


def as_trait_list(traits) -> list[Trait]:
    if traits is None:
        return []
    if isinstance(traits, Mapping):
        return [t if isinstance(t, Trait) else Trait(name, *t) for name, t in traits.items()]
    return list(traits)
