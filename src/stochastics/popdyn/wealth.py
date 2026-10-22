"""Wealth economy: a population with traits, pluggable rules, and a wealth ledger.

    econ = Economy(n=10_000, rules=[LuckEvents(), Investment(), YardSale(), Tax(0.01)])
    res = econ.run(steps=80)
    res.plot()                   # individual paths, shares, Gini, Lorenz, tails, traits
    res.leaders(10)              # the richest people and their traits
    res.trait_effects()          # which traits predict final wealth?

Each person has traits (talent, intelligence, creativity, charisma, luck, risk,
thrift by default; any ``Trait`` list works). Each step every rule changes
wealth and reports its flows:

    created / destroyed   wealth that appears or disappears (returns, losses, spending)
    transferred           wealth that changes hands (gross volume; conserves the total)
    redistributed         taxes collected and paid back out (conserves the total)

so ``total(t) - total(0) = created - destroyed`` holds exactly (the ledger
residual is reported). Rules:

LuckEvents      Talent vs Luck (Pluchino, Biondo & Rapisarda 2018): random lucky /
                unlucky events; talent decides whether a lucky break is used.
Investment      multiplicative returns (a GBM per person): skill shifts the drift,
                risk appetite scales the volatility.
Wages           additive income shaped by traits.
Consumption     spending: a subsistence amount plus a share of wealth.
YardSale        random pairs bet a share of the poorer one's wealth (Boghosian);
                left alone it concentrates wealth without bound.
KineticExchange random pairs pool and re-split their non-saved wealth
                (Chakraborti-Chakrabarti); thrift sets the saved share.
Tax             wealth tax above a threshold, redistributed equally.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import _viz as V
from .core import Result, Timer, Trait, gini, lorenz, sample_traits, top_share, trait_effects

__all__ = ['Economy', 'EconomyResult', 'LuckEvents', 'Investment', 'Wages', 'Consumption',
           'YardSale', 'KineticExchange', 'Tax', 'Market', 'DEFAULT_TRAITS']

DEFAULT_TRAITS = (
    Trait('talent', 'normal', 0.6, 0.1, clip=(0.0, 1.0)),
    Trait('intelligence', 'normal', 100.0, 15.0, clip=(40.0, 160.0)),
    Trait('creativity', 'beta', 2.0, 2.0),
    Trait('charisma', 'normal', 0.0, 1.0),
    Trait('luck', 'beta', 5.0, 5.0),
    Trait('risk', 'lognormal', 1.0, 0.5),
    Trait('thrift', 'beta', 2.0, 5.0),
)


@dataclass
class Market:
    """What a rule sees and changes."""
    wealth: np.ndarray
    traits: dict
    z: dict                     # standardised traits (mean 0, sd 1)
    rng: np.random.Generator
    dt: float
    step: int = 0
    counters: dict = field(default_factory=dict)

    def counter(self, name: str) -> np.ndarray:
        if name not in self.counters:
            self.counters[name] = np.zeros(len(self.wealth), dtype=np.int64)
        return self.counters[name]

    def trait(self, name: str, default: float = 0.0) -> np.ndarray:
        return self.traits.get(name, np.full(len(self.wealth), default))

    def score(self, weights: dict | None) -> np.ndarray:
        """Weighted sum of standardised traits."""
        out = np.zeros(len(self.wealth))
        for k, w in (weights or {}).items():
            if k in self.z:
                out += w * self.z[k]
        return out


def _flows(created=0.0, destroyed=0.0, transferred=0.0, redistributed=0.0):
    return dict(created=float(created), destroyed=float(destroyed),
                transferred=float(transferred), redistributed=float(redistributed))


def _change(m: Market, new: np.ndarray):
    d = new - m.wealth
    m.wealth[:] = new
    return _flows(created=d[d > 0].sum(), destroyed=-d[d < 0].sum())


@dataclass
class LuckEvents:
    """Each step a person meets an event with prob. ``rate``; it is lucky with
    prob. = their ``luck`` trait. A lucky break multiplies wealth by ``gain`` only if
    they seize it (prob. = ``talent``); an unlucky one multiplies it by ``loss``."""
    rate: float = 0.1
    gain: float = 2.0
    loss: float = 0.5
    talent: str = 'talent'
    luck: str = 'luck'

    def apply(self, m: Market):
        n = len(m.wealth)
        event = m.rng.random(n) < self.rate
        lucky = event & (m.rng.random(n) < np.clip(m.trait(self.luck, 0.5), 0, 1))
        unlucky = event & ~lucky
        seized = lucky & (m.rng.random(n) < np.clip(m.trait(self.talent, 0.5), 0, 1))
        m.counter('lucky_events')[lucky] += 1
        m.counter('unlucky_events')[unlucky] += 1
        new = m.wealth.copy()
        new[seized] *= self.gain
        new[unlucky] *= self.loss
        return _change(m, new)


@dataclass
class Investment:
    """Invest ``share`` of wealth: w_inv *= exp((mu_i - s_i^2/2) dt + s_i sqrt(dt) Z),
    mu_i = mu + skill . z(traits), s_i = sigma * risk_i."""
    mu: float = 0.03
    sigma: float = 0.2
    share: float = 0.5
    skill: dict = field(default_factory=lambda: {'intelligence': 0.01, 'creativity': 0.01})
    risk: str = 'risk'

    def apply(self, m: Market):
        mu = self.mu + m.score(self.skill)
        s = self.sigma * m.trait(self.risk, 1.0)
        growth = np.exp((mu - 0.5 * s * s) * m.dt + s * np.sqrt(m.dt) * m.rng.standard_normal(len(mu)))
        new = m.wealth * (1 - self.share) + m.wealth * self.share * growth
        return _change(m, new)


@dataclass
class Wages:
    """Income each step: base * exp(weights . z(traits))."""
    base: float = 1.0
    weights: dict = field(default_factory=lambda: {'intelligence': 0.2, 'creativity': 0.1,
                                                   'charisma': 0.1})

    def apply(self, m: Market):
        return _change(m, m.wealth + self.base * m.dt * np.exp(m.score(self.weights)))


@dataclass
class Consumption:
    """Spend ``subsistence`` plus ``rate`` of wealth each step (never below zero)."""
    rate: float = 0.03
    subsistence: float = 0.5

    def apply(self, m: Market):
        spend = np.minimum(m.wealth, (self.subsistence + self.rate * m.wealth) * m.dt)
        return _change(m, m.wealth - spend)


def _pairs(m: Market, fraction: float):
    n = len(m.wealth)
    k = int(n * fraction) // 2 * 2
    idx = m.rng.permutation(n)[:k]
    return idx[0::2], idx[1::2]


@dataclass
class YardSale:
    """Random pairs bet ``stake`` x the poorer one's wealth. Win probability is
    0.5 + ``edge`` * (z_i - z_j) of the chosen trait (edge 0 = a fair coin)."""
    stake: float = 0.1
    edge: float = 0.0
    trait: str = 'charisma'
    fraction: float = 1.0

    def apply(self, m: Market):
        a, b = _pairs(m, self.fraction)
        bet = self.stake * np.minimum(m.wealth[a], m.wealth[b])
        za = m.z.get(self.trait, np.zeros(len(m.wealth)))
        p = np.clip(0.5 + self.edge * (za[a] - za[b]), 0.05, 0.95)
        a_wins = m.rng.random(len(a)) < p
        delta = np.where(a_wins, bet, -bet)
        m.wealth[a] += delta
        m.wealth[b] -= delta
        return _flows(transferred=np.abs(delta).sum())


@dataclass
class KineticExchange:
    """Random pairs keep a saved share (``thrift`` trait) and randomly re-split the rest."""
    trait: str = 'thrift'
    fraction: float = 1.0

    def apply(self, m: Market):
        a, b = _pairs(m, self.fraction)
        lam = np.clip(m.trait(self.trait, 0.0), 0.0, 0.99)
        wa, wb = m.wealth[a], m.wealth[b]
        pool = (1 - lam[a]) * wa + (1 - lam[b]) * wb
        eps = m.rng.random(len(a))
        na = lam[a] * wa + eps * pool
        nb = lam[b] * wb + (1 - eps) * pool
        moved = np.abs(na - wa).sum()
        m.wealth[a], m.wealth[b] = na, nb
        return _flows(transferred=moved)


@dataclass
class Tax:
    """Wealth tax ``rate`` on wealth above ``threshold``, paid back equally to everyone."""
    rate: float = 0.01
    threshold: float = 0.0

    def apply(self, m: Market):
        tax = self.rate * np.maximum(m.wealth - self.threshold, 0.0) * m.dt
        total = tax.sum()
        m.wealth[:] = m.wealth - tax + total / len(m.wealth)
        return _flows(redistributed=total)


def _default_rules():
    return [LuckEvents(), Investment(), YardSale(stake=0.05)]


@dataclass
class Economy:
    n: int = 10_000
    traits: Sequence[Trait] = DEFAULT_TRAITS
    initial_wealth: Trait = Trait('wealth', 'constant', 10.0)
    rules: list = field(default_factory=_default_rules)
    dt: float = 0.5                  # e.g. half a year per step
    seed: int | None = None

    def run(self, steps: int = 80, record_every: int = 1) -> 'EconomyResult':
        rng = np.random.default_rng(self.seed)
        traits = sample_traits(self.traits, self.n, rng)
        z = {k: (v - v.mean()) / (v.std() or 1.0) for k, v in traits.items()}
        w0 = np.maximum(self.initial_wealth.sample(self.n, rng), 0.0)
        m = Market(w0.copy(), traits, z, rng, self.dt)
        cum = _flows()
        snaps, times, rows = [w0.copy()], [0.0], []
        peak = w0.copy()

        def record(t):
            w = m.wealth
            rows.append({'time': t, 'total': w.sum(), 'mean': w.mean(), 'median': np.median(w),
                         'max': w.max(), 'gini': gini(w), 'top 1%': top_share(w, 0.01),
                         'top 10%': top_share(w, 0.10), 'bottom 50%': 1 - top_share(w, 0.5),
                         'pareto alpha': _hill(w, 0.01), **cum,
                         'residual': w.sum() - w0.sum() - (cum['created'] - cum['destroyed'])})

        record(0.0)
        with Timer() as timer:
            for step in range(1, steps + 1):
                m.step = step
                for rule in self.rules:
                    for k, v in rule.apply(m).items():
                        cum[k] += v
                np.maximum(m.wealth, 0.0, out=m.wealth)
                np.maximum(peak, m.wealth, out=peak)
                if step % record_every == 0 or step == steps:
                    t = step * self.dt
                    snaps.append(m.wealth.copy())
                    times.append(t)
                    record(t)
        W = np.array(snaps)
        final = W[-1]
        order = np.argsort(-final)
        rank = np.empty(self.n, int)
        rank[order] = np.arange(1, self.n + 1)
        people = pd.DataFrame({'id': np.arange(self.n), **traits, 'initial': w0, 'final': final,
                               'peak': peak, 'rank': rank,
                               'percentile': 100.0 * (1 - (rank - 1) / self.n),
                               **{k: v for k, v in m.counters.items()}}).set_index('id')
        stats = pd.DataFrame(rows).set_index('time')
        ledger_cols = ['total', 'created', 'destroyed', 'transferred', 'redistributed', 'residual']
        return EconomyResult('economy', stats.drop(columns=ledger_cols[1:]), stats[ledger_cols],
                             people, [], params={'model': self},
                             extras={'wealth': W, 'times': np.array(times)},
                             runtime=timer.seconds)


def _hill(w: np.ndarray, q: float) -> float:
    """Hill estimate of the Pareto tail exponent from the top fraction q."""
    x = np.sort(w)[::-1]
    k = max(5, int(q * len(x)))
    if x[k] <= 0:
        return float('nan')
    spread = np.sum(np.log(x[:k] / x[k]))
    return float(k / spread) if spread > 0 else float('nan')


class EconomyResult(Result):
    @property
    def model_params(self) -> Economy:
        return self.params['model']

    @property
    def people(self) -> pd.DataFrame:
        return self.agents

    @property
    def trait_names(self) -> list:
        return [t.name for t in self.model_params.traits]

    def leaders(self, k: int = 10, by: str = 'final', ascending: bool = False) -> pd.DataFrame:
        cols = ['rank', 'final', 'initial', 'peak'] + self.trait_names + \
            [c for c in ('lucky_events', 'unlucky_events') if c in self.agents]
        return self.agents.sort_values(by, ascending=ascending).head(k)[cols]

    def trait_effects(self, target: str = 'final', top: float = 0.01) -> pd.DataFrame:
        cols = self.trait_names + [c for c in ('lucky_events', 'unlucky_events') if c in self.agents]
        return trait_effects(self.agents, cols, target, top)

    def summary(self) -> pd.DataFrame:
        s0, s1 = self.counts.iloc[0], self.counts.iloc[-1]
        led = self.ledger.iloc[-1]
        richest = self.agents.final.max()
        rows = [('people', len(self.agents)), ('rules', ', '.join(type(r).__name__ for r in
                                                                 self.model_params.rules)),
                ('years simulated', f'{self.counts.index[-1]:g}'),
                ('total wealth', f"{s0['total'] if 'total' in s0 else led['total']:.4g} -> {led['total']:.4g}"),
                ('Gini', f"{s0['gini']:.3f} -> {s1['gini']:.3f}"),
                ('top 1% share', f"{s0['top 1%']:.1%} -> {s1['top 1%']:.1%}"),
                ('bottom 50% share', f"{s0['bottom 50%']:.1%} -> {s1['bottom 50%']:.1%}"),
                ('richest person / median', f"{richest / max(s1['median'], 1e-12):.3g}x"),
                ('Pareto tail alpha (top 1%)', f"{s1['pareto alpha']:.2f}"),
                ('ledger residual', f"{led['residual']:+.2e}")]
        return pd.DataFrame(rows, columns=['metric', 'value']).set_index('metric')

    def plot(self, show: bool = True, highlight: int = 5):
        return plot_economy(self, show, highlight)


# ------------------------------------------------------------------ figure
def plot_economy(res: EconomyResult, show: bool = True, highlight: int = 5):
    W, times = res.extras['wealth'], res.extras['times']
    p = res.agents
    st = res.counts
    led = res.ledger
    rng = np.random.default_rng(0)
    with V.themed():
        fig = plt.figure(figsize=(18, 13))
        gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        sample = rng.choice(W.shape[1], size=min(250, W.shape[1]), replace=False)
        floor = max(np.percentile(W[W > 0], 0.5), 1e-6)
        for i in sample:
            ax.plot(times, np.maximum(W[:, i], floor), color=V.MUTED, alpha=0.18, linewidth=0.6)
        top = p.sort_values('final', ascending=False).index[:highlight]
        for k, i in enumerate(top):
            ax.plot(times, np.maximum(W[:, i], floor), color=V.SERIES[k], linewidth=1.8,
                    label=f'#{i} (rank {k + 1})')
        ax.plot(times, np.median(W, axis=1), color=V.TEXT, linewidth=2.0, label='median')
        ax.set_yscale('log')
        ax.legend(loc='upper left', fontsize=7)
        ax.set(title='Individual wealth (grey: random sample)', xlabel='years', ylabel='wealth')

        ax = fig.add_subplot(gs[0, 1])
        for k, col in enumerate(('top 1%', 'top 10%', 'bottom 50%')):
            ax.plot(st.index, 100 * st[col], color=V.SERIES[k], label=col)
        ax.legend(loc='upper left')
        ax.set(title='Share of total wealth', xlabel='years', ylabel='% of all wealth')

        ax = fig.add_subplot(gs[0, 2])
        ax.plot(st.index, st['gini'], color=V.VIOLET)
        ax.set(title='Gini coefficient', xlabel='years', ylim=(0, 1))

        ax = fig.add_subplot(gs[1, 0])
        ax.plot([0, 1], [0, 1], color=V.MUTED, linewidth=0.8)
        for k, (row, label) in enumerate(((0, 'start'), (-1, 'end'))):
            xs, ys = lorenz(W[row])
            ax.plot(xs, ys, color=V.SERIES[k], label=f'{label} (Gini {gini(W[row]):.2f})')
        ax.legend(loc='upper left')
        ax.set(title='Lorenz curve', xlabel='share of people (poorest first)',
               ylabel='share of wealth')
        ax.set_aspect('equal')

        ax = fig.add_subplot(gs[1, 1])
        final = np.sort(W[-1][W[-1] > 0])[::-1]
        ccdf = np.arange(1, len(final) + 1) / len(W[-1])
        ax.loglog(final, ccdf, color=V.BLUE, linewidth=1.6, label='final wealth')
        alpha = st['pareto alpha'].iloc[-1]
        if np.isfinite(alpha):
            k = max(5, int(0.01 * len(W[-1])))
            xk = final[k]
            xs = np.geomspace(xk, final[0], 50)
            ax.loglog(xs, (k / len(W[-1])) * (xs / xk) ** (-alpha), color=V.ORANGE, linewidth=1.4,
                      label=f'Pareto tail α = {alpha:.2f}')
        ax.legend(loc='lower left')
        ax.set(title='Tail: P(wealth > x)', xlabel='x', ylabel='fraction of people')

        ax = fig.add_subplot(gs[1, 2])
        eff = res.trait_effects()
        ax.barh(eff.index[::-1], eff['spearman_rho'][::-1], color=V.BLUE, height=0.55)
        ax.axvline(0, color=V.MUTED, linewidth=0.8)
        ax.grid(axis='y', visible=False)
        ax.set(title='Which traits predict final wealth?', xlabel='Spearman ρ')

        ax = fig.add_subplot(gs[2, 0])
        ax.barh(eff.index[::-1], eff['top_vs_all_sd'][::-1], color=V.VIOLET, height=0.55)
        ax.axvline(0, color=V.MUTED, linewidth=0.8)
        ax.grid(axis='y', visible=False)
        ax.set(title='The top 1% vs everyone', xlabel='difference in trait (sd units)')

        ax = fig.add_subplot(gs[2, 1])
        xt = 'talent' if 'talent' in p else res.trait_names[0]
        color_by = 'lucky_events' if 'lucky_events' in p else res.trait_names[-1]
        pts = ax.scatter(p[xt], np.maximum(p['final'], floor), c=p[color_by], cmap=V.SEQUENTIAL,
                         s=4, linewidths=0, alpha=0.7)
        ax.set_yscale('log')
        cb = fig.colorbar(pts, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label(color_by.replace('_', ' '))
        cb.outline.set_visible(False)
        ax.set(title=f'{xt.capitalize()} vs final wealth', xlabel=xt, ylabel='final wealth')

        ax = fig.add_subplot(gs[2, 2])
        for k, col in enumerate(('created', 'destroyed', 'transferred', 'redistributed')):
            if led[col].abs().max() > 0:
                ax.plot(led.index, led[col], color=V.SERIES[k + 2], label=col)
        ax.plot(led.index, led['total'], color=V.TEXT, linewidth=2.0, label='total wealth')
        ax.set_yscale('symlog', linthresh=max(1.0, led['total'].iloc[0] / 100))
        ax.legend(loc='upper left', fontsize=7)
        ax.set(title=f"Wealth ledger (cumulative) · residual {led['residual'].iloc[-1]:+.1e}",
               xlabel='years', ylabel='wealth')

        s = res.summary()['value']
        V.suptitle(fig, f"Economy · {len(p):,} people · {s['rules']} · Gini {s['Gini']} · "
                        f"top 1% share {s['top 1% share']}")
        fig.subplots_adjust(top=0.93, left=0.05, right=0.98, bottom=0.05)
    return V.finish(fig, show)
