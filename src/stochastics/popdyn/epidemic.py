"""Agent-based epidemic (SIR / SEIR / SIRS) on a 2D torus, with an infection tree.

People wander (Brownian motion scaled by their ``mobility`` trait). Whenever a
susceptible person is within ``radius`` of an infectious one, infection
happens with probability 1 - exp(-beta * susceptibility * infectiousness * dt).
Infectious people recover at rate gamma (or die with probability ``fatality``);
with ``sigma`` set there is an exposed (incubating) stage first; with ``omega``
immunity wanes.

Every infection records who infected whom, so the run yields the offspring
distribution: how many people each case infected. ``infectiousness`` is drawn
from a skewed gamma distribution by default, which produces superspreaders:
a minority of cases causes most infections (dispersion k < 1).

``mean_field()`` solves the matching well-mixed ODE. The gap between the two
curves is the effect of space: people mostly meet neighbours who are already
infected or immune, which slows the epidemic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

from .. import _viz as V
from .core import Result, Timer, Trait, sample_traits, trait_effects

__all__ = ['Epidemic', 'EpidemicResult']

S, E, I, R, D = range(5)
NAMES = ('S', 'E', 'I', 'R', 'D')
LABELS = {'S': 'susceptible', 'E': 'exposed', 'I': 'infectious', 'R': 'recovered', 'D': 'dead'}
COLORS = {'S': V.BLUE, 'E': V.YELLOW, 'I': V.RED, 'R': V.AQUA, 'D': V.MUTED}


def _default_traits():
    return (Trait('mobility', 'lognormal', 1.0, 0.6),
            Trait('susceptibility', 'lognormal', 1.0, 0.3),
            Trait('infectiousness', 'gamma', 1.0, 1.5))


@dataclass
class Epidemic:
    n: int = 3000
    radius: float = 0.012          # contact distance on the unit torus
    beta: float = 1.0              # infection hazard per contact per unit time
    gamma: float = 0.25            # recovery rate (mean infectious period 1/gamma)
    sigma: float | None = None     # incubation rate (SEIR); None = SIR
    omega: float = 0.0             # waning-immunity rate (SIRS)
    fatality: float = 0.0          # probability that leaving I means death
    mobility: float = 0.03         # typical displacement per sqrt(unit time)
    dt: float = 0.1
    initial_infected: int = 10
    vaccinated: float = 0.0        # fraction immune at the start
    traits: Sequence[Trait] = field(default_factory=_default_traits)
    seed: int | None = None

    @property
    def contacts(self) -> float:
        """Expected number of people within ``radius`` of someone."""
        return (self.n - 1) * np.pi * self.radius ** 2

    def run(self, duration: float = 60.0, frame_every: int | None = None) -> 'EpidemicResult':
        rng = np.random.default_rng(self.seed)
        n, dt = self.n, self.dt
        tr = sample_traits(self.traits, n, rng)
        mob = tr.get('mobility', np.ones(n))
        sus = tr.get('susceptibility', np.ones(n))
        inf = tr.get('infectiousness', np.ones(n))
        pos = rng.random((n, 2))
        state = np.full(n, S, dtype=np.int8)
        order = rng.permutation(n)
        state[order[:self.initial_infected]] = E if self.sigma else I
        state[order[self.initial_infected:self.initial_infected + int(self.vaccinated * n)]] = R
        infected_at = np.full(n, np.nan)
        infected_at[order[:self.initial_infected]] = 0.0
        infected_by = np.full(n, -1, dtype=np.int64)
        secondary = np.zeros(n, dtype=np.int64)
        times_infected = (state == (E if self.sigma else I)).astype(np.int64)
        steps = int(round(duration / dt))
        frame_every = frame_every or max(1, steps // 150)
        rows, frames = [], []
        p_inc = 1 - np.exp(-(self.sigma or 0.0) * dt)
        p_rec = 1 - np.exp(-self.gamma * dt)
        p_wane = 1 - np.exp(-self.omega * dt)

        def record(t, new):
            c = np.bincount(state, minlength=5)
            rows.append({'time': t, **{k: int(c[j]) for j, k in enumerate(NAMES)}, 'new_infections': new})

        record(0.0, 0)
        frames.append((0.0, pos.copy(), state.copy()))
        with Timer() as timer:
            for step in range(1, steps + 1):
                t = step * dt
                alive = state != D
                pos[alive] = (pos[alive] + self.mobility * mob[alive, None] * np.sqrt(dt)
                              * rng.standard_normal((alive.sum(), 2))) % 1.0
                new = 0
                infectious = state == I
                if infectious.any() and (state == S).any():
                    idx = np.flatnonzero(alive)
                    tree = cKDTree(pos[idx], boxsize=1.0)
                    pairs = tree.query_pairs(self.radius, output_type='ndarray')
                    if len(pairs):
                        a, b = idx[pairs[:, 0]], idx[pairs[:, 1]]
                        sa, sb = state[a], state[b]
                        m1 = (sa == S) & (sb == I)
                        m2 = (sa == I) & (sb == S)
                        s_ = np.concatenate([a[m1], b[m2]])
                        i_ = np.concatenate([b[m1], a[m2]])
                        hazard = self.beta * sus[s_] * inf[i_] * dt
                        hit = rng.random(len(s_)) < 1 - np.exp(-hazard)
                        s_, i_ = s_[hit], i_[hit]
                        if len(s_):
                            perm = rng.permutation(len(s_))
                            s_, i_ = s_[perm], i_[perm]
                            _, first = np.unique(s_, return_index=True)
                            s_, i_ = s_[first], i_[first]
                            state[s_] = E if self.sigma else I
                            infected_at[s_] = t
                            infected_by[s_] = i_
                            times_infected[s_] += 1
                            np.add.at(secondary, i_, 1)
                            new = len(s_)
                u = rng.random(n)
                if self.sigma:
                    incubating = (state == E) & (infected_at < t)
                    state[incubating & (u < p_inc)] = I
                    u = rng.random(n)
                leaving = (state == I) & (u < p_rec) & (infected_at < t)
                dies = leaving & (rng.random(n) < self.fatality)
                state[leaving] = R
                state[dies] = D
                if self.omega:
                    state[(state == R) & (rng.random(n) < p_wane) & ~dies] = S
                record(t, new)
                if step % frame_every == 0 or step == steps:
                    frames.append((t, pos.copy(), state.copy()))
                if not ((state == I) | (state == E)).any() and not self.omega:
                    break
        counts = pd.DataFrame(rows).set_index('time')
        people = pd.DataFrame({'id': np.arange(n), **tr, 'infected_at': infected_at,
                               'infected_by': infected_by, 'secondary': secondary,
                               'times_infected': times_infected,
                               'final_state': [NAMES[s] for s in state],
                               'x': pos[:, 0], 'y': pos[:, 1]}).set_index('id')
        return EpidemicResult('epidemic', counts, pd.DataFrame(index=counts.index), people, frames,
                              params={'model': self}, runtime=timer.seconds)

    def mean_field(self, duration: float = 60.0, points: int = 600) -> pd.DataFrame:
        """Well-mixed ODE with the same parameters (fractions of the population)."""
        rng = np.random.default_rng(0)
        means = {t.name: float(np.mean(t.sample(200_000, rng))) for t in self.traits}
        b = self.beta * means.get('susceptibility', 1.0) * means.get('infectiousness', 1.0) * self.contacts
        f0 = self.initial_infected / self.n
        r0 = self.vaccinated
        y0 = [1 - f0 - r0, f0 if self.sigma else 0.0, 0.0 if self.sigma else f0, r0, 0.0]
        sig, g, w, fat = self.sigma or 0.0, self.gamma, self.omega, self.fatality

        def rhs(_, y):
            s, e, i, r, d = y
            infect = b * s * i
            if self.sigma:
                return [-infect + w * r, infect - sig * e, sig * e - g * i, g * (1 - fat) * i - w * r,
                        g * fat * i]
            return [-infect + w * r, 0.0, infect - g * i, g * (1 - fat) * i - w * r, g * fat * i]
        t = np.linspace(0, duration, points)
        sol = solve_ivp(rhs, (0, duration), y0, t_eval=t, rtol=1e-8, atol=1e-10)
        out = pd.DataFrame(sol.y.T * self.n, index=pd.Index(t, name='time'), columns=list(NAMES))
        out.attrs['R0'] = b / g
        return out


class EpidemicResult(Result):
    @property
    def model_params(self) -> Epidemic:
        return self.params['model']

    @property
    def people(self) -> pd.DataFrame:
        return self.agents

    def offspring(self) -> pd.Series:
        """Secondary infections of every case whose infectious period is over."""
        p = self.agents
        done = p.infected_at.notna() & p.final_state.isin(['R', 'D', 'S'])
        return p.loc[done, 'secondary']

    def summary(self) -> pd.DataFrame:
        c = self.counts
        m = self.model_params
        off = self.offspring()
        mean, var = off.mean(), off.var()
        k = mean ** 2 / (var - mean) if var > mean else np.inf
        top = np.sort(off.to_numpy())[::-1]
        share20 = top[:max(1, len(top) // 5)].sum() / max(top.sum(), 1)
        ever = int(self.agents.infected_at.notna().sum())
        cases = self.agents[self.agents.infected_at.notna()].sort_values('infected_at')
        early = cases.head(max(10, len(cases) // 10))
        rows = [
            ('people', m.n), ('mean contacts within radius', round(m.contacts, 2)),
            ('R0 (well-mixed estimate)', round(m.mean_field(1, 2).attrs['R0'], 2)),
            ('early R (first 10% of cases, simulated)', round(early.secondary.mean(), 2)),
            ('ever infected', f'{ever} ({ever / m.n:.0%})'),
            ('peak infectious', f"{int(c['I'].max())} at t = {c['I'].idxmax():.1f}"),
            ('deaths', int(c['D'].iloc[-1])),
            ('dispersion k (smaller = more superspreading)', round(k, 2) if np.isfinite(k) else 'inf'),
            ('share of infections caused by top 20% of cases', f'{share20:.0%}'),
        ]
        return pd.DataFrame(rows, columns=['metric', 'value']).set_index('metric')

    def r_effective(self, bin_width: float = 2.0) -> pd.Series:
        """Mean secondary cases of people infected in each time window (cohort R)."""
        p = self.agents[self.agents.infected_at.notna()]
        bins = np.arange(0, p.infected_at.max() + bin_width, bin_width)
        cohort = pd.cut(p.infected_at, bins, include_lowest=True)
        out = p.groupby(cohort, observed=True)['secondary'].mean()
        out.index = [iv.left for iv in out.index]
        return out

    def trait_effects(self, target: str = 'secondary'):
        p = self.agents[self.agents.infected_at.notna()]
        traits = [t.name for t in self.model_params.traits]
        return trait_effects(p, traits, target)

    def plot(self, show: bool = True):
        return plot_epidemic(self, show)

    def animate(self, max_frames: int = 100, interval: int = 80, save: str | None = None):
        return animate_epidemic(self, max_frames, interval, save)


def _active_states(model: Epidemic):
    return [s for s in NAMES if (s != 'E' or model.sigma) and (s != 'D' or model.fatality)]


def plot_epidemic(res: EpidemicResult, show: bool = True):
    c = res.counts
    m = res.model_params
    ode = m.mean_field(c.index[-1])
    states = _active_states(m)
    with V.themed():
        fig = plt.figure(figsize=(17, 9.5))
        gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.25)

        ax = fig.add_subplot(gs[0, 0])
        for s in states:
            ax.plot(c.index, c[s], color=COLORS[s], linewidth=2.0, label=LABELS[s])
            ax.plot(ode.index, ode[s], color=COLORS[s], linewidth=1.0, alpha=0.6)
        ax.legend(loc='center right')
        ax.set(title='Compartments: agents (thick) vs well-mixed ODE (thin)', xlabel='time',
               ylabel='people')

        ax = fig.add_subplot(gs[0, 1])
        inc = c['new_infections'].rolling(max(1, int(1 / m.dt)), min_periods=1).sum()
        ax.fill_between(c.index, inc, color=COLORS['I'], alpha=0.15, linewidth=0)
        ax.plot(c.index, inc, color=COLORS['I'])
        ax.set(title='New infections (per unit time)', xlabel='time', ylabel='cases')

        ax = fig.add_subplot(gs[0, 2])
        reff = res.r_effective()
        ax.axhline(1, color=V.MUTED, linewidth=0.8)
        ax.plot(reff.index, reff.values, color=V.VIOLET, marker='o', markersize=4)
        ax.set(title='Effective R (by infection cohort)', xlabel='time infected',
               ylabel='mean secondary cases')

        ax = fig.add_subplot(gs[1, 0])
        off = res.offspring()
        if len(off):
            bins = np.arange(0, off.max() + 2) - 0.5
            ax.hist(off, bins=bins, color=V.VIOLET, alpha=0.85)
            ax.set_yscale('log')
        summ = res.summary()['value']
        ax.set(title='Offspring distribution (superspreading)', xlabel='people infected by one case',
               ylabel='cases (log)')
        ax.text(0.97, 0.95, f"k = {summ['dispersion k (smaller = more superspreading)']}\n"
                            f"top 20% caused {summ['share of infections caused by top 20% of cases']}",
                transform=ax.transAxes, ha='right', va='top', color=V.TEXT_2, fontsize=9)

        ax = fig.add_subplot(gs[1, 1])
        eff = res.trait_effects()
        ax.barh(eff.index, eff['spearman_rho'], color=V.VIOLET, height=0.5)
        ax.axvline(0, color=V.MUTED, linewidth=0.8)
        ax.grid(axis='y', visible=False)
        ax.set(title='Traits of people who infect others', xlabel='Spearman ρ with secondary cases')

        ax = fig.add_subplot(gs[1, 2])
        k_peak = int(np.argmin([abs(f[0] - c['I'].idxmax()) for f in res.frames]))
        _draw_people(ax, res.frames[k_peak], states)
        ax.set_title(f'Positions at the peak (t = {res.frames[k_peak][0]:.1f})')

        V.suptitle(fig, f"Epidemic · {m.n} people · R0≈{summ['R0 (well-mixed estimate)']} · "
                        f"{summ['ever infected']} ever infected · peak {summ['peak infectious']}")
        fig.subplots_adjust(top=0.91, left=0.05, right=0.98, bottom=0.06)
    return V.finish(fig, show)


def _draw_people(ax, frame, states):
    t, pos, st = frame
    for s in states:
        m = st == NAMES.index(s)
        ax.scatter(pos[m, 0], pos[m, 1], s=6 if s != 'I' else 10, color=COLORS[s], linewidths=0,
                   label=LABELS[s])
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[])
    ax.set_aspect('equal')
    ax.grid(False)
    ax.legend(loc='upper right', fontsize=7, markerscale=1.5, frameon=True, facecolor=V.SURFACE,
              edgecolor=V.GRID)


def animate_epidemic(res: EpidemicResult, max_frames=100, interval=80, save=None):
    frames = res.frames[::max(1, int(np.ceil(len(res.frames) / max_frames)))]
    c = res.counts
    states = _active_states(res.model_params)
    with V.themed():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=70,
                                      gridspec_kw={'width_ratios': [1, 1.2]})
        for s in states:
            ax2.plot(c.index, c[s], color=COLORS[s], label=LABELS[s])
        ax2.legend(loc='center right')
        ax2.set(xlabel='time', ylabel='people', title='Compartments')
        cursor = ax2.axvline(0, color=V.TEXT, linewidth=0.8)

        def update(k):
            ax.clear()
            _draw_people(ax, frames[k], states)
            ax.set_title(f't = {frames[k][0]:.1f}')
            cursor.set_xdata([frames[k][0]] * 2)
            return ()

        fig.tight_layout()
        ani = FuncAnimation(fig, update, frames=len(frames), interval=interval)
        return V.to_html_animation(fig, ani, save)
