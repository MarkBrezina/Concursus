"""Rock-paper-scissors (May-Leonard) on a lattice: cyclic dominance and spirals.

Three species A, B, C live on an L x L grid (with empty sites). A random site
interacts with a random neighbour:

* selection    (rate ``selection``):    A kills B, B kills C, C kills A -> the loser's site empties
* reproduction (rate ``reproduction``): an individual next to an empty site fills it
* exchange     (rate ``exchange``):     neighbours swap places (mobility)

No species is best, so in a well-mixed world the cycle drifts until two
species go extinct. On a lattice with low mobility the cycle organises into
rotating spiral waves that let all three coexist; above a critical mobility
(Reichenbach, Mobilia & Frey 2007: M = 2 * exchange / L^2 ~ 4.5e-4) the
spirals outgrow the box and biodiversity collapses.

Time is measured in Monte Carlo sweeps (one sweep = L^2 interactions).
Interactions are applied in conflict-free batches for speed.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from .. import _viz as V
from .core import Result, Timer

__all__ = ['RockPaperScissors', 'CyclicResult']

NEIGHBOURS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
SPECIES_COLORS = (V.BLUE, V.ORANGE, V.AQUA)
EMPTY_COLOR = '#ecebe7'


@dataclass
class RockPaperScissors:
    size: int = 200
    selection: float = 1.0
    reproduction: float = 1.0
    mobility: float = 1e-4            # M = 2 * exchange / L^2 (Reichenbach et al.)
    empty: float = 0.1                # initial empty fraction
    names: tuple = ('rock', 'scissors', 'paper')   # each beats the next, the last beats the first
    seed: int | None = None

    @property
    def exchange(self) -> float:
        return self.mobility * self.size ** 2 / 2

    def run(self, sweeps: int = 600, frame_every: int | None = None) -> 'CyclicResult':
        rng = np.random.default_rng(self.seed)
        L = self.size
        N = L * L
        grid = rng.integers(1, 4, N).astype(np.int8)
        grid[rng.random(N) < self.empty] = 0
        rates = np.array([self.selection, self.reproduction, self.exchange], float)
        p_react = rates / rates.sum()
        batch = max(64, N // 20)
        frame_every = frame_every or max(1, sweeps // 150)
        rows = []
        frames = []

        def record(t):
            c = np.bincount(grid, minlength=4)
            rows.append({'sweep': t, 'empty': c[0] / N,
                         **{name: c[k + 1] / N for k, name in enumerate(self.names)}})

        record(0)
        frames.append((0, grid.reshape(L, L).copy()))
        with Timer() as timer:
            for sweep in range(1, sweeps + 1):
                done = 0
                while done < N:
                    k = min(batch, N - done)
                    site = rng.integers(0, N, k)
                    d = NEIGHBOURS[rng.integers(0, 4, k)]
                    x, y = site % L, site // L
                    nb = ((y + d[:, 1]) % L) * L + (x + d[:, 0]) % L
                    # keep only pairs whose two cells are not touched earlier in the batch
                    first = np.full(N, k, dtype=np.int64)
                    order = np.arange(k)
                    np.minimum.at(first, site, order)
                    np.minimum.at(first, nb, order)
                    ok = (first[site] == order) & (first[nb] == order) & (site != nb)
                    site, nb = site[ok], nb[ok]
                    kind = rng.choice(3, size=len(site), p=p_react)
                    a, b = grid[site], grid[nb]
                    # selection: a beats b if b == a % 3 + 1 (1 beats 2, 2 beats 3, 3 beats 1)
                    sel = kind == 0
                    a_wins = sel & (a > 0) & (b == a % 3 + 1)
                    b_wins = sel & (b > 0) & (a == b % 3 + 1)
                    grid[nb[a_wins]] = 0
                    grid[site[b_wins]] = 0
                    rep = kind == 1
                    fill_b = rep & (a > 0) & (b == 0)
                    fill_a = rep & (b > 0) & (a == 0)
                    grid[nb[fill_b]] = a[fill_b]
                    grid[site[fill_a]] = b[fill_a]
                    ex = kind == 2
                    grid[site[ex]], grid[nb[ex]] = b[ex], a[ex]
                    done += k
                record(sweep)
                if sweep % frame_every == 0 or sweep == sweeps:
                    frames.append((sweep, grid.reshape(L, L).copy()))
        counts = pd.DataFrame(rows).set_index('sweep')
        return CyclicResult('rock-paper-scissors', counts, pd.DataFrame(index=counts.index),
                            pd.DataFrame(), frames, params={'model': self}, runtime=timer.seconds)


class CyclicResult(Result):
    @property
    def model_params(self) -> RockPaperScissors:
        return self.params['model']

    def summary(self) -> pd.DataFrame:
        m = self.model_params
        last = self.counts.iloc[-1]
        alive = [n for n in m.names if last[n] > 0]
        rows = [('lattice', f'{m.size}×{m.size}'), ('mobility M', m.mobility),
                ('critical mobility (theory)', 4.5e-4), ('sweeps', int(self.counts.index[-1])),
                ('species surviving', f"{len(alive)} ({', '.join(alive)})")]
        rows += [(f'final density: {n}', round(last[n], 3)) for n in m.names]
        return pd.DataFrame(rows, columns=['metric', 'value']).set_index('metric')

    def plot(self, show: bool = True):
        return plot_cyclic(self, show)

    def animate(self, max_frames: int = 100, interval: int = 80, save: str | None = None):
        return animate_cyclic(self, max_frames, interval, save)


_CMAP = ListedColormap([EMPTY_COLOR, *SPECIES_COLORS])


def _ternary(ax, fr: pd.DataFrame, names):
    """Trajectory of the three shares inside the simplex."""
    tot = fr[list(names)].sum(axis=1).replace(0, np.nan)
    a, b, c = (fr[n] / tot for n in names)
    x = b + 0.5 * c
    y = np.sqrt(3) / 2 * c
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=V.GRID, linewidth=1)
    ax.plot(x, y, color=V.VIOLET, linewidth=1.0)
    ax.plot(x.iloc[-1:], y.iloc[-1:], 'o', color=V.VIOLET, markersize=6)
    for (px, py), name, ha in (((0, 0), names[0], 'right'), ((1, 0), names[1], 'left'),
                               ((0.5, np.sqrt(3) / 2), names[2], 'center')):
        ax.annotate(name, (px, py), xytext=(0, -12 if py == 0 else 6), textcoords='offset points',
                    ha=ha, color=V.TEXT_2, fontsize=8.5)
    ax.plot([1 / 2], [np.sqrt(3) / 6], '+', color=V.MUTED)
    ax.set(xlim=(-0.1, 1.1), ylim=(-0.12, 0.95), xticks=[], yticks=[])
    ax.set_aspect('equal')
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_cyclic(res: CyclicResult, show: bool = True):
    m = res.model_params
    c = res.counts
    picks = [res.frames[int(i)] for i in np.linspace(0, len(res.frames) - 1, 4)]
    with V.themed():
        fig = plt.figure(figsize=(17, 8.5))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.18)
        ax = fig.add_subplot(gs[0, :2])
        for k, name in enumerate(m.names):
            ax.plot(c.index, c[name], color=SPECIES_COLORS[k], label=name)
        ax.plot(c.index, c['empty'], color=V.MUTED, linewidth=1.0, label='empty')
        ax.legend(loc='upper right', ncols=4)
        ax.set(title='Densities', xlabel='Monte Carlo sweep', ylabel='fraction of sites')
        ax = fig.add_subplot(gs[0, 2])
        _ternary(ax, c, m.names)
        ax.set_title('Shares on the simplex (centre = balance)')
        ax = fig.add_subplot(gs[0, 3])
        ax.axis('off')
        txt = res.summary()['value']
        ax.text(0, 0.95, '\n'.join(f'{k}: {v}' for k, v in txt.items()), va='top', fontsize=9.5,
                color=V.TEXT_2, family='monospace', transform=ax.transAxes)
        for j, (t, g) in enumerate(picks):
            ax = fig.add_subplot(gs[1, j])
            ax.imshow(g, cmap=_CMAP, vmin=0, vmax=3, interpolation='nearest')
            ax.set(xticks=[], yticks=[], title=f'sweep {t}')
            ax.grid(False)
        V.suptitle(fig, f'Rock-paper-scissors · {m.size}×{m.size} lattice · mobility M = {m.mobility:g}')
        fig.subplots_adjust(top=0.9, left=0.04, right=0.98, bottom=0.04)
    return V.finish(fig, show)


def animate_cyclic(res: CyclicResult, max_frames=100, interval=80, save=None):
    frames = res.frames[::max(1, int(np.ceil(len(res.frames) / max_frames)))]
    with V.themed():
        fig, ax = plt.subplots(figsize=(5.2, 5.4), dpi=70)
        im = ax.imshow(frames[0][1], cmap=_CMAP, vmin=0, vmax=3, interpolation='nearest')
        ax.set(xticks=[], yticks=[])
        ax.grid(False)
        title = ax.set_title('')

        def update(k):
            im.set_data(frames[k][1])
            title.set_text(f'sweep {frames[k][0]}')
            return (im,)

        fig.tight_layout()
        ani = FuncAnimation(fig, update, frames=len(frames), interval=interval)
        return V.to_html_animation(fig, ani, save)
