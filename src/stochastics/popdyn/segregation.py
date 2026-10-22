"""Schelling segregation: mild preferences, strong segregation.

Two (or more) groups live on a grid with some empty cells. A person is
content if at least ``tolerance`` of their occupied neighbours (8-cell
neighbourhood) belong to their own group; discontent people move to a random
empty cell. Even with tolerance 0.3 (happy as a 30% minority) the town sorts
itself into large homogeneous blocks: macro outcomes need not reflect micro
intentions.
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

__all__ = ['Schelling', 'SegregationResult']

_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


@dataclass
class Schelling:
    size: int = 80
    empty: float = 0.1
    shares: tuple = (0.5, 0.5)       # population shares of the groups
    tolerance: float = 0.35          # minimum share of like neighbours to stay
    seed: int | None = None

    def _neighbours(self, grid):
        G = len(self.shares)
        like = np.zeros(grid.shape)
        occ = np.zeros(grid.shape)
        for g in range(1, G + 1):
            mask = (grid == g).astype(float)
            n = sum(np.roll(np.roll(mask, dy, 0), dx, 1) for dy, dx in _OFFSETS)
            like += np.where(grid == g, n, 0.0)
            occ += n
        return like, occ

    def run(self, steps: int = 60, frame_every: int = 1) -> 'SegregationResult':
        rng = np.random.default_rng(self.seed)
        L = self.size
        shares = np.asarray(self.shares, float) / np.sum(self.shares)
        probs = np.concatenate([[self.empty], (1 - self.empty) * shares])
        grid = rng.choice(len(probs), size=(L, L), p=probs).astype(np.int8)
        rows, frames = [], [(0, grid.copy())]

        def measure(t):
            like, occ = self._neighbours(grid)
            people = grid > 0
            frac = np.where(occ > 0, like / np.maximum(occ, 1), 1.0)
            unhappy = people & (frac < self.tolerance)
            rows.append({'step': t, 'similarity': float(frac[people].mean()),
                         'unhappy': float(unhappy.sum() / people.sum())})
            return unhappy

        with Timer() as timer:
            unhappy = measure(0)
            for t in range(1, steps + 1):
                movers = np.flatnonzero(unhappy.ravel())
                empties = np.flatnonzero((grid == 0).ravel())
                if not len(movers) or not len(empties):
                    measure(t)
                    frames.append((t, grid.copy()))
                    break
                movers = rng.permutation(movers)[:len(empties)]
                targets = rng.choice(empties, size=len(movers), replace=False)
                flat = grid.ravel()
                flat[targets] = flat[movers]
                flat[movers] = 0
                unhappy = measure(t)
                if t % frame_every == 0:
                    frames.append((t, grid.copy()))
        counts = pd.DataFrame(rows).set_index('step')
        return SegregationResult('schelling', counts, pd.DataFrame(index=counts.index), pd.DataFrame(),
                                 frames, params={'model': self}, runtime=timer.seconds)


class SegregationResult(Result):
    @property
    def model_params(self) -> Schelling:
        return self.params['model']

    def plot(self, show: bool = True):
        m = self.model_params
        c = self.counts
        cmap = ListedColormap(['#ecebe7', *V.SERIES[:len(m.shares)]])
        with V.themed():
            fig, axs = plt.subplots(1, 3, figsize=(15, 4.8),
                                    gridspec_kw={'width_ratios': [1.3, 1, 1]})
            axs[0].plot(c.index, 100 * c['similarity'], color=V.BLUE, label='like neighbours (%)')
            axs[0].plot(c.index, 100 * c['unhappy'], color=V.ORANGE, label='unhappy (%)')
            axs[0].axhline(100 * m.tolerance, color=V.MUTED, linewidth=0.8)
            axs[0].legend(loc='center right')
            axs[0].set(title=f'Tolerance {m.tolerance:.0%}: everyone would accept being a minority',
                       xlabel='step', ylabel='%')
            for ax, (t, g) in zip(axs[1:], (self.frames[0], self.frames[-1])):
                ax.imshow(g, cmap=cmap, vmin=0, vmax=len(m.shares), interpolation='nearest')
                ax.set(xticks=[], yticks=[], title=f'step {t}')
                ax.grid(False)
            V.suptitle(fig, f"Schelling segregation · {m.size}×{m.size} · like-neighbour share "
                            f"{c['similarity'].iloc[0]:.0%} → {c['similarity'].iloc[-1]:.0%}")
            fig.subplots_adjust(top=0.84, left=0.05, right=0.98, bottom=0.1, wspace=0.15)
        return V.finish(fig, show)

    def animate(self, max_frames: int = 80, interval: int = 120, save: str | None = None):
        m = self.model_params
        cmap = ListedColormap(['#ecebe7', *V.SERIES[:len(m.shares)]])
        frames = self.frames[::max(1, int(np.ceil(len(self.frames) / max_frames)))]
        with V.themed():
            fig, ax = plt.subplots(figsize=(5, 5.3), dpi=80)
            im = ax.imshow(frames[0][1], cmap=cmap, vmin=0, vmax=len(m.shares),
                           interpolation='nearest')
            ax.set(xticks=[], yticks=[])
            ax.grid(False)
            title = ax.set_title('')

            def update(k):
                im.set_data(frames[k][1])
                title.set_text(f'step {frames[k][0]}')
                return (im,)

            fig.tight_layout()
            ani = FuncAnimation(fig, update, frames=len(frames), interval=interval)
            return V.to_html_animation(fig, ani, save)
