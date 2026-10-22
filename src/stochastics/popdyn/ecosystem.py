"""Grass - rabbits - wolves on a grid, with heritable traits and an energy ledger.

Each step:

1. rabbits move (towards grass they can see, else at random) and eat grass;
2. wolves move (towards rabbits they can see) and eat rabbits;
3. every animal burns energy: metabolism + move_cost * cells moved + vision_cost * vision;
4. animals die from starvation (energy <= 0) or old age; carcasses fertilise
   their cell, so grass grows back from dead animals (the loop closes:
   grass -> rabbits -> wolves -> grass);
5. animals with enough energy breed with probability ``fertility``; the
   child gets half the energy and the parent's traits with small mutations;
6. eaten grass regrows at random with probability ``regrow_rate``.

Heritable traits: ``speed`` (cells per step), ``vision`` (sight radius) and
``fertility``. Faster and farther-sighted animals find more food but burn more
energy, so selection has a real trade-off to work on.

The energy ledger closes exactly:
    change in (grass + rabbit + wolf energy) = sun + fertiliser
                                               - digestion loss - metabolism - detritus
"""
from __future__ import annotations

from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from .. import _viz as V
from .core import Result, Timer, mutate, trait_effects

__all__ = ['Species', 'Ecosystem', 'EcosystemResult']

DIRS = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])
CAUSES = {0: 'alive', 1: 'starved', 2: 'old age', 3: 'eaten'}
TRAITS = ('speed', 'vision', 'fertility')
GRASS_COLOR, PREY_COLOR, PREDATOR_COLOR = V.AQUA, V.ORANGE, V.BLUE


@dataclass
class Species:
    name: str
    count: int
    energy: float = 5.0          # initial energy per animal
    speed: float = 1.0           # mean cells moved per step            (heritable)
    vision: float = 3.0          # mean sight radius in cells           (heritable)
    fertility: float = 0.2       # breeding chance per step when able   (heritable)
    metabolism: float = 0.3      # base energy burned per step
    move_cost: float = 0.1       # energy per cell moved
    vision_cost: float = 0.05    # energy per step per cell of sight
    efficiency: float = 0.8      # fraction of food energy absorbed
    breed_energy: float = 8.0    # energy needed to breed (child gets half)
    max_age: int = 150
    mutation: float = 0.08       # sd of the lognormal trait mutation at birth
    flee: bool = False           # prey only: run from a visible predator before looking for food
    max_speed: float = 4.0       # physiological ceilings for the evolving traits
    max_vision: float = 8.0
    plural: str | None = None

    @property
    def plural_name(self) -> str:
        return self.plural or f'{self.name}s'


def _rabbits():
    return Species('rabbit', 300, energy=4.0, speed=1.0, vision=3.0, fertility=0.12,
                   metabolism=0.25, move_cost=0.08, vision_cost=0.04, efficiency=0.9,
                   breed_energy=6.0, max_age=120, flee=True)


def _wolves():
    return Species('wolf', 30, energy=15.0, speed=1.0, vision=2.0, fertility=0.05,
                   metabolism=0.4, move_cost=0.1, vision_cost=0.05, efficiency=0.4,
                   breed_energy=24.0, max_age=250, plural='wolves')


class _Pool:
    """Growable structure-of-arrays for every animal that ever lived."""

    FIELDS = dict(sp=np.int8, x=np.int32, y=np.int32, energy=float, age=np.int32, alive=bool,
                  born=np.int32, died=np.int32, cause=np.int8, parent=np.int64,
                  generation=np.int32, speed=float, vision=float, fertility=float,
                  food=np.int32, eaten=float, offspring=np.int32, distance=float,
                  max_energy=float)

    def __init__(self, capacity=1024):
        self.n = 0
        self.d = {k: np.zeros(capacity, dtype=t) for k, t in self.FIELDS.items()}

    def add(self, **cols) -> np.ndarray:
        k = len(next(iter(cols.values())))
        need = self.n + k
        if need > len(self.d['sp']):
            cap = max(need, 2 * len(self.d['sp']))
            for name, arr in self.d.items():
                new = np.zeros(cap, dtype=arr.dtype)
                new[:self.n] = arr[:self.n]
                self.d[name] = new
        idx = np.arange(self.n, need)
        for name in self.FIELDS:
            if name in cols:
                self.d[name][idx] = cols[name]
        self.d['died'][idx] = -1
        self.n = need
        return idx

    def __getattr__(self, name):
        d = self.__dict__['d']
        if name in d:
            return d[name][:self.__dict__['n']]
        raise AttributeError(name)


@dataclass
class Ecosystem:
    width: int = 80
    height: int = 80
    grass_energy: float = 3.0
    regrow_rate: float = 0.03
    grass_cover: float = 0.5
    fertilize: bool = True
    prey: Species = field(default_factory=_rabbits)
    predator: Species = field(default_factory=_wolves)
    seed: int | None = None

    # ---------------------------------------------------------------- run
    def run(self, steps: int = 1000, record_every: int = 2, frame_every: int | None = None
            ) -> 'EcosystemResult':
        rng = np.random.default_rng(self.seed)
        W, H = self.width, self.height
        grass = rng.random((H, W)) < self.grass_cover
        pool = _Pool()
        species = (self.prey, self.predator)
        for s_id, sp in enumerate(species):
            n = sp.count
            pool.add(sp=np.full(n, s_id), x=rng.integers(0, W, n), y=rng.integers(0, H, n),
                     energy=rng.uniform(0.5, 1.5, n) * sp.energy, age=rng.integers(0, sp.max_age // 2, n),
                     alive=np.ones(n, bool), born=np.zeros(n), cause=np.zeros(n),
                     parent=np.full(n, -1), generation=np.zeros(n),
                     speed=np.minimum(mutate(np.full(n, sp.speed), 0.25, rng), sp.max_speed),
                     vision=np.minimum(mutate(np.full(n, sp.vision), 0.25, rng), sp.max_vision),
                     fertility=np.clip(mutate(np.full(n, sp.fertility), 0.25, rng), 0, 1),
                     max_energy=np.zeros(n))
        ledger = dict(sun=0.0, fertilizer=0.0, grass_eaten=0.0, prey_eaten=0.0, digestion=0.0,
                      metabolism_prey=0.0, metabolism_predator=0.0, detritus=0.0)
        tallies = dict(grass_items=0, prey_items=0, births_prey=0, births_predator=0)
        stock0 = self._stocks(grass, pool)
        rows, frames = [], []
        frame_every = frame_every or max(1, steps // 150)

        def record(t):
            stocks = self._stocks(grass, pool)
            row = {'time': t, 'grass': int(grass.sum()),
                   self.prey.name: int(((pool.sp == 0) & pool.alive).sum()),
                   self.predator.name: int(((pool.sp == 1) & pool.alive).sum()),
                   'grass_energy': stocks[0], f'{self.prey.name}_energy': stocks[1],
                   f'{self.predator.name}_energy': stocks[2], **ledger, **tallies}
            for s_id, sp in enumerate(species):
                m = (pool.sp == s_id) & pool.alive
                for tr in TRAITS:
                    row[f'{sp.name}_{tr}'] = float(getattr(pool, tr)[m].mean()) if m.any() else np.nan
            row['residual'] = (sum(stocks) - sum(stock0)) - (
                ledger['sun'] + ledger['fertilizer'] - ledger['digestion'] - ledger['metabolism_prey']
                - ledger['metabolism_predator'] - ledger['detritus'])
            rows.append(row)

        def snapshot(t):
            live = np.flatnonzero(pool.alive)
            sp = pool.sp[live]
            frames.append((t, grass.copy(), np.c_[pool.x[live][sp == 0], pool.y[live][sp == 0]],
                           np.c_[pool.x[live][sp == 1], pool.y[live][sp == 1]]))

        record(0)
        snapshot(0)
        with Timer() as timer:
            for t in range(1, steps + 1):
                prey = np.flatnonzero(pool.alive & (pool.sp == 0))
                danger = None
                if self.prey.flee:
                    pred_now = np.flatnonzero(pool.alive & (pool.sp == 1))
                    danger = np.zeros(H * W, bool)
                    danger[pool.y[pred_now] * W + pool.x[pred_now]] = True
                    danger = danger.reshape(H, W)
                moved_prey = self._move(pool, prey, grass, rng, danger)
                self._eat_grass(pool, prey, grass, ledger, tallies, rng)
                pred = np.flatnonzero(pool.alive & (pool.sp == 1))
                prey_now = np.flatnonzero(pool.alive & (pool.sp == 0))
                occupied = np.zeros(H * W, bool)
                occupied[pool.y[prey_now] * W + pool.x[prey_now]] = True
                moved_pred = self._move(pool, pred, occupied.reshape(H, W), rng)
                self._hunt(pool, pred, prey_now, ledger, tallies, rng, t)
                self._metabolise(pool, prey, moved_prey, self.prey, ledger, 'metabolism_prey')
                self._metabolise(pool, pred, moved_pred, self.predator, ledger, 'metabolism_predator')
                self._deaths(pool, grass, ledger, t)
                self._breed(pool, rng, t, tallies)
                regrow = (~grass) & (rng.random((H, W)) < self.regrow_rate)
                grass |= regrow
                ledger['sun'] += self.grass_energy * regrow.sum()
                live = pool.alive
                pool.age[live] += 1
                pool.max_energy[live] = np.maximum(pool.max_energy[live], pool.energy[live])
                if t % record_every == 0 or t == steps:
                    record(t)
                if t % frame_every == 0 or t == steps:
                    snapshot(t)
                if not live.any():
                    record(t)
                    break
        return self._result(pool, rows, frames, steps, timer.seconds)

    # ------------------------------------------------------------ helpers
    def _stocks(self, grass, pool):
        e = pool.energy
        a = pool.alive
        return (float(grass.sum() * self.grass_energy), float(e[a & (pool.sp == 0)].sum()),
                float(e[a & (pool.sp == 1)].sum()))

    @staticmethod
    def _look(grid, x, y, vis, rng):
        """Nearest marked cell along the 8 compass rays within each animal's sight."""
        k = len(x)
        H, W = grid.shape
        vmax = int(vis.max()) if k else 0
        if vmax <= 0:
            return np.zeros(k, int), np.full(k, np.inf)
        s = np.arange(1, vmax + 1)
        cx = (x[:, None, None] + DIRS[None, :, 0, None] * s[None, None, :]) % W
        cy = (y[:, None, None] + DIRS[None, :, 1, None] * s[None, None, :]) % H
        seen = grid[cy, cx] & (s[None, None, :] <= vis[:, None, None])
        first = np.where(seen.any(axis=2), np.argmax(seen, axis=2) + 1.0, np.inf)
        first = first + 0.01 * rng.random(first.shape)                # random tie-break
        best = np.argmin(first, axis=1)
        return best, np.floor(first[np.arange(k), best])

    def _move(self, pool, idx, food, rng, danger=None):
        """Move animals ``idx``: away from visible danger, else towards visible food."""
        k = len(idx)
        if k == 0:
            return np.zeros(0)
        H, W = food.shape
        x, y = pool.x[idx], pool.y[idx]
        speed = pool.speed[idx]
        steps = np.floor(speed) + (rng.random(k) < speed - np.floor(speed))
        vis = np.clip(np.rint(pool.vision[idx]), 0, 12).astype(int)
        here = food[y, x]
        best, dist = self._look(food, x, y, vis, rng)
        found = np.isfinite(dist)
        direction = np.where(found, best, rng.integers(0, 8, k))
        move = np.where(found, np.minimum(steps, dist), steps)
        move = np.where(here, 0, move)
        if danger is not None:
            threat, tdist = self._look(danger, x, y, vis, rng)
            scared = np.isfinite(tdist)
            direction = np.where(scared, (threat + 4) % 8, direction)
            move = np.where(scared, np.maximum(steps, 1), move)
        pool.d['x'][idx] = (x + DIRS[direction, 0] * move).astype(int) % W
        pool.d['y'][idx] = (y + DIRS[direction, 1] * move).astype(int) % H
        pool.d['distance'][idx] += move
        return move

    def _eat_grass(self, pool, prey, grass, ledger, tallies, rng):
        if not len(prey):
            return
        W = self.width
        cells = pool.y[prey] * W + pool.x[prey]
        flat = grass.reshape(-1)
        on = flat[cells]
        cand = prey[on]
        if not len(cand):
            return
        cand = cand[rng.permutation(len(cand))]
        ccells = pool.y[cand] * W + pool.x[cand]
        _, first = np.unique(ccells, return_index=True)
        eaters = cand[first]
        flat[ccells[first]] = False
        gain = self.prey.efficiency * self.grass_energy
        pool.d['energy'][eaters] += gain
        pool.d['food'][eaters] += 1
        pool.d['eaten'][eaters] += self.grass_energy
        ledger['grass_eaten'] += self.grass_energy * len(eaters)
        ledger['digestion'] += (1 - self.prey.efficiency) * self.grass_energy * len(eaters)
        tallies['grass_items'] += len(eaters)

    def _hunt(self, pool, pred, prey, ledger, tallies, rng, t):
        if not len(pred) or not len(prey):
            return
        W = self.width
        pcells = pool.y[prey] * W + pool.x[prey]
        wcells = pool.y[pred] * W + pool.x[pred]
        hunting = pred[np.isin(wcells, pcells)]
        if not len(hunting):
            return
        hunting = hunting[rng.permutation(len(hunting))]
        hcells = pool.y[hunting] * W + pool.x[hunting]
        cell_ids, first = np.unique(hcells, return_index=True)
        hunters = hunting[first]
        shuffled = prey[rng.permutation(len(prey))]
        scells = pool.y[shuffled] * W + pool.x[shuffled]
        prey_cells, pfirst = np.unique(scells, return_index=True)
        victims = shuffled[pfirst][np.searchsorted(prey_cells, cell_ids)]
        e = pool.energy[victims].copy()
        pool.d['energy'][hunters] += self.predator.efficiency * e
        pool.d['food'][hunters] += 1
        pool.d['eaten'][hunters] += e
        pool.d['alive'][victims] = False
        pool.d['died'][victims] = t
        pool.d['cause'][victims] = 3
        pool.d['energy'][victims] = 0.0
        ledger['prey_eaten'] += float(e.sum())
        ledger['digestion'] += float((1 - self.predator.efficiency) * e.sum())
        tallies['prey_items'] += len(victims)

    def _metabolise(self, pool, idx, moved, sp, ledger, key):
        idx_alive = pool.alive[idx]
        idx, moved = idx[idx_alive], moved[idx_alive]
        if not len(idx):
            return
        cost = sp.metabolism + sp.move_cost * moved + sp.vision_cost * pool.vision[idx]
        before = pool.energy[idx]
        spent = np.minimum(cost, np.maximum(before, 0.0))
        pool.d['energy'][idx] = before - spent
        ledger[key] += float(spent.sum())

    def _deaths(self, pool, grass, ledger, t):
        live = np.flatnonzero(pool.alive)
        sp = pool.sp[live]
        max_age = np.where(sp == 0, self.prey.max_age, self.predator.max_age)
        starved = pool.energy[live] <= 1e-12
        old = (pool.age[live] >= max_age) & ~starved
        dead = live[starved | old]
        if not len(dead):
            return
        ledger['detritus'] += float(pool.energy[live[old]].sum())
        pool.d['cause'][live[starved]] = 1
        pool.d['cause'][live[old]] = 2
        pool.d['alive'][dead] = False
        pool.d['died'][dead] = t
        pool.d['energy'][dead] = 0.0
        if self.fertilize:
            cells = np.unique(pool.y[dead] * self.width + pool.x[dead])
            flat = grass.reshape(-1)
            new = cells[~flat[cells]]
            flat[new] = True
            ledger['fertilizer'] += self.grass_energy * len(new)

    def _breed(self, pool, rng, t, tallies):
        live = np.flatnonzero(pool.alive)
        sp = pool.sp[live]
        need = np.where(sp == 0, self.prey.breed_energy, self.predator.breed_energy)
        ok = (pool.energy[live] >= need) & (rng.random(len(live)) < pool.fertility[live])
        parents = live[ok]
        if not len(parents):
            return
        half = pool.energy[parents] / 2
        pool.d['energy'][parents] = half
        pool.d['offspring'][parents] += 1
        psp = pool.sp[parents]
        mut = np.where(psp == 0, self.prey.mutation, self.predator.mutation)
        d = DIRS[rng.integers(0, 8, len(parents))]
        k = len(parents)
        pool.add(sp=psp, x=(pool.x[parents] + d[:, 0]) % self.width,
                 y=(pool.y[parents] + d[:, 1]) % self.height, energy=half,
                 age=np.zeros(k), alive=np.ones(k, bool), born=np.full(k, t), cause=np.zeros(k),
                 parent=parents, generation=pool.generation[parents] + 1,
                 speed=np.minimum(pool.speed[parents] * np.exp(mut * rng.standard_normal(k)),
                                  np.where(psp == 0, self.prey.max_speed, self.predator.max_speed)),
                 vision=np.minimum(pool.vision[parents] * np.exp(mut * rng.standard_normal(k)),
                                   np.where(psp == 0, self.prey.max_vision, self.predator.max_vision)),
                 fertility=np.clip(pool.fertility[parents] * np.exp(mut * rng.standard_normal(k)), 0, 1),
                 max_energy=half)
        tallies['births_prey'] += int((psp == 0).sum())
        tallies['births_predator'] += int((psp == 1).sum())

    def _result(self, pool, rows, frames, steps, seconds):
        df = pd.DataFrame(rows).drop_duplicates('time').set_index('time')
        names = np.array([self.prey.name, self.predator.name])
        T = int(df.index[-1])
        died = pool.died.astype(float)
        died[died < 0] = np.nan
        agents = pd.DataFrame({
            'id': np.arange(pool.n), 'species': names[pool.sp], 'parent': pool.parent,
            'generation': pool.generation, 'born': pool.born, 'died': died,
            'lifespan': np.where(np.isnan(died), T, died) - pool.born,
            'cause': pd.Categorical([CAUSES[c] for c in pool.cause]),
            'speed': pool.speed, 'vision': pool.vision, 'fertility': pool.fertility,
            'food': pool.food, 'energy_eaten': pool.eaten, 'offspring': pool.offspring,
            'distance': pool.distance, 'max_energy': pool.max_energy, 'alive': pool.alive.copy(),
        }).set_index('id')
        counts = df[['grass', self.prey.name, self.predator.name]]
        ledger_cols = ['grass_energy', f'{self.prey.name}_energy', f'{self.predator.name}_energy',
                       'sun', 'fertilizer', 'grass_eaten', 'prey_eaten', 'digestion',
                       'metabolism_prey', 'metabolism_predator', 'detritus', 'residual']
        return EcosystemResult('ecosystem', counts, df[ledger_cols], agents, frames,
                               params={'eco': self, 'steps': steps},
                               extras={'series': df}, runtime=seconds)


class EcosystemResult(Result):
    @property
    def eco(self) -> Ecosystem:
        return self.params['eco']

    def totals(self) -> pd.DataFrame:
        """Who ate what over the whole run, births and deaths by cause."""
        s = self.extras['series'].iloc[-1]
        prey, pred = self.eco.prey.name, self.eco.predator.name
        plural = {prey: self.eco.prey.plural_name, pred: self.eco.predator.plural_name}
        a = self.agents
        rows = {
            f'grass eaten by {plural[prey]} (items)': s['grass_items'],
            f'grass eaten by {plural[prey]} (energy)': s['grass_eaten'],
            f'{plural[prey]} eaten by {plural[pred]} (items)': s['prey_items'],
            f'{plural[prey]} eaten by {plural[pred]} (energy)': s['prey_eaten'],
            f'{prey} births': s['births_prey'], f'{pred} births': s['births_predator'],
        }
        for sp in (prey, pred):
            for cause in ('starved', 'old age', 'eaten'):
                n = int(((a.species == sp) & (a.cause == cause)).sum())
                if n or cause != 'eaten' or sp == prey:
                    rows[f'{sp} deaths: {cause}'] = n
        rows['energy ledger residual'] = s['residual']
        return pd.Series(rows, name='total').to_frame()

    def trait_effects(self, species: str | None = None, target: str = 'lifespan', top: float = 0.1,
                      include_alive: bool = False):
        species = species or self.eco.prey.name
        df = self.agents[self.agents.species == species]
        if not include_alive:
            df = df[~df.alive]
        return trait_effects(df, list(TRAITS), target, top)

    def plot(self, show: bool = True):
        return plot_ecosystem(self, show)

    def animate(self, max_frames: int = 100, interval: int = 80, save: str | None = None):
        return animate_ecosystem(self, max_frames, interval, save)


# ---------------------------------------------------------------- figures
def plot_ecosystem(res: EcosystemResult, show: bool = True):
    s = res.extras['series']
    eco = res.eco
    prey, pred = eco.prey.name, eco.predator.name
    plural = {prey: eco.prey.plural_name, pred: eco.predator.plural_name}
    a = res.agents
    with V.themed():
        fig = plt.figure(figsize=(18, 11.5))
        gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.28)

        ax = fig.add_subplot(gs[0, 0])
        ax.plot(s.index, s[prey], color=PREY_COLOR, label=f'{plural[prey]}')
        ax.plot(s.index, s[pred], color=PREDATOR_COLOR, label=f'{plural[pred]}')
        ax.legend(loc='upper right')
        ax.set(title='Population', xlabel='step', ylabel='animals')

        ax = fig.add_subplot(gs[0, 1])
        ax.plot(s.index, 100 * s['grass'] / (eco.width * eco.height), color=GRASS_COLOR)
        ax.set(title='Grass cover', xlabel='step', ylabel='% of cells grown', ylim=(0, 100))

        ax = fig.add_subplot(gs[0, 2])
        pts = ax.scatter(s[prey], s[pred], c=s.index, cmap=V.SEQUENTIAL, s=6, linewidths=0)
        ax.plot(s[prey], s[pred], color=V.GRID, linewidth=0.6, zorder=0)
        cb = fig.colorbar(pts, ax=ax, fraction=0.05, pad=0.02)
        cb.set_label('step')
        cb.outline.set_visible(False)
        ax.set(title='Phase portrait', xlabel=f'{plural[prey]}', ylabel=f'{plural[pred]}')

        ax = fig.add_subplot(gs[0, 3])
        for name, color in (('grass_energy', GRASS_COLOR), (f'{prey}_energy', PREY_COLOR),
                            (f'{pred}_energy', PREDATOR_COLOR)):
            ax.plot(s.index, s[name], color=color, label=name.replace('_', ' '))
        total = s['grass_energy'] + s[f'{prey}_energy'] + s[f'{pred}_energy']
        ax.plot(s.index, total, color=V.TEXT, linewidth=2, label='total stock')
        ax.legend(loc='upper right', ncols=2, fontsize=7)
        ax.set(title='Energy stocks', xlabel='step', ylabel='energy')

        ax = fig.add_subplot(gs[1, 0])
        flows = [('sun', V.YELLOW), ('fertilizer', V.GREEN), ('grass_eaten', GRASS_COLOR),
                 ('prey_eaten', PREY_COLOR), ('digestion', V.MAGENTA),
                 ('metabolism_prey', V.VIOLET), ('metabolism_predator', V.RED)]
        for name, color in flows:
            ax.plot(s.index, s[name], color=color, label=name.replace('_', ' '), linewidth=1.4)
        ax.legend(loc='upper left', ncols=2, fontsize=7)
        ax.set(title='Energy flows (cumulative)', xlabel='step', ylabel='energy')

        ax = fig.add_subplot(gs[1, 1])
        ax.axhline(0, color=V.MUTED, linewidth=0.8)
        ax.plot(s.index, s['residual'], color=V.BLUE)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        ax.set(title='Ledger residual (should be ~0)', xlabel='step', ylabel='energy')

        ax = fig.add_subplot(gs[1, 2])
        for k, tr in enumerate(TRAITS):
            for sp, color, alpha in ((prey, PREY_COLOR, 1.0), (pred, PREDATOR_COLOR, 1.0)):
                col = s[f'{sp}_{tr}']
                base = col.dropna().iloc[0] if col.notna().any() else 1.0
                ax.plot(s.index, col / base, color=V.SERIES[[1, 0][sp == pred]],
                        linewidth=[2.0, 1.2, 0.7][k], alpha=alpha,
                        label=f'{sp} {tr}')
        ax.axhline(1, color=V.MUTED, linewidth=0.8)
        ax.legend(loc='upper left', ncols=2, fontsize=7)
        ax.set(title='Trait evolution (relative to start)', xlabel='step', ylabel='mean / initial mean')

        ax = fig.add_subplot(gs[1, 3])
        dead = a[~a.alive]
        bins = np.linspace(0, max(1, dead.lifespan.max()), 40)
        for sp, color in ((prey, PREY_COLOR), (pred, PREDATOR_COLOR)):
            ax.hist(dead[dead.species == sp].lifespan, bins=bins, histtype='step', color=color,
                    linewidth=1.6, label=f'{plural[sp]}', density=True)
        ax.legend(loc='upper right')
        ax.set(title='Lifespan of the dead', xlabel='steps lived', ylabel='density')

        ax = fig.add_subplot(gs[2, 0])
        causes = ['starved', 'old age', 'eaten']
        xs = np.arange(len(causes))
        for j, (sp, color) in enumerate(((prey, PREY_COLOR), (pred, PREDATOR_COLOR))):
            vals = [int(((a.species == sp) & (a.cause == c)).sum()) for c in causes]
            ax.bar(xs + (j - 0.5) * 0.36, vals, width=0.34, color=color, label=f'{plural[sp]}')
        ax.set_xticks(xs, causes)
        ax.grid(axis='x', visible=False)
        ax.legend(loc='upper right')
        ax.set(title='Deaths by cause', ylabel='animals')

        ax = fig.add_subplot(gs[2, 1])
        rd = dead[dead.species == prey]
        if len(rd) > 20:
            hb = ax.hexbin(rd.speed, rd.vision, C=rd.lifespan, reduce_C_function=np.mean,
                           gridsize=22, cmap=V.SEQUENTIAL, mincnt=5, linewidths=0.2,
                           edgecolors=V.SURFACE)
            cb = fig.colorbar(hb, ax=ax, fraction=0.05, pad=0.02)
            cb.set_label('mean lifespan')
            cb.outline.set_visible(False)
        ax.grid(False)
        ax.set(title=f'{prey.capitalize()} trait combos: mean lifespan', xlabel='speed',
               ylabel='vision')

        ax = fig.add_subplot(gs[2, 2])
        eff = res.trait_effects(prey, 'lifespan')
        eff2 = res.trait_effects(pred, 'lifespan') if (dead.species == pred).sum() > 10 else None
        ys = np.arange(len(TRAITS))
        ax.barh(ys + 0.18, eff.loc[list(TRAITS), 'spearman_rho'], height=0.34, color=PREY_COLOR,
                label=f'{plural[prey]}')
        if eff2 is not None:
            ax.barh(ys - 0.18, eff2.loc[list(TRAITS), 'spearman_rho'], height=0.34,
                    color=PREDATOR_COLOR, label=f'{plural[pred]}')
        ax.axvline(0, color=V.MUTED, linewidth=0.8)
        ax.set_yticks(ys, TRAITS)
        ax.grid(axis='y', visible=False)
        ax.legend(loc='lower right')
        ax.set(title='Which traits buy a long life?', xlabel='Spearman ρ with lifespan')

        ax = fig.add_subplot(gs[2, 3])
        _draw_frame(ax, res.frames[-1], eco)
        ax.set_title(f'World at step {res.frames[-1][0]}')

        tot = res.totals()['total']
        V.suptitle(fig, f'Ecosystem {eco.width}×{eco.height} · {int(s.index[-1])} steps · '
                        f'{plural[prey]} ate {int(tot.iloc[0])} grass · {plural[pred]} ate {int(tot.iloc[2])} {plural[prey]} · '
                        f'ledger residual {s["residual"].iloc[-1]:+.1e}')
        fig.subplots_adjust(top=0.93, left=0.04, right=0.98, bottom=0.05)
    return V.finish(fig, show)


_GRID_CMAP = ListedColormap([V.SURFACE, '#bfe9d6'])


def _draw_frame(ax, frame, eco, size=9):
    t, grass, prey_xy, pred_xy = frame
    ax.imshow(grass, cmap=_GRID_CMAP, vmin=0, vmax=1, origin='lower', interpolation='nearest')
    ax.scatter(prey_xy[:, 0], prey_xy[:, 1], s=size, color=PREY_COLOR, linewidths=0)
    ax.scatter(pred_xy[:, 0], pred_xy[:, 1], s=size * 2.2, color=PREDATOR_COLOR,
               edgecolors=V.SURFACE, linewidths=0.5)
    ax.set(xlim=(-0.5, eco.width - 0.5), ylim=(-0.5, eco.height - 0.5), xticks=[], yticks=[])
    ax.grid(False)


def animate_ecosystem(res: EcosystemResult, max_frames=100, interval=80, save=None):
    frames = res.frames[::max(1, int(np.ceil(len(res.frames) / max_frames)))]
    eco = res.eco
    s = res.counts
    with V.themed():
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), dpi=70,
                                      gridspec_kw={'width_ratios': [1, 1.15]})
        ax2.plot(s.index, s[eco.prey.name], color=PREY_COLOR, label=f'{eco.prey.plural_name}')
        ax2.plot(s.index, s[eco.predator.name], color=PREDATOR_COLOR, label=f'{eco.predator.plural_name}')
        ax2.legend(loc='upper right')
        ax2.set(xlabel='step', ylabel='animals', title='Population')
        cursor = ax2.axvline(0, color=V.TEXT, linewidth=0.8)

        def update(k):
            ax.clear()
            _draw_frame(ax, frames[k], eco)
            t = frames[k][0]
            ax.set_title(f'step {t}   {eco.prey.plural_name} {len(frames[k][2])}   '
                         f'{eco.predator.plural_name} {len(frames[k][3])}')
            cursor.set_xdata([t, t])
            return ()

        fig.tight_layout()
        ani = FuncAnimation(fig, update, frames=len(frames), interval=interval)
        return V.to_html_animation(fig, ani, save)
