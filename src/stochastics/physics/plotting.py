"""Static figures: initial-state overview, the run dashboard, and focused plots.

Every function shows the figure and returns ``None`` by default; pass
``show=False`` to get the ``Figure`` back (for saving or further edits).
"""
from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import _style as S
from .state import ParticleSystem

__all__ = ['plot_initial', 'plot_dashboard', 'report', 'plot_conservation', 'plot_tracks',
           'plot_velocities', 'plot_collision_regimes']


def _finish(fig, show: bool):
    if show:
        plt.show()
        return None
    return fig


def _display_table(df: pd.DataFrame) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            with pd.option_context('display.max_colwidth', None):
                display(df)
            return
    except ImportError:
        pass
    print(df.to_string())


def _hist_pair(ax, a, b, labels=('initial', 'final'), log: bool | None = None, bins: int = 30):
    a = np.asarray(a, float)
    b = np.asarray(b, float) if b is not None else np.empty(0)
    both = np.concatenate([a, b])
    both = both[np.isfinite(both)]
    if not len(both):
        return
    if np.ptp(both) == 0:
        ax.text(0.5, 0.5, f'all values = {both[0]:.4g}', ha='center', va='center',
                transform=ax.transAxes, color=S.MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    if log is None:
        log = both.min() > 0 and both.max() / both.min() > 30
    if log:
        edges = np.geomspace(both.min(), both.max() * 1.0001, bins + 1)
        ax.set_xscale('log')
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        lo, hi = both.min(), both.max()
        if hi == lo:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, bins + 1)
    ax.hist(a, bins=edges, histtype='stepfilled', color=S.SERIES[0], alpha=0.12, linewidth=0)
    ax.hist(a, bins=edges, histtype='step', color=S.SERIES[0], linewidth=1.6, label=labels[0])
    if labels[1] is not None:
        ax.hist(b, bins=edges, histtype='step', color=S.SERIES[1], linewidth=1.6, label=labels[1])
        ax.legend(loc='upper right')
    ax.set_ylabel('count')


def _scatter_clusters(ax, pos: np.ndarray, cluster_id: np.ndarray, dim3: bool, s=8, alpha=0.75):
    colors = S.cluster_colors(cluster_id)
    if dim3:
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=list(colors), s=s, alpha=alpha,
                   linewidths=0, depthshade=False)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c=list(colors), s=s, alpha=alpha, linewidths=0)
    uniq = sorted(set(int(c) for c in cluster_id))
    if 1 < len(uniq) <= 9:
        handles = [Line2D([], [], marker='o', linestyle='', markersize=6,
                          markerfacecolor=S.SERIES[c] if 0 <= c < len(S.SERIES) else S.OTHER,
                          markeredgewidth=0, label=f'cluster {c}' if c >= 0 else 'unassigned')
                   for c in uniq]
        ax.legend(handles=handles, loc='upper right', fontsize=7, handletextpad=0.2)


def _axes_labels(ax, dim3: bool):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    if dim3:
        ax.set_zlabel('z')
    else:
        ax.set_aspect('equal', adjustable='datalim')


# ------------------------------------------------------------------ initial
def plot_initial(particles: ParticleSystem, *, show: bool = True):
    """Positions by cluster, mass and charge distributions, cluster counts."""
    live = particles.live
    pos = particles.position[live]
    cid = particles.cluster_id[live]
    dim3 = particles.dimension == 3
    with S.themed():
        fig = plt.figure(figsize=(16, 4.2))
        gs = GridSpec(1, 4, figure=fig, width_ratios=[1.25, 1, 1, 0.8], wspace=0.28)
        ax0 = fig.add_subplot(gs[0], projection='3d' if dim3 else None)
        _scatter_clusters(ax0, pos, cid, dim3)
        _axes_labels(ax0, dim3)
        if dim3:
            S.style_3d(ax0)
        ax0.set_title('Positions by cluster')
        ax1 = fig.add_subplot(gs[1])
        _hist_pair(ax1, particles.mass[live], None, labels=('mass', None))
        ax1.set(title='Mass distribution', xlabel='mass')
        ax2 = fig.add_subplot(gs[2])
        _hist_pair(ax2, particles.charge[live], None, labels=('charge', None), log=False)
        ax2.set(title='Charge distribution', xlabel='charge')
        ax3 = fig.add_subplot(gs[3])
        uniq, counts = np.unique(cid, return_counts=True)
        ax3.bar([str(u) for u in uniq], counts, width=0.6,
                color=list(S.cluster_colors(uniq)), linewidth=0)
        for k, c in enumerate(counts):
            ax3.annotate(f'{c}', (k, c), textcoords='offset points', xytext=(0, 3),
                         ha='center', fontsize=8, color=S.TEXT_2)
        ax3.set(title='Particles per cluster', xlabel='cluster', ylabel='count')
        ax3.grid(axis='x', visible=False)
        fig.suptitle(f'{particles.n_live} particles · {particles.dimension}D · '
                     f'{len(uniq)} cluster(s) · species {", ".join(sorted(set(map(str, particles.species[live]))))}',
                     x=0.01, ha='left', fontsize=11.5, fontweight='bold', color=S.TEXT, y=1.02)
    print('Counts by cluster:', dict(zip(uniq.tolist(), counts.tolist())))
    return _finish(fig, show)


# ---------------------------------------------------------------- dashboard
def _first_last(history: pd.DataFrame):
    times = history['time'].unique()
    return history[history.time == times[0]], history[history.time == times[-1]]


def _track_ids(history: pd.DataFrame, max_tracks: int) -> np.ndarray:
    first, last = _first_last(history)
    survivors = last.sort_values('mass', ascending=False)['particle_id']
    survivors = survivors[survivors.isin(first['particle_id'])]
    ids = survivors.to_numpy()[:max_tracks]
    if len(ids) < max_tracks:
        extra = first.loc[~first.particle_id.isin(ids), 'particle_id'].to_numpy()
        ids = np.concatenate([ids, extra[:max_tracks - len(ids)]])
    return ids


def _plot_tracks_on(ax, history: pd.DataFrame, ids, dim3: bool, lw=0.9, alpha=0.6, box=None):
    sub = history[history.particle_id.isin(ids)].sort_values(['particle_id', 'time'])
    if box is not None:  # 3D axes do not clip lines: blank out points outside the box
        cols = ['x', 'y', 'z'][:len(box[0])]
        xyz = sub[cols].to_numpy()
        outside = np.any((xyz < box[0]) | (xyz > box[1]), axis=1)
        sub = sub.copy()
        sub.loc[outside, cols] = np.nan
    for pid, track in sub.groupby('particle_id', sort=False):
        c = int(track.cluster_id.iloc[0])
        color = S.SERIES[c] if 0 <= c < len(S.SERIES) else S.OTHER
        if dim3:
            ax.plot(track.x, track.y, track.z, color=color, linewidth=lw, alpha=alpha)
        else:
            ax.plot(track.x, track.y, color=color, linewidth=lw, alpha=alpha)


def _velocity_dispersion(history: pd.DataFrame) -> pd.DataFrame:
    vcols = [c for c in ('vx', 'vy', 'vz') if c in history]
    h = history[history.cluster_id >= 0]
    if h.empty:
        return pd.DataFrame()
    g = h.groupby(['time', 'cluster_id'])
    var = sum(g[c].var(ddof=0) for c in vcols)
    return np.sqrt(var).unstack('cluster_id')


def plot_dashboard(result, *, max_tracks: int = 40, show: bool = True):
    """One figure with positions, distributions, events, energy ledger and velocities."""
    h, d = result.history, result.diagnostics
    dim3 = result.dimension == 3
    first, last = _first_last(h)
    with S.themed():
        fig = plt.figure(figsize=(18, 12.5))
        gs = GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.27)

        # Row 1 — the system in space
        ax = fig.add_subplot(gs[0, 0], projection='3d' if dim3 else None)
        cols = ['x', 'y', 'z'] if dim3 else ['x', 'y']
        p0 = first[cols].to_numpy()
        if dim3:
            ax.scatter(*p0.T, s=4, c='#c9c8c3', linewidths=0, depthshade=False)
        else:
            ax.scatter(*p0.T, s=4, c='#c9c8c3', linewidths=0)
        p1 = last[cols].to_numpy()
        lo = np.percentile(np.vstack([p0, p1]), 2, axis=0)
        hi = np.percentile(np.vstack([p0, p1]), 98, axis=0)
        pad = 0.15 * np.maximum(hi - lo, 1e-9)
        lo, hi = lo - pad, hi + pad
        _plot_tracks_on(ax, h, _track_ids(h, max_tracks), dim3, box=(lo, hi))
        keep = np.all((p1 >= lo) & (p1 <= hi), axis=1)
        sizes = np.clip(6 + 30 * (last.mass / last.mass.max()).to_numpy(), 6, 60)[keep]
        colors = list(S.cluster_colors(last.cluster_id.to_numpy()[keep]))
        if dim3:
            ax.scatter(*p1[keep].T, s=sizes, c=colors, linewidths=0.5, edgecolors=S.SURFACE,
                       depthshade=False)
            S.style_3d(ax)
        else:
            ax.scatter(*p1[keep].T, s=sizes, c=colors, linewidths=0.5, edgecolors=S.SURFACE)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        if dim3:
            ax.set_zlim(lo[2], hi[2])
        _axes_labels(ax, dim3)
        ax.set_title('Trajectories (grey = start, dots = end)')

        ax = fig.add_subplot(gs[0, 1])
        _scatter_clusters(ax, first[['x', 'y']].to_numpy(), first.cluster_id.to_numpy(), False)
        _axes_labels(ax, False)
        ax.set_title('Initial clusters (x–y)')

        ax = fig.add_subplot(gs[0, 2])
        _hist_pair(ax, first.mass, last.mass)
        ax.set(title='Mass distribution', xlabel='mass')

        ax = fig.add_subplot(gs[0, 3])
        quantity = 'charge'
        if np.ptp(first.charge) == 0 and np.ptp(first.polarity_now) > 0:
            quantity = 'polarity_now'          # charge-free runs: show the clock values instead
        _hist_pair(ax, first[quantity], last[quantity], log=False)
        ax.set(title=f'{quantity.replace("_", " ").capitalize()} distribution', xlabel=quantity)

        # Row 2 — population, events, energy
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(d.time, d.n_active, color=S.SERIES[0])
        ax.set(title='Active particles', xlabel='time', ylabel='live particles')

        ax = fig.add_subplot(gs[1, 1])
        ev = result.events
        if ev.empty:
            ax.text(0.5, 0.5, 'no discrete events', ha='center', va='center',
                    transform=ax.transAxes, color=S.MUTED)
        else:
            for k, (name, grp) in enumerate(ev.groupby('event', sort=False)):
                color = S.EVENT_COLORS.get(name, S.SERIES[(4 + k) % len(S.SERIES)])
                t = np.sort(grp.time.to_numpy())
                ax.step(np.r_[d.time.iloc[0], t, d.time.iloc[-1]],
                        np.r_[0, np.arange(1, len(t) + 1), len(t)], where='post',
                        color=color, label=f'{name} ({len(t)})')
            ax.legend(loc='upper left')
        ax.set(title='Events (cumulative)', xlabel='time', ylabel='count')

        ax = fig.add_subplot(gs[1, 2])
        for name in ('kinetic', 'potential', 'internal', 'radiated', 'dissipated', 'binding',
                     'expansion', 'drive'):
            if name in d and np.nanmax(np.abs(d[name])) > 0:
                ax.plot(d.time, d[name], color=S.LEDGER_COLORS[name], label=name, linewidth=1.4)
        if np.isfinite(d.accounted_energy).all():
            ax.plot(d.time, d.accounted_energy, color=S.TEXT, linewidth=2.0, label='accounted')
        ax.legend(loc='best', ncols=2)
        ax.set(title='Energy ledger', xlabel='time', ylabel='energy')

        ax = fig.add_subplot(gs[1, 3])
        e = d.accounted_energy
        if np.isfinite(e).all():
            scale = max(abs(d.kinetic.iloc[0]) + abs(d.potential.iloc[0]) + abs(d.internal.iloc[0]),
                        1e-300)
            ax.axhline(0, color=S.MUTED, linewidth=0.8)
            ax.plot(d.time, (e - e.iloc[0]) / scale, color=S.SERIES[0])
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
            ax.set(ylabel='(E − E₀) / |E₀ terms|')
        else:
            ax.text(0.5, 0.5, 'no potential defined\n(non-reciprocal law)', ha='center',
                    va='center', transform=ax.transAxes, color=S.MUTED)
        ax.set(title='Energy residual (relative)', xlabel='time')

        # Row 3 — velocities and conservation
        ax = fig.add_subplot(gs[2, 0])
        _hist_pair(ax, first.speed, last.speed)
        ax.set(title='Speed distribution', xlabel='|v|')

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(d.time, d.mean_speed, color=S.SERIES[0], label='mean |v|')
        ax.plot(d.time, d.rms_speed, color=S.SERIES[1], label='rms |v|')
        ax.legend(loc='best')
        ax.set(title='Speed over time', xlabel='time', ylabel='speed')

        ax = fig.add_subplot(gs[2, 2])
        disp = _velocity_dispersion(h)
        shown = [c for c in disp.columns if 0 <= c < len(S.SERIES)] if not disp.empty else []
        for c in shown:
            ax.plot(disp.index, disp[c], color=S.SERIES[c], label=f'cluster {c}')
        if len(shown) > 1:
            ax.legend(loc='best')
        ax.set(title='Velocity dispersion by cluster', xlabel='time', ylabel='σ_v (internal)')

        ax = fig.add_subplot(gs[2, 3])
        dims = 'xyz'[:result.dimension]
        p = np.stack([d[f'p_{c}'] for c in dims], axis=1)
        p_scale = float((first.mass * first.speed).sum()) or 1.0
        q_scale = float(np.abs(first.charge).sum()) or 1.0
        floor = 1e-17
        drifts = {
            'mass': np.abs(d.mass - d.mass.iloc[0]) / d.mass.iloc[0],
            'charge': np.abs(d.charge - d.charge.iloc[0]) / q_scale,
            'momentum': np.linalg.norm(p - p[0], axis=1) / p_scale,
        }
        external = d.dissipated.iloc[-1] > 0 or d.expansion.iloc[-1] != 0
        for k, (name, val) in enumerate(drifts.items()):
            label = name + (' (drag/expansion act on it)' if name == 'momentum' and external else '')
            ax.plot(d.time, np.maximum(val, floor), color=S.SERIES[k], label=label)
        ax.set_yscale('log')
        ax.set_ylim(floor / 2, max(1e-6, max(float(np.max(v)) for v in drifts.values()) * 10))
        ax.legend(loc='upper left')
        ax.set(title='Conservation drift (relative)', xlabel='time', ylabel='|Δ| / scale')

        counts = result.event_counts()
        ev_txt = ', '.join(f'{v} {k}' for k, v in counts.items()) or 'no events'
        law = type(result.environment.law).__name__ if result.environment else ''
        resid = (e.iloc[-1] - e.iloc[0]) if np.isfinite(e).all() else float('nan')
        fig.suptitle(f'{law} · {result.dimension}D · {int(d.n_active.iloc[0])} → '
                     f'{int(d.n_active.iloc[-1])} particles · t = {d.time.iloc[-1]:.3g} · '
                     f'{ev_txt} · energy residual {resid:+.2e}',
                     x=0.01, ha='left', fontsize=12.5, fontweight='bold', color=S.TEXT, y=0.985)
        fig.subplots_adjust(top=0.93, left=0.04, right=0.98, bottom=0.05)
    return _finish(fig, show)


def report(result, *, show: bool = True, **kwargs):
    """Summary table + event counts + the dashboard, in one call."""
    _display_table(result.summary())
    return plot_dashboard(result, show=show, **kwargs)


# ----------------------------------------------------------- focused plots
def plot_conservation(result, *, show: bool = True):
    """Active particles, energy ledger and residual (the original three panels)."""
    d = result.diagnostics
    with S.themed():
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
        axs[0].plot(d.time, d.n_active, color=S.SERIES[0])
        axs[0].set(title='Active particles', xlabel='time')
        for name in ('kinetic', 'potential', 'internal', 'radiated', 'dissipated', 'binding',
                     'expansion', 'drive'):
            if np.nanmax(np.abs(d[name])) > 0:
                axs[1].plot(d.time, d[name], color=S.LEDGER_COLORS[name], label=name, linewidth=1.4)
        axs[1].plot(d.time, d.accounted_energy, color=S.TEXT, linewidth=2.0, label='accounted')
        axs[1].legend(fontsize=8, ncols=2)
        axs[1].set(title='Energy ledger', xlabel='time')
        axs[2].axhline(0, color=S.MUTED, linewidth=0.8)
        axs[2].plot(d.time, d.accounted_energy - d.accounted_energy.iloc[0], color=S.SERIES[0])
        axs[2].set(title='Energy residual from initial', xlabel='time')
        fig.tight_layout()
    return _finish(fig, show)


def plot_tracks(result, max_particles: int = 30, *, show: bool = True):
    """Trajectories of up to ``max_particles`` particles (2D or 3D)."""
    h = result.history
    dim3 = result.dimension == 3
    with S.themed():
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection='3d' if dim3 else None)
        ids = h['particle_id'].drop_duplicates().iloc[:max_particles]
        _plot_tracks_on(ax, h, ids, dim3, lw=1.1, alpha=0.8)
        _axes_labels(ax, dim3)
        if dim3:
            S.style_3d(ax)
        ax.set_title(f'Particle trajectories ({len(ids)} shown)')
    return _finish(fig, show)


def plot_velocities(result, *, show: bool = True):
    """Speed distribution, speed over time, dispersion by cluster."""
    h, d = result.history, result.diagnostics
    first, last = _first_last(h)
    with S.themed():
        fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
        _hist_pair(axs[0], first.speed, last.speed)
        axs[0].set(title='Speed distribution', xlabel='|v|')
        axs[1].plot(d.time, d.mean_speed, color=S.SERIES[0], label='mean |v|')
        axs[1].plot(d.time, d.rms_speed, color=S.SERIES[1], label='rms |v|')
        axs[1].legend()
        axs[1].set(title='Speed over time', xlabel='time')
        disp = _velocity_dispersion(h)
        if not disp.empty:
            for c in disp.columns:
                if 0 <= c < len(S.SERIES):
                    axs[2].plot(disp.index, disp[c], color=S.SERIES[c], label=f'cluster {c}')
            if disp.shape[1] > 1:
                axs[2].legend()
        axs[2].set(title='Velocity dispersion by cluster', xlabel='time')
        fig.tight_layout()
    return _finish(fig, show)


def plot_collision_regimes(policy=None, *, max_speed: float = 2.0, reduced_mass: float = 1.0,
                           pair_potential: float = 0.0, resolution: int = 301,
                           examples: Sequence[tuple[str, Sequence[int], Sequence[int]]] | None = None,
                           constituent_charge: Sequence[float] = (1.0, -1.0), show: bool = True):
    """Outcome map of a :class:`CompositionAffinity` policy for head-on contacts.

    x: constituent affinity w (-1 all parts repel ... +1 all parts attract);
    y: approach speed. Bounces are shaded by restitution e (dark = elastic),
    captures are orange. ``examples`` marks real compositions, e.g.
    ``[('3+2- vs 0+5-', (3, 2), (0, 5))]``.
    """
    from matplotlib.colors import Normalize
    from .collisions import CompositionAffinity
    policy = policy or CompositionAffinity()
    w = np.linspace(-1, 1, resolution)
    v = np.linspace(0, max_speed, resolution)
    restitution = np.full((len(v), len(w)), np.nan)
    merged = np.zeros((len(v), len(w)), dtype=bool)
    for a, speed in enumerate(v):
        ke = 0.5 * reduced_mass * speed ** 2
        for b, wv in enumerate(w):
            out = policy.classify(wv, ke, ke, reduced_mass, pair_potential)
            if out.kind == 'merge':
                merged[a, b] = True
            else:
                restitution[a, b] = out.restitution
    q = np.asarray(constituent_charge, float)
    K = (np.asarray(policy.affinity_matrix, float) if getattr(policy, 'affinity_matrix', None)
         is not None else -np.outer(q, q) + getattr(policy, 'cohesion', 0.0))
    extent = (-1, 1, 0, max_speed)
    with S.themed():
        fig, axs = plt.subplots(1, 2, figsize=(13.5, 4.8), gridspec_kw={'width_ratios': [1.4, 1]})
        ax = axs[0]
        im = ax.imshow(restitution, origin='lower', aspect='auto', extent=extent, cmap=S.SEQUENTIAL,
                       norm=Normalize(0, 1), interpolation='nearest')
        ax.imshow(np.where(merged, 1.0, np.nan), origin='lower', aspect='auto', extent=extent,
                  cmap=ListedColormap([S.EVENT_COLORS['merge']]), vmin=0, vmax=1,
                  interpolation='nearest', alpha=0.9)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label('bounce restitution e   (1 = elastic)')
        cb.outline.set_visible(False)
        ax.legend(handles=[Patch(color=S.EVENT_COLORS['merge'], alpha=0.9, label='merge (captured)')],
                  loc='upper right', frameon=True, facecolor=S.SURFACE, edgecolor=S.GRID)
        ax.set(xlabel='constituent affinity w   (−1: all parts repel · +1: all parts attract)',
               ylabel='approach speed', title='Collision outcome map (head-on contact)')
        ax2 = axs[1]
        e = np.array([policy.restitution(x) for x in w])
        ax2.plot(w, e, color=S.SERIES[0], linewidth=2)
        ax2.set(xlabel='constituent affinity w', ylabel='restitution e', ylim=(-0.03, 1.08),
                xlim=(-1.05, 1.05), title='Restitution: repulsive → elastic, attractive → sticky')
        if examples:
            for label, ca, cb_ in examples:
                ca, cb_ = np.asarray(ca, float), np.asarray(cb_, float)
                wx = float(np.clip(ca @ K @ cb_ / (ca.sum() * cb_.sum()), -1, 1))
                ax.axvline(wx, color=S.TEXT, linewidth=0.9, alpha=0.8)
                ax2.axvline(wx, color=S.GRID, linewidth=0.9, zorder=0)
                inside = wx < 0.85
                ax.annotate(label, (wx, 0.03 * max_speed), rotation=90, va='bottom',
                            ha='left' if inside else 'right', fontsize=8, color=S.TEXT,
                            xytext=(3 if inside else -3, 0), textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.15', fc=S.SURFACE, ec='none', alpha=0.85))
                ax2.plot([wx], [policy.restitution(wx)], 'o', color=S.SERIES[0], markersize=7,
                         markeredgecolor=S.SURFACE, markeredgewidth=1.5)
                ax2.annotate(label, (wx, policy.restitution(wx)), textcoords='offset points',
                             xytext=(6, 4), fontsize=8, color=S.TEXT_2)
        fig.tight_layout()
    return _finish(fig, show)
