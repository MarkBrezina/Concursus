"""2D and 3D animations of a :class:`SimulationResult`.

Both return an ``IPython.display.HTML`` player in notebooks (or the raw
``FuncAnimation`` outside IPython) and can also write a ``.gif``/``.mp4``.
``animate_2d`` works on 3D runs too, projected onto ``plane``.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

from . import _style as S

__all__ = ['animate_2d', 'animate_3d']


def _frames(history, max_frames: int):
    groups = list(history.groupby('time', sort=True))
    stride = max(1, int(np.ceil(len(groups) / max_frames)))
    picked = groups[::stride]
    if picked[-1][0] != groups[-1][0]:
        picked.append(groups[-1])
    return picked


def _limits(history, cols, robust: bool):
    """Axis limits. ``robust`` frames the bulk (mass-weighted 2-98%) so a few
    ejected particles leave the box instead of shrinking everything else."""
    xyz = history[cols].to_numpy()
    if robust and len(xyz) > 50:
        w = history['mass'].to_numpy()
        lo, hi = [], []
        for d in range(xyz.shape[1]):
            order = np.argsort(xyz[:, d])
            cdf = np.cumsum(w[order]) / w.sum()
            lo.append(xyz[order[np.searchsorted(cdf, 0.02)], d])
            hi.append(xyz[order[min(np.searchsorted(cdf, 0.98), len(order) - 1)], d])
        lo, hi = np.array(lo), np.array(hi)
        pad = 0.25
    else:
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        pad = 0.08
    span = np.maximum(hi - lo, 1e-9)
    return lo - pad * span, hi + pad * span


def _event_times(result):
    ev = result.events
    return np.sort(ev.time.to_numpy()) if not ev.empty else np.empty(0)


def _colorer(history, color_by):
    color_by = S.auto_color_by(history) if color_by == 'auto' else color_by
    colors, _, cmap, label = S.color_values(history.iloc[:1], color_by)
    norm = S.make_norm(history[color_by], color_by) if cmap is not None else None

    def colors_for(frame):
        if cmap is None:
            return list(S.color_values(frame, color_by)[0])
        return cmap(norm(frame[color_by].to_numpy()))
    return color_by, cmap, norm, colors_for


def _output(fig, ani, save, fps, frame_format):
    if save:
        writer = PillowWriter(fps=fps) if str(save).endswith('.gif') else FFMpegWriter(fps=fps)
        ani.save(str(save), writer=writer)
    try:
        from IPython.display import HTML
    except ImportError:
        return ani
    # JPEG frames keep the embedded player ~3x smaller than PNG
    with mpl.rc_context({'animation.frame_format': frame_format,
                         'animation.embed_limit': 60.0}):
        html = HTML(ani.to_jshtml(fps=fps))
    plt.close(fig)
    return html


def _sizer(history, span, axis_pt, size_by):
    """Marker areas from true radii, or from mass when radii are too small to see."""
    if size_by == 'auto':
        true = S.marker_sizes(history.radius.to_numpy(), span, axis_pt, floor=0.0)
        size_by = 'radius' if np.median(true) >= 12.0 else 'mass'
    m_max = float(history.mass.max())

    def sizes(frame):
        if size_by == 'radius':
            return S.marker_sizes(frame.radius.to_numpy(), span, axis_pt)
        return 8.0 + 70.0 * (frame.mass.to_numpy() / m_max) ** (2.0 / 3.0)
    return sizes


def _trail_positions(history, frames, cols):
    """(frames, particles, d) positions with NaN where a particle is absent."""
    ids = np.unique(np.concatenate([f.particle_id.to_numpy() for _, f in frames]))
    index = {pid: k for k, pid in enumerate(ids)}
    arr = np.full((len(frames), len(ids), len(cols)), np.nan)
    for t, (_, f) in enumerate(frames):
        rows = np.array([index[p] for p in f.particle_id.to_numpy()])
        arr[t, rows] = f[cols].to_numpy()
    return arr


def animate_2d(result, max_frames: int = 120, interval: int = 70, *, color_by: str = 'auto',
               plane: str = 'xy', trail: int = 0, robust_limits: bool = True,
               figsize=(6.4, 6.4), dpi: int = 90, save: str | None = None, fps: int = 15,
               frame_format: str = 'jpeg', size_by: str = 'auto'):
    """Animated 2D view. ``color_by``: 'auto', 'cluster', 'charge', 'polarity_now',
    'mass' or 'species'. ``trail`` draws each particle's last N frames. ``size_by``:
    'radius' (true size), 'mass', or 'auto' (mass when radii would be invisible)."""
    h = result.history
    cols = list(plane)
    if not set(cols).issubset(h.columns):
        raise ValueError(f'plane {plane!r} not available in a {result.dimension}D result')
    frames = _frames(h, max_frames)
    lo, hi = _limits(h, cols, robust_limits)
    color_by, cmap, norm, colors_for = _colorer(h, color_by)
    events = _event_times(result)
    trails = _trail_positions(h, frames, cols) if trail else None
    with S.themed():
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel=cols[0], ylabel=cols[1])
        ax.set_aspect('equal', adjustable='box')
        span = float(max(hi - lo))
        axis_pt = figsize[0] * 72 * 0.78
        sizes = _sizer(h, span, axis_pt, size_by)
        lc = LineCollection([], colors=S.MUTED, linewidths=0.7, alpha=0.45)
        ax.add_collection(lc)
        scat = ax.scatter([], [], s=[], linewidths=0.5, edgecolors=S.SURFACE, alpha=0.9)
        if cmap is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label=color_by)
        title = ax.set_title('')

        def update(k):
            t, f = frames[k]
            scat.set_offsets(f[cols].to_numpy())
            scat.set_sizes(sizes(f))
            scat.set_facecolors(colors_for(f))
            if trails is not None:
                seg = trails[max(0, k - trail):k + 1].transpose(1, 0, 2)
                lc.set_segments([s[~np.isnan(s).any(axis=1)] for s in seg])
            n_ev = int(np.searchsorted(events, t, side='right'))
            title.set_text(f't = {t:.2f}   live = {len(f)}   events = {n_ev}')
            return scat, lc, title

        ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
        return _output(fig, ani, save, fps, frame_format)


def animate_3d(result, max_frames: int = 120, interval: int = 70, *, color_by: str = 'auto',
               rotate: float = 90.0, elev: float = 22.0, azim: float = -60.0, trail: int = 0,
               robust_limits: bool = True, figsize=(7.0, 6.6), dpi: int = 90,
               save: str | None = None, fps: int = 15, frame_format: str = 'jpeg',
               size_by: str = 'auto'):
    """Animated 3D view with a slowly orbiting camera (``rotate`` degrees in total)."""
    h = result.history
    if 'z' not in h:
        raise ValueError('animate_3d needs a 3D simulation (dimension=3); use animate_2d')
    cols = ['x', 'y', 'z']
    frames = _frames(h, max_frames)
    lo, hi = _limits(h, cols, robust_limits)
    color_by, cmap, norm, colors_for = _colorer(h, color_by)
    events = _event_times(result)
    trails = _trail_positions(h, frames, cols) if trail else None
    with S.themed():
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(projection='3d')
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel='x', ylabel='y', zlabel='z')
        ax.set_box_aspect(np.maximum((hi - lo) / float(max(hi - lo)), 0.45), zoom=1.0)
        fig.subplots_adjust(left=0.0, right=0.9 if cmap is not None else 0.96, top=0.95, bottom=0.02)
        S.style_3d(ax)
        span = float(max(hi - lo))
        axis_pt = figsize[0] * 72 * 0.62
        sizes = _sizer(h, span, axis_pt, size_by)
        if cmap is not None:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.08, shrink=0.7, label=color_by)
        state = {'scat': None, 'lines': []}
        title = ax.set_title('')

        def update(k):
            t, f = frames[k]
            n_live = len(f)
            if state['scat'] is not None:
                state['scat'].remove()
            for ln in state['lines']:
                ln.remove()
            state['lines'] = []
            if trails is not None:
                seg = trails[max(0, k - trail):k + 1]
                for p in range(seg.shape[1]):
                    path = seg[:, p]
                    path = path[~np.isnan(path).any(axis=1)]
                    if len(path) > 1:
                        state['lines'] += ax.plot(*path.T, color=S.MUTED, linewidth=0.6, alpha=0.4)
            xyz = f[cols].to_numpy()
            inside = np.all((xyz >= lo) & (xyz <= hi), axis=1)
            f = f[inside]
            xyz = xyz[inside]
            state['scat'] = ax.scatter(*xyz.T, s=sizes(f),
                                       c=colors_for(f), linewidths=0.4, edgecolors=S.SURFACE,
                                       alpha=0.9, depthshade=True)
            ax.view_init(elev=elev, azim=azim + rotate * k / max(len(frames) - 1, 1))
            n_ev = int(np.searchsorted(events, t, side='right'))
            title.set_text(f't = {t:.2f}   live = {n_live}   events = {n_ev}')
            return (state['scat'],)

        ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
        return _output(fig, ani, save, fps, frame_format)
