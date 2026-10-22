"""Shared visual language for plots and animations (light theme).

Categorical slots are used in a fixed order and never cycled: cluster k gets
slot k; anything past eight clusters folds into a neutral grey. Charge and
polarity use a blue <-> red diverging map with a grey midpoint; magnitudes
(mass) use a single-hue blue ramp.
"""
from __future__ import annotations

import contextlib

import matplotlib as mpl
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, TwoSlopeNorm

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
MUTED = '#8f8e89'
GRID = '#e6e5e1'
OTHER = '#a3a29d'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
DIVERGING = LinearSegmentedColormap.from_list(
    'stochastics_diverging', ['#104281', '#2a78d6', '#86b6ef', '#c9c8c3', '#f09a9a', '#e34948', '#a52a2a'])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'stochastics_sequential', ['#b7d3f6', '#6da7ec', '#2a78d6', '#1c5cab', '#0d366b'])

# Energy-ledger series keep the same colour everywhere (colour follows the entity).
LEDGER_COLORS = {
    'kinetic': SERIES[0], 'potential': SERIES[1], 'internal': SERIES[2], 'radiated': SERIES[3],
    'dissipated': SERIES[4], 'binding': SERIES[5], 'expansion': SERIES[6], 'drive': SERIES[7],
    'accounted_energy': TEXT,
}
EVENT_COLORS = {'merge': SERIES[1], 'inelastic': SERIES[2], 'elastic': SERIES[0],
                'emit': SERIES[3], 'pass': OTHER}

RC = {
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.labelcolor': TEXT_2,
    'axes.titlecolor': TEXT, 'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left', 'axes.titlepad': 8, 'axes.labelsize': 9,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-',
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.color': TEXT_2, 'ytick.color': TEXT_2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'xtick.major.size': 0, 'ytick.major.size': 0, 'legend.frameon': False,
    'legend.fontsize': 8, 'legend.labelcolor': TEXT_2, 'lines.linewidth': 1.6,
    'lines.solid_capstyle': 'round', 'font.size': 9, 'text.color': TEXT,
    'axes.prop_cycle': mpl.cycler(color=list(SERIES)),
}


@contextlib.contextmanager
def themed():
    with mpl.rc_context(RC):
        yield


def cluster_colors(cluster_ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(cluster_ids, dtype=int)
    out = np.empty(len(ids), dtype=object)
    out[:] = OTHER
    for k, color in enumerate(SERIES):
        out[ids == k] = color
    return out


def color_values(frame, color_by: str):
    """Return (colors, norm, cmap, label) for a frame-like table."""
    if color_by == 'cluster':
        return cluster_colors(frame['cluster_id']), None, None, 'cluster'
    if color_by in ('charge', 'polarity', 'polarity_now'):
        return None, None, DIVERGING, color_by
    if color_by == 'mass':
        return None, None, SEQUENTIAL, 'mass'
    if color_by == 'species':
        species = frame['species'].astype(str)
        uniq = sorted(set(species))
        lut = {s: SERIES[k] if k < len(SERIES) else OTHER for k, s in enumerate(uniq)}
        return np.array([lut[s] for s in species], dtype=object), None, None, 'species'
    raise ValueError("color_by must be 'cluster', 'charge', 'polarity_now', 'mass' or 'species'")


def make_norm(values: np.ndarray, color_by: str):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if color_by in ('charge', 'polarity', 'polarity_now'):
        lim = float(np.percentile(np.abs(finite), 98)) if len(finite) else 1.0
        lim = lim if lim > 0 else 1.0
        return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    if color_by == 'mass':
        lo, hi = (float(finite.min()), float(finite.max())) if len(finite) else (1.0, 1.0)
        if lo > 0 and hi / lo > 20:
            return LogNorm(lo, hi)
        return Normalize(lo, hi if hi > lo else lo + 1.0)
    return None


def auto_color_by(history) -> str:
    n_clusters = history.loc[history.cluster_id >= 0, 'cluster_id'].nunique()
    if n_clusters > 1:
        return 'cluster'
    if 'charge' in history and np.any(np.abs(history['charge']) > 1e-12):
        return 'charge'
    if 'polarity_now' in history and np.any(np.abs(history['polarity_now']) > 1e-12):
        return 'polarity_now'
    return 'mass'


def marker_sizes(radius: np.ndarray, span: float, axis_points: float, floor: float = 7.0,
                 cap: float = 900.0) -> np.ndarray:
    """Marker area (pt^2) so a particle's drawn diameter matches its radius on screen."""
    diameter_pt = 2.0 * np.asarray(radius, float) / max(span, 1e-12) * axis_points
    return np.clip(diameter_pt ** 2, floor, cap)


def style_3d(ax) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(mpl.colors.to_rgba(SURFACE, 1.0))
        axis._axinfo['grid'].update(color=GRID, linewidth=0.6)
        axis.line.set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=7, pad=0)
