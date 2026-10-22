"""Shared plotting helpers for ``baseline`` and ``popdyn`` (light theme).

Kept separate from ``physics._style`` so each subpackage stays independent:
breaking one never stops another from importing or plotting.

Categorical colours are used in a fixed order (never cycled); identity
past eight series folds into grey. Diverging maps are blue <-> red with a grey
midpoint; magnitudes use a single-hue blue ramp.
"""
from __future__ import annotations

import contextlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = '#fcfcfb'
TEXT = '#0b0b0b'
TEXT_2 = '#52514e'
MUTED = '#8f8e89'
GRID = '#e6e5e1'
OTHER = '#a3a29d'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES
DIVERGING = LinearSegmentedColormap.from_list(
    'stochastics_div', ['#104281', '#2a78d6', '#86b6ef', '#c9c8c3', '#f09a9a', '#e34948', '#a52a2a'])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    'stochastics_seq', ['#e9f1fc', '#b7d3f6', '#6da7ec', '#2a78d6', '#1c5cab', '#0d366b'])

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


def finish(fig, show: bool):
    """Show and return None (avoids double display in notebooks), or return the figure."""
    if show:
        plt.show()
        return None
    return fig


def suptitle(fig, text: str, y: float = 0.995):
    fig.suptitle(text, x=0.01, ha='left', fontsize=12, fontweight='bold', color=TEXT, y=y)


def display_table(df) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import display
        if get_ipython() is not None:
            import pandas as pd
            with pd.option_context('display.max_colwidth', None):
                display(df)
            return
    except ImportError:
        pass
    print(df.to_string())


def to_html_animation(fig, ani, save=None, fps: int = 12):
    """Embed an animation as a JPEG-frame HTML player (or return it outside IPython)."""
    from matplotlib.animation import FFMpegWriter, PillowWriter
    if save:
        writer = PillowWriter(fps=fps) if str(save).endswith('.gif') else FFMpegWriter(fps=fps)
        ani.save(str(save), writer=writer)
    try:
        from IPython.display import HTML
    except ImportError:
        return ani
    with mpl.rc_context({'animation.frame_format': 'jpeg', 'animation.embed_limit': 60.0}):
        html = HTML(ani.to_jshtml(fps=fps))
    plt.close(fig)
    return html


def robust_limits(values: np.ndarray, lo_q: float = 0.5, hi_q: float = 99.5, pad: float = 0.05):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return -1.0, 1.0
    lo, hi = np.percentile(v, [lo_q, hi_q])
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    span = hi - lo
    return lo - pad * span, hi + pad * span
