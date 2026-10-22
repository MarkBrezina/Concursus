"""Spin / polarity clocks.

Every particle carries a clock: ``spin`` is its speed (theta-like), ``polarity``
its signed amplitude (charge-like), ``phase`` its position in the cycle. The
value the external policy sees is

    polarity_now = polarity * waveform(phase),   phase += spin * dt / 4 per step

With polarity = -1, spin = 0.5 and dt = 1 the triangle waveform produces
-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, ...
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .. import _style as S
from ..state import ParticleSystem, waveform

__all__ = ['polarity_sequence', 'plot_polarity_clocks']


def polarity_sequence(polarity: float, spin: float, steps: int, dt: float = 1.0,
                      phase: float = 0.0, kind: str = 'triangle') -> np.ndarray:
    """``polarity_now`` at t = 0, dt, ..., steps*dt for a single clock."""
    t = np.arange(steps + 1) * dt
    return polarity * waveform(phase + spin * t / 4.0, kind) + 0.0  # + 0.0 drops -0.0


def plot_polarity_clocks(particles: ParticleSystem, *, duration: float = 16.0, dt: float = 0.05,
                         max_particles: int = 6, show: bool = True):
    """Plot the next ``duration`` time units of ``polarity_now`` for a few particles."""
    live = particles.live[:max_particles]
    t = np.arange(0, duration + dt / 2, dt)
    with S.themed():
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.axhline(0, color=S.MUTED, linewidth=0.8)
        for k, slot in enumerate(live):
            values = particles.polarity[slot] * waveform(
                particles.phase[slot] + particles.spin[slot] * t / 4.0, particles.waveform)
            ax.plot(particles.time + t, values, color=S.SERIES[k % len(S.SERIES)],
                    label=f'id {particles.ids[slot]}: spin {particles.spin[slot]:g}, '
                          f'polarity {particles.polarity[slot]:g}')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0))
        ax.set(title=f'Polarity clocks ({particles.waveform} waveform)', xlabel='time',
               ylabel='polarity_now')
        fig.tight_layout()
    if show:
        plt.show()
        return None
    return fig
