"""Population presets: dimensionless, editable distributions for initial states.

``quantum``, ``em``, ``macro`` and ``astro`` are *parameter regimes* for a
classical toy simulator, not faithful physical solvers. Use
``dataclasses.replace(PRESETS['em'], ...)`` to explore a variation without
editing the defaults.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ['ParticlePreset', 'PRESETS', 'get_preset', 'positive_lognormal']


@dataclass(frozen=True)
class ParticlePreset:
    """Distributions used by :meth:`ParticleSystem.random`.

    Masses and radii are lognormal (positive, no pile-up at zero), charges
    are normal, spins and polarities are drawn from discrete state sets.
    ``expansion_rate`` belongs to the environment; pass it to
    :class:`SpaceExpansion`.
    """
    name: str = 'classical'
    mass_mean: float = 1.0
    mass_std: float = 0.28
    charge_mean: float = 0.0
    charge_std: float = 0.0
    radius_mean: float = 0.045
    radius_std: float = 0.0
    spin_states: tuple[float, ...] = (0.0,)
    polarity_states: tuple[float, ...] = (-1.0, 1.0)
    constituents_mean: float = 3.0          # average number of sub-particles
    n_clusters: int = 1
    cluster_center_std: float = 0.0
    cluster_std: float = 0.7
    velocity_std: float = 0.1
    expansion_rate: float = 0.0             # inverse simulation time

    def __post_init__(self):
        if self.mass_mean <= 0 or self.radius_mean <= 0:
            raise ValueError('mass_mean and radius_mean must be positive')
        for name in ('mass_std', 'charge_std', 'radius_std', 'cluster_center_std',
                     'cluster_std', 'velocity_std'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative')
        if self.n_clusters < 1 or len(self.spin_states) < 1 or len(self.polarity_states) < 1:
            raise ValueError('Need >=1 cluster, spin state and polarity state')
        if self.constituents_mean < 1:
            raise ValueError('constituents_mean must be >= 1')
        if not np.isfinite([self.mass_mean, self.charge_mean, self.radius_mean,
                            self.expansion_rate]).all():
            raise ValueError('Preset parameters must be finite')
        if not (np.isfinite(self.spin_states).all() and np.isfinite(self.polarity_states).all()):
            raise ValueError('spin_states and polarity_states must be finite')


# Illustrative regimes for exploring rules; NOT measured physics.
PRESETS: dict[str, ParticlePreset] = {
    'classical': ParticlePreset(),
    'quantum': ParticlePreset(
        name='quantum', mass_mean=0.5, mass_std=0.15,
        charge_mean=0, charge_std=0.75, radius_mean=0.018, radius_std=0.005,
        spin_states=(-0.5, 0.5), polarity_states=(-1.0, 1.0), constituents_mean=3,
        n_clusters=10, cluster_center_std=1.3, cluster_std=0.25,
        velocity_std=0.25, expansion_rate=0.0,
    ),
    'em': ParticlePreset(
        name='em', mass_mean=2.0, mass_std=0.5,
        charge_mean=0, charge_std=1.2, radius_mean=0.06, radius_std=0.015,
        spin_states=(-1.0, 0.0, 1.0), polarity_states=(-1.0, -0.5, 0.5, 1.0),
        constituents_mean=4, n_clusters=10, cluster_center_std=2.0,
        cluster_std=0.35, velocity_std=0.15, expansion_rate=0.0,
    ),
    'macro': ParticlePreset(
        name='macro', mass_mean=12.0, mass_std=3.0,
        charge_mean=0, charge_std=0.02, radius_mean=0.12, radius_std=0.025,
        spin_states=(0.0,), constituents_mean=6,
        n_clusters=6, cluster_center_std=2.5, cluster_std=0.45,
        velocity_std=0.12, expansion_rate=0.0,
    ),
    'astro': ParticlePreset(
        name='astro', mass_mean=100.0, mass_std=25.0,
        charge_mean=0, charge_std=0.0, radius_mean=0.15, radius_std=0.03,
        spin_states=(0.0,), constituents_mean=8,
        n_clusters=5, cluster_center_std=3.0, cluster_std=0.65,
        velocity_std=0.08, expansion_rate=0.02,
    ),
}


def get_preset(preset: str | ParticlePreset) -> ParticlePreset:
    if isinstance(preset, ParticlePreset):
        return preset
    if isinstance(preset, str):
        try:
            return PRESETS[preset.lower()]
        except KeyError:
            raise KeyError(f'Unknown preset {preset!r}; choose from {sorted(PRESETS)}') from None
    raise TypeError('preset must be a ParticlePreset or a preset name')


def positive_lognormal(rng: np.random.Generator, mean: float, std: float, n: int) -> np.ndarray:
    """Positive draws matching the requested mean and std (lognormal moments)."""
    if std == 0:
        return np.full(n, float(mean))
    sigma2 = np.log1p((std / mean) ** 2)
    mu = np.log(mean) - 0.5 * sigma2
    return rng.lognormal(mu, np.sqrt(sigma2), size=n)
