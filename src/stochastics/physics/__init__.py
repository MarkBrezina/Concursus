"""stochastics.physics — classical particle simulations with collisions and an energy ledger.

Quick start::

    from stochastics.physics import *

    particles = ParticleSystem.random(
        n=300, preset='astro', seed=210,
        clusters=[Cluster(mean=[0, 0, 0], std=1.0, rotation=0.4),
                  Cluster(mean=[4, -2, 0], std=0.8, velocity=[-0.3, 0.15, 0])])
    env = Environment(law=Gravity(G=0.002), rules=[Collisions()])
    result = simulate(particles, env, steps=2000, dt=0.01, record_every=10)

    report(result)                    # summary table + dashboard
    animate_3d(result, max_frames=100)

The experimental policy lab lives in :mod:`stochastics.physics.policies`.
"""
from .presets import PRESETS, ParticlePreset, get_preset, positive_lognormal
from .state import (Cluster, EnergyLedger, Particle, ParticleSystem, ParticleView, waveform,
                    WAVEFORMS)
from .forces import (CombinedForce, Coulomb, ForceLaw, Gaussian, Gravity, Kernel, NoForce,
                     PairInteraction, Plummer, Spring, Yukawa, as_law, evaluate_law)
from .mechanics import bounce, emit, merge
from .environment import (Drag, Environment, Friction, PolarityClock, Rule, SpaceExpansion,
                          StepContext, advance_clocks)
from .collisions import (Collisions, CompositionAffinity, Contact, ContactPolicy,
                         MergeWithinDistance, Outcome, find_contacts)
from .engine import SimulationResult, measure, simulate, snapshot
from .plotting import (plot_collision_regimes, plot_conservation, plot_dashboard, plot_initial,
                       plot_tracks, plot_velocities, report)
from .animation import animate_2d, animate_3d

from . import policies  # noqa: E402  (the experimental policy lab)

ParticleConfig = Particle  # alias matching the original design sketch

__all__ = [
    'PRESETS', 'ParticlePreset', 'get_preset', 'positive_lognormal',
    'Cluster', 'EnergyLedger', 'Particle', 'ParticleConfig', 'ParticleSystem', 'ParticleView',
    'waveform', 'WAVEFORMS',
    'CombinedForce', 'Coulomb', 'ForceLaw', 'Gaussian', 'Gravity', 'Kernel', 'NoForce',
    'PairInteraction', 'Plummer', 'Spring', 'Yukawa', 'as_law', 'evaluate_law',
    'bounce', 'emit', 'merge',
    'Drag', 'Environment', 'Friction', 'PolarityClock', 'Rule', 'SpaceExpansion', 'StepContext',
    'advance_clocks',
    'Collisions', 'CompositionAffinity', 'Contact', 'ContactPolicy', 'MergeWithinDistance',
    'Outcome', 'find_contacts',
    'SimulationResult', 'measure', 'simulate', 'snapshot',
    'plot_collision_regimes', 'plot_conservation', 'plot_dashboard', 'plot_initial',
    'plot_tracks', 'plot_velocities', 'report',
    'animate_2d', 'animate_3d', 'policies',
]
