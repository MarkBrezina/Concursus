"""Regenerate the README figures:  python docs/make_figures.py"""
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

from stochastics.physics import (PRESETS, Cluster, Collisions, CompositionAffinity, Environment,
                                 Gravity, ParticleSystem, animate_3d, plot_collision_regimes,
                                 plot_dashboard, simulate)

HERE = Path(__file__).parent
G = 1e-4
astro_cfg = replace(PRESETS['astro'], radius_mean=0.003, radius_std=0.0008, velocity_std=0.03)
particles = ParticleSystem.random(
    n=400, preset=astro_cfg, seed=210, dimension=3,
    clusters=[Cluster(mean=[-3.5, -1.0, 0.0], std=[1.0, 1.0, 0.15], orbital_G=G, rotation=+1,
                      velocity=[0.2, 0.12, 0.0]),
              Cluster(mean=[3.5, 1.0, 0.3], std=[0.8, 0.8, 0.12], orbital_G=G,
                      rotation=[0.4, 0, -1], weight=0.8, velocity=[-0.2, -0.12, 0.0])])
env = Environment(law=Gravity(G=G, softening=0.12), rules=[Collisions(CompositionAffinity())])
result = simulate(particles, env, steps=3200, dt=0.01, record_every=20)

plot_dashboard(result, show=False).savefig(HERE / 'dashboard.png', dpi=80)
plot_collision_regimes(max_speed=1.2, show=False, examples=[
    ('5+ vs 5+', (5, 0), (5, 0)), ('3+2− vs 3+2−', (3, 2), (3, 2)),
    ('3+2− vs 0+5−', (3, 2), (0, 5)), ('5+ vs 5−', (5, 0), (0, 5))]).savefig(
    HERE / 'collision_regimes.png', dpi=90)
animate_3d(result, max_frames=60, dpi=72, save=str(HERE / 'galaxies_3d.gif'), fps=12)
print('figures written to', HERE)
