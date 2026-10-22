"""The environment: a force law, per-step rules, and background conditions.

    env = Environment(law=Gravity(), rules=[Collisions(), Drag(0.05)],
                      expansion=SpaceExpansion(rate=0.02))
    result = env.run(particles, steps=10_000, dt=0.01, record_every=10)

A *rule* is anything with ``apply(state, time, ctx) -> list[dict]`` (events).
Rules that move particles, change charges/polarities, or add/remove particles
must call ``state.touch()`` (the mechanics helpers do), so the engine can
re-evaluate forces and book the potential-energy jump into the rule's
``ledger`` channel.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .forces import as_law
from .state import ParticleSystem

__all__ = ['Rule', 'StepContext', 'Environment', 'SpaceExpansion', 'Drag', 'Friction',
           'PolarityClock', 'advance_clocks']


@dataclass
class StepContext:
    """What a rule can see during a step."""
    dt: float
    step: int
    law: Any
    rng: np.random.Generator
    previous_position: np.ndarray        # positions at the start of the step
    settle: Callable[[str], None]        # book potential change so far into a ledger channel


class Rule:
    """Base class for per-step rules. Override :meth:`apply`."""
    ledger: str = 'binding'

    def apply(self, state: ParticleSystem, time: float, ctx: StepContext) -> list[dict]:
        return []


def advance_clocks(state: ParticleSystem, dt: float, speed: float = 1.0) -> bool:
    """Advance every live particle's spin/polarity clock by one step.

    ``spin`` is the rate at which the *normalised* polarity moves per unit
    time, so the phase advances by ``speed * spin * dt / 4`` cycles (a full
    cycle sweeps the normalised value through a distance of 4). With
    ``polarity=-1, spin=0.5, dt=1`` the triangle waveform gives
    -1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, ...
    """
    live = state.live
    rate = state.spin[live]
    if not np.any(rate):
        return False
    state.phase[live] = np.mod(state.phase[live] + speed * rate * dt / 4.0, 1.0)
    state.touch()
    return True


@dataclass
class PolarityClock(Rule):
    """Rule form of :func:`advance_clocks`; energy pumped in/out is booked as ``drive``."""
    speed: float = 1.0
    ledger = 'drive'

    def apply(self, state, time, ctx):
        advance_clocks(state, ctx.dt, self.speed)
        return []


@dataclass
class Drag(Rule):
    """Linear drag (friction with a background medium): v <- v * exp(-gamma dt).

    Exact for the drag term in isolation (operator splitting). Lost kinetic
    energy goes to the ``dissipated`` ledger, or into the particles' internal
    energy with ``heat_particles=True``. With a conservative law, drag turns
    the dynamics into a descent on the potential: particles settle into
    minima, i.e. they *optimise*.
    """
    gamma: float = 0.1
    heat_particles: bool = False
    ledger = 'dissipated'

    def apply(self, state, time, ctx):
        if self.gamma < 0:
            raise ValueError('gamma must be nonnegative')
        if self.gamma == 0:
            return []
        live = state.live
        f = np.exp(-self.gamma * ctx.dt)
        v = state.velocity[live]
        lost = 0.5 * state.mass[live] * np.einsum('ij,ij->i', v, v) * (1.0 - f * f)
        state.velocity[live] = v * f
        if self.heat_particles:
            state.internal_energy[live] += lost
        else:
            state.ledger.dissipated += float(lost.sum())
        return []


Friction = Drag


@dataclass(frozen=True)
class SpaceExpansion:
    """Toy homogeneous expansion: separations scale by exp(H dt) each step.

    Velocities are *peculiar* velocities and are left unchanged. Potential
    energy changes caused by the stretch are booked to the ``expansion``
    ledger. Not FLRW cosmology: bound structures are stretched too.
    """
    rate: float = 0.0
    anchor: str = 'center_of_mass'  # or 'origin'

    def apply(self, state: ParticleSystem, dt: float) -> None:
        if self.anchor not in ('center_of_mass', 'origin'):
            raise ValueError('anchor must be center_of_mass or origin')
        if not np.isfinite(self.rate) or dt < 0 or not np.isfinite(dt):
            raise ValueError('expansion rate and dt must be finite; dt nonnegative')
        live = state.live
        if len(live) == 0 or self.rate == 0 or dt == 0:
            return
        if self.anchor == 'center_of_mass':
            center = np.average(state.position[live], axis=0, weights=state.mass[live])
        else:
            center = np.zeros(state.dimension)
        factor = np.exp(self.rate * dt)
        if not np.isfinite(factor):
            raise OverflowError('expansion rate * dt is too large')
        state.position[live] = center + (state.position[live] - center) * factor
        state.touch()


@dataclass
class Environment:
    """Bundle of law + rules + background. ``law`` may be a list of laws."""
    law: Any = None
    rules: list = field(default_factory=list)
    expansion: SpaceExpansion | None = None

    def __post_init__(self):
        self.law = as_law(self.law)
        self.rules = list(self.rules or [])

    def run(self, particles: ParticleSystem, **kwargs):
        from .engine import simulate
        return simulate(particles, self, **kwargs)

    def __repr__(self) -> str:
        rules = ', '.join(type(r).__name__ for r in self.rules) or 'none'
        exp = f'H={self.expansion.rate}' if self.expansion else 'none'
        return f'Environment(law={type(self.law).__name__}, rules=[{rules}], expansion={exp})'
