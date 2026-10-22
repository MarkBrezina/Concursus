"""policy_internal — what happens inside a particle, and when two particles touch.

An internal policy has three hooks; override any of them::

    evolve(state, t, dt, ctx) -> events   every step: clocks, decay, emission
    decide(state, contact, t) -> Outcome  on contact: merge / bounce(e) / pass
    combine(state, i, j, t)   -> dict     internal state of a merged particle

Policies only *decide*; the mechanics (:mod:`stochastics.physics.mechanics`)
execute merges, bounces and emissions so that mass, charge, momentum and
constituents are conserved and every unit of energy lands in the ledger.
That is what makes it safe to experiment here: a policy can be wrong about
physics, but it cannot silently break the books.

Pass it as ``simulate(particles, internal=policy_internal, ...)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..collisions import CompositionAffinity, Contact, Outcome
from ..environment import advance_clocks
from ..mechanics import emit
from ..state import ParticleSystem

__all__ = ['InternalPolicy', 'ThermalEmission']


@dataclass
class InternalPolicy(CompositionAffinity):
    """Base ``policy_internal``: spin/polarity clocks + the composition contact rule.

    Inherits the outcome gradient of :class:`CompositionAffinity`
    (elastic -> inelastic -> merge as the parts' affinity rises) so the
    default already gives elastic and inelastic collisions, capture and
    radiation. ``clock_speed=0`` freezes the clocks.
    """
    clock_speed: float = 1.0

    def evolve(self, state: ParticleSystem, t: float, dt: float, ctx) -> list[dict]:
        if self.clock_speed:
            advance_clocks(state, dt, self.clock_speed)
        return []

    def decide(self, state: ParticleSystem, contact: Contact, t: float) -> Outcome:
        return super().decide(state, contact, t)

    def combine(self, state: ParticleSystem, i: int, j: int, t: float) -> dict:
        return {}


@dataclass
class ThermalEmission(InternalPolicy):
    """Example policy: hot composites shed constituents; others absorb them.

    * Collisions heat particles (inelastic heat -> internal energy).
    * When internal energy per unit mass exceeds ``threshold`` and a particle
      has >= 2 constituents, it emits the constituent least bound to the rest,
      releasing ``release`` of its excess energy as kinetic energy (with recoil).
    * An emitted quantum that is captured by another particle is logged as
      ``absorb`` instead of ``merge``.

    Emission + absorption + inelastic heating form a closed energy cycle that
    the ledger tracks exactly.
    """
    threshold: float = 0.02
    release: float = 0.5
    max_emissions_per_step: int = 8
    quantum_species: str = 'quantum'

    def evolve(self, state, t, dt, ctx):
        events = super().evolve(state, t, dt, ctx)
        live = state.live
        hot = live[(state.internal_energy[live] > self.threshold * state.mass[live])
                   & (state.n_constituents[live] >= 2)]
        if not len(hot):
            return events
        hot = hot[np.argsort(-state.internal_energy[hot] / state.mass[hot])]
        K = self.matrix(state)
        for slot in hot[:self.max_emissions_per_step]:
            comp = state.composition[slot]
            present = np.flatnonzero(comp > 0)
            # bond of one constituent of type k to the rest of the particle
            bonds = np.array([K[k] @ (comp - np.eye(len(comp), dtype=int)[k]) for k in present])
            k = int(present[np.argmin(bonds)])
            parts = np.zeros_like(comp)
            parts[k] = 1
            excess = state.internal_energy[slot] - self.threshold * state.mass[slot]
            direction = ctx.rng.normal(size=state.dimension)
            events.append(emit(state, int(slot), energy=self.release * excess, direction=direction,
                               time=t, parts=parts, species=self.quantum_species))
        return events

    def decide(self, state, contact, t):
        outcome = super().decide(state, contact, t)
        if outcome.kind == 'merge' and self.quantum_species in (state.species[contact.i],
                                                                state.species[contact.j]):
            return Outcome.merge(outcome.radiated_fraction, label='absorb', **outcome.info)
        return outcome
