"""Conservation-guaranteeing primitives: merge, bounce, emit.

Policies *decide* what happens; these functions *execute* it. Each one
conserves mass, charge, linear momentum and constituent counts exactly, and
books every unit of kinetic energy it removes or creates into the particles'
internal energy or the system ledger. Potential-energy jumps caused by moving
or replacing particles are booked by the engine (``ctx.settle``).
"""
from __future__ import annotations

import numpy as np

from .state import ParticleSystem

__all__ = ['merge', 'bounce', 'emit']


def merge(state: ParticleSystem, i: int, j: int, *, time: float,
          radiated_fraction: float = 0.0, overrides: dict | None = None,
          label: str = 'merge') -> dict:
    """Perfectly inelastic union of slots i and j into a new particle.

    Relative kinetic energy ``0.5 * mu * |v_i - v_j|^2`` becomes heat; a
    ``radiated_fraction`` of it leaves through the radiation ledger, the rest
    is stored as the child's internal energy.

    Default conventions for internal state (override via ``overrides``):
    charge, polarity and composition add; spin is mass-weighted; phase,
    cluster and identity of the heavier parent are inherited.
    """
    if not 0.0 <= radiated_fraction <= 1.0:
        raise ValueError('radiated_fraction must be in [0, 1]')
    mi, mj = float(state.mass[i]), float(state.mass[j])
    M = mi + mj
    mu = mi * mj / M
    dv = state.velocity[i] - state.velocity[j]
    heat = 0.5 * mu * float(dv @ dv)
    radiated = radiated_fraction * heat
    heavy = i if mi >= mj else j
    d = state.dimension
    child = dict(
        species='composite', mass=M,
        charge=float(state.charge[i] + state.charge[j]),
        radius=float((state.radius[i] ** d + state.radius[j] ** d) ** (1.0 / d)),
        spin=float((mi * state.spin[i] + mj * state.spin[j]) / M),
        polarity=float(state.polarity[i] + state.polarity[j]),
        phase=float(state.phase[heavy]),
        position=(mi * state.position[i] + mj * state.position[j]) / M,
        velocity=(mi * state.velocity[i] + mj * state.velocity[j]) / M,
        internal_energy=float(state.internal_energy[i] + state.internal_energy[j] + heat - radiated),
        composition=state.composition[i] + state.composition[j],
        cluster_id=int(state.cluster_id[heavy]),
    )
    if overrides:
        unknown = set(overrides) - set(child)
        if unknown:
            raise KeyError(f'cannot override {sorted(unknown)} on merge')
        child.update(overrides)
    ids = int(state.ids[i]), int(state.ids[j])
    distance = float(np.linalg.norm(state.position[i] - state.position[j]))
    state.deactivate(i)
    state.deactivate(j)
    child_id = state.add_particle(**child)
    state.ledger.radiated += radiated
    return {'time': time, 'event': label, 'a': ids[0], 'b': ids[1], 'child': child_id,
            'distance': distance, 'heat': heat, 'radiated': radiated,
            'mass_a': mi, 'mass_b': mj}


def bounce(state: ParticleSystem, i: int, j: int, normal: np.ndarray, restitution: float, *,
           time: float, radiated_fraction: float = 0.0, remaining_time: float = 0.0,
           label: str | None = None) -> dict | None:
    """Impulse along ``normal`` (unit, pointing from j to i).

    ``restitution`` e in [0, 1]: 1 is elastic, 0 sticks along the normal.
    Heat ``0.5 * mu * v_n^2 * (1 - e^2)`` goes to internal energy (split by
    mass) minus the radiated fraction. ``remaining_time`` re-advances the
    pair along the post-collision path (continuous collision detection).
    Returns ``None`` if the pair is already separating.
    """
    if not 0.0 <= restitution <= 1.0:
        raise ValueError('restitution must be in [0, 1]')
    n = np.asarray(normal, dtype=float)
    mi, mj = float(state.mass[i]), float(state.mass[j])
    vn = float((state.velocity[i] - state.velocity[j]) @ n)
    if vn >= 0.0:
        return None
    mu = mi * mj / (mi + mj)
    impulse = -(1.0 + restitution) * mu * vn
    dvi, dvj = (impulse / mi) * n, -(impulse / mj) * n
    state.velocity[i] += dvi
    state.velocity[j] += dvj
    if remaining_time > 0.0:
        state.position[i] += dvi * remaining_time
        state.position[j] += dvj * remaining_time
        state.touch()
    heat = 0.5 * mu * vn * vn * (1.0 - restitution ** 2)
    radiated = radiated_fraction * heat
    kept = heat - radiated
    state.internal_energy[i] += kept * mi / (mi + mj)
    state.internal_energy[j] += kept * mj / (mi + mj)
    state.ledger.radiated += radiated
    if label is None:
        label = 'elastic' if restitution >= 0.999 else 'inelastic'
    return {'time': time, 'event': label, 'a': int(state.ids[i]), 'b': int(state.ids[j]),
            'restitution': restitution, 'approach_speed': -vn, 'heat': heat,
            'radiated': radiated, 'mass_a': mi, 'mass_b': mj}


def emit(state: ParticleSystem, i: int, *, energy: float, direction: np.ndarray, time: float,
         mass_fraction: float | None = None, parts: np.ndarray | None = None,
         species: str = 'fragment', separation: float = 1.05) -> dict:
    """Split slot i into itself (the residual) plus a new fragment.

    Either ``parts`` (constituent counts handed to the fragment) or
    ``mass_fraction`` sets the fragment's share; with ``parts`` the mass,
    charge and polarity follow the constituent-count share. ``energy`` is
    taken from i's internal energy and released as relative kinetic energy,
    with recoil so momentum is conserved. The residual keeps i's ID.
    """
    total_parts = int(state.composition[i].sum())
    if parts is not None:
        parts = np.asarray(parts, dtype=np.int64)
        if np.any(parts < 0) or np.any(parts > state.composition[i]) or parts.sum() < 1:
            raise ValueError('parts must be a non-empty subset of the parent composition')
        if parts.sum() >= total_parts:
            raise ValueError('cannot emit every constituent; leave a residual')
        share = parts.sum() / total_parts
    else:
        if mass_fraction is None or not 0.0 < mass_fraction < 1.0:
            raise ValueError('give parts=... or 0 < mass_fraction < 1')
        share = float(mass_fraction)
        parts = np.zeros_like(state.composition[i])
    available = float(state.internal_energy[i])
    energy = float(np.clip(energy, 0.0, max(available, 0.0)))
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    m = float(state.mass[i])
    mf, mr = share * m, (1.0 - share) * m
    mu = mf * mr / m
    v0, x0 = state.velocity[i].copy(), state.position[i].copy()
    vrel = np.sqrt(2.0 * energy / mu)
    dim = state.dimension
    rf, rr = state.radius[i] * share ** (1 / dim), state.radius[i] * (1 - share) ** (1 / dim)
    gap = (rf + rr) * separation
    frag = dict(
        species=species, mass=mf, charge=float(state.charge[i] * share),
        radius=float(rf), spin=float(state.spin[i]), polarity=float(state.polarity[i] * share),
        phase=float(state.phase[i]), position=x0 + (mr / m) * gap * d,
        velocity=v0 + (mr / m) * vrel * d, internal_energy=0.0, composition=parts,
        cluster_id=int(state.cluster_id[i]),
    )
    state.mass[i] = mr
    state.charge[i] *= (1.0 - share)
    state.polarity[i] *= (1.0 - share)
    state.radius[i] = rr
    state.composition[i] = state.composition[i] - parts
    state.position[i] = x0 - (mf / m) * gap * d
    state.velocity[i] = v0 - (mf / m) * vrel * d
    state.internal_energy[i] = available - energy
    parent_id = int(state.ids[i])
    child_id = state.add_particle(**frag)
    return {'time': time, 'event': 'emit', 'a': parent_id, 'child': child_id,
            'energy': energy, 'mass_a': m, 'mass_b': mf}
