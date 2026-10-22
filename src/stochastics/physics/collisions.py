"""Contact detection, contact policies and the collision rule.

The collision outcome is a *gradient* driven by what the particles are made
of. Each particle carries a composition: counts of sub-particles per
constituent type. Constituent types attract or repel through an affinity
matrix K (default: opposite constituent charges attract, like charges repel).

``CompositionAffinity`` turns that into an outcome:

1. affinity    w = (c_i . K . c_j) / (n_i n_j)  in [-1, 1]
               (+1: every part of i attracts every part of j; -1: all repel)
2. restitution e = e_min + (e_max - e_min) * (1 - w) / 2
               repulsive interiors bounce elastically, attractive ones stick
3. after a bounce with e, the pair keeps relative kinetic energy
               K' = K_rel - (1 - e^2) * 0.5 * mu * v_n^2
4. the pair's binding energy at contact
               B = -U_law(i, j) + w * 0.5 * mu * bond_speed^2
               (long-range law, e.g. gravity, plus short-range part bonds)
5. K' < B  -> MERGE (captured: cannot escape each other)
   else    -> BOUNCE with e (elastic if e = 1, inelastic otherwise)

So 3 attracting + 2 repelling parts against a mostly opposite partner gives
a soft, lossy bounce or a capture; fully repulsive interiors give a clean
elastic bounce. Heat from inelastic contacts goes to internal energy, minus a
radiated fraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree

from .environment import Rule, StepContext
from .forces import pair_potential
from .mechanics import bounce, merge
from .state import ParticleSystem

__all__ = ['Contact', 'Outcome', 'ContactPolicy', 'CompositionAffinity', 'Collisions',
           'MergeWithinDistance', 'find_contacts']


@dataclass
class Contact:
    """A detected touching pair, evaluated at the moment of contact."""
    i: int                      # array slot
    j: int                      # array slot
    normal: np.ndarray          # unit vector from j to i
    approach_speed: float       # closing speed along the normal (> 0)
    reduced_mass: float
    impact_energy: float        # 0.5 * mu * v_n^2 (the part a collision can dissipate)
    relative_kinetic: float     # 0.5 * mu * |v_i - v_j|^2
    tau: float                  # time of contact inside the step, in [0, dt]
    remaining_time: float       # dt - tau
    distance: float             # centre distance at contact
    pair_potential: float = 0.0  # law's U_ij at contact (filled in by Collisions)


@dataclass(frozen=True)
class Outcome:
    """A contact policy's decision. Mechanics guarantee conservation."""
    kind: str                           # 'merge' | 'bounce' | 'pass'
    restitution: float = 1.0
    radiated_fraction: float = 0.0
    label: str | None = None
    info: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def merge(radiated_fraction: float = 0.0, label: str = 'merge', **info) -> 'Outcome':
        return Outcome('merge', 0.0, radiated_fraction, label, info)

    @staticmethod
    def bounce(restitution: float, radiated_fraction: float = 0.0, label: str | None = None,
               **info) -> 'Outcome':
        return Outcome('bounce', float(restitution), radiated_fraction, label, info)

    @staticmethod
    def elastic(**info) -> 'Outcome':
        return Outcome('bounce', 1.0, 0.0, 'elastic', info)

    @staticmethod
    def ignore(**info) -> 'Outcome':
        return Outcome('pass', info=info)


class ContactPolicy:
    """Decides what happens when two particles touch. Subclass and override.

    ``decide``  -> Outcome for one contact
    ``combine`` -> overrides for a merged particle's internal state
    ``evolve``  -> (optional) per-step internal dynamics, run before contacts
    """
    def decide(self, state: ParticleSystem, contact: Contact, time: float) -> Outcome:
        return Outcome.elastic()

    def combine(self, state: ParticleSystem, i: int, j: int, time: float) -> dict:
        return {}


@dataclass
class CompositionAffinity(ContactPolicy):
    """Outcome gradient from the particles' sub-particle composition (see module doc).

    ``allow_merge=False`` keeps the restitution gradient but never fuses particles
    (hard spheres with composition-dependent bounciness).
    """
    e_min: float = 0.1
    e_max: float = 1.0
    bond_speed: float = 0.5
    radiated_fraction: float = 0.1
    cohesion: float = 0.0
    affinity_matrix: np.ndarray | None = None
    allow_merge: bool = True

    def matrix(self, state: ParticleSystem) -> np.ndarray:
        """Constituent affinity K (positive = attract)."""
        if self.affinity_matrix is not None:
            return np.asarray(self.affinity_matrix, dtype=float)
        q = state.constituent_charge
        return -np.outer(q, q) + self.cohesion

    def affinity(self, state: ParticleSystem, i: int, j: int) -> float:
        """Normalised cross-affinity w between the parts of i and the parts of j."""
        ci, cj = state.composition[i], state.composition[j]
        pairs = ci.sum() * cj.sum()
        if pairs == 0:
            return 0.0
        return float(np.clip(ci @ self.matrix(state) @ cj / pairs, -1.0, 1.0))

    def cohesion_of(self, state: ParticleSystem, i: int) -> float:
        """Normalised internal cohesion: how strongly i's own parts attract each other."""
        c = state.composition[i].astype(float)
        n = c.sum()
        if n < 2:
            return 0.0
        K = self.matrix(state)
        within = (c @ K @ c - c @ np.diag(K) * 1.0) / 2.0
        return float(within / (n * (n - 1) / 2.0))

    def restitution(self, w: float) -> float:
        return float(np.clip(self.e_min + (self.e_max - self.e_min) * (1.0 - w) / 2.0, 0.0, 1.0))

    def classify(self, w: float, impact_energy: float, relative_kinetic: float,
                 reduced_mass: float, pair_potential: float = 0.0) -> Outcome:
        """The decision rule without a state (used by the regime map)."""
        e = self.restitution(w)
        k_after = relative_kinetic - impact_energy * (1.0 - e * e)
        binding = -pair_potential + w * 0.5 * reduced_mass * self.bond_speed ** 2
        info = {'affinity': w, 'binding': binding}
        if self.allow_merge and k_after < binding:
            return Outcome.merge(self.radiated_fraction, label='merge', restitution=e, **info)
        return Outcome.bounce(e, self.radiated_fraction, **info)

    def decide(self, state, contact, time):
        w = self.affinity(state, contact.i, contact.j)
        return self.classify(w, contact.impact_energy, contact.relative_kinetic,
                             contact.reduced_mass, contact.pair_potential)


def find_contacts(state: ParticleSystem, previous_position: np.ndarray | None, dt: float,
                  continuous: bool = True) -> list[Contact]:
    """Touching, approaching pairs, earliest contact first.

    With ``continuous=True`` each particle is swept along its straight path
    over the step, so fast particles cannot tunnel through each other; the
    contact time ``tau`` is the first moment the spheres touch.
    """
    live = state.live
    if len(live) < 2:
        return []
    x1 = state.position[live]
    v = state.velocity[live]
    r = state.radius[live]
    x0 = x1 - v * dt
    if previous_position is not None and continuous:
        known = live < len(previous_position)
        x0[known] = previous_position[live[known]]
    if not continuous:
        x0 = x1
    disp = np.linalg.norm(x1 - x0, axis=1)
    search = 2.0 * float(r.max()) + 2.0 * float(disp.max())
    if search <= 0:
        return []
    pairs = cKDTree(x1).query_pairs(search, output_type='ndarray')
    if not len(pairs):
        return []
    a, b = pairs[:, 0], pairs[:, 1]
    R = r[a] + r[b]
    d0 = x0[a] - x0[b]
    d1 = x1[a] - x1[b]
    if continuous:
        u = d1 - d0
        A = np.einsum('ij,ij->i', u, u)
        B = 2.0 * np.einsum('ij,ij->i', d0, u)
        C = np.einsum('ij,ij->i', d0, d0) - R * R
        overlap0 = C <= 0.0
        disc = B * B - 4.0 * A * C
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (-B - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * A)
        swept = (A > 0) & (disc >= 0) & (s >= 0.0) & (s <= 1.0)
        hit = overlap0 | swept
        s = np.where(overlap0, 0.0, s)
    else:
        hit = np.einsum('ij,ij->i', d1, d1) < R * R
        s = np.ones(len(a))
    if not hit.any():
        return []
    a, b, s = a[hit], b[hit], s[hit]
    xa = x0[a] + (x1[a] - x0[a]) * s[:, None]
    xb = x0[b] + (x1[b] - x0[b]) * s[:, None]
    sep = xa - xb
    dist = np.linalg.norm(sep, axis=1)
    vr = v[a] - v[b]
    fallback = dist < 1e-12
    normal = np.where(fallback[:, None], 0.0, sep / np.where(fallback, 1.0, dist)[:, None])
    if fallback.any():  # coincident centres: push apart against the relative velocity
        speed = np.linalg.norm(vr[fallback], axis=1)
        alt = -vr[fallback] / np.where(speed > 0, speed, 1.0)[:, None]
        alt[speed == 0] = np.eye(state.dimension)[0]
        normal[fallback] = alt
    vn = np.einsum('ij,ij->i', vr, normal)
    approaching = vn < 0.0
    order = np.lexsort((dist, s))
    out: list[Contact] = []
    for k in order:
        if not approaching[k]:
            continue
        i, j = int(live[a[k]]), int(live[b[k]])
        mi, mj = state.mass[i], state.mass[j]
        mu = float(mi * mj / (mi + mj))
        out.append(Contact(i=i, j=j, normal=normal[k], approach_speed=float(-vn[k]),
                           reduced_mass=mu, impact_energy=0.5 * mu * float(vn[k]) ** 2,
                           relative_kinetic=0.5 * mu * float(vr[k] @ vr[k]),
                           tau=float(s[k] * dt), remaining_time=float((1.0 - s[k]) * dt),
                           distance=float(dist[k])))
    return out


@dataclass
class Collisions(Rule):
    """Radius-based contact rule: detect, ask the policy, execute conservatively.

    Each particle takes part in at most one contact per step (earliest first);
    simultaneous multi-body contacts resolve over consecutive steps.
    If the policy defines ``evolve(state, time, dt, ctx)`` (an internal policy),
    it runs first and its potential-energy effect is booked as ``drive``.
    """
    policy: Any = field(default_factory=CompositionAffinity)
    continuous: bool = True
    max_events_per_step: int | None = None
    log_passes: bool = False
    ledger = 'binding'

    def apply(self, state: ParticleSystem, time: float, ctx: StepContext) -> list[dict]:
        events: list[dict] = []
        evolve = getattr(self.policy, 'evolve', None)
        if evolve is not None:
            events.extend(evolve(state, time, ctx.dt, ctx) or [])
            ctx.settle('drive')
        contacts = find_contacts(state, ctx.previous_position, ctx.dt, self.continuous)
        used: set[int] = set()
        for c in contacts:
            if self.max_events_per_step is not None and len(used) >= 2 * self.max_events_per_step:
                break
            if c.i in used or c.j in used or not (state.active[c.i] and state.active[c.j]):
                continue
            c.pair_potential = pair_potential(ctx.law, state, c.i, c.j, time)
            outcome = self.policy.decide(state, c, time)
            event = self._resolve(state, c, outcome, time)
            if event is None:
                continue
            used.update((c.i, c.j))
            events.append(event)
        return events

    def _resolve(self, state, c: Contact, outcome: Outcome, time: float) -> dict | None:
        base = {'impact_energy': c.impact_energy, 'approach_speed': c.approach_speed,
                'tau': c.tau, **dict(outcome.info)}
        if outcome.kind == 'pass':
            if not self.log_passes:
                return None
            return {'time': time, 'event': outcome.label or 'pass',
                    'a': int(state.ids[c.i]), 'b': int(state.ids[c.j]), **base}
        if outcome.kind == 'merge':
            combine = getattr(self.policy, 'combine', None)
            overrides = combine(state, c.i, c.j, time) if combine else None
            ev = merge(state, c.i, c.j, time=time, radiated_fraction=outcome.radiated_fraction,
                       overrides=overrides or None, label=outcome.label or 'merge')
            return {**ev, **base}
        if outcome.kind == 'bounce':
            ev = bounce(state, c.i, c.j, c.normal, outcome.restitution, time=time,
                        radiated_fraction=outcome.radiated_fraction,
                        remaining_time=c.remaining_time, label=outcome.label)
            return None if ev is None else {**base, **ev}
        raise ValueError(f'unknown outcome kind {outcome.kind!r}')


@dataclass
class MergeWithinDistance(Rule):
    """Proximity capture: merge pairs closer than ``distance`` (ignores radii).

    Kept from the first prototype. Conserves mass, charge, momentum; kinetic
    energy lost goes to internal energy and radiation; the potential-energy
    jump is booked as ``binding`` so accounted energy stays conserved.
    """
    distance: float = 0.18
    radiated_fraction: float = 0.25
    max_merges_per_step: int | None = None
    ledger = 'binding'

    def apply(self, state: ParticleSystem, time: float, ctx: StepContext | None = None):
        if not 0 <= self.radiated_fraction <= 1:
            raise ValueError('radiated_fraction must be in [0, 1]')
        live = state.live
        if len(live) < 2:
            return []
        pairs = cKDTree(state.position[live]).query_pairs(self.distance, output_type='ndarray')
        if not len(pairs):
            return []
        sep = np.linalg.norm(state.position[live[pairs[:, 0]]] - state.position[live[pairs[:, 1]]],
                             axis=1)
        order = np.lexsort((pairs[:, 1], pairs[:, 0], sep))
        used: set[int] = set()
        events: list[dict] = []
        for k in order:
            i, j = int(live[pairs[k, 0]]), int(live[pairs[k, 1]])
            if i in used or j in used:
                continue
            if self.max_merges_per_step is not None and len(events) >= self.max_merges_per_step:
                break
            used.update((i, j))
            ev = merge(state, i, j, time=time, radiated_fraction=self.radiated_fraction)
            ev['kinetic_to_other'] = ev['heat']
            ev['radiated_energy'] = ev['radiated']
            events.append(ev)
        return events
