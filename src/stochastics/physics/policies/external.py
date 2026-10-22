"""policy_external — who is drawn to whom.

An external policy turns a pairwise *affinity* into motion::

    A[i, j] > 0   i is pulled toward j        A[i, j] < 0   i is pushed away

    force on i  = strength * sum_j A[i, j] * g(r_ij) * (x_j - x_i)
    potential   = -strength * sum_{i<j} A[i, j] * phi(r_ij)      (if symmetric)

``g``/``phi`` come from the kernel: ``'plummer'`` (inverse square, default),
``'gaussian'`` (local, good for agents), ``'yukawa'`` (screened) or ``'spring'``.
Write a policy by overriding one method::

    class MyPolicy(ExternalPolicy):
        def affinity(self, a, b):                       # a, b: Agents views
            return -np.outer(a.polarity_now, b.polarity_now)

Plug ``ElectromagneticPolicy()`` in and the system *is* electromagnetism;
``GravitationalPolicy()`` is gravity. Everything else is your hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..forces import PairInteraction
from ..state import ParticleSystem, waveform

__all__ = ['Agents', 'ExternalPolicy', 'OppositesAttract', 'ElectromagneticPolicy',
           'GravitationalPolicy']

_ARRAYS = {'mass': 'mass', 'charge': 'charge', 'radius': 'radius', 'spin': 'spin',
           'polarity': 'polarity', 'phase': 'phase', 'energy': 'internal_energy',
           'composition': 'composition', 'position': 'position', 'velocity': 'velocity',
           'species': 'species', 'cluster_id': 'cluster_id', 'ids': 'ids'}


class Agents:
    """Read-only view of a subset of particles (array slots ``index``) at time ``t``.

    Attributes: mass, charge, radius, spin, polarity, phase, polarity_now,
    energy, composition, position, velocity, species, cluster_id, ids, n.
    """
    __slots__ = ('_state', 'index', 't', '_cache')

    def __init__(self, state: ParticleSystem, index: np.ndarray, t: float):
        self._state, self.index, self.t, self._cache = state, np.asarray(index), t, {}

    @property
    def n(self) -> int:
        return len(self.index)

    @property
    def polarity_now(self) -> np.ndarray:
        if 'polarity_now' not in self._cache:
            s = self._state
            self._cache['polarity_now'] = s.polarity[self.index] * waveform(s.phase[self.index],
                                                                             s.waveform)
        return self._cache['polarity_now']

    @property
    def state(self) -> ParticleSystem:
        return self._state

    def __getattr__(self, name: str):
        try:
            source = _ARRAYS[name]
        except KeyError:
            raise AttributeError(name) from None
        cache = self._cache
        if name not in cache:
            cache[name] = getattr(self._state, source)[self.index]
        return cache[name]


@dataclass
class ExternalPolicy(PairInteraction):
    """Base class for ``policy_external``. Override :meth:`affinity`."""
    strength: float = 1.0

    def affinity(self, a: Agents, b: Agents) -> np.ndarray:
        raise NotImplementedError('return an (a.n, b.n) matrix; > 0 attracts i toward j')

    def coupling(self, state, I, J, t):
        A = np.asarray(self.affinity(Agents(state, I, t), Agents(state, J, t)), dtype=float)
        if A.shape != (len(I), len(J)):
            raise ValueError(f'affinity must return shape {(len(I), len(J))}, got {A.shape}')
        return -self.strength * A


@dataclass
class OppositesAttract(ExternalPolicy):
    """Starting hypothesis: particles seek partners with opposite characteristics.

    affinity = - w_p * p_i(t) p_j(t)            opposite current polarity attracts
               - w_q * q_i q_j                  opposite static charge attracts
               - w_phi * cos(2 pi (phase_i - phase_j))   anti-phase clocks attract

    ``p(t) = polarity_now`` cycles with each particle's spin clock, so a pair
    can attract, go neutral and repel as their clocks drift in and out of step.
    """
    polarity_weight: float = 1.0
    charge_weight: float = 0.0
    phase_weight: float = 0.0

    def affinity(self, a, b):
        A = -self.polarity_weight * np.outer(a.polarity_now, b.polarity_now)
        if self.charge_weight:
            A = A - self.charge_weight * np.outer(a.charge, b.charge)
        if self.phase_weight:
            A = A - self.phase_weight * np.cos(2 * np.pi * (a.phase[:, None] - b.phase[None, :]))
        return A


@dataclass
class ElectromagneticPolicy(ExternalPolicy):
    """Coulomb's law written as a policy. ``source='charge'`` reproduces
    :class:`Coulomb` exactly; ``source='polarity_now'`` uses the cycling polarity."""
    k: float = 0.2
    source: str = 'charge'

    def affinity(self, a, b):
        return -self.k * np.outer(getattr(a, self.source), getattr(b, self.source))


@dataclass
class GravitationalPolicy(ExternalPolicy):
    """Newtonian gravity written as a policy (reproduces :class:`Gravity`)."""
    G: float = 0.5

    def affinity(self, a, b):
        return self.G * np.outer(a.mass, b.mass)
