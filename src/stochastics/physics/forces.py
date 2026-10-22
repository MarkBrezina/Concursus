"""Force laws built on one pairwise kernel.

Every central pair interaction has the form

    U_ij = C_ij * phi(r_ij)          F_i = sum_j C_ij * g(r_ij) * (x_i - x_j)

with ``g = -phi'(r)/r``. A law only supplies the *coupling* matrix ``C_ij``
(positive = repulsive); the kernel supplies the distance profile. Gravity is
``C = -G m_i m_j``, Coulomb is ``C = k q_i q_j``, and the policy lab's
external policies are ``C = -strength * affinity_ij``. One chunked routine
computes accelerations and the matching potential in a single pass, so the
energy ledger is always consistent with the forces actually applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ['ForceLaw', 'Kernel', 'Plummer', 'Gaussian', 'Spring', 'Yukawa', 'make_kernel',
           'PairInteraction', 'NoForce', 'Gravity', 'Coulomb', 'CombinedForce',
           'as_law', 'evaluate_law', 'pair_potential']


@runtime_checkable
class ForceLaw(Protocol):
    def __call__(self, state, time: float) -> np.ndarray: ...
    def potential(self, state) -> float: ...


# ------------------------------------------------------------------ kernels
class Kernel:
    """Distance profile: ``phi(r2)`` and ``g(r2) = -phi'(r)/r`` (r2 = r**2)."""
    def phi(self, r2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def g(self, r2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def phi_g(self, r2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.phi(r2), self.g(r2)


@dataclass(frozen=True)
class Plummer(Kernel):
    """Softened inverse-square law: phi = 1/sqrt(r^2 + eps^2)."""
    softening: float = 0.15

    def phi(self, r2):
        return (r2 + self.softening ** 2) ** -0.5

    def g(self, r2):
        return (r2 + self.softening ** 2) ** -1.5

    def phi_g(self, r2):
        inv = 1.0 / np.sqrt(r2 + self.softening ** 2)
        return inv, inv * inv * inv


@dataclass(frozen=True)
class Gaussian(Kernel):
    """Short-range bump: phi = exp(-r^2 / 2L^2). Good for local agent interactions."""
    length: float = 1.0

    def phi(self, r2):
        return np.exp(-r2 / (2.0 * self.length ** 2))

    def g(self, r2):
        return self.phi(r2) / self.length ** 2


@dataclass(frozen=True)
class Spring(Kernel):
    """Linear spring: attraction grows with distance (U = |C| r^2 / 2)."""
    def phi(self, r2):
        return -0.5 * r2

    def g(self, r2):
        return np.ones_like(r2)


@dataclass(frozen=True)
class Yukawa(Kernel):
    """Screened inverse-square: phi = exp(-s/L)/s with s = sqrt(r^2 + eps^2)."""
    screening: float = 1.0
    softening: float = 0.05

    def phi(self, r2):
        s = np.sqrt(r2 + self.softening ** 2)
        return np.exp(-s / self.screening) / s

    def g(self, r2):
        s = np.sqrt(r2 + self.softening ** 2)
        return self.phi(r2) * (1.0 / self.screening + 1.0 / s) / s


def make_kernel(kernel: Kernel | str | None, softening: float) -> Kernel:
    if isinstance(kernel, Kernel):
        return kernel
    if kernel in (None, 'plummer', 'inverse_square'):
        return Plummer(softening)
    if kernel == 'gaussian':
        return Gaussian()
    if kernel == 'spring':
        return Spring()
    if kernel == 'yukawa':
        return Yukawa(softening=softening)
    raise ValueError(f'unknown kernel {kernel!r}')


# ------------------------------------------------------------ pair engine
@dataclass
class PairInteraction:
    """Base class: subclass and implement ``coupling(state, I, J, t)``.

    The base options (``softening``, ``chunk_size``, ``kernel``, ``reciprocal``)
    are keyword-only, so subclasses keep their own parameters positional:
    ``Gravity(0.5)`` sets G, ``Coulomb(0.2)`` sets k.

    ``I``/``J`` are array slots; return a ``(len(I), len(J))`` matrix of
    energy scales (positive = repulsive). Memory is O(chunk_size * N);
    runtime is O(N^2). ``reciprocal=False`` declares a non-symmetric
    coupling (A chases B while B flees A); then no potential exists and
    the energy diagnostics report NaN.
    """
    softening: float = field(default=0.15, kw_only=True)
    chunk_size: int = field(default=256, kw_only=True)
    kernel: Any = field(default=None, kw_only=True)
    reciprocal: bool = field(default=True, kw_only=True)

    def coupling(self, state, I: np.ndarray, J: np.ndarray, t: float) -> np.ndarray:
        raise NotImplementedError

    @property
    def profile(self) -> Kernel:
        return make_kernel(self.kernel, self.softening)

    def evaluate(self, state, t: float | None = None, need_acc: bool = True):
        """Return ``(acceleration (N, d), potential energy)`` in one pass."""
        t = state.time if t is None else t
        acc = np.zeros_like(state.position)
        live = state.live
        n = len(live)
        if n < 2:
            return acc, 0.0
        x, m = state.position[live], state.mass[live]
        prof = self.profile
        U = 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            for lo in range(0, n, self.chunk_size):
                hi = min(lo + self.chunk_size, n)
                delta = x[lo:hi, None, :] - x[None, :, :]
                r2 = np.einsum('ijk,ijk->ij', delta, delta)
                C = np.asarray(self.coupling(state, live[lo:hi], live, t), dtype=float)
                rows, cols = np.arange(hi - lo), np.arange(lo, hi)
                if need_acc and self.reciprocal:
                    phi, g = prof.phi_g(r2)
                elif need_acc:
                    phi, g = None, prof.g(r2)
                else:
                    phi, g = prof.phi(r2), None
                if need_acc:
                    w = C * g
                    w[rows, cols] = 0.0
                    acc[live[lo:hi]] = np.einsum('ij,ijk->ik', w, delta) / m[lo:hi, None]
                if self.reciprocal:
                    phi[rows, cols] = 0.0
                    U += 0.5 * float(np.sum(C * phi))
        return acc, (U if self.reciprocal else float('nan'))

    def __call__(self, state, time: float | None = None) -> np.ndarray:
        return self.evaluate(state, time)[0]

    def potential(self, state, time: float | None = None) -> float:
        return self.evaluate(state, time, need_acc=False)[1]

    def pair_potential(self, state, i: int, j: int, t: float | None = None) -> float:
        """Interaction energy of the single pair (i, j) at its current separation."""
        t = state.time if t is None else t
        d = state.position[i] - state.position[j]
        r2 = np.array([[d @ d]])
        cij = float(self.coupling(state, np.array([i]), np.array([j]), t)[0, 0])
        cji = cij if self.reciprocal else float(self.coupling(state, np.array([j]),
                                                              np.array([i]), t)[0, 0])
        return 0.5 * (cij + cji) * float(self.profile.phi(r2)[0, 0])


@dataclass
class NoForce:
    """Free motion."""
    reciprocal: bool = True

    def evaluate(self, state, t=None, need_acc=True):
        return np.zeros_like(state.position), 0.0

    def __call__(self, state, time=None):
        return np.zeros_like(state.position)

    def potential(self, state, time=None) -> float:
        return 0.0

    def pair_potential(self, state, i, j, t=None) -> float:
        return 0.0


@dataclass
class Gravity(PairInteraction):
    """Softened Newtonian gravity (arbitrary, internally consistent units)."""
    G: float = 0.5

    def coupling(self, state, I, J, t):
        return -self.G * np.outer(state.mass[I], state.mass[J])


@dataclass
class Coulomb(PairInteraction):
    """Softened electrostatics; like charges repel. Uses inertial mass for acceleration."""
    k: float = 0.2

    def coupling(self, state, I, J, t):
        return self.k * np.outer(state.charge[I], state.charge[J])


@dataclass
class CombinedForce:
    """Sum of several laws (``law=[Gravity(), Coulomb()]`` builds one)."""
    laws: list

    @property
    def reciprocal(self) -> bool:
        return all(getattr(law, 'reciprocal', True) for law in self.laws)

    def evaluate(self, state, t=None, need_acc=True):
        acc = np.zeros_like(state.position)
        U = 0.0
        for law in self._fused():
            a, u = evaluate_law(law, state, t, need_acc)
            acc += a
            U += u
        return acc, U

    def _fused(self) -> list:
        """Pair laws sharing a kernel are summed into one pass (half the cost)."""
        groups: dict = {}
        out = []
        for law in self.laws:
            if isinstance(law, PairInteraction):
                try:
                    key = (law.profile, law.chunk_size, law.reciprocal)
                    hash(key)
                except TypeError:
                    out.append(law)
                    continue
                groups.setdefault(key, []).append(law)
            else:
                out.append(law)
        for (profile, chunk, reciprocal), laws in groups.items():
            out.append(laws[0] if len(laws) == 1 else
                       _SummedCoupling(kernel=profile, chunk_size=chunk, reciprocal=reciprocal,
                                       parts=laws))
        return out

    def __call__(self, state, time=None):
        return self.evaluate(state, time)[0]

    def potential(self, state, time=None) -> float:
        return self.evaluate(state, time, need_acc=False)[1]

    def pair_potential(self, state, i, j, t=None) -> float:
        return sum(pair_potential(law, state, i, j, t) for law in self.laws)


@dataclass
class _SummedCoupling(PairInteraction):
    parts: list = field(default_factory=list)

    def coupling(self, state, I, J, t):
        return sum(np.asarray(p.coupling(state, I, J, t), dtype=float) for p in self.parts)


def as_law(law) -> Any:
    """Normalise ``None`` / a list of laws / a single law into one law object."""
    if law is None:
        return NoForce()
    if isinstance(law, (list, tuple)):
        laws = [as_law(item) for item in law]
        return laws[0] if len(laws) == 1 else CombinedForce(laws)
    return law


def evaluate_law(law, state, t: float | None = None, need_acc: bool = True):
    """Evaluate any law, including legacy ones that only define ``__call__``/``potential``."""
    t = state.time if t is None else t
    if hasattr(law, 'evaluate'):
        acc, U = law.evaluate(state, t, need_acc)
    else:
        acc = np.asarray(law(state, t), dtype=float) if need_acc else np.zeros_like(state.position)
        pot = getattr(law, 'potential', None)
        try:
            U = float(pot(state)) if pot is not None else float('nan')
        except TypeError:
            U = float(pot(state, t))
    acc = np.asarray(acc, dtype=float)
    if acc.shape != state.position.shape:
        raise ValueError('force law must return an (N, dimension) acceleration array')
    return acc, float(U)


def pair_potential(law, state, i: int, j: int, t: float | None = None) -> float:
    fn = getattr(law, 'pair_potential', None)
    if fn is None:
        return 0.0
    value = float(fn(state, i, j, t))
    return value if np.isfinite(value) else 0.0
