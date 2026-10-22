"""Particle state: a structure-of-arrays system with an object-style API.

* ``ParticleSystem`` stores every property as a NumPy array (fast, vectorised).
* ``system[k]`` returns a live :class:`ParticleView` of the k-th *live*
  particle; assigning to its attributes writes through to the arrays.
* :class:`Particle` is a detached, plain configuration record used to build
  systems by hand: ``ParticleSystem.from_particles([...])``.

IDs are never reused. Merging deactivates the parents and appends a child,
so array slots are stable for a particle's whole lifetime.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .presets import ParticlePreset, get_preset, positive_lognormal

__all__ = ['Particle', 'ParticleView', 'Cluster', 'EnergyLedger', 'ParticleSystem',
           'waveform', 'WAVEFORMS']

WAVEFORMS = ('triangle', 'cosine', 'square')


def waveform(phase: np.ndarray | float, kind: str = 'triangle') -> np.ndarray:
    """Unit-amplitude clock shape c(phase) with c(0)=1, c(1/4)=0, c(1/2)=-1.

    ``phase`` is measured in cycles. ``polarity_now = polarity * c(phase)``.
    The triangle wave reproduces the stepwise sequence
    -1, -0.5, 0, 0.5, 1, 0.5, 0, ... for polarity=-1, spin=0.5, dt=1.
    """
    p = np.mod(np.asarray(phase, dtype=float), 1.0)
    if kind == 'triangle':
        return 4.0 * np.abs(p - 0.5) - 1.0
    if kind == 'cosine':
        return np.cos(2.0 * np.pi * p)
    if kind == 'square':
        return np.where(np.cos(2.0 * np.pi * p) >= 0.0, 1.0, -1.0)
    raise ValueError(f'waveform must be one of {WAVEFORMS}')


@dataclass
class EnergyLedger:
    """Energy that has left (K + U + internal) through a named channel.

    ``kinetic + potential + internal + ledger.total()`` is the *accounted*
    energy; with a conservative law it stays constant up to integrator error.

    radiated   : carried away as untracked radiation (collisions, mergers)
    dissipated : given to a background medium (drag / friction)
    binding    : potential energy released or absorbed by discrete events
                 (mergers, emissions, contact displacements)
    expansion  : potential-energy change caused by background expansion
    drive      : potential-energy change caused by internal clocks / policies
    """
    radiated: float = 0.0
    dissipated: float = 0.0
    binding: float = 0.0
    expansion: float = 0.0
    drive: float = 0.0

    def total(self) -> float:
        return self.radiated + self.dissipated + self.binding + self.expansion + self.drive

    def as_dict(self) -> dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}

    def copy(self) -> 'EnergyLedger':
        return EnergyLedger(**self.as_dict())


@dataclass(frozen=True)
class Cluster:
    """Spatial cluster used by ``ParticleSystem.random(clusters=[...])``.

    mean         : centre, length = dimension
    std          : positional spread (scalar or per-axis)
    weight       : relative share of particles (normalised across clusters)
    velocity     : bulk drift velocity of the whole cluster
    rotation     : bulk (solid-body) angular velocity; a scalar spins about z
    velocity_std : internal random velocity dispersion (None -> preset value)
    orbital_G    : if set, add approximately circular orbital speeds about the
                   cluster centre for a gravity law with this G (enclosed-mass
                   estimate); ``rotation`` then only sets the axis and sense (default +z)
    """
    mean: Sequence[float]
    std: float | Sequence[float] = 1.0
    weight: float = 1.0
    velocity: Sequence[float] | None = None
    rotation: float | Sequence[float] = 0.0
    velocity_std: float | None = None
    orbital_G: float | None = None

    @classmethod
    def coerce(cls, spec: Any) -> 'Cluster':
        if isinstance(spec, Cluster):
            return spec
        if isinstance(spec, Mapping):
            return cls(**spec)
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            return cls(mean=spec[0], std=spec[1])
        raise TypeError('cluster spec must be a Cluster, a dict, or a (mean, std) pair')


@dataclass
class Particle:
    """A detached particle configuration (edit freely, then add to a system).

    ``energy`` is the particle's internal energy reservoir: collision heat goes
    in, emission spends it. The cycling quantity driven by ``spin`` and
    ``polarity`` is available as ``ParticleSystem.polarity_now``.
    """
    mass: float = 1.0
    charge: float = 0.0
    position: Sequence[float] = (0.0, 0.0, 0.0)
    radius: float = 0.05
    spin: float = 0.0
    polarity: float = 0.0
    energy: float = 0.0
    velocity: Sequence[float] | None = None
    phase: float = 0.0
    composition: Sequence[int] | None = None
    species: str = 'particle'
    cluster_id: int = -1
    id: int | None = None


_VIEW_FIELDS = {
    'mass': 'mass', 'charge': 'charge', 'radius': 'radius', 'spin': 'spin',
    'polarity': 'polarity', 'phase': 'phase', 'energy': 'internal_energy',
    'internal_energy': 'internal_energy', 'species': 'species',
    'cluster_id': 'cluster_id', 'position': 'position', 'velocity': 'velocity',
    'composition': 'composition',
}


class ParticleView:
    """Live view of one particle; attribute writes go straight to the arrays."""
    __slots__ = ('_system', '_slot')

    def __init__(self, system: 'ParticleSystem', slot: int):
        object.__setattr__(self, '_system', system)
        object.__setattr__(self, '_slot', int(slot))

    @property
    def id(self) -> int:
        return int(self._system.ids[self._slot])

    @property
    def slot(self) -> int:
        return self._slot

    @property
    def active(self) -> bool:
        return bool(self._system.active[self._slot])

    @property
    def polarity_now(self) -> float:
        s = self._system
        return float(s.polarity[self._slot] * waveform(s.phase[self._slot], s.waveform))

    @property
    def kinetic_energy(self) -> float:
        v = self._system.velocity[self._slot]
        return float(0.5 * self._system.mass[self._slot] * v @ v)

    def __getattr__(self, name: str):
        try:
            arr = getattr(self._system, _VIEW_FIELDS[name])
        except KeyError:
            raise AttributeError(name) from None
        value = arr[self._slot]
        if isinstance(value, np.ndarray):
            return value  # row view: in-place edits write through
        return value.item() if hasattr(value, 'item') else value

    def __setattr__(self, name: str, value) -> None:
        if name not in _VIEW_FIELDS:
            raise AttributeError(f'cannot set {name!r} on a particle')
        arr = getattr(self._system, _VIEW_FIELDS[name])
        arr[self._slot] = value
        self._system.touch()

    def to_particle(self) -> Particle:
        s, k = self._system, self._slot
        return Particle(mass=float(s.mass[k]), charge=float(s.charge[k]),
                        position=s.position[k].copy(), radius=float(s.radius[k]),
                        spin=float(s.spin[k]), polarity=float(s.polarity[k]),
                        energy=float(s.internal_energy[k]), velocity=s.velocity[k].copy(),
                        phase=float(s.phase[k]), composition=s.composition[k].copy(),
                        species=str(s.species[k]), cluster_id=int(s.cluster_id[k]),
                        id=int(s.ids[k]))

    def __repr__(self) -> str:
        s, k = self._system, self._slot
        fmt = lambda a: '[' + ', '.join(f'{x:.4g}' for x in a) + ']'
        comp = ', '.join(f'{n}:{c}' for n, c in zip(s.constituent_names, s.composition[k]) if c)
        return (f'Particle(id={self.id}, species={s.species[k]!r}, mass={s.mass[k]:.4g}, '
                f'charge={s.charge[k]:.4g}, radius={s.radius[k]:.4g}, spin={s.spin[k]:.4g}, '
                f'polarity={s.polarity[k]:.4g}, polarity_now={self.polarity_now:.4g}, '
                f'energy={s.internal_energy[k]:.4g}, position={fmt(s.position[k])}, '
                f'velocity={fmt(s.velocity[k])}, composition={{{comp}}}, '
                f'cluster_id={s.cluster_id[k]}, active={bool(s.active[k])})')


_PER_PARTICLE = ('ids', 'species', 'mass', 'charge', 'radius', 'spin', 'polarity', 'phase',
                 'internal_energy', 'active', 'cluster_id')


@dataclass
class ParticleSystem:
    """Structure-of-arrays particle state (positions are ``(N, dimension)``).

    ``composition[k]`` counts particle k's sub-particles per constituent type
    (``constituent_names`` / ``constituent_charge``); collision policies use it
    to decide how strongly the *parts* of two particles attract.
    """
    ids: np.ndarray
    species: np.ndarray
    mass: np.ndarray
    charge: np.ndarray
    radius: np.ndarray
    spin: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    internal_energy: np.ndarray
    active: np.ndarray
    polarity: np.ndarray | None = None
    phase: np.ndarray | None = None
    cluster_id: np.ndarray | None = None
    composition: np.ndarray | None = None
    constituent_names: tuple[str, ...] = ('+', '-')
    constituent_charge: np.ndarray | None = None
    ledger: EnergyLedger = field(default_factory=EnergyLedger)
    time: float = 0.0
    waveform: str = 'triangle'
    next_id: int = 0
    revision: int = 0

    # ------------------------------------------------------------------ setup
    def __post_init__(self):
        n = len(self.ids)
        self.ids = np.asarray(self.ids, dtype=np.int64)
        self.species = np.asarray(self.species, dtype=object)
        for name in ('mass', 'charge', 'radius', 'spin', 'internal_energy'):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.active = np.asarray(self.active, dtype=bool)
        self.polarity = (np.zeros(n) if self.polarity is None
                         else np.asarray(self.polarity, dtype=float))
        self.phase = np.zeros(n) if self.phase is None else np.asarray(self.phase, dtype=float)
        self.cluster_id = (np.zeros(n, dtype=np.int64) if self.cluster_id is None
                           else np.asarray(self.cluster_id, dtype=np.int64))
        self.constituent_names = tuple(self.constituent_names)
        k = len(self.constituent_names)
        if self.constituent_charge is None:
            default = {'+': 1.0, '-': -1.0, '0': 0.0}
            self.constituent_charge = np.array([default.get(c, 0.0) for c in self.constituent_names])
        self.constituent_charge = np.asarray(self.constituent_charge, dtype=float)
        if self.composition is None:
            self.composition = _default_composition(self.charge, self.constituent_charge)
        self.composition = np.asarray(self.composition, dtype=np.int64).reshape(n, k)
        if self.next_id <= 0 and n:
            self.next_id = int(self.ids.max()) + 1
        if self.waveform not in WAVEFORMS:
            raise ValueError(f'waveform must be one of {WAVEFORMS}')

    # ------------------------------------------------------------- properties
    @property
    def dimension(self) -> int:
        return self.position.shape[1]

    @property
    def live(self) -> np.ndarray:
        """Array slots of active particles."""
        return np.flatnonzero(self.active)

    @property
    def n_live(self) -> int:
        return int(self.active.sum())

    @property
    def energy(self) -> np.ndarray:
        """Alias for ``internal_energy`` (the per-particle energy reservoir)."""
        return self.internal_energy

    @property
    def polarity_now(self) -> np.ndarray:
        """Instantaneous cycled polarity: ``polarity * waveform(phase)``."""
        return self.polarity * waveform(self.phase, self.waveform)

    @property
    def n_constituents(self) -> np.ndarray:
        return self.composition.sum(axis=1)

    @property
    def radiation_energy(self) -> float:
        return self.ledger.radiated

    @radiation_energy.setter
    def radiation_energy(self, value: float) -> None:
        self.ledger.radiated = float(value)

    def touch(self) -> None:
        """Mark positions/charges/membership as changed (invalidates cached forces)."""
        self.revision += 1

    # ------------------------------------------------------------- validation
    def validate(self) -> None:
        n = len(self.ids)
        for name in _PER_PARTICLE:
            if len(getattr(self, name)) != n:
                raise ValueError(f'{name} length is not {n}')
        if self.composition.shape != (n, len(self.constituent_names)):
            raise ValueError('composition must be (N, n_constituent_types)')
        if self.position.shape != self.velocity.shape or self.position.shape[0] != n:
            raise ValueError('position and velocity must both be (N, dimension)')
        if self.dimension not in (2, 3):
            raise ValueError('Use 2D or 3D coordinates')
        if len(np.unique(self.ids)) != n:
            raise ValueError('Particle IDs must be unique')
        a = self.active
        if np.any(self.mass[a] <= 0) or np.any(self.radius[a] < 0):
            raise ValueError('Active particle masses must be positive; radii nonnegative')
        if np.any(self.composition[a] < 0):
            raise ValueError('composition counts must be nonnegative')
        for values in (self.position[a], self.velocity[a], self.mass[a], self.charge[a],
                       self.internal_energy[a], self.polarity[a], self.spin[a], self.phase[a]):
            if not np.isfinite(values).all():
                raise ValueError('Non-finite active-particle state')

    # ----------------------------------------------------------- construction
    @classmethod
    def random(cls, n: int = 60, dimension: int | None = None, seed: int = 11,
               position_std: float | None = None, velocity_std: float | None = None, *,
               preset: str | ParticlePreset = 'classical',
               clusters: int | Sequence[Cluster | Mapping | tuple] | None = None,
               n_clusters: int | None = None, cluster_sampling: str = 'uniform',
               cluster_probability: float = 0.5, random_phase: bool = False,
               constituents: Sequence[tuple[str, float]] | None = None) -> 'ParticleSystem':
        """Sample a population from a preset, optionally in explicit clusters.

        ``clusters`` may be a list of :class:`Cluster` / dicts / ``(mean, std)``
        pairs, e.g. ``[Cluster(mean=[0,0,0], std=100),
        Cluster(mean=[100,-100,0], std=150)]``, or an int for that many
        random centres drawn from the preset. ``dimension`` is inferred from
        explicit cluster means when not given (otherwise 2).

        Composition: each particle gets ``1 + Poisson(constituents_mean - 1)``
        sub-particles; types are drawn with probabilities proportional to
        ``exp(charge * type_charge / charge_scale)`` so positively charged
        particles are mostly made of '+' parts.
        """
        cfg = get_preset(preset)
        explicit = clusters is not None and not isinstance(clusters, (int, np.integer))
        specs = [Cluster.coerce(c) for c in clusters] if explicit else None
        if dimension is None:
            dimension = len(specs[0].mean) if specs else 2
        if n < 1 or dimension not in (2, 3):
            raise ValueError('n >= 1 and dimension must be 2 or 3')
        if cluster_sampling not in ('uniform', 'binomial'):
            raise ValueError('cluster_sampling must be uniform or binomial')
        if not 0 <= cluster_probability <= 1:
            raise ValueError('cluster_probability must be in [0, 1]')
        xs = cfg.cluster_std if position_std is None else position_std
        vs = cfg.velocity_std if velocity_std is None else velocity_std
        if not np.isfinite([xs, vs]).all() or xs < 0 or vs < 0:
            raise ValueError('position_std and velocity_std must be nonnegative')
        rng = np.random.default_rng(seed)

        if specs:
            for s in specs:
                if len(s.mean) != dimension:
                    raise ValueError(f'cluster mean {list(s.mean)} does not match dimension={dimension}')
            weights = np.array([max(float(s.weight), 0.0) for s in specs])
            if weights.sum() <= 0:
                raise ValueError('cluster weights must have a positive sum')
            counts = rng.multinomial(n, weights / weights.sum())
            groups = np.repeat(np.arange(len(specs)), counts)
            means = np.array([np.asarray(s.mean, float) for s in specs])
            stds = np.array([np.broadcast_to(np.asarray(s.std, float), (dimension,)) for s in specs])
            position = means[groups] + rng.normal(size=(n, dimension)) * stds[groups]
        else:
            c = n_clusters if n_clusters is not None else (clusters if clusters is not None
                                                           else cfg.n_clusters)
            if not isinstance(c, (int, np.integer)) or c < 1:
                raise ValueError('n_clusters must be a positive integer')
            centers = rng.normal(0, cfg.cluster_center_std, (c, dimension))
            if cluster_sampling == 'binomial':
                groups = rng.binomial(c - 1, cluster_probability, size=n)
            else:
                groups = rng.integers(0, c, size=n)
            position = centers[groups] + rng.normal(0, xs, size=(n, dimension))

        mass = positive_lognormal(rng, cfg.mass_mean, cfg.mass_std, n)
        charge = rng.normal(cfg.charge_mean, cfg.charge_std, size=n)
        radius = positive_lognormal(rng, cfg.radius_mean, cfg.radius_std, n)
        spin = rng.choice(np.asarray(cfg.spin_states, float), size=n)
        polarity = rng.choice(np.asarray(cfg.polarity_states, float), size=n)
        phase = rng.uniform(0, 1, n) if random_phase else np.zeros(n)

        velocity = np.zeros((n, dimension))
        if specs:
            for g, s in enumerate(specs):
                idx = np.flatnonzero(groups == g)
                if not len(idx):
                    continue
                sv = vs if s.velocity_std is None else float(s.velocity_std)
                velocity[idx] = rng.normal(0, sv, (len(idx), dimension))
                if s.velocity is not None:
                    velocity[idx] += np.asarray(s.velocity, float)
                offset = position[idx] - np.asarray(s.mean, float)
                if not s.orbital_G:
                    velocity[idx] += _rotation_velocity(offset, s.rotation, dimension)
                else:
                    velocity[idx] += _orbital_velocity(offset, mass[idx], float(s.orbital_G),
                                                       s.rotation, dimension,
                                                       0.1 * float(np.mean(s.std)))
        else:
            velocity = rng.normal(0, vs, (n, dimension))

        names, qc = _constituent_table(constituents)
        n_parts = 1 + rng.poisson(cfg.constituents_mean - 1, size=n)
        scale = cfg.charge_std if cfg.charge_std > 0 else 1.0
        logits = np.outer(charge / scale, qc)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        composition = np.array([rng.multinomial(k, p) for k, p in zip(n_parts, probs)])

        system = cls(
            ids=np.arange(n, dtype=np.int64), species=np.full(n, cfg.name, dtype=object),
            mass=mass, charge=charge, radius=radius, spin=spin, position=position,
            velocity=velocity, internal_energy=np.zeros(n), active=np.ones(n, dtype=bool),
            polarity=polarity, phase=phase, cluster_id=groups.astype(np.int64),
            composition=composition, constituent_names=names, constituent_charge=qc, next_id=n,
        )
        system.validate()
        return system

    @classmethod
    def from_particles(cls, particles: Iterable[Particle], dimension: int | None = None,
                       constituents: Sequence[tuple[str, float]] | None = None,
                       waveform: str = 'triangle') -> 'ParticleSystem':
        """Build a system from hand-written :class:`Particle` records."""
        plist = list(particles)
        if not plist:
            raise ValueError('need at least one particle')
        if dimension is None:
            dimension = len(plist[0].position)
        names, qc = _constituent_table(constituents)
        n = len(plist)
        system = cls(
            ids=np.arange(n), species=np.array([p.species for p in plist], dtype=object),
            mass=[p.mass for p in plist], charge=[p.charge for p in plist],
            radius=[p.radius for p in plist], spin=[p.spin for p in plist],
            position=np.zeros((n, dimension)), velocity=np.zeros((n, dimension)),
            internal_energy=[p.energy for p in plist], active=np.ones(n, dtype=bool),
            polarity=[p.polarity for p in plist], phase=[p.phase for p in plist],
            cluster_id=[p.cluster_id for p in plist],
            composition=np.zeros((n, len(names)), dtype=np.int64),
            constituent_names=names, constituent_charge=qc, waveform=waveform, next_id=n,
        )
        for k, p in enumerate(plist):
            system.position[k] = _vector(p.position, dimension, 'position')
            system.velocity[k] = (_vector(p.velocity, dimension, 'velocity')
                                  if p.velocity is not None else 0.0)
            system.composition[k] = _composition_for(p.composition, p.charge, qc)
        system.validate()
        return system

    def copy(self) -> 'ParticleSystem':
        """Deep copy (arrays and ledger)."""
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        for name, value in kw.items():
            if isinstance(value, np.ndarray):
                kw[name] = value.copy()
        kw['ledger'] = self.ledger.copy()
        return ParticleSystem(**kw)

    # ------------------------------------------------------- sequence protocol
    def __len__(self) -> int:
        return self.n_live

    def __getitem__(self, index):
        live = self.live
        if isinstance(index, slice):
            return [ParticleView(self, s) for s in live[index]]
        return ParticleView(self, live[index])

    def __setitem__(self, index: int, particle: Particle) -> None:
        self._write(int(self.live[index]), particle)

    def __iter__(self) -> Iterator[ParticleView]:
        return (ParticleView(self, s) for s in self.live)

    def get(self, particle_id: int) -> ParticleView:
        """Look a particle up by its stable ID (works for inactive ones too)."""
        slot = np.flatnonzero(self.ids == particle_id)
        if not len(slot):
            raise KeyError(particle_id)
        return ParticleView(self, slot[0])

    def slot_of(self, particle_id: int) -> int:
        return int(self.get(particle_id).slot)

    # ------------------------------------------------------------- mutation
    def add(self, particle: Particle) -> int:
        """Append a :class:`Particle`; returns its new ID."""
        return self.add_particle(
            species=particle.species, mass=particle.mass, charge=particle.charge,
            radius=particle.radius, spin=particle.spin, position=particle.position,
            velocity=(particle.velocity if particle.velocity is not None
                      else np.zeros(self.dimension)),
            internal_energy=particle.energy, polarity=particle.polarity, phase=particle.phase,
            composition=_composition_for(particle.composition, particle.charge,
                                         self.constituent_charge),
            cluster_id=particle.cluster_id)

    def add_particle(self, *, species: str, mass: float, charge: float, radius: float,
                     spin: float, position, velocity, internal_energy: float,
                     polarity: float = 0.0, phase: float = 0.0, composition=None,
                     cluster_id: int = -1) -> int:
        """Append one particle without reusing an ID. Returns the new ID."""
        new_id = self.next_id
        self.next_id += 1
        self.ids = np.append(self.ids, new_id)
        self.species = np.append(self.species, np.array([species], dtype=object))
        for name, value in (('mass', mass), ('charge', charge), ('radius', radius),
                            ('spin', spin), ('internal_energy', internal_energy),
                            ('polarity', polarity), ('phase', phase)):
            setattr(self, name, np.append(getattr(self, name), float(value)))
        self.position = np.vstack((self.position, _vector(position, self.dimension, 'position')))
        self.velocity = np.vstack((self.velocity, _vector(velocity, self.dimension, 'velocity')))
        self.active = np.append(self.active, True)
        self.cluster_id = np.append(self.cluster_id, int(cluster_id))
        comp = (np.zeros(len(self.constituent_names), dtype=np.int64) if composition is None
                else np.asarray(composition, dtype=np.int64))
        self.composition = np.vstack((self.composition, comp[None, :]))
        self.touch()
        return new_id

    def deactivate(self, slot: int) -> None:
        self.active[slot] = False
        self.touch()

    def _write(self, slot: int, p: Particle) -> None:
        self.mass[slot], self.charge[slot], self.radius[slot] = p.mass, p.charge, p.radius
        self.spin[slot], self.polarity[slot], self.phase[slot] = p.spin, p.polarity, p.phase
        self.internal_energy[slot] = p.energy
        self.position[slot] = _vector(p.position, self.dimension, 'position')
        self.velocity[slot] = (_vector(p.velocity, self.dimension, 'velocity')
                               if p.velocity is not None else 0.0)
        self.composition[slot] = _composition_for(p.composition, p.charge, self.constituent_charge)
        self.species[slot], self.cluster_id[slot] = p.species, p.cluster_id
        self.touch()

    # ------------------------------------------------------------ inspection
    def to_frame(self, live_only: bool = True) -> pd.DataFrame:
        idx = self.live if live_only else np.arange(len(self.ids))
        data = {
            'particle_id': self.ids[idx], 'species': self.species[idx],
            'cluster_id': self.cluster_id[idx], 'mass': self.mass[idx],
            'charge': self.charge[idx], 'radius': self.radius[idx], 'spin': self.spin[idx],
            'polarity': self.polarity[idx], 'polarity_now': self.polarity_now[idx],
            'phase': self.phase[idx], 'energy': self.internal_energy[idx],
            'n_constituents': self.n_constituents[idx],
        }
        for d, name in enumerate('xyz'[:self.dimension]):
            data[name] = self.position[idx, d]
        for d, name in enumerate('xyz'[:self.dimension]):
            data['v' + name] = self.velocity[idx, d]
        if not live_only:
            data['active'] = self.active[idx]
        return pd.DataFrame(data)

    def describe(self) -> pd.DataFrame:
        """Per-cluster summary of the live population."""
        df = self.to_frame()
        speed = np.linalg.norm(self.velocity[self.live], axis=1)
        df = df.assign(speed=speed)
        out = df.groupby('cluster_id').agg(
            count=('particle_id', 'size'), mass_total=('mass', 'sum'),
            mass_mean=('mass', 'mean'), charge_mean=('charge', 'mean'),
            speed_mean=('speed', 'mean'),
            **{f'{c}_mean': (c, 'mean') for c in 'xyz'[:self.dimension]})
        return out

    def __repr__(self) -> str:
        clusters = len(np.unique(self.cluster_id[self.live])) if self.n_live else 0
        return (f'ParticleSystem(live={self.n_live}, slots={len(self.ids)}, '
                f'dimension={self.dimension}, t={self.time:.4g}, clusters={clusters}, '
                f'constituents={self.constituent_names})')


# --------------------------------------------------------------------- helpers
def _vector(value, dimension: int, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=float).reshape(-1)
    if v.shape != (dimension,):
        raise ValueError(f'{name} must have length {dimension}, got {v.shape[0]}')
    return v


def _constituent_table(spec) -> tuple[tuple[str, ...], np.ndarray]:
    if spec is None:
        return ('+', '-'), np.array([1.0, -1.0])
    names = tuple(str(name) for name, _ in spec)
    return names, np.array([float(q) for _, q in spec])


def _default_composition(charge: np.ndarray, qc: np.ndarray) -> np.ndarray:
    return np.array([_composition_for(None, q, qc) for q in np.asarray(charge)],
                    dtype=np.int64).reshape(len(charge), len(qc))


def _composition_for(composition, charge: float, qc: np.ndarray) -> np.ndarray:
    k = len(qc)
    if composition is not None:
        c = np.asarray(composition, dtype=np.int64).reshape(-1)
        if c.shape != (k,):
            raise ValueError(f'composition must have {k} counts (one per constituent type)')
        return c
    out = np.zeros(k, dtype=np.int64)
    if charge > 0 and np.any(qc > 0):
        out[int(np.argmax(qc))] = 1
    elif charge < 0 and np.any(qc < 0):
        out[int(np.argmin(qc))] = 1
    elif np.any(qc == 0):
        out[int(np.flatnonzero(qc == 0)[0])] = 1
    else:  # neutral particle built from one + and one - part
        out[int(np.argmax(qc))] += 1
        out[int(np.argmin(qc))] += 1
    return out


def _orbital_velocity(offset: np.ndarray, mass: np.ndarray, G: float, rotation, dimension: int,
                      eps: float) -> np.ndarray:
    """Circular speeds sqrt(G M(<R) R^2 / (R^2 + eps^2)^1.5) around the rotation axis."""
    omega = np.asarray(rotation, dtype=float)
    if dimension == 2:
        sign = -1.0 if float(omega.reshape(-1)[-1]) < 0 else 1.0
        R_vec = offset
        tangent = sign * np.column_stack((-offset[:, 1], offset[:, 0]))
    else:
        axis = (np.array([0.0, 0.0, -1.0 if float(omega) < 0 else 1.0]) if omega.ndim == 0
                else omega.reshape(3) / (np.linalg.norm(omega) or 1.0))
        if omega.ndim == 0 and float(omega) == 0:
            axis = np.array([0.0, 0.0, 1.0])
        R_vec = offset - np.outer(offset @ axis, axis)
        tangent = np.cross(axis, R_vec)
    R = np.linalg.norm(R_vec, axis=1)
    order = np.argsort(R)
    enclosed = np.empty_like(R)
    enclosed[order] = np.cumsum(mass[order]) - mass[order]
    speed = np.sqrt(G * enclosed * R ** 2 / (R ** 2 + eps ** 2) ** 1.5)
    unit = tangent / np.where(R > 0, R, 1.0)[:, None]
    return unit * speed[:, None]


def _rotation_velocity(offset: np.ndarray, rotation, dimension: int) -> np.ndarray:
    omega = np.asarray(rotation, dtype=float)
    if not np.any(omega):
        return np.zeros_like(offset)
    if dimension == 2:
        w = float(omega.reshape(-1)[-1])
        return w * np.column_stack((-offset[:, 1], offset[:, 0]))
    w3 = np.array([0.0, 0.0, float(omega)]) if omega.ndim == 0 else omega.reshape(3)
    return np.cross(w3, offset)
