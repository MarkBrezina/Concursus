"""Time loop, recording and diagnostics.

Each step:

1. velocity-Verlet kick/drift/kick with the force law (one fused
   force+potential evaluation per step; reused while nothing restructures);
2. background expansion (potential change -> ``expansion`` ledger);
3. rules in order: collisions, mergers, drag, clocks, internal policies
   (potential change -> each rule's ledger channel);
4. recording every ``record_every`` steps.

``accounted_energy = kinetic + potential + internal + ledger`` should be flat
for a conservative law; its drift is integrator error plus whatever a rule
did without booking it.
"""
from __future__ import annotations

import inspect
import time as _time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .collisions import Collisions
from .environment import Environment, StepContext
from .forces import as_law, evaluate_law
from .state import ParticleSystem

__all__ = ['simulate', 'SimulationResult', 'snapshot', 'measure']

LEDGER_COLUMNS = ('radiated', 'dissipated', 'binding', 'expansion', 'drive')


@dataclass
class SimulationResult:
    history: pd.DataFrame          # one row per live particle per recorded frame
    diagnostics: pd.DataFrame      # one row per recorded frame: invariants and energy ledger
    events: pd.DataFrame           # one row per discrete event (merge, bounce, emit, ...)
    final_state: ParticleSystem
    initial_state: ParticleSystem | None = None
    environment: Environment | None = None
    settings: dict = field(default_factory=dict)
    runtime: float = 0.0

    @property
    def dimension(self) -> int:
        return self.final_state.dimension

    def event_counts(self) -> pd.Series:
        if self.events.empty:
            return pd.Series(dtype=int, name='count')
        return self.events['event'].value_counts().rename('count')

    @property
    def merges(self) -> pd.DataFrame:
        return self._events_of('merge')

    @property
    def collisions(self) -> pd.DataFrame:
        return self._events_of('elastic', 'inelastic')

    def _events_of(self, *names) -> pd.DataFrame:
        if self.events.empty:
            return self.events
        return self.events[self.events['event'].isin(names)]

    def summary(self) -> pd.DataFrame:
        """Key numbers of the run as a two-column table."""
        d = self.diagnostics
        first, last = d.iloc[0], d.iloc[-1]
        e0 = first.accounted_energy
        scale = max(abs(first.kinetic) + abs(first.potential) + abs(first.internal), 1e-300)
        dims = 'xyz'[:self.dimension]
        p0 = np.array([first[f'p_{c}'] for c in dims])
        p1 = np.array([last[f'p_{c}'] for c in dims])
        h0 = self.history[self.history.time == first.time]
        pscale = float((h0.mass * h0.speed).sum()) or 1.0   # sum of |p_i|: a natural scale
        rows = [
            ('simulated time', f"{first.time:.4g} -> {last.time:.4g}"),
            ('steps x dt', f"{self.settings.get('steps')} x {self.settings.get('dt')}"),
            ('runtime', f'{self.runtime:.2f} s'),
            ('particles (live)', f'{int(first.n_active)} -> {int(last.n_active)}'),
        ]
        counts = self.event_counts()
        if counts.empty:
            rows.append(('events', 'none'))
        rows += [(f'events: {k}', f'{v}') for k, v in counts.items()]
        rows += [
            ('mass drift', f'{last.mass - first.mass:+.3e}'),
            ('charge drift', f'{last.charge - first.charge:+.3e}'),
            ('momentum drift |dp|', f'{np.linalg.norm(p1 - p0):.3e}  (scale {pscale:.3g})'
                                    + ('  [drag removes momentum]' if last.dissipated > 0 else '')),
        ]
        if np.isfinite(e0):
            rel = (last.accounted_energy - e0) / scale
            rows.append(('energy residual', f'{last.accounted_energy - e0:+.3e}  ({rel:+.2e} relative)'))
            if abs(rel) > 1e-2:
                rows.append(('note', 'residual > 1%: reduce dt or increase softening '
                                     '(close encounters are under-resolved)'))
        else:
            rows.append(('energy residual', 'n/a (non-reciprocal law has no potential)'))
        for name in LEDGER_COLUMNS:
            if abs(last[name]) > 0:
                rows.append((f'ledger: {name}', f'{last[name]:+.4g}'))
        return pd.DataFrame(rows, columns=['metric', 'value']).set_index('metric')

    def report(self, **kwargs):
        from .plotting import report
        return report(self, **kwargs)

    def plot(self, **kwargs):
        from .plotting import plot_dashboard
        return plot_dashboard(self, **kwargs)

    def animate_2d(self, **kwargs):
        from .animation import animate_2d
        return animate_2d(self, **kwargs)

    def animate_3d(self, **kwargs):
        from .animation import animate_3d
        return animate_3d(self, **kwargs)

    def __repr__(self) -> str:
        d = self.diagnostics
        return (f'SimulationResult(frames={len(d)}, t={d.time.iloc[0]:.4g}->{d.time.iloc[-1]:.4g}, '
                f'live={int(d.n_active.iloc[0])}->{int(d.n_active.iloc[-1])}, '
                f'events={len(self.events)}, dimension={self.dimension})')


# ----------------------------------------------------------------- recording
def snapshot(state: ParticleSystem, time: float) -> pd.DataFrame:
    """DataFrame of the live particles at ``time``."""
    return pd.DataFrame(_frame_columns(state, time))


def _frame_columns(state: ParticleSystem, time: float) -> dict:
    i = state.live
    v = state.velocity[i]
    data = {
        'time': np.full(len(i), float(time)), 'particle_id': state.ids[i].copy(),
        'species': state.species[i].copy(), 'cluster_id': state.cluster_id[i].copy(),
        'mass': state.mass[i].copy(), 'charge': state.charge[i].copy(),
        'radius': state.radius[i].copy(), 'spin': state.spin[i].copy(),
        'polarity': state.polarity[i].copy(), 'polarity_now': state.polarity_now[i],
        'phase': state.phase[i].copy(), 'internal_energy': state.internal_energy[i].copy(),
        'n_constituents': state.n_constituents[i],
    }
    for d, name in enumerate('xyz'[:state.dimension]):
        data[name] = state.position[i, d].copy()
    for d, name in enumerate('xyz'[:state.dimension]):
        data['v' + name] = v[:, d].copy()
    data['speed'] = np.sqrt(np.einsum('ij,ij->i', v, v))
    return data


def measure(state: ParticleSystem, time: float, law=None, potential: float | None = None) -> dict:
    """Invariants and the full energy ledger at one instant."""
    i = state.live
    m, v, x = state.mass[i], state.velocity[i], state.position[i]
    kinetic = 0.5 * float(np.sum(m * np.einsum('ij,ij->i', v, v)))
    if potential is None:
        potential = evaluate_law(as_law(law), state, time, need_acc=False)[1]
    internal = float(np.sum(state.internal_energy[i]))
    ledger = state.ledger.as_dict()
    momentum = (m[:, None] * v).sum(axis=0)
    speed = np.sqrt(np.einsum('ij,ij->i', v, v))
    out = {
        'time': float(time), 'n_active': len(i), 'mass': float(m.sum()),
        'charge': float(state.charge[i].sum()), 'kinetic': kinetic,
        'potential': float(potential), 'internal': internal, **ledger,
        'accounted_energy': kinetic + float(potential) + internal + sum(ledger.values()),
        'mean_speed': float(speed.mean()) if len(i) else 0.0,
        'rms_speed': float(np.sqrt(np.mean(speed ** 2))) if len(i) else 0.0,
        'virial_ratio': (2 * kinetic / abs(potential) if potential and np.isfinite(potential)
                         else float('nan')),
    }
    out.update({f'p_{c}': float(momentum[d]) for d, c in enumerate('xyz'[:state.dimension])})
    if state.dimension == 2:
        out['L_z'] = float(np.sum(m * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])))
    else:
        L = np.sum(m[:, None] * np.cross(x, v), axis=0)
        out.update({'L_x': float(L[0]), 'L_y': float(L[1]), 'L_z': float(L[2])})
    return out


class _Recorder:
    def __init__(self):
        self.frames: list[dict] = []
        self.measures: list[dict] = []

    def record(self, state, time, potential):
        self.frames.append(_frame_columns(state, time))
        self.measures.append(measure(state, time, potential=potential))

    def history(self) -> pd.DataFrame:
        keys = self.frames[0].keys()
        return pd.DataFrame({k: np.concatenate([f[k] for f in self.frames]) for k in keys})


class _Tracker:
    """Keeps the cached acceleration/potential in sync with the state."""

    def __init__(self, law, state: ParticleSystem, t: float):
        self.law, self.state = law, state
        self.acc, self.U = evaluate_law(law, state, t)
        self.rev = state.revision

    def settle(self, channel: str, t: float) -> None:
        s = self.state
        if s.revision == self.rev:
            return
        acc, U = evaluate_law(self.law, s, t)
        if np.isfinite(U) and np.isfinite(self.U):
            setattr(s.ledger, channel, getattr(s.ledger, channel) - (U - self.U))
        self.acc, self.U, self.rev = acc, U, s.revision


def _rule_caller(rule):
    try:
        params = inspect.signature(rule.apply).parameters
    except (TypeError, ValueError):
        return lambda s, t, ctx: rule.apply(s, t, ctx)
    if len(params) >= 3:
        return lambda s, t, ctx: rule.apply(s, t, ctx)
    return lambda s, t, ctx: rule.apply(s, t)


def _resolve_environment(environment, law, rules, expansion, external, internal) -> Environment:
    if environment is not None and not isinstance(environment, Environment):
        raise TypeError('second argument must be an Environment (or use law=/rules= keywords)')
    env = environment or Environment()
    laws = [env.law] if environment is not None else []
    if law is not None:
        laws.append(as_law(law))
    if external is not None:
        laws.append(as_law(external))
    laws = [l for l in laws if type(l).__name__ != 'NoForce'] or laws[:1]
    combined = as_law(laws) if laws else as_law(None)
    all_rules = list(env.rules) + list(rules or [])
    if internal is not None:
        all_rules.insert(0, internal if hasattr(internal, 'apply') else Collisions(internal))
    return Environment(law=combined, rules=all_rules,
                       expansion=expansion if expansion is not None else env.expansion)


def simulate(state: ParticleSystem, environment: Environment | None = None, *,
             law=None, rules: list | None = None, expansion=None,
             external=None, internal=None,
             steps: int = 300, dt: float = 0.02, record_every: int = 1,
             seed: int | None = 0) -> SimulationResult:
    """Run the time loop. ``state`` is advanced in place (``result.final_state``).

    Either pass an :class:`Environment` positionally, or the pieces as
    keywords (``law=``, ``rules=``, ``expansion=``). ``external=`` is a force
    law / external policy (added to ``law``); ``internal=`` is an internal
    policy (wrapped in :class:`Collisions` and run first among the rules).

    Contacts are detected along each step's straight-line path; very large
    ``dt`` still degrades accuracy of the force integration itself.
    """
    if steps < 1 or dt <= 0 or record_every < 1:
        raise ValueError('steps and record_every must be >= 1; dt > 0')
    env = _resolve_environment(environment, law, rules, expansion, external, internal)
    state.validate()
    initial = state.copy()
    rng = np.random.default_rng(seed)
    callers = [(rule, _rule_caller(rule), getattr(rule, 'ledger', 'binding')) for rule in env.rules]
    t0 = float(state.time)
    tracker = _Tracker(env.law, state, t0)
    rec = _Recorder()
    rec.record(state, t0, tracker.U)
    events: list[dict] = []
    started = _time.perf_counter()
    for step in range(1, steps + 1):
        t_new = t0 + step * dt
        live = state.live
        x_prev = state.position.copy()
        a_old = tracker.acc
        state.position[live] += dt * state.velocity[live] + 0.5 * dt * dt * a_old[live]
        a_new, U_new = evaluate_law(env.law, state, t_new)
        if not np.isfinite(a_new[live]).all():
            raise FloatingPointError(f'non-finite acceleration at step {step}; reduce dt or '
                                     'increase softening')
        state.velocity[live] += 0.5 * dt * (a_old[live] + a_new[live])
        state.time = t_new
        tracker.acc, tracker.U, tracker.rev = a_new, U_new, state.revision
        if env.expansion is not None:
            env.expansion.apply(state, dt)
            tracker.settle('expansion', t_new)
        ctx = StepContext(dt=dt, step=step, law=env.law, rng=rng, previous_position=x_prev,
                          settle=lambda channel, _t=t_new: tracker.settle(channel, _t))
        for rule, call, channel in callers:
            out = call(state, t_new, ctx)
            if out:
                events.extend(out)
            tracker.settle(channel, t_new)
        if step % record_every == 0 or step == steps:
            rec.record(state, t_new, tracker.U)
    runtime = _time.perf_counter() - started
    ev = pd.DataFrame(events)
    if not ev.empty:
        front = [c for c in ('time', 'event', 'a', 'b', 'child') if c in ev]
        ev = ev[front + [c for c in ev.columns if c not in front]]
    return SimulationResult(
        history=rec.history(), diagnostics=pd.DataFrame(rec.measures), events=ev,
        final_state=state, initial_state=initial, environment=env,
        settings={'steps': steps, 'dt': dt, 'record_every': record_every, 'seed': seed},
        runtime=runtime)
