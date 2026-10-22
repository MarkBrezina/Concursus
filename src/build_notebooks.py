"""Generate the two notebooks (run from the repo root: python notebooks/build_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# ============================================================== showcase
showcase = [
md(r"""
# `stochastics.physics` — particle interactions lab

A classical many-body simulator built as a small, tested package:

| layer | what it does |
|---|---|
| `ParticleSystem` | structure-of-arrays state; `particles[0]` is a live, editable particle; explicit clusters |
| `Environment` | force **law** (gravity, Coulomb, any pairwise kernel) + **rules** (collisions, mergers, drag, clocks) + background **expansion** |
| `simulate` | velocity-Verlet time loop with continuous (swept-sphere) collision detection |
| energy ledger | `kinetic + potential + internal + radiated + dissipated + binding + expansion + drive` is conserved, so every run audits itself |
| `report` / `animate_2d` / `animate_3d` | one-call dashboard and animations |

The design rule that makes it safe to experiment: **policies decide, mechanics conserve.** A collision policy
only says *merge*, *bounce with restitution e* or *pass*; the mechanics execute it while conserving mass,
charge, momentum and constituents and booking every unit of energy.

*All quantities are dimensionless; this is a classical toy model, not a validated astrophysics or plasma code.*
"""),
code(r"""
import numpy as np
import pandas as pd
from dataclasses import replace

import stochastics
from stochastics import physics
from stochastics.physics import *

print('stochastics', stochastics.__version__)
"""),
md(r"""
## 1. Particles

Two rotating clusters in 3D. `orbital_G` gives each member an approximately circular orbital speed about its
cluster centre (for the gravity constant it will feel), `velocity` sets the bulk drift, and `rotation` the
spin axis / sense. Masses, radii, charges, spins and polarities come from the `astro` preset.
"""),
code(r"""
G = 1e-4
astro_cfg = replace(PRESETS['astro'], radius_mean=0.003, radius_std=0.0008, velocity_std=0.03)

particles = ParticleSystem.random(
    n=400, preset=astro_cfg, seed=210, dimension=3,
    clusters=[
        Cluster(mean=[-3.5, -1.0, 0.0], std=[1.0, 1.0, 0.15], orbital_G=G, rotation=+1,
                velocity=[0.2, 0.12, 0.0]),
        Cluster(mean=[3.5, 1.0, 0.3], std=[0.8, 0.8, 0.12], orbital_G=G, rotation=[0.4, 0, -1],
                weight=0.8, velocity=[-0.2, -0.12, 0.0]),
    ])
particles
"""),
md(r"""
Every particle is addressable as an object. `particles[0]` is a *live view*: assigning `particles[0].mass = ...`
writes straight into the arrays. `composition` counts its sub-particles per constituent type — this is what
drives the collision outcome later.
"""),
code(r"""
particles[0]
"""),
code(r"""
plot_initial(particles)
particles.describe()
"""),
md(r"""
## 2. Environment and run

The environment bundles the law, the rules and background conditions. Here: softened Newtonian gravity plus
radius-based collisions whose outcome is decided by the particles' composition.
"""),
code(r"""
env = Environment(law=Gravity(G=G, softening=0.12),
                  rules=[Collisions(CompositionAffinity())])
result = simulate(particles, env, steps=3200, dt=0.01, record_every=20)
result
"""),
md(r"""
## 3. Report

`report(result)` = summary table + a 12-panel dashboard: trajectories, initial clusters, mass / charge /
speed distributions (initial vs final), active particles, cumulative events by type, the full energy ledger,
the relative energy residual, speed over time, velocity dispersion per cluster, and conservation drift.
"""),
code(r"""
report(result)
"""),
md(r"""
## 4. Animations

Both work on the same result; `animate_2d` projects a 3D run onto a plane (`plane='xz'` etc.).
`color_by` accepts `'cluster'`, `'charge'`, `'polarity_now'`, `'mass'` or `'species'`; `trail=N` draws the
last N frames of each path; `save='run.gif'` writes a file.
"""),
code(r"""
animate_2d(result, max_frames=80, trail=6)
"""),
code(r"""
animate_3d(result, max_frames=80)
"""),
md(r"""
## 5. Collisions: a gradient driven by composition

Each particle is made of sub-particles. With constituent types `+` and `−` (opposites attract, likes repel),
take particle A = **3 `+` and 2 `−`**. Against a partner B, some of A's parts are attracted to B's parts and
some are repelled, while A's own parts hold A together. `CompositionAffinity` turns that into one number and
an outcome:

1. **affinity** `w = (c_A · K · c_B) / (n_A n_B)` ∈ [−1, 1] — the average pull between A's parts and B's parts;
2. **restitution** `e = e_min + (e_max − e_min)(1 − w)/2` — repulsive interiors bounce elastically, attractive ones stick;
3. after a bounce with `e` the pair keeps relative kinetic energy `K' = K_rel − (1 − e²)·½μv_n²`;
4. **binding** at contact `B = −U_law(A, B) + w·½μ·bond_speed²` — long-range law (gravity, Coulomb) plus short-range part bonds;
5. `K' < B` → **merge** (captured), otherwise **bounce** with `e` (elastic if `e = 1`, inelastic otherwise).

Heat from inelastic contacts goes into the particles' internal `energy`, minus a radiated fraction.
"""),
code(r"""
policy = CompositionAffinity()
plot_collision_regimes(policy, max_speed=1.2, examples=[
    ('5+ vs 5+', (5, 0), (5, 0)),
    ('3+2− vs 3+2−', (3, 2), (3, 2)),
    ('3+2− vs 0+5−', (3, 2), (0, 5)),
    ('5+ vs 5−', (5, 0), (0, 5)),
])
"""),
md(r"""
The same thing measured in the engine: head-on collisions of two unit masses (no long-range force), for a
grid of compositions and approach speeds.
"""),
code(r"""
def head_on(comp_a, comp_b, speed):
    pair = ParticleSystem.from_particles([
        Particle(mass=1, position=[-0.5, 0, 0], velocity=[speed / 2, 0, 0], radius=0.1, composition=comp_a),
        Particle(mass=1, position=[0.5, 0, 0], velocity=[-speed / 2, 0, 0], radius=0.1, composition=comp_b)])
    r = simulate(pair, law=NoForce(), rules=[Collisions(policy)], steps=int(1.5 / speed / 0.01) + 50, dt=0.01)
    ev = r.events.iloc[0]
    return {'A': comp_a, 'B': comp_b, 'speed': speed, 'outcome': ev.event,
            'affinity w': round(ev.affinity, 3), 'restitution': round(ev.restitution, 3),
            'heat': round(ev.heat, 4), 'energy residual': r.diagnostics.accounted_energy.iloc[-1] - r.diagnostics.accounted_energy.iloc[0]}

pairs = [((5, 0), (5, 0)), ((3, 2), (3, 2)), ((3, 2), (1, 4)), ((3, 2), (0, 5)), ((5, 0), (0, 5))]
pd.DataFrame([head_on(a, b, s) for a, b in pairs for s in (0.2, 0.8)])
"""),
md(r"""
## 6. Electromagnetic plasma in 3D

Charged particles from the `em` preset under Coulomb's law with composition collisions. Opposite charges
pull together and, because a particle's charge also biases what it is made of, they tend to stick on
contact (recombination): the mean |charge| of the survivors falls. Like charges are pushed apart and
bounce. Colour = charge.
"""),
code(r"""
em_cfg = replace(PRESETS['em'], radius_mean=0.04, radius_std=0.01)
plasma = ParticleSystem.random(
    n=300, preset=em_cfg, seed=21, dimension=3,
    clusters=[Cluster(mean=[-1.2, 0, 0], std=0.5, velocity=[0.15, 0, 0]),
              Cluster(mean=[1.2, 0.3, 0], std=0.5, velocity=[-0.15, 0, 0]),
              Cluster(mean=[0, 1.5, 0.5], std=0.4)])
env_em = Environment(law=Coulomb(k=0.02, softening=0.05),
                     rules=[Collisions(CompositionAffinity(bond_speed=0.3))])
result_em = simulate(plasma, env_em, steps=2000, dt=0.005, record_every=20)
report(result_em)
first, last = (result_em.history[result_em.history.time == t] for t in result_em.diagnostics.time.iloc[[0, -1]])
print(f'mean |charge|: {first.charge.abs().mean():.3f} -> {last.charge.abs().mean():.3f}')
"""),
code(r"""
animate_3d(result_em, max_frames=80, color_by='charge')
"""),
md(r"""
## 7. Validation

Each run audits itself. The table below collects the invariants from the runs above; `pytest` in the repo
root checks these and more (force = −∇U, Coulomb/gravity against an independent reference implementation,
policy equivalence, swept-sphere detection, emission conservation, second-order convergence of the residual).
"""),
code(r"""
def invariants(name, r):
    d = r.diagnostics
    first = r.history[r.history.time == d.time.iloc[0]]
    p = np.stack([d[f'p_{c}'] for c in 'xyz'[:r.dimension]], axis=1)
    scale = abs(d.kinetic.iloc[0]) + abs(d.potential.iloc[0]) + abs(d.internal.iloc[0])
    return {'run': name, 'events': len(r.events),
            'mass drift': abs(d.mass.iloc[-1] - d.mass.iloc[0]) / d.mass.iloc[0],
            'charge drift': abs(d.charge.iloc[-1] - d.charge.iloc[0]),
            'momentum drift': np.linalg.norm(p[-1] - p[0]) / (first.mass * first.speed).sum(),
            'energy residual (rel.)': (d.accounted_energy.iloc[-1] - d.accounted_energy.iloc[0]) / scale}

pd.DataFrame([invariants('galaxies', result), invariants('plasma', result_em)]).set_index('run')
"""),
md(r"""
## 8. Background expansion (toy)

`SpaceExpansion(rate=H)` stretches separations by `exp(H·dt)` per step (velocities are peculiar). Without
forces, every separation grows exactly as `d(t) = d₀·e^{Ht}`; with gravity, the potential-energy change it
causes is booked in the `expansion` ledger.
"""),
code(r"""
astro_exp = replace(PRESETS['astro'], expansion_rate=0.2)
world_exp = ParticleSystem.random(n=50, preset=astro_exp, seed=210, dimension=3)
world_exp.velocity[:] = 0.0
d0 = np.linalg.norm(world_exp.position[0] - world_exp.position[1])
result_exp = simulate(world_exp, law=NoForce(), steps=1000, dt=0.01, record_every=10,
                      expansion=SpaceExpansion(rate=astro_exp.expansion_rate))
T = result_exp.diagnostics.time.iloc[-1]
d1 = np.linalg.norm(world_exp.position[0] - world_exp.position[1])
print(f'separation {d0:.4f} -> {d1:.4f}; expected d0*exp(H*T) = {d0 * np.exp(astro_exp.expansion_rate * T):.4f}')
"""),
]

# ============================================================== policy lab
lab = [
md(r"""
# Policy lab — designing the laws

The physics package fixes the *bookkeeping* (integrator, contacts, conservation, ledger, plots). This lab
leaves the *laws* open. Two slots drive everything:

```python
policy_external = ...   # who is drawn to whom            -> forces between particles
policy_internal = ...   # what happens inside / on contact -> clocks, merge, bounce, emission
result = simulate(particles, external=policy_external, internal=policy_internal, ...)
```

Plug in `ElectromagneticPolicy()` and the system *is* electromagnetism. Anything else is a hypothesis you
can test with the same engine, ledger and plots. Because mechanics conserve whatever a policy decides, a
policy can be wrong about physics without silently breaking the books.
"""),
code(r"""
import numpy as np
import pandas as pd
from dataclasses import dataclass

from stochastics.physics import *
from stochastics.physics.policies import *
"""),
md(r"""
## 1. Spin and polarity clocks

Each particle has a clock. **spin** is its speed (θ-like), **polarity** its signed amplitude (charge-like).
The value other particles feel is `polarity_now = polarity · waveform(phase)`, with
`phase += spin · dt / 4` per step. With polarity −1, spin 0.5 and dt = 1 the triangle waveform reproduces
the sequence from the brief:
"""),
code(r"""
pd.DataFrame({'t': range(13), 'polarity_now': polarity_sequence(polarity=-1, spin=0.5, steps=12)}).T
"""),
md(r"""
`energy` stays the particle's internal energy reservoir (collision heat goes in, emission spends it), so the
energy ledger keeps balancing; the cycling value lives in `polarity_now`. Waveforms: `'triangle'` (default),
`'cosine'`, `'square'`.
"""),
code(r"""
particles = ParticleSystem.random(
    n=160, preset='em', seed=210, dimension=3, random_phase=True,
    clusters=[Cluster(mean=[0, 0, 0], std=0.8), Cluster(mean=[2.0, -1.0, 0], std=0.6)])
particles.charge[:] = 0.0          # only the clocks drive attraction in this lab
particles.spin[:] = np.random.default_rng(1).choice([0.25, 0.5, 1.0], particles.n_live)
particles[0]
"""),
code(r"""
plot_polarity_clocks(particles, duration=16)
"""),
md(r"""
## 2. Sanity anchor: the EM policy *is* Coulomb

Same particles, same engine: once with the built-in `Coulomb` law, once with `ElectromagneticPolicy` as the
external policy. The trajectories agree to the last bit.
"""),
code(r"""
charged = ParticleSystem.random(n=80, preset='em', seed=5, dimension=3)
a = simulate(charged.copy(), law=Coulomb(k=0.2), steps=400, dt=0.01, record_every=40)
b = simulate(charged.copy(), external=ElectromagneticPolicy(k=0.2), steps=400, dt=0.01, record_every=40)
print('max |Δposition| =', np.abs(a.history[['x', 'y', 'z']].to_numpy() - b.history[['x', 'y', 'z']].to_numpy()).max())
"""),
md(r"""
## 3. Starting hypothesis: opposites attract, with clocks

`OppositesAttract` sets affinity `A_ij = −p_i(t)·p_j(t)` (optionally plus static charge and anti-phase
terms). As clocks drift in and out of step, a pair attracts, goes neutral, then repels. The clocks pump
energy in and out of the system; that work is booked in the **`drive`** ledger, so the accounted energy
still stays flat. Starting cold with drag lets the system settle — with a potential plus friction, the
dynamics descends the policy's energy landscape (the "optimising" behaviour), while the clocks keep
reshaping that landscape. Opposite pairs that touch slowly enough are captured into composites.
"""),
code(r"""
particles.velocity[:] = 0.0                                   # cold start
policy_external = OppositesAttract(strength=0.05, softening=0.1)
policy_internal = InternalPolicy(bond_speed=0.3)

result = simulate(particles, external=policy_external, internal=policy_internal,
                  rules=[Drag(gamma=0.3)], steps=3000, dt=0.01, record_every=25)
report(result)
"""),
code(r"""
animate_3d(result, max_frames=80, color_by='polarity_now')
"""),
md(r"""
## 4. Write your own

Override one method per idea. Below: an external policy that adds a short-range "same-spin" preference on top
of opposite polarity, and an internal policy where particles that bounce **phase-lock** their clocks
(a Kuramoto-like rule). Both are a few lines; conservation is still guaranteed by the mechanics.
"""),
code(r"""
@dataclass
class OppositePolaritySameSpin(ExternalPolicy):
    spin_weight: float = 0.5

    def affinity(self, a, b):
        polarity = -np.outer(a.polarity_now, b.polarity_now)        # opposites attract
        same_spin = np.isclose(a.spin[:, None], b.spin[None, :])    # kin attract a little
        return polarity + self.spin_weight * same_spin


@dataclass
class PhaseLocking(InternalPolicy):
    coupling: float = 0.5        # 0 = independent clocks, 1 = fully synchronise on contact

    def decide(self, state, contact, t):
        outcome = super().decide(state, contact, t)
        if outcome.kind == 'bounce':
            i, j = contact.i, contact.j
            gap = (state.phase[j] - state.phase[i] + 0.5) % 1.0 - 0.5
            state.phase[i] += 0.5 * self.coupling * gap
            state.phase[j] -= 0.5 * self.coupling * gap
            state.touch()           # tell the engine charges changed -> energy re-booked
        return outcome


lab = ParticleSystem.random(n=160, preset='em', seed=3, dimension=2, random_phase=True,
                            clusters=[Cluster(mean=[0, 0], std=1.0)])
lab.charge[:] = 0.0
lab.radius[:] = 0.05
custom = simulate(lab, external=OppositePolaritySameSpin(strength=0.02, softening=0.1),
                  internal=PhaseLocking(coupling=0.6, bond_speed=0.2),
                  rules=[Drag(gamma=0.1)], steps=2500, dt=0.01, record_every=25)
custom.summary()
"""),
code(r"""
animate_2d(custom, max_frames=80, color_by='polarity_now', trail=5)
"""),
md(r"""
## 5. Emission and absorption

`ThermalEmission` is an example internal policy: collisions heat particles; when a composite's internal
energy per unit mass passes a threshold it **emits** its least-bound constituent (with recoil), and quanta
captured by other particles are logged as **absorb**. Every step of that cycle is in the ledger.
"""),
code(r"""
hot = ParticleSystem.random(n=100, preset='em', seed=7, dimension=2)
emission = simulate(hot, external=OppositesAttract(strength=0.05, softening=0.1),
                    internal=ThermalEmission(threshold=0.02, release=0.5),
                    steps=1500, dt=0.01, record_every=25)
display(emission.event_counts().to_frame().T)
emission.summary()
"""),
md(r"""
## 6. Toward agents: non-reciprocal policies

Nothing in the engine requires Newton's third law. Setting `reciprocal=False` allows A to chase B while B
flees A — predator/prey, or traders chasing liquidity that moves away. There is no potential then, so the
energy residual is reported as *n/a*, but momentum/mass bookkeeping and all plots still work. A Gaussian
kernel makes the interaction local, as it usually is for agents.
"""),
code(r"""
@dataclass
class ChaseAndEvade(ExternalPolicy):
    def affinity(self, a, b):
        predator_a = (a.cluster_id == 0)[:, None]
        predator_b = (b.cluster_id == 0)[None, :]
        return np.where(predator_a & ~predator_b, 1.0,          # predators chase prey
               np.where(~predator_a & predator_b, -1.5,         # prey flee predators
               np.where(predator_a & predator_b, -0.2, 0.3)))   # predators spread out, prey flock

herd = ParticleSystem.random(n=120, seed=4, dimension=2, clusters=[
    Cluster(mean=[-2, 0], std=0.4, weight=0.15), Cluster(mean=[1, 0], std=1.0, weight=0.85)])
chase = simulate(herd, external=ChaseAndEvade(strength=0.3, kernel=Gaussian(length=0.8), reciprocal=False),
                 rules=[Drag(gamma=0.6)], steps=1500, dt=0.01, record_every=15)
animate_2d(chase, max_frames=80, color_by='cluster', trail=6)
"""),
md(r"""
## Where to take it next

* **External policy:** distance-dependent sign changes (attract far / repel near) via a custom `Kernel`;
  affinities that depend on internal energy; learnable affinity matrices fitted to observed trajectories.
* **Internal policy:** rules for how spin and polarity combine on merge (`combine`), energy-dependent spin,
  decay chains (`emit` with several constituent types), phase-locking as an interaction in its own right.
  `InternalPolicy(allow_merge=False)` keeps the restitution gradient but never fuses particles.
* **Agents:** replace `polarity_now` with any state (inventory, attention, opinion) and the same machinery
  runs population, market or attention dynamics — the ledger then becomes a budget of the conserved quantity.
"""),
]


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    nbf.write(nb, HERE / name)


if __name__ == '__main__':
    write(showcase, '01_physics_showcase.ipynb')
    write(lab, '02_policy_lab.ipynb')
    print('written')
