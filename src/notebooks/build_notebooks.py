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


# ============================================================== workbench
workbench = [
md(r"""
# Workbench — the whole `stochastics` package in one place

Run this notebook to check that every part imports, run the tests for any part, and try quick experiments
while you develop. Code is reloaded automatically when you save a file (`%autoreload 2`), and the tests
always run in a fresh process, so they see exactly what is on disk.

| part | what it is | tests |
|---|---|---|
| `physics` | particles, gravity/EM, composition collisions, energy ledger, policy lab | `tests/test_physics.py` |
| `baseline` | stochastic processes: random walk … GBM, OU, CIR, Poisson, SDE schemes, jumps, fBM, Hawkes | `tests/test_baseline.py` |
| `popdyn` | ecosystems, epidemics, rock-paper-scissors, wealth economies, segregation | `tests/test_popdyn.py` |
| `finance` | insurance & ruin, market models (v1–4), limit order book, agent-based exchange (v5–7), real-data benchmark | `tests/test_finance*.py` |
"""),
code(r"""
%load_ext autoreload
%autoreload 2

import numpy as np
import pandas as pd

import stochastics
from stochastics import dev, physics, baseline, popdyn, finance
from stochastics.finance import insurance, market, lob, exchange, realdata, benchmark

print('stochastics', stochastics.__version__, 'loaded from', stochastics.__file__)
dev.health()
"""),
md(r"""
## Tests

`dev.test()` runs everything; `dev.test('baseline')` runs one part; extra pytest arguments pass through,
e.g. `dev.test('popdyn', '-k', 'wealth')` or `dev.test('physics', '-x')` (stop at the first failure).
"""),
code(r"""
dev.test()
"""),
code(r"""
dev.test('baseline', '-k', 'convolution or moments')
"""),
md(r"""
## physics — quick check
"""),
code(r"""
from stochastics.physics import ParticleSystem, Environment, Gravity, Collisions, simulate

p = ParticleSystem.random(n=120, preset='em', dimension=2, seed=1)
r = simulate(p, Environment(law=Gravity(G=0.05), rules=[Collisions()]), steps=400, dt=0.01, record_every=10)
r.summary()
"""),
md(r"""
## baseline — quick check
"""),
code(r"""
from stochastics.baseline import process, combine

X = process(t=100, n=1000, seed=1)
X.randomwalk()
print('final mean', round(X.mean.iloc[-1], 3), '| final std', round(X.std.iloc[-1], 3))
X.plot()
"""),
code(r"""
P = process(t=250, n=2000, seed=2)
S = combine([0.5, 0.2, 0.3], [P.gbm(mu=1, sigma=2), P.abm(mu=2, sigma=0.5), P.gbm(mu=1, sigma=3)])
S.describe()
"""),
md(r"""
## popdyn — quick check
"""),
code(r"""
from stochastics.popdyn import Ecosystem, Economy

eco = Ecosystem(width=50, height=50, seed=3).run(400)
display(eco.totals())
econ = Economy(n=5000, seed=1).run(60)
econ.summary()
"""),
md(r"""
## finance — quick checks

Full tours are in notebooks 05 (insurance), 06 (market models, with the real-data benchmark on daily data) and 07
(agent market, with the benchmark on intraday data). Here: one line from each.
"""),
code(r"""
claims = insurance.ClaimProcess(insurance.Poisson(rate=120), insurance.weibull_with_mean(1.4, 1000))
print('E[N] E[X] =', round(claims.expected_total(T=1)))
firm = insurance.Company.priced(claims, capital=30_000, principle='std_dev', beta=0.5)
display(insurance.premium_table(claims.aggregate(T=1))[['premium', 'loading in sd']])
firm.ruin_exact([0, 30_000, 60_000])
"""),
code(r"""
market.compare_versions(streams=100, steps=2000, seed=1).table.round(3)
"""),
code(r"""
day = exchange.simulate('minimal', seed=1, ticks=600, schedule=False)
display(day.summary())
display(day.by_type().round(1))
day.conservation()
"""),
md(r"""
The benchmark needs no download for this check: it scores two models against a simulated version-4 market. (With
real data: `realdata.sample('sp500')`, see notebooks 06 and 07.)
"""),
code(r"""
fake = market.Market(version=4, steps=3000, seed=1).run().returns
benchmark.compare(fake, [benchmark.Gaussian(), benchmark.GARCH()], replicates=8).table[['rms error', 'p-value']].round(2)
"""),
code(r"""
dev.test('finance')
"""),
md(r"""
## Scratch
"""),
code(r"""
"""),
]

# ============================================================== baseline
baseline_nb = [
md(r"""
# `stochastics.baseline` — stochastic processes

```python
X = process(t=100, n=1000)     # t time steps, n runs; horizon T = 1, so dt = T / t
X.randomwalk()                 # generate (returns the Paths, and X remembers them)
X.mean, X.std, X.var           # statistics across runs, as Series over time
X.plot()                       # paths on the left, terminal distribution on the right
```

Every process carries its **theory** (mean, variance, and the terminal density where known), so plots overlay
theory on simulation and `describe()` compares them. Exact transition sampling is used wherever it exists;
Euler–Maruyama and Milstein are there for any SDE and for comparing discretisation error.
"""),
code(r"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stochastics.baseline import *
"""),
md(r"""
## 1. Random walk — your API
Each step is +1 with probability p and −1 otherwise. After k steps the mean is k(2p − 1) and the variance
4p(1 − p)k; the orange dots on the right are the exact binomial probabilities.
"""),
code(r"""
X = process(t=100, n=1000, seed=1)
X.randomwalk()
pd.DataFrame({'mean': X.mean, 'std': X.std, 'var': X.var}).iloc[::20]
"""),
code(r"""
X.plot()
"""),
md(r"""
## 2. The core continuous processes

| process | SDE | use |
|---|---|---|
| Wiener / Brownian motion | dW, increments N(0, dt) | the building block of everything below |
| arithmetic BM | dX = μ dt + σ dW | additive noise with drift; normal at every t |
| geometric BM | dX = μX dt + σX dW | prices: positive, lognormal, returns independent of level |
| Ornstein–Uhlenbeck | dX = θ(μ − X) dt + σ dW | mean reversion (spreads, rates, Hurst < 0.5 behaviour) |
| CIR | dX = κ(θ − X) dt + σ√X dW | mean-reverting and non-negative (rates, variance) |
| Poisson | N(t) jumps by 1 at rate λ | counting arrivals (orders, defaults, events) |
"""),
code(r"""
P = process(t=250, n=3000, seed=7)
P.gbm(mu=0.4, sigma=0.5).plot()
"""),
code(r"""
P.ou(theta=4, mu=1.0, sigma=0.6, x0=-1).plot()
"""),
code(r"""
P.cir(kappa=2, theta=0.04, sigma=0.3, x0=0.09).plot()
"""),
code(r"""
P.poisson(rate=8).plot()
"""),
md(r"""
`describe()` puts simulated terminal moments next to theory; `z (mean)` is the sample mean's distance from the
theoretical mean in standard errors (|z| < 3 is what sampling noise looks like).
"""),
code(r"""
pd.concat({name: getattr(P, name)(**kw).describe()[['simulated', 'theory']].loc[['mean', 'std']]
           for name, kw in [('wiener', {}), ('abm', dict(mu=2, sigma=0.5)), ('gbm', dict(mu=0.4, sigma=0.5)),
                            ('ou', dict(theta=4, mu=1, sigma=0.6)), ('cir', {}), ('poisson', dict(rate=8))]})
"""),
md(r"""
## 3. Euler–Maruyama and Milstein for any SDE

`sde(drift, diffusion, x0, scheme=...)` solves dX = a(t, X) dt + b(t, X) dW for any vectorised a and b.
Below, both schemes integrate GBM with the *same* Brownian increments as the exact solution, so the pathwise
error is pure discretisation error: Euler converges like √dt, Milstein like dt.
"""),
code(r"""
rows = []
for steps in (8, 16, 32, 64, 128, 256):
    Q = process(t=steps, n=4000, seed=3)
    noise = Q.noise()
    exact = Q.gbm(0.5, 0.8, noise=noise).final
    for scheme in ('euler', 'milstein'):
        approx = Q.sde(lambda t, x: 0.5 * x, lambda t, x: 0.8 * x, x0=1.0, noise=noise, scheme=scheme).final
        rows.append({'dt': 1 / steps, 'scheme': scheme, 'strong error': np.mean(np.abs(approx - exact))})
err = pd.DataFrame(rows).pivot(index='dt', columns='scheme', values='strong error')
slopes = {c: round(float(np.polyfit(np.log(err.index), np.log(err[c]), 1)[0]), 2) for c in err}
ax = err.plot(loglog=True, marker='o', figsize=(6, 4), title='Strong error vs dt  (slope = order)')
ax.set_ylabel('E|X_approx − X_exact| at T')
print('estimated orders:', slopes)
"""),
md(r"""
## 4. More processes worth having

* **Brownian bridge** — BM pinned at both ends (interpolation, barrier crossing between observations).
* **Compound Poisson / Merton jump diffusion** — GBM plus lognormal jumps: fat tails and gaps.
* **Heston** — price with its own mean-reverting variance; ρ < 0 gives the leverage effect.
* **Fractional BM** — memory: Hurst H > ½ trends, H < ½ mean-reverts ("rough").
* **Hawkes** — self-exciting arrivals: each event raises the chance of the next (order flow, trade clustering).
* **Regime switching** — drift and volatility follow a hidden Markov chain (calm ↔ stressed).
* **GARCH(1,1)** — volatility clustering in discrete returns.
"""),
code(r"""
Q = process(t=300, n=400, seed=11)
plot_many([Q.brownian_bridge(0, 1), Q.compound_poisson(rate=6, jump_mean=0.1, jump_std=0.3),
           Q.merton(mu=0.1, sigma=0.2, jump_rate=3, jump_mean=-0.1, jump_std=0.1),
           Q.heston(), Q.fbm(hurst=0.25), Q.fbm(hurst=0.8),
           Q.hawkes(mu=2, alpha=4, beta=6), Q.regime_switching(), Q.garch()])
"""),
code(r"""
F = process(t=1000, n=500, seed=4)
pd.DataFrame({'H (true)': [0.25, 0.5, 0.75],
              'H (estimated from paths)': [F.fbm(h).hurst() for h in (0.25, 0.5, 0.75)]})
"""),
code(r"""
H = process(t=500, n=2000, seed=5).heston(rho=-0.7)
fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
for i in range(20):
    axs[0].plot(H.times, H.values[i], lw=0.7)
    axs[1].plot(H.times, H.extras['variance'].values[i], lw=0.7)
axs[0].set_title('Heston price paths'); axs[1].set_title('their variance v(t) (CIR)')
plt.show()
dv = np.diff(H.extras['variance'].values, axis=1)
print('corr(return, change in variance) =', round(np.corrcoef(H.log_returns.ravel(), dv.ravel())[0, 1], 3))
"""),
md(r"""
## 5. Combining processes: weighted sums are convolutions

`0.5*X + 0.2*Y + 0.3*Z` builds a new set of paths, run by run (a portfolio of the three). When X, Y and Z are
driven by **independent** noise:

* means add: E[S] = 0.5 E[X] + 0.2 E[Y] + 0.3 E[Z];
* variances add with squared weights;
* the density of S at T is the **convolution** of the densities of 0.5X_T, 0.2Y_T and 0.3Z_T. The package
  computes it numerically (exact per-bin probabilities, then FFT convolution) and overlays it in orange.
"""),
code(r"""
P = process(t=250, n=5000, seed=1)
X = P.gbm(mu=1, sigma=2)        # 'geometric Brownian motion with mean 1 and std 2'
Y = P.abm(mu=2, sigma=0.5)      # 'arithmetic Brownian motion with mean 2, std 0.5'
Z = P.gbm(mu=1, sigma=3)
S = 0.5 * X + 0.2 * Y + 0.3 * Z   # same as combine([0.5, 0.2, 0.3], [X, Y, Z])
S.plot()
S.describe()
"""),
md(r"""
GBM with σ = 2–3 is extremely skewed (median far below mean), so the sample mean converges slowly; the
theory column is the exact value. The plot switches to a log axis for heavy tails and shows the density per
log₁₀ unit.

### Sum vs mixture
A **mixture** is a different blend: each run *is* X with probability 0.5, Y with 0.2, Z with 0.3. Same mean,
but its density is the weighted *average* of the component densities, not their convolution. With three
normal components the difference is stark: the sum is one narrow normal, the mixture is three humps.
"""),
code(r"""
M = mixture([0.5, 0.2, 0.3], [X, Y, Z], seed=3)
plot_terminal(S, M, labels=['0.5X + 0.2Y + 0.3Z (sum → convolution)', 'mixture (average of densities)'])
"""),
code(r"""
Q = process(t=200, n=5000, seed=2)
A, B, C = Q.abm(-2, 0.6), Q.abm(0.5, 0.6), Q.abm(3, 0.6)
plot_terminal(combine([0.5, 0.2, 0.3], [A, B, C]), mixture([0.5, 0.2, 0.3], [A, B, C], seed=1),
              labels=['weighted sum', 'mixture'], title='Three normals: sum vs mixture')
"""),
md(r"""
### Correlated drivers
Independence is what makes the variances add. With shared or correlated noise the package knows the
variance is no longer additive (`S.independent` is False and the theory variance is dropped), and
`process.correlated(k, rho)` gives correlated Brownian drivers for building dependent processes.
"""),
code(r"""
Q = process(t=250, n=20000, seed=8)
rows = []
for rho in (-0.9, -0.5, 0.0, 0.5, 0.9):
    a, b = Q.correlated(2, rho)
    port = 0.5 * Q.gbm(0.1, 0.3, noise=a) + 0.5 * Q.gbm(0.1, 0.3, noise=b)
    rows.append({'rho': rho, 'portfolio sd': port.final.std(), 'independent?': port.independent})
pd.DataFrame(rows).set_index('rho')
"""),
md(r"""
## 6. Path analytics
"""),
code(r"""
G = process(t=500, n=5000, seed=9).gbm(mu=0.1, sigma=0.3)
hit = G.first_passage(1.5)
dd = G.max_drawdown()
pd.Series({'P(hit 1.5 before T)': np.mean(~np.isnan(hit)), 'median time to hit': np.nanmedian(hit),
           'median max drawdown': np.median(dd), '95% max drawdown': np.quantile(dd, 0.95),
           'lag-1 autocorrelation of increments': G.acf(5).iloc[1]})
"""),
]

# ============================================================== popdyn
popdyn_nb = [
md(r"""
# `stochastics.popdyn` — population dynamics

Agent-based models where individuals have traits, interact locally, and the population-level picture emerges.
Every result has `.counts` (time series), `.agents` (one row per individual with traits and life history),
`.plot()` (a dashboard) and, for spatial models, `.animate()`. Where something is conserved (energy in the
ecosystem, wealth under exchange) there is a ledger that must close, and it is checked.

1. **Ecosystem** — grass → rabbits → wolves → grass, with heritable speed, vision and fertility
2. **Epidemic** — SIR/SEIR on a 2D world, with the full infection tree and superspreaders
3. **Rock-paper-scissors** — cyclic dominance on a lattice: spiral waves and the mobility threshold
4. **Wealth economy** — 10,000 people with traits; luck, talent, investment, trading and tax
5. **Schelling segregation** — mild preferences, strongly segregated outcome
"""),
code(r"""
import numpy as np
import pandas as pd
from stochastics.popdyn import *
"""),
md(r"""
## 1. Ecosystem: grass, rabbits, wolves

Rabbits look for grass and flee wolves they can see; wolves look for rabbits. Every animal burns energy
(more when it moves fast or sees far), breeds when it has enough energy, and passes its traits to its
children with small mutations, so the population evolves. Dead animals fertilise the ground, so grass grows
back from them and the loop closes. The energy ledger (sun + fertiliser in; digestion, metabolism and
detritus out) must balance to rounding error, and it does.
"""),
code(r"""
eco = Ecosystem(seed=7).run(steps=2000)
eco.totals()
"""),
code(r"""
eco.plot()
"""),
code(r"""
eco.animate(max_frames=45)
"""),
md(r"""
**Who ate more?** The totals table above answers it in items and in energy. **Which rabbits did best?**
The longest-lived and the hungriest rabbits, with their traits, and how each trait relates to lifespan:
"""),
code(r"""
cols = ['generation', 'lifespan', 'cause', 'speed', 'vision', 'fertility', 'food', 'offspring']
display(eco.leaders('lifespan', 8, species='rabbit')[cols])
display(eco.leaders('food', 8, species='rabbit')[cols])
eco.trait_effects('rabbit', target='lifespan')
"""),
md(r"""
## 2. Epidemic with superspreaders

3,000 people wander a 2D world. Infectiousness is drawn from a skewed (gamma) distribution, so a minority
infects most others. Thick lines are the agents; thin lines are the classic well-mixed SIR equations with the
same parameters — the gap is the effect of space: people mostly meet neighbours who are already infected or
immune.
"""),
code(r"""
epi = Epidemic(n=3000, sigma=0.5, fatality=0.01, seed=3).run(80)
display(epi.summary())
epi.plot()
"""),
code(r"""
epi.animate(max_frames=45)
"""),
code(r"""
# the biggest spreaders and their traits
epi.leaders('secondary', 8)[['mobility', 'susceptibility', 'infectiousness', 'secondary', 'infected_at']]
"""),
code(r"""
# mobility is a policy lever: same disease, different mixing
cols = ['ever infected', 'peak infectious', 'early R (first 10% of cases, simulated)']
pd.DataFrame({m: Epidemic(n=3000, mobility=m, seed=3).run(80).summary()['value'][cols]
              for m in (0.005, 0.03, 0.15)}).T.rename_axis('mobility')
"""),
md(r"""
## 3. Rock-paper-scissors: spirals keep everyone alive

Rock beats scissors, scissors beats paper, paper beats rock. In a well-mixed world the cycle drifts until one
species is left. On a lattice with little mobility, the cycle organises into rotating spiral waves that let
all three coexist; above a critical mobility (≈ 4.5 × 10⁻⁴) the spirals outgrow the world and diversity
collapses.
"""),
code(r"""
rps = RockPaperScissors(size=200, mobility=1e-4, seed=2).run(800)
rps.plot()
"""),
code(r"""
rps.animate(max_frames=50)
"""),
code(r"""
pd.DataFrame({M: RockPaperScissors(size=60, mobility=M, seed=3).run(3000, frame_every=3000).summary()['value']
              for M in (1e-4, 5e-3, 2e-2)}).T[['species surviving']].rename_axis('mobility M')
"""),
md(r"""
## 4. Wealth: 10,000 people, traits and luck

Each person has talent, intelligence, creativity, charisma, luck, risk appetite and thrift (all editable via
`Trait`). Rules act each step (half a year):

* `LuckEvents` — lucky/unlucky events; a lucky break only pays off if you have the talent to seize it
  (Pluchino, Biondo & Rapisarda, *Talent vs Luck*, 2018);
* `Investment` — multiplicative returns; skill shifts the drift, risk appetite scales the volatility;
* `YardSale` — random pairs bet a share of the poorer one's wealth.

The wealth ledger (created − destroyed = change in total) closes to rounding error.
"""),
code(r"""
econ = Economy(n=10_000, rules=[LuckEvents(), Investment(), YardSale(stake=0.05)], seed=1).run(steps=80)
econ.summary()
"""),
code(r"""
econ.plot()
"""),
md(r"""
**Is there anyone with a trait set that made them richer than everyone else?** The richest ten, with their
traits and how many lucky / unlucky events they had, then the population-wide trait effects:
"""),
code(r"""
display(econ.leaders(10).round(2))
econ.trait_effects().round(3)
"""),
md(r"""
The richest are rarely the most talented: they are moderately talented people who had many lucky breaks and
almost no unlucky ones. Talent matters (positive ρ), luck matters far more.

### Policy experiments
Same population, different rules. Pure exchange conserves wealth exactly; tax redistributes it; luck and
investment create and destroy it.
"""),
code(r"""
setups = {
    'kinetic exchange (savings)': [KineticExchange()],
    'yard sale': [YardSale()],
    'yard sale + 2% wealth tax': [YardSale(), Tax(0.02)],
    'luck': [LuckEvents()],
    'luck + investment + 1% tax': [LuckEvents(), Investment(), Tax(0.01)],
    'wages + investment + spending': [Wages(), Investment(), Consumption()],
}
cols = ['Gini', 'top 1% share', 'bottom 50% share', 'Pareto tail alpha (top 1%)', 'ledger residual']
pd.DataFrame({name: Economy(n=10_000, rules=rules, seed=2).run(100).summary()['value'][cols]
              for name, rules in setups.items()}).T
"""),
md(r"""
Write your own rule with one method, `apply(market)`, that changes `market.wealth` and returns the flows:
"""),
code(r"""
class Inheritance:
    # people from wealthy families get a yearly transfer from outside the economy
    def apply(self, m):
        gift = 0.02 * m.wealth * m.dt * (m.traits['family'] > 0.9)
        m.wealth += gift
        return {'created': gift.sum()}

traits = [*DEFAULT_TRAITS, Trait('family', 'uniform', 0, 1)]
econ2 = Economy(n=10_000, traits=traits, rules=[Inheritance(), LuckEvents(), Investment()], seed=4).run(80)
econ2.trait_effects().round(3).loc[['luck', 'family', 'talent', 'intelligence']]
"""),
md(r"""
## 5. Schelling segregation

Two groups; everyone is content as long as at least 35% of their neighbours are like them (they would happily
be a minority). Discontent people move. The town still sorts itself into large single-group blocks.
"""),
code(r"""
seg = Schelling(size=80, tolerance=0.35, seed=1).run(60)
seg.plot()
"""),
code(r"""
seg.animate()
"""),
md(r"""
## More models to build on this

* **Sugarscape** (Epstein & Axtell) — agents with vision and metabolism harvest a renewable landscape:
  wealth inequality from ecology; a natural bridge between the ecosystem and the economy here.
* **Opinion dynamics** — voter model, bounded confidence (Deffuant / Hegselmann–Krause): polarisation and
  consensus; the seed for the social-media attention model.
* **Evolutionary games** — prisoner's dilemma on a lattice: when does cooperation survive?
* **Kinetic wealth with debt** — let wealth go negative (credit) and add bankruptcy.
* **Fitting** — calibrate rule parameters to real distributions (wealth surveys, case counts) with
  approximate Bayesian computation.
"""),
]


# ============================================================== finance: insurance
insurance_nb = [
md(r"""
# `stochastics.finance.insurance` — claims, premiums and survival

The classic **collective risk model**: claims *arrive* by a counting process and each has a *size*,

$$S(t)=\sum_{i=1}^{N(t)} X_i,\qquad \mathbb E[S(t)]=\mathbb E[N(t)]\;\mathbb E[X].$$

```python
from stochastics.finance import insurance as ins

claims = ins.ClaimProcess(ins.Poisson(rate=120), ins.weibull(shape=1.4, scale=900))
claims.expected_total(T=1)            # E[N] E[X] for one year
A = claims.aggregate(T=1)             # the whole distribution of the year's total
ins.premium_table(A)                  # every premium principle side by side
firm = ins.Company.priced(claims, capital=30_000, principle='std_dev', beta=0.5)
firm.plot()                           # surplus paths, survival curve, ruin vs capital
```

| piece | choices |
|---|---|
| arrivals | `Poisson(rate)` (interarrival times Exp), `Renewal(gap)` (any gap law), `Hawkes(mu, alpha, beta)` (claims cluster) |
| sizes | any frozen `scipy.stats` law; shortcuts `weibull`, `weibull_with_mean`, `lognormal`, `gamma`, `pareto`, `exponential` |
| premium principles | expected value, variance, standard deviation, percentile (VaR), TVaR, exponential, Esscher, proportional hazard, Wang |
| survival | surplus $U(t)=u+ct-S(t)$; ruin is detected *exactly* at claim epochs; Lundberg and Pollaczek–Khinchine for the exact answer |

*Time is in years. Not modelled: interest on the surplus, expenses, reinsurance, dependence between claim sizes.*
"""),
code(r"""
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from stochastics.finance import insurance as ins

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda v: f'{v:,.0f}' if abs(v) >= 1e4 else f'{v:.4g}')
"""),
md(r"""
## 1. Arrivals × sizes

120 claims a year on average, Weibull sizes with shape 1.4 (a little lighter than exponential at the very
small end, a longer tail than a normal) and mean 1,000. The expected total is $\mathbb E[N]\,\mathbb E[X]=120{,}000$,
and because the arrivals are Poisson the variance is also exact: $\mathrm{Var}\,S=\lambda\,\mathbb E[X^2]$.
"""),
code(r"""
claims = ins.ClaimProcess(ins.Poisson(rate=120), ins.weibull_with_mean(shape=1.4, mean=1000))

print('expected total claims per year  E[N] E[X] =', f'{claims.expected_total(T=1):,.0f}')
var, exact = claims.variance_total(T=1)
print('standard deviation of the year  =', f'{var ** 0.5:,.0f}', '(exact)' if exact else '(approximate)')
claims.loss_ratio_table(1)
"""),
md(r"""
**Check by simulation.** `sample` draws whole years of claims (arrival times *and* sizes). The mean of the
yearly totals should match $\mathbb E[N]\mathbb E[X]$, interarrival times should be Exp(120), and the
distribution of the year's total claims is known exactly (FFT of the compound-Poisson characteristic
function), so we can compare the empirical CDF with it.
"""),
code(r"""
sample = claims.sample(T=1, n=20_000, seed=1)
A = claims.aggregate(T=1)

fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
gaps = sample.interarrival()
ax[0].hist(gaps, bins=80, density=True, alpha=.6, label='simulated')
g = np.linspace(0, gaps.max(), 200)
ax[0].plot(g, 120 * np.exp(-120 * g), 'k', label='Exp(120)')
ax[0].set(title='time between claims (years)', xlim=(0, 0.06)); ax[0].legend()

n = sample.counts()
k = np.arange(n.min(), n.max() + 1)
ax[1].hist(n, bins=np.arange(n.min(), n.max() + 2) - .5, density=True, alpha=.6, label='simulated')
ax[1].plot(k, stats.poisson(120).pmf(k), 'k.-', label='Poisson(120)')
ax[1].set(title='claims per year'); ax[1].legend()

tot = np.sort(sample.totals())
ax[2].plot(tot, np.arange(1, len(tot) + 1) / len(tot), label='simulated')
ax[2].plot(tot, A.cdf(tot), 'k--', label='exact (FFT)')
ax[2].set(title='total claims per year: CDF'); ax[2].legend()
plt.tight_layout(); plt.show()

print(f'simulated mean {tot.mean():,.0f}   exact {A.mean:,.0f}   E[N]E[X] {claims.expected_total(1):,.0f}')
A.summary()
"""),
md(r"""
## 2. Does the *pattern* of arrivals matter?

Three arrival processes with the **same expected number of claims per year** (120):

* **Poisson** — claims independent, counts have variance = mean.
* **Renewal with Weibull gaps (shape 3)** — claims arrive more regularly than Poisson (variance < mean).
* **Hawkes** — every claim raises the chance of the next for a while (storms, epidemics, aftershocks). Counts have variance well above the mean.

Same mean total claims, very different risk.
"""),
code(r"""
sev = ins.weibull_with_mean(shape=1.4, mean=1000)
h0 = ins.Hawkes(mu=60, alpha=10, beta=20)                     # branching ratio alpha/beta = 0.5
hawkes = ins.Hawkes(mu=60 * 120 / h0.mean_count(1), alpha=10, beta=20)   # scale mu so E[N(1)] = 120

arrivals = {
    'Poisson': ins.Poisson(rate=120),
    'Renewal (Weibull gaps, k=3)': ins.Renewal(ins.weibull_with_mean(shape=3, mean=1 / 120)),
    'Hawkes (branching 0.5)': hawkes,
}
rows, aggs = {}, {}
for name, arr in arrivals.items():
    cp = ins.ClaimProcess(arr, sev)
    s = cp.sample(T=1, n=20_000, seed=2)
    n, tot = s.counts(), s.totals()
    rows[name] = {'mean claims': n.mean(), 'var of claims': n.var(), 'Fano (var/mean)': n.var() / n.mean(),
                  'mean total': tot.mean(), 'sd total': tot.std(), 'skew': stats.skew(tot),
                  '99% quantile': np.quantile(tot, .99)}
    aggs[name] = cp.aggregate(T=1, n=40_000, seed=3)
pd.DataFrame(rows).T
"""),
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name, cp_agg in aggs.items():
    x = np.linspace(80_000, 190_000, 400)
    ax[0].plot(x, cp_agg.cdf(x), label=name)
    ax[1].plot(x, 1 - cp_agg.cdf(x), label=name)
ax[0].set(title='CDF of the year\'s total claims'); ax[0].legend()
ax[1].set(title='tail: P(S > x)', yscale='log', ylim=(1e-5, 1)); ax[1].legend()
plt.tight_layout(); plt.show()
"""),
md(r"""
The means agree; the tails do not. Under a premium principle that charges for variance or tail risk,
clustered claims therefore cost **more** for the same expected loss, and an insurer that ignores clustering
under-prices and over-states its survival.
"""),
code(r"""
prem = pd.DataFrame({name: ins.premium_table(a)['premium'] for name, a in aggs.items()})
prem
"""),
md(r"""
## 3. Premium principles

`premium_table(A, base=0.5)` calibrates *every* principle so that its first-order loading is
$0.5\,\mathrm{sd}[S]$. The first three coincide by construction (they differ only in which
statistic they scale), so the interesting differences are in the **tails**: the percentile and TVaR
principles charge for capital to survive the 99th percentile, the exponential and Esscher premiums
weigh the right tail through the moment generating function, and the proportional-hazard premium is the
cheapest here.

* `variance`: $\mathbb E S+\alpha\,\mathrm{Var}\,S$ — the variance premium (what the brief calls the "variance" or "var-squared" principle)
* `std_dev`: $\mathbb E S+\beta\,\mathrm{sd}\,S$
* `percentile`: the $q$-quantile; `tvar`: expected loss beyond it
"""),
code(r"""
ins.premium_table(claims.aggregate(T=1))
"""),
md(r"""
**Heavy tails and the MGF.** The exponential-utility and Esscher premiums are built from the moment generating
function $\mathbb E\,e^{rS}$, which **does not exist** for lognormal, Pareto, or Weibull sizes with shape below 1.
The package says so (`inf`) instead of returning a number that depends on where the simulation stopped.
"""),
code(r"""
ln = ins.ClaimProcess(ins.Poisson(120), ins.lognormal(mu=6.4, sigma=1.0))
print('lognormal claims: mean claim size', round(ln.severity.mean()), '| moment generating function exists:', ln.mgf_ok(1e-4), '(lognormal) vs', claims.mgf_ok(1e-4), '(Weibull k=1.4)')
ins.premium_table(ln.aggregate(T=1, n=100_000, seed=1))
"""),
md(r"""
## 4. The insurer's survival

An insurer starts with capital $u$ and collects premium at rate $c$:
$U(t)=u+ct-S(t)$. Ruin is the first time $U<0$. Since $U$ only falls at claims, checking just after every claim
is **exact** — no time grid, no missed dips between steps.

`Company.priced(claims, capital, principle, ...)` sets the premium by a principle; `firm.plot()` shows
(1) surplus paths and their 5–95% fan, (2) the survival curve $P(\text{not ruined by } t)$ with a confidence band
and the infinite-horizon Pollaczek–Khinchine level, (3) ruin probability against starting capital — Monte-Carlo
against the exact bracket and the Lundberg bound — and (4) the distribution of one year's claims against the premium and VaR.
"""),
code(r"""
firm = ins.Company.priced(claims, capital=30_000, principle='std_dev', beta=0.5)
print(f'premium rate {firm.premium_rate:,.0f}/yr   safety loading {firm.safety_loading:.2%}   claims/yr {firm.claim_rate:,.0f}')
print('Lundberg adjustment coefficient R =', f'{firm.adjustment_coefficient():.3e}')
firm.plot(horizon=10, n=4000, seed=1)
"""),
code(r"""
res = firm.simulate(horizon=10, n=4000, seed=1)
res.summary()
"""),
md(r"""
**Exact ruin probabilities.** For Poisson arrivals the infinite-horizon ruin probability is the
Pollaczek–Khinchine geometric sum over ladder heights. `ruin_exact` discretises the ladder-height law in
two ways (rounding up and down) and Panjer-recurses each, so the true value lies **between** `lower` and
`upper`. The Lundberg bound $e^{-Ru}$ is an upper bound that is tight for large $u$. The Monte-Carlo column is
a 25-year horizon with a 95% confidence interval, so it can sit slightly *below* the infinite-horizon answer.
"""),
code(r"""
u = np.array([0, 10e3, 20e3, 30e3, 45e3, 60e3])
exact = firm.ruin_exact(u)
mc = []
for i, x in enumerate(u):
    r = ins.Company(claims, capital=float(x), premium_rate=firm.premium_rate).simulate(horizon=25, n=4000, seed=10 + i)
    p = r.ruin_probability()
    mc.append((p, *r.interval(p)))
exact[['MC, 25 years', 'MC low', 'MC high']] = mc
exact
"""),
md(r"""
## 5. What does the premium principle do to survival?

Same claims, same starting capital (30,000), and a premium set by each principle. The first-order-loading
principles all charge about the same, so their survival is about the same; the quantile-based premiums
charge 27-32% more and the insurer essentially never fails. The proportional-hazard premium is the cheapest
and the least safe. `compare_principles` uses common random numbers
(the same claim histories for every principle), so the differences are not noise.
"""),
code(r"""
cmp = ins.compare_principles(claims, capital=30_000, horizon=10, n=4000, seed=1)
cmp
"""),
md(r"""
## 6. Heavier tails cost survival

Hold the **mean claim and the premium fixed** (set by the standard-deviation principle for the Weibull k=1.4 case)
and make the claim sizes heavier-tailed (shape 0.6) or lighter (shape 3). Same expected loss and same income, very
different survival: ruin is a tail event.
"""),
code(r"""
rate = firm.premium_rate
rows = {}
for k in (0.6, 1.0, 1.4, 3.0):
    cp = ins.ClaimProcess(ins.Poisson(120), ins.weibull_with_mean(k, 1000))
    co = ins.Company(cp, capital=30_000, premium_rate=rate)
    r = co.simulate(horizon=10, n=4000, seed=5)
    p = r.ruin_probability()
    lo, hi = r.interval(p)
    rows[f'Weibull k={k}'] = {'sd of annual claims': cp.variance_total(1)[0] ** .5,
                              'ruin P (10y)': p, '95% CI low': lo, '95% CI high': hi}
pd.DataFrame(rows).T
"""),
md(r"""
## Next steps / ideas

* **Reinsurance** — excess-of-loss or quota-share layers inside `Company` (the surplus equation just changes).
* **Interest and dividends** — a barrier strategy: pay out surplus above a level, and ask for the expected discounted dividends *and* the ruin time.
* **Dependence** — claims whose sizes share a common factor, or a Hawkes process whose *marks* (sizes) excite it.
* **Link to the markets** — an insurer that invests its surplus in one of the stochastic markets of `finance.market`.
"""),
]

# ============================================================== finance: markets
markets_nb = [
md(r"""
# `stochastics.finance.market` — market models of growing complexity (versions 1–4)

```python
from stochastics.finance import market

r = market.Market(version=3, streams=100, steps=2000, seed=1).run()
r.price, r.streams, r.value          # the market, its constituent streams, the internal value
r.facts()                            # stylised facts (tails, volatility clustering, leverage...)
market.compare_versions(streams=100, steps=2000, seed=1).plot()
```

| version | what the price is |
|---|---|
| **1** | one Heston stochastic-volatility path |
| **2** | the **average** of 10 / 100 / 1000 streams (ABM, GBM, Brownian bridge, Heston, Merton), sharing a common factor |
| **3** | one **internal value** (a bridge, GBM, ABM or Heston path); the other streams are *traders* with views that fluctuate around it |
| **4** | as 3, but the internal value also receives **news**: jumps (lognormal size, random sign) arriving by a Poisson or Hawkes process; traders follow the value and get jumpier right after news |

Units: time in years, volatility annualised.
"""),
code(r"""
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stochastics.finance import market

pd.set_option('display.width', 160)
pd.set_option('display.float_format', lambda v: f'{v:,.0f}' if abs(v) >= 1e4 else f'{v:.4g}')
"""),
md(r"""
## All four versions, same seed

`stylized_facts` checks what real return series show: **excess kurtosis > 0** with tail index 3–5 (fat tails),
**acf(r) ≈ 0** (no linear predictability), **acf(|r|) clearly positive** (volatility clustering), and a
**negative leverage** correlation (falls raise volatility).
"""),
code(r"""
cmp = market.compare_versions(streams=100, steps=2000, seed=1)
cmp.table
"""),
code(r"""
cmp.plot()
"""),
md(r"""
## Version 1 — a single Heston path

Stochastic volatility with mean reversion and correlation between price and variance shocks gives
volatility clustering and a negative leverage effect from a single equation.
"""),
code(r"""
v1 = market.Market(version=1, steps=2000, seed=3).run()
print(v1.facts().round(3).to_string())
v1.plot()
"""),
md(r"""
## Version 2 — what averaging many streams does

Averaging $N$ **independent** noise sources divides the variance by $N$ and, by the central limit theorem,
*removes* fat tails and clustering: the average of 1,000 independent streams is nearly a smooth curve. The
streams therefore share a **common factor** (`correlation`), which is what keeps the average moving. The table
shows the effect of both knobs.
"""),
code(r"""
rows = {}
for n in (10, 100, 1000):
    for rho in (0.0, 0.25):
        f = market.Market(version=2, streams=n, correlation=rho, steps=2000, seed=11).run().facts()
        rows[f'N={n}, common factor {rho}'] = f[['annualised vol', 'excess kurtosis', 'tail index (Hill, top 5%)', 'acf(|r|), lag 1']]
pd.DataFrame(rows).T
"""),
md(r"""
## Version 3 — one internal value, traders around it

Pick the central stream (`value='gbm' | 'bridge' | 'abm' | 'heston'`); every other stream is a trader whose
view deviates from it (an Ornstein–Uhlenbeck gap, fast reversion = well-informed). The market price is the
average view. `tracking()` reports how closely the price follows the value.
"""),
code(r"""
for value in ('gbm', 'bridge', 'heston'):
    r = market.Market(version=3, streams=100, value=value, steps=2000, seed=21).run()
    print(f'{value:7s}', r.tracking().round(3).to_dict())
r3 = market.Market(version=3, streams=100, value='bridge', steps=2000, seed=21).run()
r3.plot()
"""),
md(r"""
## Version 4 — news

Poisson (`event_process='poisson'`) or Hawkes (`'hawkes'`, news clusters) arrivals; each event adds a signed
lognormal jump to the internal value. Traders under-react (`underreaction`) and become noisier right after
news (`panic`). Jumps are applied in log space and capped (`jump_cap`): a lognormal has no exponential moments,
so an uncapped $e^{+Y}$ would have infinite mean.
"""),
code(r"""
r4 = market.Market(version=4, streams=100, value='gbm', event_process='hawkes', steps=2000, seed=31).run()
print(len(r4.events), 'news events; first few:')
print(r4.events.head().to_string())
r4.plot()
"""),
code(r"""
cols = ['annualised vol', 'excess kurtosis', 'tail index (Hill, top 5%)', 'acf(|r|), lag 1', 'leverage corr(r, next r^2)', 'max drawdown']
variants = {
    'v1: single Heston': dict(version=1),
    'v3: no news': dict(version=3),
    'v3: Heston internal value': dict(version=3, value='heston'),
    'v4: Poisson news': dict(version=4, event_process='poisson'),
    'v4: Hawkes news': dict(version=4, event_process='hawkes'),
    'v4: Hawkes news, strong panic': dict(version=4, event_process='hawkes', panic=4.0),
    'v4: Hawkes news + Heston value': dict(version=4, event_process='hawkes', value='heston'),
    'v4: big jumps (median 4%)': dict(version=4, event_process='poisson', jump_median=0.04),
}
rows = {}
for name, kw in variants.items():          # average over 6 seeds: one path is too noisy to compare
    facts = [market.Market(streams=100, steps=2000, seed=100 + s, **kw).run().facts()[cols] for s in range(6)]
    rows[name] = pd.DataFrame(facts).mean()
pd.DataFrame(rows).T
"""),
md(r"""
## What to take from this

* A single Heston path already has volatility clustering (acf of |r| near 0.15); **averaging makes a market too well-behaved** unless the streams share a factor.
* Version 3 is Gaussian with no clustering. Fat tails and clustering have to come from somewhere: the internal value's own stochastic volatility, or the news.
* Version 4's **news** brings back fat tails (excess kurtosis rises with Poisson news and much more with clustered Hawkes news), and `panic` adds a little clustering. But in these runs **volatility clustering stays weak unless the internal value itself has stochastic volatility** (`value='heston'`). If you want clustering from the news alone, the next step is to make the *trader noise* scale with a slowly decaying news-intensity state instead of a one-off multiplier.

The agent-based market of notebook 07 produces these effects from the *behaviour* of traders instead of by construction.
"""),
]

# ============================================================== finance: agent market
agents_nb = [
md(r"""
# `stochastics.finance` — order book, agents and the exchange (versions 5–7)

The agent-based market is built from three layers, each usable on its own:

| module | what it is |
|---|---|
| `lob.OrderBook` | limit order book with price–time priority (a tested port of your `OrderBook.js`) |
| `agents` | trader types (your `bots.js` plus the brief's personalities and a utility trader), each a small class with an `on_tick(m)` |
| `exchange.simulate` | a population of agents trading through the book for a simulated day: the schedule, news, attention, results |
| `profiles` | version 7: traders defined by a **psychology vector** mapped to behaviour by replaceable functions |

```python
from stochastics.finance import exchange
res = exchange.simulate('game', seed=1)      # the website's ~1,000-agent day
res.summary(); res.by_type(); res.plot()
```
"""),
code(r"""
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from stochastics.finance import exchange, agents
from stochastics.finance.lob import OrderBook, BUY, SELL
from stochastics.finance.profiles import Profile, ProfiledTrader, Behavior, sample_profiles

pd.set_option('display.width', 180)
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', lambda v: f'{v:,.0f}' if abs(v) >= 1e4 else f'{v:.4g}')
"""),
md(r"""
## The order book

Limit orders match what crosses and the rest rests; market orders never rest; a trade happens at the
*resting* order's price; within a price level the oldest order goes first.
"""),
code(r"""
book = OrderBook()
book.on_trade(lambda t: print('trade', t))
book.limit('maker', SELL, 100.05, 50)
book.limit('maker', BUY, 99.95, 50)
book.limit('other', SELL, 100.10, 30)
print(book.market('taker', BUY, 60))          # eats 50 @ 100.05 and 10 @ 100.10
print('bid', book.best_bid(), 'ask', book.best_ask(), 'mid', book.mid(), 'spread', book.spread())
print(book.levels(SELL, 3), book.levels(BUY, 3))
book.check()
"""),
md(r"""
Your JavaScript `OrderBook.js` and this book were run on the same 11,689 random operations: every trade,
fill, resting quantity and book level was identical.
"""),
md(r"""
## A trading day (version 5 and 6)

`'game'` is the website's population (about 1,000 agents: 90% noise traders of four kinds, 10% market makers,
whales, trend followers, mean reverters...), with the day's schedule: an opening crowd, hourly arrivals and
departures, the US-open directional cohort at 14:30 and a closing crowd. Time runs in 15-second ticks from
09:30 to 18:00. Python's random numbers are not bit-identical to the JavaScript ones, so the *distributions*
agree, not each path (see the end of the notebook).
"""),
code(r"""
res = exchange.simulate('game', seed=1)
res.summary()
"""),
code(r"""
res.plot()
"""),
md(r"""
**Who makes money?** Profit is marked to the closing mid. Every trade has a buyer and a seller, so the books must
balance exactly; `conservation()` checks it.
"""),
code(r"""
display(res.by_type())
res.conservation()
"""),
md(r"""
The market makers earn the most, and most of them finish the day ahead; the large crowd of noise traders pays
in total. The rule-based bots are mixed, and that is the point of the exercise: **a strategy's profit depends on who
else is in the market**, so change the mix and the ranking changes. One day is one draw: the ordering of the large
groups is stable from day to day, the small groups (a handful of whales) are not.

**Stylised facts of the price this population produces:**
"""),
code(r"""
res.facts().round(3)
"""),
md(r"""
## The brief's personalities

`'brief'` is the version-6 mix: noise traders split into 20% value-anchored, 50% random and 30% genuine investors
(a running mean with stop-loss and take-profit), plus market makers, mean reverters (20/40 window cross,
price below 60MA − 1σ), whales, informed whales (which see some of the hidden fair value), trend followers (x% rise → buy,
sell after y% more, stop at z% fall) and volatility traders.
"""),
code(r"""
brief = exchange.simulate('brief', seed=2)
display(brief.summary())
brief.by_type()
"""),
md(r"""
## Your own population, your own agent

A population is a dict `{type: count}`. Extra agents join with `extra=[...]`; `PolicyAgent` is an empty slot:
its `policy(agent, m)` is called every tick with the market state `m` (price, `m.sma(n)`, `m.dev_z(n)`,
`m.move_z(k)`, `m.imbalance`, `m.value` ...).
"""),
code(r"""
def contrarian(agent, m):
    # lean against a stretched price with a limit order, flatten when the stretch is gone
    if abs(m.dev_z(60)) < 0.3:
        agent.cancel_all(); agent.flatten(); return
    if m.dev_z(60) > 2 and agent.shares > -300:
        agent.limit(SELL, m.mid + m.tick_sigma, 50, 20)
    if m.dev_z(60) < -2 and agent.shares < 300:
        agent.limit(BUY, m.mid - m.tick_sigma, 50, 20)

mine = [(agents.PolicyAgent, {'policy': contrarian, 'params': {'z': 2.0}}) for _ in range(25)]
r = exchange.simulate('game', seed=1, extra=mine)
r.by_type().loc[['policy', 'marketMaker', 'noise', 'whale']]
"""),
md(r"""
## Version 5 — agents with internal wealth, risk aversion and learning

`UtilityTrader` holds a target fraction of its wealth in the stock: the mean-variance / Kelly share
$\;\mathrm{clip}\!\big((\mu/\sigma^2)/\text{risk aversion}\big)$, where $\mu$ is the gap between its **desired
price** and the market, and the desired price itself learns toward the market at `learning_rate`. It has a
stop-loss and a take-profit. More risk-averse traders trade much less. Their profits do not line up with risk
aversion in one day (the traders share a single price path, so the standard errors understate the day-to-day noise):
a single day cannot rank them.
"""),
code(r"""
rows = {}
for ra in (0.5, 2.0, 8.0):
    r = exchange.simulate('game', seed=3, ticks=900, record_trades=False,
                          extra=[(agents.UtilityTrader, {'risk_aversion': ra}) for _ in range(40)])
    a = r.agents[r.agents['type'] == 'utility']
    rows[f'risk aversion {ra}'] = {'shares traded per trader': a['volume'].mean(), 'mean pnl': a['pnl'].mean(),
                                   'pnl std error': a['pnl'].std() / np.sqrt(len(a))}
pd.DataFrame(rows).T
"""),
md(r"""
## Version 7 (first draft) — traders defined by their psychology

A `Profile` is just numbers; the work is in the *functions* that turn the numbers into trading. They live in
`Behavior`, one method each, so any of them can be replaced:

| trait | default mapping |
|---|---|
| `patience` | looks at the market `1/patience` as often; must hold `20·patience` ticks before a *signal* exit; patient traders rest limit orders, impatient ones pay the spread |
| `greed` | how stretched the signal must be to enter (`0.3 + 1.2·greed` σ) and how much profit it waits for |
| `resilience` | stop-loss distance; `1 − resilience` is the per-tick chance of bailing out of a losing position |
| `hypothesis` | `mean_reversion`, `trend`, `buy_hold` (belief only) or `noise` |
| `desired_price`, `internal_value` | the price it trades around; its perceived fundamental |
| `info` | how much of the market's hidden fair value leaks into its belief |
| `capital` | order size: bigger → whale-like |

```python
Profile(resilience=.5, greed=.5, patience=1, hypothesis='mean_reversion', capital=50_000)
```
"""),
code(r"""
mix = exchange.simulate('profiles', seed=4)           # noise + makers + 400 random profiles
by_h = mix.by_type(by='label')
by_h[by_h.index.str.startswith('profiled')]
"""),
md(r"""
One day is one draw. To see which **traits** pay, pool the profile traders over several days (each run ~4 s):
"""),
code(r"""
frames = []
for seed in range(6):
    r = exchange.simulate('profiles', seed=seed, ticks=1200, record_trades=False)
    ids = r.agents.index[r.agents['type'] == 'profiled']
    d = r.traits.loc[ids].join(r.agents.loc[ids, ['pnl', 'volume', 'label']])
    d['day'] = seed
    d['pnl'] = d['pnl'] - d.groupby('day')['pnl'].transform('mean')       # remove the day's common level
    frames.append(d)
pooled = pd.concat(frames)
rows = []
for col in ['resilience', 'greed', 'patience', 'log_capital', 'info']:
    rho, p = stats.spearmanr(pooled[col], pooled['pnl'])
    rows.append({'trait': col, 'spearman with pnl': rho, 'p-value': p})
display(pd.DataFrame(rows).set_index('trait'))
pooled.groupby('label')[['pnl', 'volume']].mean()
"""),
md(r"""
**Reading this honestly.** **Patience is the one trait that clearly pays** (rank correlation about +0.16 with profit after removing each day's
average): patient traders look at the market less often and rest limit orders instead of paying the spread, so they
trade less. Larger capital does worse in dollars (about -0.24), the most likely reason being that bigger orders pay
more spread and impact. Resilience and greed matter little, and `info` is slightly *negative*. By hypothesis,
buy-and-hold, which trades least, comes out best and the noise hypothesis and trend following worst.

The last question is why `info` does not pay: a trader who sees the market's hidden fair value should win. The
gap between that value and the price, and what the price does next:
"""),
code(r"""
rows = {5: [], 20: [], 60: []}
for seed in range(10):                                     # ten days; one day's correlation is very noisy
    r = exchange.simulate('game', seed=seed, ticks=1500, record_trades=False)
    gap = (r.value - r.price) / r.price
    for k in rows:
        rows[k].append(np.corrcoef(gap[:-k], r.price[k:] / r.price[:-k] - 1)[0, 1])
pd.DataFrame({f'{k} ticks ahead': {'mean corr(gap, later return)': np.mean(v),
                                   'std error': np.std(v, ddof=1) / np.sqrt(len(v)),
                                   'days with corr > 0': int(np.sum(np.array(v) > 0))}
              for k, v in rows.items()}).T
"""),
md(r"""
In this engine the hidden value is **not a leading indicator**: averaged over days the gap has a *negative*
correlation with the following return, growing with the horizon (it is not stable from one day to the next).
That is what you would expect if the value lags the price (it is pulled 2% per tick toward the mid) while the price
has some short-term momentum, so a positive gap mostly means the price has just fallen. I also tried the obvious repair, making the makers quote around
the price instead of the value: then the news has no way into the price at all and informed traders still lose.
For information to pay, the price has to react to the value *with a lag*, for example through a group of traders who
see the news late. That is a design decision about your market, so I have left it as it is.

### A learned policy

`ProfiledTrader(policy=...)` replaces the rule engine; `trader.features(m)` is a fixed-length observation vector
(market z-scores, order-book imbalance, volatility rank, position, P&L, holding time, the trader's own traits).
That is where a GRU or an RL agent plugs in — no neural-network dependency is needed in the package.
"""),
code(r"""
from stochastics.finance.profiles import FEATURE_NAMES
seen = []
def peek(trader, m):
    seen.append(trader.features(m))
exchange.simulate('minimal', seed=1, ticks=300, schedule=False, record_trades=False,
                  extra=[(ProfiledTrader, {'profile': Profile(patience=2.0, greed=.3), 'policy': peek})])
pd.DataFrame(seen, columns=FEATURE_NAMES).describe().T[['mean', 'std', 'min', 'max']]
"""),
md(r"""
## How close is this to your JavaScript engine?

Both engines were run for 40 seeds each (the JS one in Node, unchanged, apart from a stub for `ai/RLTrader.js`).
Means (standard deviation across seeds in brackets) of the day's statistics:

| statistic | JavaScript | Python | Welch p |
|---|---|---|---|
| trades | 116,687 (8,918) | 111,447 (12,107) | 0.03 |
| shares traded | 1.137M (87k) | 1.099M (118k) | 0.10 |
| annualised vol | 0.219 (0.053) | 0.205 (0.037) | 0.19 |
| excess kurtosis | 3.46 (5.2) | 3.36 (2.7) | 0.91 |
| mean spread (cents) | 5.20 (0.48) | 5.09 (0.35) | 0.24 |
| high–low range | 2.11% (0.59%) | 2.03% (0.64%) | 0.56 |
| acf(\|r\|), lag 1 | 0.237 (0.059) | 0.235 (0.076) | 0.89 |

Volatility, tails, spread, range and volatility clustering agree; trade counts differ by about 5% (an earlier
16-seed comparison showed a difference of similar size in the opposite direction, so I treat it as sampling noise
rather than a porting error — but it is not bit-identical, because the random-number generators differ).
"""),
md(r"""
## Ideas

* Make news move the **value** first, the price follow (so informed traders can earn).
* GRU / RL traders via `ProfiledTrader(policy=...)` and `features(m)`; train on simulated days; evaluate against the rule-based types.
* Fit the psychology distribution (resilience, greed, patience, ...) so the simulated market reproduces the stylised facts, then ask which traits survive.
  The scoreboard above (`stochastics.finance.benchmark`) is the yardstick and `benchmark.calibrate` the search, but at about six seconds per simulated day
  the search needs a cheaper surrogate first (60-second step, half the population: about one second a day).
* Version 5 over-reverts and version 7 over-trends (lag-1 autocorrelation and 5-bar variance ratio, opposite signs): adjust the mix of mean-reverting and trend-following rules until both are inside the noise.
* Add sign-dependent news reactions to version 4 (and jumps to the exchange): skew, leverage and tail thickness are what every model above still misses.
* Run a population over many days with entry and exit of traders by wealth (the particle-model analogy).
"""),
]


# ============================================================== real-data sections (06 and 07)
from realdata_cells import daily_cells, intraday_cells  # noqa: E402
from realdata_results import RESULTS  # noqa: E402

markets_nb = markets_nb + daily_cells(RESULTS)
agents_nb = agents_nb[:-1] + intraday_cells(RESULTS) + agents_nb[-1:]


def write(cells, name):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
    nb.metadata['language_info'] = {'name': 'python'}
    nbf.write(nb, HERE / name)


NOTEBOOKS = {
    '00_workbench.ipynb': workbench,
    '01_physics_showcase.ipynb': showcase,
    '02_policy_lab.ipynb': lab,
    '03_baseline_processes.ipynb': baseline_nb,
    '04_population_dynamics.ipynb': popdyn_nb,
    '05_insurance.ipynb': insurance_nb,
    '06_market_models.ipynb': markets_nb,
    '07_agent_market.ipynb': agents_nb,
}


if __name__ == '__main__':
    # python notebooks/build_notebooks.py             -> write only notebooks that do not exist yet
    # python notebooks/build_notebooks.py 03 --force  -> rebuild 03 even if it exists (overwrites it!)
    import sys
    args = [a for a in sys.argv[1:] if a != '--force']
    force = '--force' in sys.argv
    for name, cells in NOTEBOOKS.items():
        if args and not any(name.startswith(a) for a in args):
            continue
        if (HERE / name).exists() and not force:
            print('kept (exists):', name)
            continue
        write(cells, name)
        print('written:', name)
