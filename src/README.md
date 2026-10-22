# stochastics

A research toolkit for stochastic and many-body systems, organised as one package with independent parts:

| part | what it does | notebook |
|---|---|---|
| `stochastics.physics` | particles under gravity / electromagnetism, composition-driven collisions, an energy ledger that audits every run, dashboards, 2D/3D animation, and a **policy lab** for designing interaction laws | `01_physics_showcase`, `02_policy_lab` |
| `stochastics.baseline` | stochastic processes with theory attached: random walk, Wiener, ABM, GBM, OU, CIR, Poisson, Euler–Maruyama / Milstein for any SDE, Brownian bridge, compound Poisson, Merton jumps, Heston, fractional BM, Hawkes, regime switching, GARCH; weighted sums (convolutions) and mixtures | `03_baseline_processes` |
| `stochastics.popdyn` | agent-based populations: grass–rabbit–wolf ecosystem with evolving traits, SIR/SEIR epidemic with superspreaders, rock-paper-scissors spirals, a 10,000-person wealth economy, Schelling segregation | `04_population_dynamics` |
| `stochastics.finance` | **insurance** (claim arrivals × sizes, premium principles, ruin and survival), **market models** v1–4 (Heston → ensembles → internal value with traders → news jumps), a **limit order book** (port of `OrderBook.js`) and an **agent-based exchange** v5–7 (ported `bots.js` / `BotFleet.js`, the brief's trader personalities, psychology-profile traders), and a **real-data benchmark** that scores every model against equity, FX and crypto histories and weighs accuracy against run time | `05_insurance`, `06_market_models`, `07_agent_market` |
| `stochastics.dev` | workbench helpers: `dev.health()`, `dev.test('baseline')` | `00_workbench` |

Each part imports on its own, so a half-finished module in one never breaks the others.

## physics

![Two rotating clusters colliding in 3D](docs/galaxies_3d.gif)

```python
from dataclasses import replace
import stochastics
from stochastics import physics
from stochastics.physics import *

# 1. particles: explicit clusters from any (editable) preset
astro_cfg = replace(PRESETS['astro'], radius_mean=0.003, radius_std=0.0008, velocity_std=0.03)
particles = ParticleSystem.random(
    n=400, preset=astro_cfg, seed=210, dimension=3,
    clusters=[Cluster(mean=[-3.5, -1, 0], std=[1, 1, 0.15], orbital_G=1e-4, velocity=[0.2, 0.12, 0]),
              Cluster(mean=[3.5, 1, 0.3], std=[0.8, 0.8, 0.12], orbital_G=1e-4, velocity=[-0.2, -0.12, 0],
                      rotation=[0.4, 0, -1], weight=0.8)])
particles[0]            # Particle(id=0, mass=..., charge=..., spin=..., polarity=..., energy=..., composition={+:4, -:2}, ...)
particles[0].mass = 120 # live view: writes straight into the arrays

# 2. environment: law + rules + background
env = Environment(law=Gravity(G=1e-4, softening=0.12),
                  rules=[Collisions(CompositionAffinity()), Drag(gamma=0.0)],
                  expansion=None)

# 3. time loop
result = simulate(particles, env, steps=3200, dt=0.01, record_every=20)

# 4. plots and animations, separately
report(result)                        # summary table + 12-panel dashboard
animate_2d(result, max_frames=100)    # 3D runs are projected (plane='xy' | 'xz' | 'yz')
animate_3d(result, max_frames=100)
```

## baseline

```python
from stochastics.baseline import process, combine, mixture

X = process(t=100, n=1000)          # t steps, n runs, horizon T = 1 (dt = T/t)
X.randomwalk()
X.mean, X.std, X.var                # Series over time
X.plot()                            # paths | terminal distribution, theory overlaid

P = process(t=250, n=5000, seed=1)
S = 0.5 * P.gbm(mu=1, sigma=2) + 0.2 * P.abm(mu=2, sigma=0.5) + 0.3 * P.gbm(mu=1, sigma=3)
S.plot()                            # terminal density = numerical convolution of the parts
S.describe()                        # simulated vs exact moments
mixture([0.5, 0.2, 0.3], [...])     # the other kind of blend: average of densities
```

Exact transition sampling wherever it exists (ABM, GBM, OU, CIR via non-central χ², Poisson, Merton, fBM via
Davies–Harte); `method='euler'|'milstein'` and `sde(drift, diffusion)` for anything else.

## popdyn

```python
from stochastics.popdyn import *

eco = Ecosystem(seed=7).run(2000)        # energy ledger closes to ~1e-9
eco.totals(); eco.plot(); eco.animate()
eco.leaders('lifespan', 10, species='rabbit'); eco.trait_effects('rabbit')

epi = Epidemic(sigma=0.5, seed=3).run(80)    # infection tree -> superspreaders, ODE comparison
econ = Economy(n=10_000, rules=[LuckEvents(), Investment(), YardSale()]).run(80)
econ.plot(); econ.leaders(10); econ.trait_effects()
RockPaperScissors(seed=2).run(800).animate(); Schelling().run(60).plot()
```

## finance

```python
from stochastics.finance import insurance as ins, market, exchange
from stochastics.finance.lob import OrderBook

# insurance: arrivals x sizes, premiums, survival
claims = ins.ClaimProcess(ins.Poisson(rate=120), ins.weibull_with_mean(1.4, 1000))
claims.expected_total(T=1)                     # E[N] E[X]
ins.premium_table(claims.aggregate(T=1))       # expected value, variance, std dev, VaR, TVaR, exponential, Esscher, PH, Wang
firm = ins.Company.priced(claims, capital=30_000, principle='std_dev', beta=0.5)
firm.plot(); firm.ruin_exact([0, 30_000, 60_000])   # Monte Carlo, Lundberg and exact Pollaczek-Khinchine bracket
ins.Hawkes(mu=60, alpha=10, beta=20)           # clustered claims; Renewal(gap) for any interarrival law

# market models, versions 1-4
market.compare_versions(streams=100, steps=2000, seed=1).plot()
market.Market(version=4, streams=100, event_process='hawkes', value='gbm').run().facts()

# agent-based exchange, versions 5-7
res = exchange.simulate('game', seed=1)        # the website's ~1,000-agent day; 'brief', 'profiles', or {type: count}
res.summary(); res.by_type(); res.conservation(); res.plot()
res = exchange.simulate('profiles', extra=[(agents.PolicyAgent, {'policy': my_policy})])

# how close is each model to a real market, and what does the fidelity cost?
from stochastics.finance import realdata, benchmark as bm
spx = realdata.sample('sp500')                                   # public sample sets download on first use
mine = realdata.load_csv('SPY_5min.csv', asset='equity')         # or your own file (realdata.yahoo needs yfinance)
models = [bm.Gaussian(), bm.StudentT(), bm.GARCH(), bm.Heston(), bm.Traders(), bm.News()]
board = bm.compare(spx, models, replicates=24, calibrate_budget=70)      # tune each model to the series, score on fresh numbers
board.table; board.plot()                                        # ranking, p-values, cost; fact-by-fact heatmap
sw = bm.sweep({'S&P': spx, 'EURUSD': realdata.sample('eur_d')}, models, n_jobs=2, calibrate_budget=70)
bm.plot_frontier(sw.cost_accuracy())                             # error against seconds per replicate, Pareto and knee
```

| module | what it does |
|---|---|
| `finance.insurance` | `ClaimProcess` = arrivals (`Poisson`, `Renewal`, `Hawkes`) × severity (any `scipy.stats` law; `weibull`, `lognormal`, `pareto`, ...); exact aggregate law by FFT; nine premium principles calibrated to one loading; `Company` surplus paths, exact ruin detection, Lundberg and Pollaczek–Khinchine |
| `finance.market` | `Market(version=1..4)`; ensembles of ABM/GBM/bridge/Heston/Merton streams with a common factor; an internal-value stream with traders around it; news jumps by Poisson or Hawkes clocks; `stylized_facts` |
| `finance.lob` | `OrderBook`: price–time priority, market/limit orders, expiry, aggressor side on every trade |
| `finance.agents` | the trader types (ported bots, the brief's personalities, `UtilityTrader`, `PolicyAgent`); `register()` adds your own |
| `finance.exchange` | `simulate()`: a population trading through the book for a day with schedule, news, attention; `by_type`, `trait_effects`, `conservation`, dashboard |
| `finance.profiles` | version 7: `Profile` (resilience, greed, patience, hypothesis, ...) mapped to behaviour by replaceable `Behavior` methods; `ProfiledTrader` with a `policy` / `features(m)` hook for learned (GRU, RL) traders |
| `finance.realdata` | `RealSeries` (prices plus asset class, bar length, sessions); `load_csv` (Date+Time columns, epoch timestamps, long-format files, gzip), optional `yahoo`; `resample`, `window`; eleven public sample sets (`sample('sp500')`, `'eur_d'`, `'btc_d'`, `'index_5m'`, `'btc_1m'`, ...) downloaded on first use and cached |
| `finance.benchmark` | the twelve-fact `fingerprint`; candidates (`Gaussian`, `StudentT`, `GARCH`, `Heston`, `Ensemble`, `Traders`, `News`, `Exchange`); `assess`, `compare`, `sweep` (error in units of the real series' own sampling noise, Monte-Carlo p-value); `calibrate` (simulated method of moments, out-of-sample re-scoring); `frontier`, `pareto`, `knee`, `budget_curve`, `convergence` for speed against accuracy |

### What the real-data benchmark found (notebooks 06 and 07)

Every model is tuned (simulated method of moments) to each real series and scored on fresh random numbers against
twelve scale-free facts (tails, skew, return and absolute-return autocorrelation, leverage, vol of vol, variance
ratios), with the error expressed in units of the real series' own sampling noise (below about 2: indistinguishable).

* **Volatility clustering splits the models in two.** Gaussian, Student-t, version 2 and version 3 sit 3-4 noise
  units from daily equities, FX and crypto; GARCH-t, Heston, version 4 and version 3 with a stochastic-volatility value
  sit at 1.6-2.0 (daily) and 1.4-1.8 (intraday). The split is the same in all three asset classes. Version 2 fails by
  construction (averaging washes the structure out) and version 3 because its internal value is Gaussian.
* **Among the four that work there is no stable winner.** Held-out (tune on the first half, score the second) they are
  tied at 1.9-2.1. The calmest series (EUR/USD) is best served by the simplest model, GARCH-t.
* **Nobody gets skew, leverage and the lag-1 return correlation right**: the models are too symmetric. That is the
  concrete place to add structure (sign-dependent news reactions).
* **The agent-based exchange, untuned, scores like a tuned GARCH-t on 5-minute data** (version 6 and the web game: 2.1-2.3 on an
  equity index, 1.6-2.1 on BTC) and produces volatility clustering without any volatility process in the code. Version 5 over-reverts,
  version 7 over-trends, version 6 improves on both by 0.4-0.5. It costs 3,000 to 40,000 times more per replicate than the tuned
  quick models (version 3-SV and 4 at one end, GARCH-t at the other).
* **Speed against accuracy.** At daily resolution accuracy is nearly free (<= 0.08 s per replicate); the knee is
  version 4. More traders do not help the *aggregate* price (flat from 10 to 1,000); fitting plateaus after about 80
  evaluations; 24 replicates resolve gaps of only about 0.2. For the exchange: a 60-second step is 4 times cheaper with no loss,
  half the population at 60 s matches the default at about a sixth of the cost, at least 8 simulated days are needed.

### Data

`realdata.sample(name)` downloads small public sample sets from GitHub on first use (`bashtage/arch`, `kernc/backtesting.py`,
`mementum/backtrader`, `datasets/exchange-rates`, `coinmetrics/data`, `ff137/bitstamp-btcusd-minute-data`) and caches them
(folder: environment variable `STOCHASTICS_DATA`, default `~/.cache/stochastics/data`). Nothing is bundled in this repository. Check each source's licence before
redistributing the data or derived tables (e.g. in a public portfolio). For longer or intraday histories use your own files
(`realdata.load_csv`) or `realdata.yahoo(...)` (needs `pip install yfinance`; not exercised in the automated tests, which do not touch the network).

## Install and run

Install once in *editable* mode; `import stochastics` then works from any folder or notebook, and edits
to the source are picked up on the next kernel restart (no reinstall needed, also for new modules).
From a notebook cell (installs into the kernel's own environment):

```python
%pip install -e "C:\Users\Mark Brezina\Stock\stochastics-lab"
```

or from a terminal in the project folder:

```bash
pip install -e ".[notebook,dev]"
pytest -q                                   # 200 tests across all parts, ~1.5 min
```

Requires Python ≥ 3.10, NumPy, SciPy, pandas, Matplotlib. Re-run the install only if the project
folder is moved or renamed.

## Layout

```
stochastics/
├── __init__.py         import stochastics; subpackages load on first use
├── dev.py              health() and test() for the workbench
├── _viz.py             shared plot style for baseline / popdyn
├── baseline/
│   ├── processes.py    Process (process): every generator, Noise, correlated drivers
│   ├── paths.py        Paths: statistics, theory, arithmetic, convolution, plots
│   └── combine.py      combine (weighted sum) and mixture
├── popdyn/
│   ├── core.py         Trait, sample_traits, gini / lorenz / top_share, trait_effects, Result
│   ├── ecosystem.py    Ecosystem, Species (grass–rabbits–wolves, evolving traits, energy ledger)
│   ├── epidemic.py     Epidemic (SIR/SEIR/SIRS on a torus, infection tree, mean-field ODE)
│   ├── cyclic.py       RockPaperScissors (May–Leonard lattice)
│   ├── wealth.py       Economy + rules: LuckEvents, Investment, Wages, Consumption, YardSale,
│   │                   KineticExchange, Tax
│   └── segregation.py  Schelling
├── finance/
│   ├── insurance.py    claim processes, aggregate claims, premium principles, Company (surplus, ruin)
│   ├── market.py       Market versions 1-4, stylized_facts, compare_versions   (_streams.py: the kernels)
│   ├── lob.py          OrderBook (price-time priority)
│   ├── agents.py       trader types and BOT_TYPES registry
│   ├── exchange.py     MarketState, Fleet, simulate(), ExchangeResult
│   ├── profiles.py     version 7: Profile, Behavior, ProfiledTrader
│   ├── realdata.py     RealSeries, load_csv, yahoo, the public sample sets
│   └── benchmark.py    fingerprint, candidates, assess / compare / sweep, calibrate, speed-accuracy frontier
└── physics/
    ├── state.py        Particle, ParticleView, Cluster, ParticleSystem, EnergyLedger, waveform
    ├── presets.py      classical / quantum / em / macro / astro population presets
    ├── forces.py       one pairwise kernel engine → Gravity, Coulomb, any coupling × kernel
    ├── mechanics.py    merge / bounce / emit — the only code that changes momentum or energy
    ├── collisions.py   swept-sphere contact detection, Outcome, CompositionAffinity, Collisions
    ├── environment.py  Environment, Rule, Drag, SpaceExpansion, PolarityClock
    ├── engine.py       simulate(), SimulationResult, recording and diagnostics
    ├── plotting.py     plot_initial, plot_dashboard, report, plot_collision_regimes, ...
    ├── animation.py    animate_2d, animate_3d (HTML player, .gif/.mp4 export)
    └── policies/       the lab: ExternalPolicy, InternalPolicy, clocks, examples
```

## Physics: design

**Policies decide, mechanics conserve.** A collision policy only returns an `Outcome`: *merge*,
*bounce with restitution e*, or *pass*. `mechanics.merge/bounce/emit` execute it and conserve mass,
charge, linear momentum and constituent counts exactly, booking every unit of energy. New rules can be
wrong about physics without silently breaking the bookkeeping.

**One pairwise kernel.** Every central law is `U_ij = C_ij · φ(r)`, `F_i = Σ_j C_ij · g(r) · (x_i − x_j)`.
A law supplies only the coupling matrix `C` (gravity `−G m_i m_j`, Coulomb `k q_i q_j`, a policy
`−strength · affinity`); the kernel (`Plummer`, `Gaussian`, `Yukawa`, `Spring`) supplies the profile. Forces
and the matching potential come out of one chunked pass (O(N²) time, O(chunk·N) memory), and laws that
share a kernel are fused into a single pass.

**An energy ledger that closes.**

| channel | what it holds |
|---|---|
| `kinetic`, `potential`, `internal` | state of the particles (internal = collision heat stored in each particle's `energy`) |
| `radiated` | energy carried away by radiation in mergers / inelastic contacts |
| `dissipated` | energy given to the medium by drag |
| `binding` | potential-energy jumps from discrete events (a merger removes a pair term) |
| `expansion` | potential-energy change caused by background expansion |
| `drive` | energy pumped in or out by internal clocks / policies |

`accounted_energy` (the sum) is flat to integrator precision: the residual shrinks ~4× when `dt` halves.
The engine books these automatically: any rule that moves particles or changes charges calls
`state.touch()`, and the engine re-evaluates the potential and charges the difference to that rule's channel.

**Collisions as a gradient.** Each particle is built from sub-particles (`composition`, e.g. 3 `+` and 2 `−`).
`CompositionAffinity` computes how strongly A's parts attract B's parts:

1. affinity `w = (c_A · K · c_B) / (n_A n_B)` ∈ [−1, 1]
2. restitution `e = e_min + (e_max − e_min)(1 − w)/2` — repulsive interiors bounce elastically, attractive ones stick
3. after a bounce the pair keeps `K' = K_rel − (1 − e²)·½μv_n²`
4. binding at contact `B = −U_law(A,B) + w·½μ·bond_speed²` (long-range law + short-range part bonds)
5. `K' < B` → **merge**, else **bounce** (elastic if e = 1, inelastic otherwise)

![Collision outcome map](docs/collision_regimes.png)

Contacts are found along each particle's straight path over the step (swept spheres), so fast particles
cannot tunnel through each other; the pair is rewound to the moment of contact, resolved, and re-advanced.

## Physics: policy lab

```python
from stochastics.physics.policies import *

policy_external = OppositesAttract(strength=0.05)   # who is drawn to whom
policy_internal = InternalPolicy()                  # clocks, contacts, merging, emission

result = simulate(particles, external=policy_external, internal=policy_internal,
                  rules=[Drag(0.3)], steps=3000, dt=0.01, record_every=25)
```

* **Clocks.** `spin` (θ-like speed) and `polarity` (charge-like amplitude) drive
  `polarity_now = polarity · waveform(phase)`, `phase += spin·dt/4`. With polarity −1, spin 0.5, dt 1 the
  triangle waveform gives −1, −0.5, 0, 0.5, 1, 0.5, 0, −0.5, −1, … A particle's `energy` remains its internal
  energy reservoir, so the ledger keeps closing.
* **External policies** override one method, `affinity(a, b) → matrix` (positive = attract).
  `ElectromagneticPolicy()` reproduces `Coulomb` bit-for-bit and `GravitationalPolicy()` reproduces `Gravity`,
  so any new idea can be benchmarked against known physics in the same engine. `reciprocal=False` allows
  non-reciprocal (agent-like) interactions such as chase/evade.
* **Internal policies** override `evolve` (per step), `decide` (per contact) and `combine` (merged state).
  `ThermalEmission` is a worked example: collisions heat particles, hot composites emit their least-bound
  constituent with recoil, and other particles absorb it.

## Validation (`pytest -q`, or `dev.test()` from the workbench)

**baseline** — terminal mean and variance of every process against theory; exact terminal distributions
(Kolmogorov–Smirnov for GBM, CIR, OU); Euler converges to the exact GBM path and Milstein beats Euler with
the same noise; fBM recovers its Hurst exponent; Hawkes mean count; the numerical convolution matches both
the exact normal-sum density and the simulated quantiles of 0.5·GBM + 0.2·ABM + 0.3·GBM; shared noise
correctly disables variance additivity; mixture moments.

**popdyn** — the ecosystem energy ledger closes; births, deaths and predation counts agree between the
time series and the per-animal table; runs are reproducible from a seed; evolving traits respect their
ceilings; the epidemic conserves people and its infection tree is consistent (cases caused = cases with an
infector); mixing speeds spread; rock-paper-scissors loses diversity above the critical mobility and keeps it
below; exchange rules and tax conserve total wealth exactly and every rule set closes the wealth ledger;
luck predicts wealth better than talent in the Talent-vs-Luck setup; Gini / Lorenz / top-share sanity checks;
Schelling segregates; every dashboard and animation renders.

**physics** — Force = −∇U by finite differences; Gravity/Coulomb against an independent reference implementation;
EM/gravity policies equal the built-in laws; mass, charge, momentum and constituents exact under 3D
collisions; accounted energy exact for pure mergers and second-order convergent with forces; drag and
expansion ledgers; `d(t) = d₀e^{Ht}` for expansion; elastic bounces swap velocities; the composition
gradient (elastic → inelastic → merge, and the speed dependence of capture); swept-sphere detection prevents
tunnelling; emission conserves everything; the polarity clock reproduces the specified sequence.

**finance** — Wald's formula and the Weibull mean against simulation; interarrival times are exponential; the exact
compound-Poisson law against Monte Carlo; Hawkes and renewal claim counts; clustered claims are riskier than
Poisson at the same mean; premium-principle algebra, the closed-form exponential premium for exponential claims,
and `inf` where the moment generating function does not exist; ruin against the closed form for exponential
claims and the exact bracket against simulation for Weibull claims; survival falls with horizon and rises with
capital. Markets: the GBM, ABM, bridge, Heston and OU kernels against theory; versions 1-4 (averaging loses noise
unless the streams share a factor, traders scatter around the internal value, news adds tails and
under-reaction adds drift); reproducibility. Order book: priority, partial fills, sweeps, expiry, cancels and
invariants under random operations (separately checked identical to `OrderBook.js` on 11,689 random operations).
Exchange: cash and shares conserved exactly, the same seed gives the same day in any Python session, the book is
never crossed, every trader type runs, market makers earn in the full mix; profile mappings move the right way and
a learned policy can replace the rule engine.
Real data and benchmark (no network in the tests): CSV loaders for the common layouts, sessions and resampling, downloads from `file://`,
the fingerprint against known answers (white noise, an AR(1), a GARCH path, short sessions and undefined lags), every candidate's
parameter space and bounds, bootstrap errors, the p-value, the calibration search (recovers known parameters, reports
at-bound fits, fresh-seed re-scoring, hold-out), Pareto set and knee, convergence in replicates, the exchange's day pool
(shared between copies, extensible, parallel), sweeps and the plots.

## Scope and limits

* `popdyn` models are deliberately simple, transparent rule sets for exploring mechanisms, not calibrated
  forecasts. The ecosystem resolves at most one meal per grid cell per step; the rock-paper-scissors lattice
  applies interactions in conflict-free batches (a close approximation to one-at-a-time updates).
* `baseline` Hawkes and Heston are simulated on the time grid (accurate when intensity × dt and dt are small);
  everything else listed as exact samples the exact transition law.

* Dimensionless toy units. Presets named `quantum` / `astro` are parameter regimes, not quantum mechanics or
  cosmology. `SpaceExpansion` is a homogeneous stretch, not FLRW.
* Pairwise forces are O(N²): comfortable to ~1–2k particles in a notebook. Barnes–Hut or a GPU kernel is the
  next step for larger N.
* Each particle takes part in at most one contact per step; simultaneous multi-body contacts resolve over
  consecutive steps. Mergers conserve linear momentum but do not track the spin angular momentum the pair's
  orbital motion would become.
* Explicitly time-dependent laws are allowed, but put time-dependence in particle state updated by a rule
  (as the clocks do) if you want the ledger to stay exact.

* `finance.insurance` ignores interest on the surplus, expenses, reinsurance and dependence between claim sizes.
  The exponential and Esscher premiums need a moment generating function and return `inf` for lognormal, Pareto
  and Weibull (shape < 1) claims instead of a number that depends on where a simulation stopped.
* `finance.market`: averaging independent streams removes fat tails and clustering (so version 2 shares a common
  factor); news jumps are applied in log space and capped, and are not drift-compensated; volatility clustering in
  versions 3-4 appears only if the internal value has stochastic volatility.
* `finance.benchmark`: the twelve facts are a deliberately small summary, not a full test of a model; the score does
  not penalise parameters (read the hold-out); the Monte-Carlo p-value is lenient to models whose own spread is wide;
  bootstrap noise for the real series understates long-memory dependence; intraday time-of-day patterns are not removed.
  The public samples are small (daily series of 2,500-5,000 days; intraday: 21 equity-index sessions in 2006, 10 days of BTC, hourly EUR/USD),
  there is no free minute-level FX in them, so FX is scored for versions 1-4 and the baselines only. The exchange
  is assessed as designed, untuned (a simulated day costs seconds), from one pool of days per population, so its numbers
  are less certain (about 0.1-0.2) than the quick models', and the exchange's 8.5-hour day is compared with BTC, which trades continuously.
* `finance.exchange` is a port of the website engine with Python's random numbers, so seeds do not reproduce the
  JavaScript paths, only their statistics. The hidden fair value follows the price (it is pulled toward the mid),
  so informed traders get no edge in this engine. The brief's version 7 is a first draft: the mapping from
  psychology to behaviour is one reading of the brief, meant to be replaced piece by piece.
