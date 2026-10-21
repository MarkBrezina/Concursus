# stochastics

A Python simulation package built around one idea: stochastic processes, population dynamics, finance and physics are all simulation problems, so they belong together as subpackages of a single library.

```python
import stochastics
from stochastics import physics, popdyn, finance
```

## Structure

| Subpackage | Focus |
|---|---|
| core | Baseline stochastic processes |
| `physics` | Particle simulations: gravity, electromagnetism, collisions and mergers, spin and polarity |
| `popdyn` | Population dynamics, including agent-based epidemic (SIR) models |
| `finance` | Market models, agent-based exchange and limit order book simulation |

## How this was built

Large parts of the code were written or rewritten with AI assistance. I want to be upfront about that, and equally clear about what is mine:

- the architecture and package layout, including the decision to treat processes, population dynamics, finance and physics as subpackages of one simulation library
- the API and command design
- the ideas, models and design decisions the code implements
- three earlier pieces of work I built myself, which this package grew out of (see [Legacy](#legacy))

In practice my role has shifted away from writing every line and towards the parts that decide whether a simulation is any good: design, modelling decisions, testing and architecture. I won't always be able to recite every equation or implementation detail on the spot, but I can explain what each piece does, why it is built the way it is, and how it fits into the whole.

## Design approach

Most of my time goes into deciding how a process should behave, down to the equations, parameters and dynamics.

Take the SIR model in `popdyn`. Individuals differ in sociability, infectiousness, the size of their friend and family network, recovery speed and prior resistance. For each of those traits I decide how it feeds into infection spread and recovery, and whether the model works better as an agent-based simulation or a collective (compartmental) one.

The constant trade-off is realism against compute: I want the simulations as close to reality as possible without grinding the user's machine to a halt. That also means validating against real-world data, which raises its own questions:

- Where does suitable data come from?
- Can parts of a process be split out and tested separately?
- For each part, what does it cost in compute, how closely does it match reality, and how faithfully does it implement the intended behaviour?

## Roadmap

Both `physics` and `finance` will grow over time, as each ties into other projects of mine.

**Finance.** The next step is market models that sit closer to reality. That means tailoring agent behaviour to how traders, investors and financial institutions actually act, and modelling market events more faithfully. I'm also experimenting with moving performance-critical parts to C++. Improvements here feed into [Tomorrow Markets](https://tomorrowmarkets.github.io/), a multiplayer trading game built on the same engine ideas.

**On hold.** Economics and population dynamics deserve proper research: more realistic dynamics, corrected equations, finer behavioural mapping for individuals and better architecture. That work waits until I have genuine time to spare.

## Subpackages

### physics

Updated with AI assistance, but built on my earlier work on astrophysics and particle swarms. The animations, the particle set-ups, and the ideas around polarity, spin and the energy that results from them all come from that earlier work.

## Legacy

`legacy/` holds the three earlier pieces this package grew out of, kept as originally written for reference. This includes the original `SSP` class (2023): discrete simulations of a random walk, Wiener process, arithmetic and exponential (geometric) Brownian motion, Brownian bridge, variance gamma and square-root diffusion.

The legacy code is not maintained and is not part of the installable package.
