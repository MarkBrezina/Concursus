"""Population dynamics: agent-based models of ecology, disease, competition and wealth.

    from stochastics.popdyn import *

    eco = Ecosystem(seed=7).run(2000)          # grass -> rabbits -> wolves -> grass
    eco.plot(); eco.animate(); eco.totals(); eco.trait_effects('rabbit')

    epi = Epidemic(sigma=0.5, seed=3).run(80)  # SEIR on a torus, infection tree
    epi.plot(); epi.summary()

    rps = RockPaperScissors(seed=1).run(800)   # cyclic dominance -> spiral waves
    rps.plot(); rps.animate()

    econ = Economy(rules=[LuckEvents(), Investment(), YardSale()]).run(80)
    econ.plot(); econ.leaders(10); econ.trait_effects()

    seg = Schelling(tolerance=0.35).run(60)
    seg.plot()

Every result has ``.counts`` (time series), ``.ledger`` (conserved-quantity
bookkeeping where it applies), ``.agents`` (one row per individual with traits
and life history), ``.plot()`` and, for spatial models, ``.animate()``.
"""
from .core import (Result, Trait, gini, leaders, lorenz, mutate, sample_traits, top_share,
                   trait_effects)
from .ecosystem import Ecosystem, EcosystemResult, Species
from .epidemic import Epidemic, EpidemicResult
from .cyclic import CyclicResult, RockPaperScissors
from .wealth import (DEFAULT_TRAITS, Consumption, Economy, EconomyResult, Investment,
                     KineticExchange, LuckEvents, Market, Tax, Wages, YardSale)
from .segregation import Schelling, SegregationResult

__all__ = ['Result', 'Trait', 'gini', 'leaders', 'lorenz', 'mutate', 'sample_traits', 'top_share',
           'trait_effects', 'Ecosystem', 'EcosystemResult', 'Species', 'Epidemic',
           'EpidemicResult', 'CyclicResult', 'RockPaperScissors', 'DEFAULT_TRAITS', 'Consumption',
           'Economy', 'EconomyResult', 'Investment', 'KineticExchange', 'LuckEvents', 'Market',
           'Tax', 'Wages', 'YardSale', 'Schelling', 'SegregationResult']
