"""Finance: insurance, market models, the order book, the agent-based exchange and the real-data benchmark.

    from stochastics.finance import insurance, market, exchange, realdata, benchmark
    from stochastics.finance.lob import OrderBook

=============  ======================================================================================
module         what is in it
=============  ======================================================================================
insurance      claim arrivals (Poisson / renewal / Hawkes) x claim sizes (Weibull, lognormal, ...),
               expected claims E[N]E[X], premium principles, surplus paths and survival / ruin
market         price models of growing complexity, versions 1-4 (Heston; ensembles; an internal-value
               stream with traders around it; news jumps), ``compare_versions``
lob            limit order book with price-time priority (port of ``OrderBook.js``)
agents         the trader types (port of ``bots.js`` plus the brief's personalities), version 5
exchange       a population of agents trading through the book over a simulated day, versions 5-6
profiles       traders defined by a psychology vector (resilience, greed, patience, ...), version 7
realdata       real price series: CSV / Yahoo loaders, sessions, resampling, a few public sample sets
benchmark      how closely does each model behave like a real market, and what does the fidelity cost:
               fingerprints, calibration, scoreboards across assets, speed-accuracy frontiers
=============  ======================================================================================

Submodules load when first used, so a problem in one never stops the others from importing.
"""
from __future__ import annotations

import importlib

__all__ = ['insurance', 'market', 'lob', 'agents', 'exchange', 'profiles', 'realdata', 'benchmark']


def __getattr__(name: str):
    if name in __all__:
        module = importlib.import_module(f'{__name__}.{name}')
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
