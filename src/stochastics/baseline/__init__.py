"""Baseline stochastic processes.

    from stochastics.baseline import Process, combine, mixture

    X = Process(t=100, n=1000)      # t steps, n runs, horizon T=1 (dt = T/t)
    X.randomwalk()
    X.mean, X.std, X.var            # Series over time
    X.plot()                        # paths | terminal distribution

    P = Process(t=250, n=2000, seed=1)
    S = combine([0.5, 0.2, 0.3], [P.gbm(1, 2), P.abm(2, 0.5), P.gbm(1, 3)])
    S.plot()                        # terminal density = convolution of the parts

Processes: randomwalk, wiener (brownian), abm, gbm, ou, cir, poisson,
sde / euler_maruyama / milstein (any SDE), brownian_bridge, compound_poisson,
merton (jumps), heston (stochastic volatility), fbm (fractional, Hurst H),
hawkes (self-exciting), regime_switching, garch.
"""
from .paths import (Paths, Theory, convolved_density, plot_many, plot_terminal,
                    terminal_density)
from .processes import Noise, Process, process
from .combine import combine, mixture

__all__ = ['Process', 'process', 'Paths', 'Theory', 'Noise', 'combine', 'mixture',
           'terminal_density', 'convolved_density', 'plot_terminal', 'plot_many']
