"""stochastics — a research toolkit for stochastic and many-body systems.

Subpackages
-----------
physics    Particle simulations: gravity, electromagnetism, composition-driven
           collisions, an exact energy ledger, dashboards and 2D/3D animation.
           ``physics.policies`` is the open lab for designing interaction laws.
baseline   Baseline stochastic processes (Brownian motion, GBM, OU, ...).
finance    Finance setups, including the limit order book (``finance.lob``).
popdyn     Population dynamics: ecosystems, epidemics, cyclic competition, wealth, segregation.
dev        Workbench helpers: ``dev.health()``, ``dev.test('baseline')``.

    import stochastics
    from stochastics import physics, baseline, finance, popdyn
    from stochastics.physics import ParticleSystem, Environment, simulate

Subpackages load on first use, so an error inside one of them (say, while
``finance`` is half-written) does not stop ``import stochastics`` or the others.
"""
import importlib

__version__ = '0.5.0'

SUBPACKAGES = ('physics', 'baseline', 'finance', 'popdyn', 'dev')
__all__ = [*SUBPACKAGES, '__version__']


def __getattr__(name):
    if name in SUBPACKAGES:
        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(SUBPACKAGES))
