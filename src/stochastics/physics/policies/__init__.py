"""The policy lab: design the laws instead of assuming them.

Two open slots drive every simulation here::

    policy_external = OppositesAttract()      # who is drawn to whom (forces)
    policy_internal = InternalPolicy()        # clocks, contacts, merging, emission

    result = simulate(particles, external=policy_external, internal=policy_internal,
                      steps=4000, dt=0.01, record_every=20)

Sanity anchors: ``ElectromagneticPolicy()`` reproduces Coulomb exactly and
``GravitationalPolicy()`` reproduces gravity, so a new policy can always be
compared against known physics with the same engine, ledger and plots.
"""
from .external import (Agents, ElectromagneticPolicy, ExternalPolicy, GravitationalPolicy,
                       OppositesAttract)
from .internal import InternalPolicy, ThermalEmission
from .clocks import plot_polarity_clocks, polarity_sequence
from ..collisions import Contact, ContactPolicy, Outcome
from ..environment import PolarityClock, advance_clocks

__all__ = ['Agents', 'ExternalPolicy', 'OppositesAttract', 'ElectromagneticPolicy',
           'GravitationalPolicy', 'InternalPolicy', 'ThermalEmission', 'plot_polarity_clocks',
           'polarity_sequence', 'Contact', 'ContactPolicy', 'Outcome', 'PolarityClock',
           'advance_clocks']
