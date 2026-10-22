"""Weighted sums (convolutions) and mixtures of processes.

Two different ways to "blend" X, Y, Z with weights (0.5, 0.2, 0.3):

* ``combine``  ->  S = 0.5 X + 0.2 Y + 0.3 Z, path by path (a portfolio).
  If X, Y, Z are driven by independent noise, the density of S_T is the
  **convolution** of the densities of 0.5 X_T, 0.2 Y_T and 0.3 Z_T. Means add;
  variances add with squared weights.

* ``mixture``  ->  each run *is* X with probability 0.5, Y with 0.2, Z with 0.3
  (an uncertain regime / model choice). The density of M_T is the weighted
  **average** of the component densities: same mean as the sum, but far more
  spread, and multi-modal when the components differ.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .paths import Paths, Theory

__all__ = ['combine', 'mixture']


def _as_paths(c):
    return c.paths if hasattr(c, 'paths') and not isinstance(c, Paths) else c


def combine(weights: Sequence[float], components: Sequence, offset: float = 0.0) -> Paths:
    """Pathwise weighted sum ``offset + sum_k w_k X_k`` (theory: see module doc)."""
    if len(weights) != len(components):
        raise ValueError('need one weight per component')
    comps = [_as_paths(c) for c in components]
    total = offset
    for w, c in zip(weights, comps):
        total = total + w * c
    total.name = ' + '.join(f'{w:g}·{c.name}' for w, c in zip(weights, comps))
    return total


def mixture(weights: Sequence[float], components: Sequence, seed: int | None = None) -> Paths:
    """Each run is drawn from component k with probability ``weights[k]``."""
    comps = [_as_paths(c) for c in components]
    w = np.asarray(weights, float)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError('mixture weights must be non-negative and sum to 1')
    shape = comps[0].values.shape
    for c in comps[1:]:
        if c.values.shape != shape or not np.allclose(c.times, comps[0].times):
            raise ValueError('components must share runs and time grid')
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(comps), size=shape[0], p=w)
    values = np.stack([c.values for c in comps])[pick, np.arange(shape[0])]

    mean = var = pdf = None
    if all(c.theory.mean is not None for c in comps):
        def mean(t):
            return sum(wk * np.asarray(c.theory.mean(t), float) for wk, c in zip(w, comps))
        if all(c.theory.var is not None for c in comps):
            def var(t):
                m = mean(t)
                second = sum(wk * (np.asarray(c.theory.var(t), float)
                                   + np.asarray(c.theory.mean(t), float) ** 2)
                             for wk, c in zip(w, comps))
                return second - m ** 2
    if all(c.theory.pdf is not None for c in comps):
        def pdf(x, t):
            return sum(wk * np.asarray(c.theory.pdf(x, t), float) for wk, c in zip(w, comps))
    name = 'Mixture[' + ', '.join(f'{wk:g}:{c.name}' for wk, c in zip(w, comps)) + ']'
    out = Paths(values, comps[0].times, name, {'weights': w.tolist()},
                Theory(mean=mean, var=var, pdf=pdf),
                {'component': pick}, frozenset().union(*[c.groups for c in comps]))
    return out
