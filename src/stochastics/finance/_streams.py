"""Vectorised stream generators with a separate parameter for every stream.

``baseline.Process`` draws many paths from *one* parameter set.  A market built
from hundreds of heterogeneous streams needs one set per stream, so these
kernels take parameter arrays of shape ``(n,)`` and shocks of shape ``(n, steps)``
and return prices of shape ``(n, steps + 1)``.  Everything uses exact
transitions except Heston (full-truncation Euler, like ``baseline``).
Tests check the kernels against the closed-form moments.
"""
from __future__ import annotations

import numpy as np

__all__ = ['cum0', 'gbm', 'abm', 'bridge', 'heston', 'merton_jumps', 'ou_deviation']


def _col(x, n):
    return np.broadcast_to(np.asarray(x, float), (n,))[:, None]


def cum0(inc: np.ndarray) -> np.ndarray:
    """Cumulative sum along time with a leading zero column."""
    out = np.zeros((inc.shape[0], inc.shape[1] + 1))
    np.cumsum(inc, axis=1, out=out[:, 1:])
    return out


def gbm(s0, mu, sigma, dW, dt):
    """``dS = mu S dt + sigma S dW`` (exact)."""
    n = dW.shape[0]
    mu, sigma = _col(mu, n), _col(sigma, n)
    return _col(s0, n) * np.exp(cum0((mu - 0.5 * sigma ** 2) * dt + sigma * dW))


def abm(s0, mu, sigma, dW, dt):
    """Arithmetic Brownian motion ``dS = mu s0 dt + sigma s0 dW``: drift and volatility are relative to the start."""
    n = dW.shape[0]
    s0 = _col(s0, n)
    return s0 * (1.0 + cum0(_col(mu, n) * dt + _col(sigma, n) * dW))


def bridge(s0, target, sigma, dW, dt):
    """Brownian bridge from ``s0`` to ``target`` at the end of the horizon (volatility relative to ``s0``)."""
    n, steps = dW.shape
    s0, target, sigma = _col(s0, n), _col(target, n), _col(sigma, n)
    W = cum0(dW)
    frac = np.arange(steps + 1)[None, :] / steps
    return s0 + (target - s0) * frac + sigma * s0 * (W - frac * W[:, -1:])


def heston(s0, mu, kappa, theta, xi, rho, v0, dW, dZ, dt):
    """Heston stochastic volatility, full-truncation Euler.  ``dW`` drives the price, ``dZ`` is independent.

    The variance shock is ``rho dW + sqrt(1 - rho^2) dZ`` (rho < 0: the leverage effect).
    Returns ``(price, variance)``.
    """
    n, steps = dW.shape
    mu, kappa, theta, xi, rho = (_col(a, n)[:, 0] for a in (mu, kappa, theta, xi, rho))
    root = np.sqrt(1.0 - rho ** 2)
    v = np.broadcast_to(np.asarray(v0, float), (n,)).copy()
    logS = np.zeros((n, steps + 1))
    V = np.empty((n, steps + 1))
    V[:, 0] = v
    for k in range(steps):
        vp = np.maximum(v, 0.0)
        sq = np.sqrt(vp)
        logS[:, k + 1] = logS[:, k] + (mu - 0.5 * vp) * dt + sq * dW[:, k]
        v = v + kappa * (theta - vp) * dt + xi * sq * (rho * dW[:, k] + root * dZ[:, k])
        V[:, k + 1] = np.maximum(v, 0.0)
    return _col(s0, n) * np.exp(logS), V


def merton_jumps(rate, jump_mean, jump_std, steps, dt, rng):
    """Per-step log-jump totals of a compound Poisson process, one row per stream."""
    n = np.size(rate)
    k = rng.poisson(np.asarray(rate, float)[:, None] * dt, (n, steps))
    return k * _col(jump_mean, n) + np.sqrt(k) * _col(jump_std, n) * rng.standard_normal((n, steps))


def ou_deviation(theta, eta, bias, z0, z, dt, noise_scale=None, kick=None, kappa=None):
    """Exact Ornstein-Uhlenbeck deviations ``d`` (log price minus log value), one row per trader.

    ``dd = -theta (d - bias) dt + eta m(t) dW``; ``eta`` is the volatility of the deviation and
    ``m(t)`` an optional noise multiplier per step (``noise_scale``).  Each row starts from its
    stationary law (``z0``: ``n`` standard normals); ``z`` holds the step shocks, shape ``(n, steps)``.
    The stationary standard deviation is ``eta / sqrt(2 theta)``.

    ``kick`` (per step, e.g. a jump in the value) times ``kappa`` (per trader, in [0, 1]) is subtracted
    from the deviation: a trader with ``kappa = 0.4`` has reflected only 60% of a jump in the value at
    first and then closes the gap at speed ``theta`` (under-reaction followed by catch-up).
    """
    n, steps = z.shape
    theta, eta, bias = (_col(a, n)[:, 0] for a in (theta, eta, bias))
    phi = np.exp(-theta * dt)
    sd = eta * np.sqrt((1.0 - phi ** 2) / (2.0 * theta))
    d = np.empty((n, steps + 1))
    d[:, 0] = bias + eta / np.sqrt(2.0 * theta) * z0
    for k in range(steps):
        m = 1.0 if noise_scale is None else noise_scale[k]
        d[:, k + 1] = bias + (d[:, k] - bias) * phi + sd * m * z[:, k]
        if kick is not None and kick[k] != 0.0:
            d[:, k + 1] -= kappa * kick[k]
    return d
