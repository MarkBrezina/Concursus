"""Stochastic process generators.

    X = Process(t=100, n=1000)        # t time steps, n runs (T = 1 by default, dt = T / t)
    X.randomwalk()                    # fills X and returns the Paths
    X.mean, X.std, X.var              # statistics across runs (Series over time)
    X.plot()                          # paths on the left, terminal distribution on the right

    P = Process(t=250, n=2000, seed=7)
    G = P.gbm(mu=1, sigma=2)          # each call draws fresh, independent noise
    A = P.abm(mu=2, sigma=0.5)
    C = 0.5 * G + 0.2 * A             # pathwise combination (see paths.Paths)

Every generator returns a :class:`Paths` with the theoretical mean, variance
and terminal density attached where they are known, so plots and
``describe()`` compare simulation against theory.

Exact transition sampling is used whenever it exists (ABM, GBM, OU, CIR,
Poisson, Brownian bridge, Merton); ``method='euler'`` / ``'milstein'`` switches
to discretisation so the schemes can be compared against the exact answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy import stats

from .paths import Paths, Process_like, Theory, new_group

__all__ = ['Process', 'process', 'Noise']


@dataclass
class Noise:
    """Brownian increments ``dW`` of shape ``(n, t)``; ``group`` marks shared sources."""
    dW: np.ndarray
    group: int


class Process(Process_like):
    """Time grid + random source. Generators fill ``self.paths`` and return it."""

    def __init__(self, t: int = 100, n: int = 1000, T: float = 1.0, dt: float | None = None,
                 seed: int | None = None):
        if t < 1 or n < 1:
            raise ValueError('t (steps) and n (runs) must be >= 1')
        self.t, self.n = int(t), int(n)
        self.dt = float(dt) if dt is not None else float(T) / self.t
        self.T = self.dt * self.t
        self.times = np.arange(self.t + 1) * self.dt
        self.rng = np.random.default_rng(seed)
        self.paths: Paths | None = None

    # ------------------------------------------------------------- plumbing
    def __getattr__(self, name):
        # X.mean / X.plot() / X.final ... act on the most recent paths
        if name.startswith('_') or name in ('paths',):
            raise AttributeError(name)
        paths = self.__dict__.get('paths')
        if paths is None:
            raise AttributeError(f'{name!r}: generate paths first, e.g. X.randomwalk()')
        return getattr(paths, name)

    def __add__(self, other): return self.paths + other
    def __radd__(self, other): return self.paths.__radd__(other)
    def __sub__(self, other): return self.paths - other
    def __rsub__(self, other): return self.paths.__rsub__(other)
    def __mul__(self, other): return self.paths * other
    def __rmul__(self, other): return self.paths * other
    def __neg__(self): return -self.paths

    def __repr__(self) -> str:
        last = f', last={self.paths.name}' if self.paths is not None else ''
        return f'Process(t={self.t}, n={self.n}, T={self.T:g}, dt={self.dt:g}{last})'

    def noise(self) -> Noise:
        """A fresh, independent set of Brownian increments."""
        return Noise(self.rng.normal(0.0, np.sqrt(self.dt), (self.n, self.t)), new_group())

    def correlated(self, k: int, corr) -> list[Noise]:
        """``k`` Brownian drivers with correlation matrix ``corr`` (or a single rho for k=2)."""
        C = np.full((k, k), float(corr)) + (1 - float(corr)) * np.eye(k) if np.ndim(corr) == 0 \
            else np.asarray(corr, float)
        L = np.linalg.cholesky(C)
        Z = self.rng.normal(0.0, np.sqrt(self.dt), (k, self.n, self.t))
        dW = np.einsum('ij,jnt->int', L, Z)
        group = new_group() if not np.allclose(C, np.eye(k)) else None
        return [Noise(dW[i], group if group is not None else new_group()) for i in range(k)]

    def _dW(self, noise: Noise | None):
        noise = noise or self.noise()
        if noise.dW.shape != (self.n, self.t):
            raise ValueError(f'noise must have shape {(self.n, self.t)}')
        return noise.dW, frozenset({noise.group})

    def _done(self, values, name, params, theory=None, groups=frozenset(), discrete=False,
              extras=None) -> Paths:
        self.paths = Paths(values, self.times.copy(), name, params, theory or Theory(),
                           extras or {}, groups, discrete)
        return self.paths

    @staticmethod
    def _cum(x0, increments):
        out = np.empty((increments.shape[0], increments.shape[1] + 1))
        out[:, 0] = x0
        np.cumsum(increments, axis=1, out=out[:, 1:])
        out[:, 1:] += x0
        return out

    # ===================================================== 1. random walk
    def randomwalk(self, p: float = 0.5, step: float = 1.0, x0: float = 0.0) -> Paths:
        """Simple random walk: +step with probability p, -step otherwise, every time step."""
        moves = np.where(self.rng.random((self.n, self.t)) < p, step, -step)
        dt = self.dt

        def k(t):
            return np.rint(np.asarray(t) / dt)
        theory = Theory(
            mean=lambda t: x0 + k(t) * (2 * p - 1) * step,
            var=lambda t: 4 * p * (1 - p) * k(t) * step ** 2,
            pmf=lambda x, t: stats.binom.pmf(np.rint(((np.asarray(x) - x0) / step + k(t)) / 2),
                                             int(k(t)), p)
            * (np.mod(np.rint((np.asarray(x) - x0) / step + k(t)), 2) == 0))
        return self._done(self._cum(x0, moves), f'RandomWalk(p={p:g})',
                          dict(p=p, step=step, x0=x0), theory, frozenset({new_group()}), True)

    # ================================================ 2. Wiener / Brownian
    def wiener(self, x0: float = 0.0, noise: Noise | None = None) -> Paths:
        """Standard Brownian motion W: independent N(0, dt) increments."""
        dW, g = self._dW(noise)
        theory = Theory(mean=lambda t: np.full_like(np.asarray(t, float), x0),
                        var=lambda t: np.asarray(t, float),
                        pdf=lambda x, t: stats.norm.pdf(x, x0, np.sqrt(t)),
                        cdf=lambda x, t: stats.norm.cdf(x, x0, np.sqrt(t)))
        return self._done(self._cum(x0, dW), 'Wiener', dict(x0=x0), theory, g)

    brownian = wiener

    # ======================================= 4. arithmetic Brownian motion
    def abm(self, mu: float = 0.0, sigma: float = 1.0, x0: float = 0.0,
            noise: Noise | None = None) -> Paths:
        """dX = mu dt + sigma dW (exact)."""
        dW, g = self._dW(noise)
        theory = Theory(mean=lambda t: x0 + mu * np.asarray(t, float),
                        var=lambda t: sigma ** 2 * np.asarray(t, float),
                        pdf=lambda x, t: stats.norm.pdf(x, x0 + mu * t, sigma * np.sqrt(t)),
                        cdf=lambda x, t: stats.norm.cdf(x, x0 + mu * t, sigma * np.sqrt(t)))
        return self._done(self._cum(x0, mu * self.dt + sigma * dW), f'ABM(μ={mu:g}, σ={sigma:g})',
                          dict(mu=mu, sigma=sigma, x0=x0), theory, g)

    # ======================================== 3. geometric Brownian motion
    def gbm(self, mu: float = 0.05, sigma: float = 0.2, x0: float = 1.0, method: str = 'exact',
            noise: Noise | None = None) -> Paths:
        """dX = mu X dt + sigma X dW. ``method``: 'exact' | 'euler' | 'milstein'."""
        dW, g = self._dW(noise)
        dt = self.dt
        if method == 'exact':
            values = x0 * np.exp(self._cum(0.0, (mu - 0.5 * sigma ** 2) * dt + sigma * dW))
        else:
            values = np.empty((self.n, self.t + 1))
            values[:, 0] = x0
            for k in range(self.t):
                x = values[:, k]
                step = mu * x * dt + sigma * x * dW[:, k]
                if method == 'milstein':
                    step += 0.5 * sigma ** 2 * x * (dW[:, k] ** 2 - dt)
                elif method != 'euler':
                    raise ValueError("method must be 'exact', 'euler' or 'milstein'")
                values[:, k + 1] = x + step

        theory = Theory(
            mean=lambda t: x0 * np.exp(mu * np.asarray(t, float)),
            var=lambda t: x0 ** 2 * np.exp(2 * mu * np.asarray(t, float))
            * np.expm1(sigma ** 2 * np.asarray(t, float)),
            pdf=lambda x, t: stats.lognorm.pdf(x, s=sigma * np.sqrt(t),
                                               scale=x0 * np.exp((mu - 0.5 * sigma ** 2) * t)),
            cdf=lambda x, t: stats.lognorm.cdf(x, s=sigma * np.sqrt(t),
                                               scale=x0 * np.exp((mu - 0.5 * sigma ** 2) * t)),
            support=(0.0, np.inf) if x0 > 0 else (-np.inf, 0.0))
        tag = '' if method == 'exact' else f', {method}'
        return self._done(values, f'GBM(μ={mu:g}, σ={sigma:g}{tag})',
                          dict(mu=mu, sigma=sigma, x0=x0, method=method), theory, g)

    # ============================================== 5. Ornstein-Uhlenbeck
    def ou(self, theta: float = 1.0, mu: float = 0.0, sigma: float = 1.0, x0: float = 0.0,
           method: str = 'exact', noise: Noise | None = None) -> Paths:
        """dX = theta (mu - X) dt + sigma dW: mean reversion to mu at speed theta.

        Half-life of a deviation: ln 2 / theta. Stationary sd: sigma / sqrt(2 theta).
        """
        if theta <= 0:
            raise ValueError('theta must be positive')
        dW, g = self._dW(noise)
        dt = self.dt
        values = np.empty((self.n, self.t + 1))
        values[:, 0] = x0
        a = np.exp(-theta * dt)
        s = sigma * np.sqrt((1 - a * a) / (2 * theta))
        for k in range(self.t):
            x = values[:, k]
            if method == 'exact':
                values[:, k + 1] = mu + (x - mu) * a + s * dW[:, k] / np.sqrt(dt)
            elif method == 'euler':
                values[:, k + 1] = x + theta * (mu - x) * dt + sigma * dW[:, k]
            else:
                raise ValueError("method must be 'exact' or 'euler'")

        def m(t):
            return mu + (x0 - mu) * np.exp(-theta * np.asarray(t, float))

        def v(t):
            return sigma ** 2 / (2 * theta) * (1 - np.exp(-2 * theta * np.asarray(t, float)))
        theory = Theory(mean=m, var=v, pdf=lambda x, t: stats.norm.pdf(x, m(t), np.sqrt(v(t))),
                        cdf=lambda x, t: stats.norm.cdf(x, m(t), np.sqrt(v(t))),
                        notes=f'half-life {np.log(2) / theta:.3g}, stationary sd '
                              f'{sigma / np.sqrt(2 * theta):.3g}')
        tag = '' if method == 'exact' else f', {method}'
        return self._done(values, f'OU(θ={theta:g}, μ={mu:g}, σ={sigma:g}{tag})',
                          dict(theta=theta, mu=mu, sigma=sigma, x0=x0, method=method), theory, g)

    # ======================================================== 6. CIR
    def cir(self, kappa: float = 2.0, theta: float = 0.04, sigma: float = 0.2, x0: float = 0.04,
            method: str = 'exact', noise: Noise | None = None) -> Paths:
        """dX = kappa (theta - X) dt + sigma sqrt(X) dW: mean-reverting and non-negative.

        Used for interest rates and variance. Stays strictly positive when the
        Feller condition 2 kappa theta >= sigma^2 holds. 'exact' samples the
        non-central chi-square transition; 'euler' uses full truncation.
        """
        dt = self.dt
        values = np.empty((self.n, self.t + 1))
        values[:, 0] = x0
        if method == 'exact' and noise is None:
            c = sigma ** 2 * (1 - np.exp(-kappa * dt)) / (4 * kappa)
            df = 4 * kappa * theta / sigma ** 2
            for k in range(self.t):
                nonc = values[:, k] * np.exp(-kappa * dt) / c
                values[:, k + 1] = c * self.rng.noncentral_chisquare(df, np.maximum(nonc, 1e-300))
            g = frozenset({new_group()})
        else:
            dW, g = self._dW(noise)
            v = np.full(self.n, float(x0))
            for k in range(self.t):
                vp = np.maximum(v, 0.0)
                v = v + kappa * (theta - vp) * dt + sigma * np.sqrt(vp) * dW[:, k]
                values[:, k + 1] = np.maximum(v, 0.0)
            method = 'euler'

        def m(t):
            return theta + (x0 - theta) * np.exp(-kappa * np.asarray(t, float))

        def var(t):
            t = np.asarray(t, float)
            e = np.exp(-kappa * t)
            return x0 * sigma ** 2 / kappa * (e - e * e) + theta * sigma ** 2 / (2 * kappa) * (1 - e) ** 2

        def pdf(x, t):
            c = sigma ** 2 * (1 - np.exp(-kappa * t)) / (4 * kappa)
            return stats.ncx2.pdf(np.asarray(x) / c, 4 * kappa * theta / sigma ** 2,
                                  x0 * np.exp(-kappa * t) / c) / c

        def cdf(x, t):
            c = sigma ** 2 * (1 - np.exp(-kappa * t)) / (4 * kappa)
            return stats.ncx2.cdf(np.asarray(x) / c, 4 * kappa * theta / sigma ** 2,
                                  x0 * np.exp(-kappa * t) / c)

        feller = 2 * kappa * theta >= sigma ** 2
        theory = Theory(mean=m, var=var, pdf=pdf, cdf=cdf, support=(0.0, np.inf),
                        notes='Feller condition holds' if feller else 'Feller condition fails: '
                        'the process can touch 0')
        tag = '' if method == 'exact' else f', {method}'
        return self._done(values, f'CIR(κ={kappa:g}, θ={theta:g}, σ={sigma:g}{tag})',
                          dict(kappa=kappa, theta=theta, sigma=sigma, x0=x0, method=method,
                               feller=feller), theory, g)

    # =================================================== 7. Poisson process
    def poisson(self, rate: float = 5.0, x0: float = 0.0) -> Paths:
        """Counting process N(t): independent Poisson(rate * dt) increments."""
        inc = self.rng.poisson(rate * self.dt, (self.n, self.t)).astype(float)
        theory = Theory(mean=lambda t: x0 + rate * np.asarray(t, float),
                        var=lambda t: rate * np.asarray(t, float),
                        pmf=lambda x, t: stats.poisson.pmf(np.asarray(x) - x0, rate * t))
        return self._done(self._cum(x0, inc), f'Poisson(λ={rate:g})', dict(rate=rate, x0=x0),
                          theory, frozenset({new_group()}), True)

    # ============================================ 8. general SDE schemes
    def sde(self, drift: Callable, diffusion: Callable, x0: float = 0.0, *,
            scheme: str = 'euler', diffusion_dx: Callable | None = None,
            noise: Noise | None = None, name: str = 'SDE') -> Paths:
        """Solve dX = a(t, X) dt + b(t, X) dW for any vectorised a, b.

        scheme='euler'    Euler-Maruyama: X += a dt + b dW            (strong order 0.5)
        scheme='milstein' adds 0.5 b b' (dW^2 - dt)                   (strong order 1)
        ``diffusion_dx`` = db/dx for Milstein (numerical derivative if omitted).
        """
        dW, g = self._dW(noise)
        dt = self.dt
        values = np.empty((self.n, self.t + 1))
        values[:, 0] = x0
        for k in range(self.t):
            t = self.times[k]
            x = values[:, k]
            b = diffusion(t, x)
            x_new = x + drift(t, x) * dt + b * dW[:, k]
            if scheme == 'milstein':
                if diffusion_dx is not None:
                    bx = diffusion_dx(t, x)
                else:
                    h = 1e-6 * (1 + np.abs(x))
                    bx = (diffusion(t, x + h) - diffusion(t, x - h)) / (2 * h)
                x_new = x_new + 0.5 * b * bx * (dW[:, k] ** 2 - dt)
            elif scheme != 'euler':
                raise ValueError("scheme must be 'euler' or 'milstein'")
            values[:, k + 1] = x_new
        return self._done(values, f'{name} ({scheme})', dict(scheme=scheme, x0=x0), Theory(), g)

    def euler_maruyama(self, drift: Callable, diffusion: Callable, x0: float = 0.0, **kw) -> Paths:
        """Alias for ``sde(..., scheme='euler')``."""
        return self.sde(drift, diffusion, x0, scheme='euler', **kw)

    def milstein(self, drift: Callable, diffusion: Callable, x0: float = 0.0, **kw) -> Paths:
        """Alias for ``sde(..., scheme='milstein')``."""
        return self.sde(drift, diffusion, x0, scheme='milstein', **kw)

    # ================================================= extra: bridges & jumps
    def brownian_bridge(self, x0: float = 0.0, x1: float = 0.0, sigma: float = 1.0) -> Paths:
        """Brownian motion pinned to x0 at t=0 and x1 at t=T."""
        dW, g = self._dW(None)
        W = self._cum(0.0, sigma * dW)
        T = self.T
        s = self.times / T
        values = x0 + W - s * W[:, -1:] + s * (x1 - x0)
        theory = Theory(mean=lambda t: x0 + (x1 - x0) * np.asarray(t, float) / T,
                        var=lambda t: sigma ** 2 * np.asarray(t, float) * (T - np.asarray(t, float)) / T)
        return self._done(values, f'BrownianBridge({x0:g}→{x1:g})', dict(x0=x0, x1=x1, sigma=sigma),
                          theory, g)

    def compound_poisson(self, rate: float = 5.0, jump_mean: float = 0.0, jump_std: float = 1.0,
                         x0: float = 0.0) -> Paths:
        """Sum of N(t) ~ Poisson(rate t) jumps, each N(jump_mean, jump_std^2)."""
        k = self.rng.poisson(rate * self.dt, (self.n, self.t))
        inc = k * jump_mean + np.sqrt(k) * jump_std * self.rng.standard_normal((self.n, self.t))
        theory = Theory(mean=lambda t: x0 + rate * np.asarray(t, float) * jump_mean,
                        var=lambda t: rate * np.asarray(t, float) * (jump_std ** 2 + jump_mean ** 2))
        return self._done(self._cum(x0, inc), f'CompoundPoisson(λ={rate:g})',
                          dict(rate=rate, jump_mean=jump_mean, jump_std=jump_std, x0=x0), theory,
                          frozenset({new_group()}))

    def merton(self, mu: float = 0.05, sigma: float = 0.2, jump_rate: float = 1.0,
               jump_mean: float = -0.1, jump_std: float = 0.15, x0: float = 1.0) -> Paths:
        """Merton jump diffusion: GBM plus lognormal jumps (crashes / gaps).

        Drift is compensated so E[X_t] = x0 e^(mu t), as for GBM.
        """
        dW, g = self._dW(None)
        dt = self.dt
        kbar = np.exp(jump_mean + 0.5 * jump_std ** 2) - 1
        k = self.rng.poisson(jump_rate * dt, (self.n, self.t))
        jumps = k * jump_mean + np.sqrt(k) * jump_std * self.rng.standard_normal((self.n, self.t))
        log_inc = (mu - 0.5 * sigma ** 2 - jump_rate * kbar) * dt + sigma * dW + jumps
        values = x0 * np.exp(self._cum(0.0, log_inc))
        drift = mu - 0.5 * sigma ** 2 - jump_rate * kbar

        def var(t):
            t = np.asarray(t, float)
            second = x0 ** 2 * np.exp(2 * drift * t + 2 * sigma ** 2 * t
                                      + jump_rate * t * (np.exp(2 * jump_mean + 2 * jump_std ** 2) - 1))
            return second - (x0 * np.exp(mu * t)) ** 2

        def mix(dist, x, t):
            lam = jump_rate * t
            kmax = int(lam + 10 * np.sqrt(lam + 1) + 10)
            x = np.asarray(x, float)
            out = np.zeros_like(x)
            for j in range(kmax + 1):
                s = np.sqrt(sigma ** 2 * t + j * jump_std ** 2)
                out += stats.poisson.pmf(j, lam) * getattr(stats.lognorm, dist)(
                    x, s=s, scale=x0 * np.exp(drift * t + j * jump_mean))
            return out
        theory = Theory(mean=lambda t: x0 * np.exp(mu * np.asarray(t, float)), var=var,
                        pdf=lambda x, t: mix('pdf', x, t), cdf=lambda x, t: mix('cdf', x, t),
                        support=(0.0, np.inf))
        return self._done(values, f'Merton(μ={mu:g}, σ={sigma:g}, λ={jump_rate:g})',
                          dict(mu=mu, sigma=sigma, jump_rate=jump_rate, jump_mean=jump_mean,
                               jump_std=jump_std, x0=x0), theory, g)

    # ================================================ extra: stochastic vol
    def heston(self, mu: float = 0.05, kappa: float = 2.0, theta: float = 0.04, xi: float = 0.3,
               rho: float = -0.7, v0: float = 0.04, x0: float = 1.0) -> Paths:
        """Heston: price with CIR variance, correlated shocks (rho < 0: leverage effect).

        dS = mu S dt + sqrt(v) S dW1,  dv = kappa (theta - v) dt + xi sqrt(v) dW2,
        corr(dW1, dW2) = rho. Full-truncation Euler for v, log-Euler for S.
        The variance path is in ``paths.extras['variance']``.
        """
        n1, n2 = self.correlated(2, rho)
        dt = self.dt
        logS = np.zeros((self.n, self.t + 1))
        V = np.empty((self.n, self.t + 1))
        V[:, 0] = v0
        v = np.full(self.n, float(v0))
        for k in range(self.t):
            vp = np.maximum(v, 0.0)
            logS[:, k + 1] = logS[:, k] + (mu - 0.5 * vp) * dt + np.sqrt(vp) * n1.dW[:, k]
            v = v + kappa * (theta - vp) * dt + xi * np.sqrt(vp) * n2.dW[:, k]
            V[:, k + 1] = np.maximum(v, 0.0)
        g = frozenset({n1.group})
        var_theory = Theory(mean=lambda t: theta + (v0 - theta) * np.exp(-kappa * np.asarray(t, float)))
        variance = Paths(V, self.times.copy(), 'variance v(t)', {}, var_theory, {}, g)
        theory = Theory(mean=lambda t: x0 * np.exp(mu * np.asarray(t, float)),
                        notes='Feller ' + ('holds' if 2 * kappa * theta >= xi ** 2 else 'fails'))
        return self._done(x0 * np.exp(logS), f'Heston(κ={kappa:g}, θ={theta:g}, ξ={xi:g}, ρ={rho:g})',
                          dict(mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0, x0=x0),
                          theory, g, extras={'variance': variance})

    # ============================================ extra: fractional BM
    def fbm(self, hurst: float = 0.7, x0: float = 0.0, sigma: float = 1.0) -> Paths:
        """Fractional Brownian motion (Davies-Harte, exact on the grid).

        H > 0.5: persistent increments (trends), H < 0.5: anti-persistent
        (mean-reverting, 'rough'), H = 0.5: ordinary Brownian motion.
        Var[B_H(t)] = sigma^2 t^(2H).
        """
        if not 0 < hurst < 1:
            raise ValueError('hurst must be in (0, 1)')
        N = self.t
        k = np.arange(N + 1)
        gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst) - 2 * np.abs(k) ** (2 * hurst)
                       + np.abs(k - 1) ** (2 * hurst))
        row = np.concatenate([gamma[:N + 1], gamma[1:N][::-1]])   # circulant of size 2N
        lam = np.fft.fft(row).real
        if np.any(lam < -1e-10):
            raise FloatingPointError('circulant embedding failed (negative eigenvalues)')
        lam = np.clip(lam, 0.0, None)
        m = len(row)
        Z = (self.rng.standard_normal((self.n, m)) + 1j * self.rng.standard_normal((self.n, m)))
        fgn = np.fft.fft(np.sqrt(lam / m) * Z, axis=1).real[:, :N]
        inc = sigma * fgn * self.dt ** hurst
        theory = Theory(mean=lambda t: np.full_like(np.asarray(t, float), x0),
                        var=lambda t: sigma ** 2 * np.asarray(t, float) ** (2 * hurst),
                        pdf=lambda x, t: stats.norm.pdf(x, x0, sigma * t ** hurst),
                        cdf=lambda x, t: stats.norm.cdf(x, x0, sigma * t ** hurst))
        return self._done(self._cum(x0, inc), f'fBM(H={hurst:g})', dict(hurst=hurst, x0=x0, sigma=sigma),
                          theory, frozenset({new_group()}))

    # ============================================ extra: Hawkes process
    def hawkes(self, mu: float = 2.0, alpha: float = 4.0, beta: float = 6.0) -> Paths:
        """Self-exciting counting process: each event raises the intensity by alpha,
        which decays at rate beta.  lambda(t) = mu + sum_i alpha e^(-beta (t - t_i)).

        Models clustered arrivals (order flow, trades, aftershocks). Stationary
        when the branching ratio alpha / beta < 1. Simulated on the grid (needs
        lambda * dt << 1); the intensity is in ``paths.extras['intensity']``.
        """
        if alpha >= beta:
            raise ValueError('need alpha < beta (branching ratio < 1) for a stable process')
        dt = self.dt
        lam = np.full(self.n, float(mu))
        counts = np.zeros((self.n, self.t + 1))
        intensity = np.empty((self.n, self.t + 1))
        intensity[:, 0] = lam
        decay = np.exp(-beta * dt)
        for k in range(self.t):
            events = self.rng.poisson(lam * dt)
            counts[:, k + 1] = counts[:, k] + events
            lam = mu + (lam - mu) * decay + alpha * events
            intensity[:, k + 1] = lam
        ratio = alpha / beta
        lam_inf = mu / (1 - ratio)
        r = beta - alpha

        def mean(t):
            t = np.asarray(t, float)
            return lam_inf * t + (mu - lam_inf) * (1 - np.exp(-r * t)) / r
        g = frozenset({new_group()})
        theory = Theory(mean=mean, notes=f'branching ratio {ratio:.2f}, stationary intensity {lam_inf:.3g}')
        inten = Paths(intensity, self.times.copy(), 'intensity λ(t)', {},
                      Theory(mean=lambda t: lam_inf + (mu - lam_inf) * np.exp(-r * np.asarray(t, float))),
                      {}, g)
        return self._done(counts, f'Hawkes(μ={mu:g}, α={alpha:g}, β={beta:g})',
                          dict(mu=mu, alpha=alpha, beta=beta), theory, g, True,
                          extras={'intensity': inten})

    # ======================================= extra: regime switching
    def regime_switching(self, mus: Sequence[float] = (0.3, -0.4), sigmas: Sequence[float] = (0.15, 0.45),
                         switch_rates: Sequence[Sequence[float]] | None = None, x0: float = 1.0,
                         start: int = 0) -> Paths:
        """GBM whose drift and volatility follow a hidden Markov chain of regimes.

        ``switch_rates[i][j]`` is the rate (per unit time) of jumping from regime
        i to j (default: calm <-> stressed, mean stay of ~0.5 time units).
        The regime path is in ``paths.extras['regime']``.
        """
        mus, sigmas = np.asarray(mus, float), np.asarray(sigmas, float)
        R = len(mus)
        Q = np.full((R, R), 2.0) if switch_rates is None else np.asarray(switch_rates, float)
        np.fill_diagonal(Q, 0.0)
        from scipy.linalg import expm
        np.fill_diagonal(Q, -Q.sum(axis=1))
        P = expm(Q * self.dt)
        cum = np.cumsum(P, axis=1)
        reg = np.empty((self.n, self.t + 1), dtype=int)
        reg[:, 0] = start
        dW, g = self._dW(None)
        logx = np.zeros((self.n, self.t + 1))
        for k in range(self.t):
            r = reg[:, k]
            logx[:, k + 1] = logx[:, k] + (mus[r] - 0.5 * sigmas[r] ** 2) * self.dt + sigmas[r] * dW[:, k]
            u = self.rng.random(self.n)
            reg[:, k + 1] = (u[:, None] > cum[r]).sum(axis=1)
        return self._done(x0 * np.exp(logx), f'RegimeSwitching({R} regimes)',
                          dict(mus=mus.tolist(), sigmas=sigmas.tolist(), x0=x0), Theory(), g,
                          extras={'regime': reg})

    # ================================================ extra: GARCH(1,1)
    def garch(self, omega: float = 0.05, alpha: float = 0.08, beta: float = 0.9, mu: float = 0.0,
              x0: float = 100.0) -> Paths:
        """GARCH(1,1) returns (per step, in %): r_k = mu + s_k e_k,
        s_k^2 = omega + alpha (r_{k-1} - mu)^2 + beta s_{k-1}^2.

        Volatility clusters: calm and turbulent stretches. Price = x0 * exp(cumsum r / 100).
        Conditional volatility is in ``paths.extras['volatility']``.
        """
        if alpha + beta >= 1:
            raise ValueError('need alpha + beta < 1 for a stationary variance')
        s2 = np.full(self.n, omega / (1 - alpha - beta))
        r_prev = np.zeros(self.n)
        rets = np.empty((self.n, self.t))
        vol = np.empty((self.n, self.t + 1))
        vol[:, 0] = np.sqrt(s2)
        for k in range(self.t):
            s2 = omega + alpha * r_prev ** 2 + beta * s2
            r_prev = np.sqrt(s2) * self.rng.standard_normal(self.n)
            rets[:, k] = mu + r_prev
            vol[:, k + 1] = np.sqrt(s2)
        g = frozenset({new_group()})
        volp = Paths(vol, self.times.copy(), 'conditional vol (%)', {},
                     Theory(mean=None), {}, g)
        return self._done(x0 * np.exp(self._cum(0.0, rets / 100.0)),
                          f'GARCH(ω={omega:g}, α={alpha:g}, β={beta:g})',
                          dict(omega=omega, alpha=alpha, beta=beta, mu=mu, x0=x0,
                               long_run_vol=float(np.sqrt(omega / (1 - alpha - beta)))),
                          Theory(), g, extras={'volatility': volp, 'returns': rets})


process = Process  # lower-case alias: X = process(t=100, n=1000)
