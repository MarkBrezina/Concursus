"""Baseline processes: moments against theory, schemes, combinations, plots."""
import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest
from scipy import stats

from stochastics.baseline import (Process, combine, convolved_density, mixture, plot_many,
                                  plot_terminal, process)

N = 20_000


def z_mean(paths):
    """z-score of the terminal sample mean against theory."""
    x = paths.final
    return (x.mean() - paths.theory_mean([paths.T])[0]) / (x.std(ddof=1) / np.sqrt(len(x)))


def var_ratio(paths):
    return paths.final.var(ddof=1) / paths.theory_var([paths.T])[0]


@pytest.fixture(scope='module')
def P():
    return Process(t=200, n=N, T=1.0, seed=11)


def test_user_style_api():
    X = process(t=100, n=1000, seed=1)
    X.randomwalk()
    assert len(X.mean) == 101 and X.std.iloc[-1] > 0 and X.var.iloc[0] == 0
    assert X.final.shape == (1000,)
    assert X.plot(show=False) is not None


@pytest.mark.parametrize('name, kwargs', [
    ('randomwalk', dict(p=0.6)),
    ('wiener', {}),
    ('abm', dict(mu=2, sigma=0.5)),
    ('gbm', dict(mu=0.3, sigma=0.4)),
    ('ou', dict(theta=3, mu=1, sigma=0.5, x0=-1)),
    ('cir', dict(kappa=2, theta=0.04, sigma=0.2, x0=0.09)),
    ('poisson', dict(rate=7)),
    ('brownian_bridge', dict(x0=0, x1=1)),
    ('compound_poisson', dict(rate=5, jump_mean=0.2, jump_std=0.5)),
    ('merton', {}),
    ('fbm', dict(hurst=0.7)),
])
def test_terminal_moments_match_theory(P, name, kwargs):
    paths = getattr(P, name)(**kwargs)
    if paths.theory_var([paths.T])[0] > 0:
        assert abs(z_mean(paths)) < 4.5
        assert 0.94 < var_ratio(paths) < 1.06
    else:
        assert np.allclose(paths.final, paths.theory_mean([paths.T])[0])


def test_exact_terminal_distributions(P):
    g = P.gbm(mu=0.2, sigma=0.5, x0=2.0)
    s, scale = 0.5, 2.0 * np.exp((0.2 - 0.125))
    assert stats.kstest(g.final, 'lognorm', args=(s, 0, scale)).pvalue > 1e-3
    c = P.cir(kappa=2, theta=0.04, sigma=0.3, x0=0.09)
    assert stats.kstest(c.final, lambda x: c.theory.cdf(x, c.T)).pvalue > 1e-3
    o = P.ou(theta=2, mu=1, sigma=0.7)
    assert stats.kstest(o.final, lambda x: o.theory.cdf(x, o.T)).pvalue > 1e-3


def test_euler_converges_to_exact_gbm():
    errs = []
    for steps in (10, 40, 160):
        P = Process(t=steps, n=4000, seed=5)
        noise = P.noise()
        exact = P.gbm(0.5, 0.8, noise=noise).final
        euler = P.gbm(0.5, 0.8, method='euler', noise=noise).final
        errs.append(np.mean(np.abs(exact - euler)))
    assert errs[0] > errs[1] > errs[2]                      # strong convergence
    assert errs[0] / errs[2] > 2.5                         # ~ order 0.5 or better


def test_milstein_beats_euler_pathwise():
    P = Process(t=50, n=4000, seed=6)
    noise = P.noise()
    exact = P.gbm(0.5, 0.8, noise=noise).final
    eu = P.sde(lambda t, x: 0.5 * x, lambda t, x: 0.8 * x, x0=1.0, noise=noise).final
    mi = P.sde(lambda t, x: 0.5 * x, lambda t, x: 0.8 * x, x0=1.0, noise=noise,
               scheme='milstein').final
    assert np.mean(np.abs(mi - exact)) < 0.5 * np.mean(np.abs(eu - exact))


def test_fbm_hurst_recovered(P):
    for h in (0.3, 0.5, 0.75):
        assert abs(P.fbm(h).hurst() - h) < 0.03


def test_ou_mean_reverts_and_cir_stays_positive(P):
    o = P.ou(theta=5, mu=2, sigma=0.3, x0=-3)
    assert abs(o.mean.iloc[-1] - 2) < 0.05
    assert (P.cir(kappa=1, theta=0.02, sigma=0.4, x0=0.02).values >= 0).all()


def test_hawkes_and_heston_run(P):
    h = P.hawkes(mu=2, alpha=4, beta=6)
    assert abs(h.final.mean() / h.theory_mean([1.0])[0] - 1) < 0.05
    assert (np.diff(h.values, axis=1) >= 0).all()
    s = P.heston()
    assert abs(s.final.mean() / np.exp(0.05) - 1) < 0.01
    assert (s.extras['variance'].values >= 0).all()


def test_combination_is_a_convolution(P):
    X, Y, Z = P.gbm(1, 2), P.abm(2, 0.5), P.gbm(1, 3)
    S = 0.5 * X + 0.2 * Y + 0.3 * Z
    S2 = combine([0.5, 0.2, 0.3], [X, Y, Z])
    assert np.allclose(S.values, S2.values) and S.independent
    assert abs(S.theory_mean([1.0])[0] - (0.5 * np.e + 0.4 + 0.3 * np.e)) < 1e-12
    grid, dens = convolved_density([0.5, 0.2, 0.3], [X, Y, Z])
    assert abs(np.trapezoid(dens, grid) - 1) < 2e-3
    cdf = np.cumsum(dens) * (grid[1] - grid[0])
    for q in (0.1, 0.5, 0.9):
        xq = np.quantile(S.final, q)
        assert abs(np.interp(xq, grid, cdf) - q) < 0.02


def test_sum_of_normals_density_exact(P):
    A, B = P.abm(0, 1), P.abm(1, 2)
    grid, dens = convolved_density([1, 1], [A, B])
    assert np.max(np.abs(dens - stats.norm.pdf(grid, 1, np.sqrt(5)))) < 1e-5


def test_shared_noise_breaks_independence(P):
    n = P.noise()
    A, B = P.abm(0, 1, noise=n), P.abm(0, 2, noise=n)
    S = A + B
    assert not S.independent and S.theory.var is None
    assert abs(S.final.var() / 9 - 1) < 0.05              # perfectly correlated: (1+2)^2


def test_correlated_drivers(P):
    a, b = P.correlated(2, -0.6)
    assert abs(np.corrcoef(a.dW.ravel(), b.dW.ravel())[0, 1] + 0.6) < 0.01


def test_mixture_moments(P):
    A, B = P.abm(-2, 0.5), P.abm(3, 0.5)
    M = mixture([0.3, 0.7], [A, B], seed=1)
    assert abs(M.final.mean() - (0.3 * -2 + 0.7 * 3)) < 0.05
    assert abs(M.final.var() / M.theory_var([1.0])[0] - 1) < 0.05


def test_plots_render(P):
    S = 0.5 * P.gbm(1, 2) + 0.5 * P.abm(2, 0.5)
    assert S.plot(show=False) is not None
    assert P.poisson(5).plot(show=False) is not None
    assert plot_terminal(S, P.abm(1, 1), show=False) is not None
    assert plot_many([P.wiener(), P.hawkes()], show=False) is not None
