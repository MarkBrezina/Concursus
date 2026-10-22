"""finance.insurance: claim processes, aggregate claims, premium principles, ruin."""
import math

import numpy as np
import pytest
from scipy import stats

from stochastics.finance import insurance as ins


def test_finance_imports():
    from stochastics import finance
    from stochastics.finance import lob

    assert finance.__name__ == 'stochastics.finance' and lob.__name__ == 'stochastics.finance.lob'


@pytest.fixture(scope='module')
def book():
    return ins.ClaimProcess(ins.Poisson(120), ins.weibull_with_mean(1.4, 1000))


def test_weibull_mean_and_total_claims(book):
    assert book.severity.mean() == pytest.approx(1000)
    assert book.expected_total(1) == pytest.approx(120_000)          # E[N] * E[X]
    var, exact = book.variance_total(1)
    assert exact and var == pytest.approx(120 * float(book.severity.moment(2)))   # lambda E[X^2]


def test_poisson_interarrival_times_are_exponential():
    s = ins.ClaimProcess(ins.Poisson(5.0), ins.exponential(1)).sample(T=200, n=20, seed=1)
    gaps = s.interarrival()
    assert stats.kstest(gaps, 'expon', args=(0, 0.2)).pvalue > 0.01
    assert s.counts().mean() == pytest.approx(5.0 * 200, rel=0.02)


def test_exact_pmf_matches_monte_carlo(book):
    A = book.aggregate(1)
    M = book.aggregate(1, method='mc', n=40_000, seed=3)
    assert A.source == 'pmf' and M.source == 'sample'
    assert A.mean == pytest.approx(book.expected_total(1), rel=1e-4)
    assert A.var == pytest.approx(book.variance_total(1)[0], rel=2e-3)
    assert M.mean == pytest.approx(A.mean, rel=0.01)
    assert M.std == pytest.approx(A.std, rel=0.03)
    assert M.quantile(0.99) == pytest.approx(A.quantile(0.99), rel=0.02)


def test_hawkes_and_renewal_counts():
    h = ins.Hawkes(mu=40, alpha=15, beta=30)                          # branching ratio 0.5
    path, _ = h.sample(2.0, 4000, np.random.default_rng(0))
    assert np.bincount(path, minlength=4000).mean() == pytest.approx(h.mean_count(2.0), rel=0.02)
    assert h.rate == pytest.approx(80.0)
    r = ins.Renewal(stats.weibull_min(1.5, scale=0.01))
    path, _ = r.sample(5.0, 2000, np.random.default_rng(1))
    assert np.bincount(path, minlength=2000).mean() == pytest.approx(r.mean_count(5.0), rel=0.02)
    with pytest.raises(ValueError):
        ins.Hawkes(1, 2, 1)                                           # branching ratio >= 1


def test_clustered_claims_are_riskier_than_poisson():
    sev = ins.exponential(1000)
    plain = ins.ClaimProcess(ins.Poisson(80), sev)
    clustered = ins.ClaimProcess(ins.Hawkes(mu=40, alpha=15, beta=30), sev)     # same long-run rate 80
    a, b = plain.aggregate(1, n=30_000, seed=1), clustered.aggregate(1, n=30_000, seed=1)
    assert a.mean == pytest.approx(b.mean, rel=0.05)
    assert b.std > 1.2 * a.std


def test_premium_principles_algebra(book):
    A = book.aggregate(1)
    assert ins.premium(A, 'expected_value', theta=0.1) == pytest.approx(1.1 * A.mean)
    assert ins.premium(A, 'variance', alpha=1e-5) == pytest.approx(A.mean + 1e-5 * A.var)
    assert ins.premium(A, 'std_dev', beta=0.7) == pytest.approx(A.mean + 0.7 * A.std)
    assert ins.premium(A, 'esscher', h=0.0) == pytest.approx(A.mean)
    assert ins.premium(A, 'proportional_hazard', rho=1.0) == pytest.approx(A.mean, rel=1e-3)
    assert ins.premium(A, 'wang', lam=0.0) == pytest.approx(A.mean, rel=1e-3)
    assert A.distortion(lambda p: p) == pytest.approx(A.mean, rel=1e-3)
    assert A.mean < ins.premium(A, 'percentile', q=0.9) < ins.premium(A, 'percentile', q=0.99)
    assert ins.premium(A, 'tvar', q=0.99) > ins.premium(A, 'percentile', q=0.99)
    with pytest.raises(KeyError):
        ins.premium(A, 'nonsense')


def test_exponential_principle_has_closed_form_for_exponential_claims():
    # compound Poisson: log E[e^{aS}] = lambda T (M_X(a) - 1), M_X(a) = 1 / (1 - a m)
    c = ins.ClaimProcess(ins.Poisson(50), ins.exponential(2.0))
    A = c.aggregate(1)
    a = 0.1
    closed = 50 * (1 / (1 - a * 2.0) - 1) / a
    assert ins.premium(A, 'exponential', a=a) == pytest.approx(closed, rel=2e-3)
    # esscher premium is the derivative of the cumulant: lambda T M'(h) = lambda T m / (1 - h m)^2
    assert ins.premium(A, 'esscher', h=0.1) == pytest.approx(50 * 2.0 / (1 - 0.2) ** 2, rel=2e-3)
    assert ins.premium(A, 'exponential', a=0.1) >= A.mean


def test_no_mgf_means_no_exponential_premium():
    for sev in (ins.weibull(0.8, 1000), ins.lognormal(6, 1.0), ins.pareto(3.0, 500)):
        A = ins.ClaimProcess(ins.Poisson(50), sev).aggregate(1, n=5000, seed=0, method='mc')
        assert ins.premium(A, 'exponential', a=1e-4) == math.inf
        assert ins.premium(A, 'esscher', h=1e-4) == math.inf
        assert math.isfinite(ins.premium(A, 'tvar', q=0.99))


def test_premium_table_is_matched_to_first_order_loading(book):
    t = ins.premium_table(book.aggregate(1))
    assert set(t.index) == set(ins.PRINCIPLES)
    for name in ('expected_value', 'variance', 'std_dev'):
        assert t.loc[name, 'loading in sd'] == pytest.approx(0.5, rel=1e-6)
    assert (t['premium'] >= t.loc['expected_value', 'premium'] * 0.95).all()


def test_ruin_exponential_claims_against_closed_form():
    claims = ins.ClaimProcess(ins.Poisson(1.0), ins.exponential(1.0))
    firm = ins.Company(claims, capital=5.0, premium_rate=1.2)
    rho = 1 / 1.2
    exact = lambda u: rho * math.exp(-(1 - rho) * u)  # noqa: E731
    assert firm.safety_loading == pytest.approx(0.2)
    assert firm.adjustment_coefficient() == pytest.approx(1 - rho, rel=1e-6)   # R = 1/m - lambda/c
    us = [0.0, 2.0, 5.0, 10.0, 20.0]
    br = firm.ruin_exact(us)
    for u, (lo, hi, lund) in zip(us, br[['lower', 'upper', 'lundberg']].to_numpy()):
        assert lo - 1e-9 <= exact(u) <= hi + 1e-9 and hi - lo < 5e-3
        assert lund >= exact(u) - 1e-9                                  # Lundberg is an upper bound
    res = firm.simulate(horizon=150, n=15_000, seed=5)
    lo, hi = res.interval(res.ruin_probability())
    assert lo - 0.012 <= exact(5.0) <= hi + 0.012          # finite horizon sits a hair below infinite horizon


def test_ruin_exact_for_weibull_claims_brackets_simulation():
    claims = ins.ClaimProcess(ins.Poisson(2.0), ins.weibull_with_mean(0.8, 1.0))
    firm = ins.Company(claims, capital=3.0, premium_rate=2.6)
    br = firm.ruin_exact([3.0], n_grid=3000)
    lo, hi = br['lower'].iloc[0], br['upper'].iloc[0]
    assert 0 < lo <= hi and hi - lo < 0.01
    assert math.isnan(firm.adjustment_coefficient())                     # shape < 1: no Lundberg exponent
    res = firm.simulate(horizon=300, n=6000, seed=2)
    assert lo - 0.025 <= res.ruin_probability() <= hi + 0.01


def test_survival_decreases_with_horizon_and_rises_with_capital():
    claims = ins.ClaimProcess(ins.Poisson(20), ins.exponential(1.0))
    thin = ins.Company(claims, capital=2.0, premium_rate=21.0)
    rich = ins.Company(claims, capital=10.0, premium_rate=21.0)
    a = thin.simulate(5, 8000, seed=1)
    b = rich.simulate(5, 8000, seed=1)
    surv = a.survival(np.linspace(0, 5, 20))
    assert np.all(np.diff(surv) <= 1e-12) and surv[0] == 1.0
    assert b.ruin_probability() < a.ruin_probability()
    assert thin.simulate(5, 100, seed=1).summary()['paths'] == 100


def test_no_ruin_without_positive_loading_is_flagged():
    claims = ins.ClaimProcess(ins.Poisson(10.0), ins.exponential(1.0))
    firm = ins.Company(claims, capital=5.0, premium_rate=9.0)
    assert firm.safety_loading < 0 and math.isnan(firm.adjustment_coefficient())
    assert (firm.ruin_exact([0.0, 5.0])['upper'] == 1.0).all()
    assert firm.ruin_probability(horizon=50, n=500, seed=0) > 0.9


def test_priced_company_and_principle_comparison():
    claims = ins.ClaimProcess(ins.Poisson(60), ins.weibull_with_mean(1.4, 1000))
    firm = ins.Company.priced(claims, capital=30_000, principle='std_dev', beta=1.0)
    A = claims.aggregate(1)
    assert firm.premium_rate == pytest.approx(A.mean + A.std)
    table = ins.compare_principles(claims, capital=30_000, horizon=5, n=4000, seed=1)
    assert table['ruin probability'].between(0, 1).all()
    # a bigger premium (VaR 99%) must not make ruin likelier than the cheapest one
    assert table.loc['percentile', 'ruin probability'] <= table.loc['expected_value', 'ruin probability']


def test_plots_render():
    import matplotlib
    matplotlib.use('Agg')
    claims = ins.ClaimProcess(ins.Poisson(30), ins.weibull_with_mean(1.4, 1000))
    firm = ins.Company.priced(claims, capital=20_000, principle='std_dev', beta=0.8)
    assert firm.plot(horizon=5, n=500, show=False) is not None
    A = claims.aggregate(1)
    assert A.plot(premiums={'EV': ins.premium(A, 'expected_value')}) is not None
