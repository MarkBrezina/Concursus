"""finance.market: versions 1-4 (stochastic markets) and their building blocks."""
import math

import numpy as np
import pytest
from scipy import stats

from stochastics.finance import _streams as K
from stochastics.finance import market


def shocks(n, steps, dt, seed=0):
    return np.random.default_rng(seed).normal(0, math.sqrt(dt), (n, steps))


# ------------------------------------------------------------------ kernels
def test_gbm_and_abm_kernels_match_theory():
    n, steps, T = 40_000, 50, 1.0
    dt = T / steps
    dW = shocks(n, steps, dt, 1)
    g = K.gbm(100.0, 0.08, 0.3, dW, dt)
    assert g.shape == (n, steps + 1) and np.all(g[:, 0] == 100.0)
    assert g[:, -1].mean() == pytest.approx(100 * math.exp(0.08), rel=0.01)
    assert np.log(g[:, -1] / 100).std() == pytest.approx(0.3, rel=0.02)
    a = K.abm(100.0, 0.08, 0.3, dW, dt)
    assert a[:, -1].mean() == pytest.approx(108.0, abs=0.4)
    assert a[:, -1].std() == pytest.approx(30.0, rel=0.02)


def test_heterogeneous_parameters_per_stream():
    dt = 0.01
    dW = shocks(2, 100, dt, 2)
    S = K.gbm(100.0, np.array([0.0, 0.0]), np.array([0.0, 0.5]), dW, dt)
    assert np.allclose(S[0], 100.0)             # sigma = 0 row is flat
    assert S[1].std() > 1.0


def test_bridge_pins_the_end_and_has_bridge_variance():
    n, steps = 30_000, 40
    dt = 1.0 / steps
    B = K.bridge(100.0, 120.0, 0.25, shocks(n, steps, dt, 3), dt)
    assert np.allclose(B[:, 0], 100.0) and np.allclose(B[:, -1], 120.0)
    mid = B[:, steps // 2]
    assert mid.mean() == pytest.approx(110.0, abs=0.3)
    assert mid.std() == pytest.approx(0.25 * 100 * math.sqrt(0.25), rel=0.03)      # sigma s0 sqrt(t(T-t)/T)


def test_heston_mean_variance_and_leverage():
    n, steps, T = 20_000, 100, 1.0
    dt = T / steps
    rng = np.random.default_rng(4)
    dW, dZ = rng.normal(0, math.sqrt(dt), (2, n, steps))
    S, v = K.heston(100.0, 0.05, 3.0, 0.04, 0.4, -0.7, 0.09, dW, dZ, dt)
    assert S[:, -1].mean() == pytest.approx(100 * math.exp(0.05), rel=0.01)
    expected_v = 0.04 + (0.09 - 0.04) * np.exp(-3.0 * T)
    assert v[:, -1].mean() == pytest.approx(expected_v, rel=0.05)
    dv = v[:, -1] - v[:, 0]
    assert np.corrcoef(np.log(S[:, -1] / 100), dv)[0, 1] < -0.4          # rho < 0: leverage effect


def test_ou_deviation_stationary_law_and_catch_up():
    n, steps, dt = 4000, 400, 0.002
    rng = np.random.default_rng(5)
    theta, eta = 40.0, 0.2
    d = K.ou_deviation(theta, eta, 0.0, rng.standard_normal(n), rng.standard_normal((n, steps)), dt)
    target_sd = eta / math.sqrt(2 * theta)
    assert d[:, 0].std() == pytest.approx(target_sd, rel=0.05)
    assert d[:, -1].std() == pytest.approx(target_sd, rel=0.05)
    kick = np.zeros(steps)
    kick[100] = 0.05
    dk = K.ou_deviation(theta, 1e-9, 0.0, np.zeros(n), np.zeros((n, steps)), dt, kick=kick, kappa=np.full(n, 0.5))
    assert dk[0, 101] == pytest.approx(-0.025)                          # half the jump not priced in at once
    assert abs(dk[0, 300]) < 1e-4                                       # ... and the gap has closed later


# ---------------------------------------------------------------- facts
def test_stylized_facts_on_known_series():
    rng = np.random.default_rng(6)
    gauss = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 20_000)))
    f = market.stylized_facts(gauss, dt=0.001)
    assert abs(f['excess kurtosis']) < 0.2 and abs(f['acf(|r|), lag 1']) < 0.03
    assert f['annualised vol'] == pytest.approx(0.01 / math.sqrt(0.001), rel=0.03)
    fat = 100 * np.exp(np.cumsum(rng.standard_t(3, 20_000) * 0.005))
    assert market.stylized_facts(fat)['tail index (Hill, top 5%)'] < 4.5
    # a GARCH-like series clusters volatility
    sig, r = 0.01, []
    for _ in range(20_000):
        x = sig * rng.standard_normal()
        r.append(x)
        sig = math.sqrt(1e-6 + 0.1 * x * x + 0.85 * sig * sig)
    assert market.stylized_facts(100 * np.exp(np.cumsum(r)))['acf(|r|), lag 1'] > 0.1


# ------------------------------------------------------------ the versions
def test_validation_and_reproducibility():
    with pytest.raises(ValueError):
        market.Market(version=5)
    with pytest.raises(ValueError):
        market.Market(version=2, kinds=('gbm', 'nope'))
    with pytest.raises(ValueError):
        market.Market(version=4, event_process='hawkes', event_alpha=2, event_beta=1)
    a = market.Market(version=4, streams=20, steps=200, seed=3).run()
    b = market.Market(version=4, streams=20, steps=200, seed=3).run()
    c = market.Market(version=4, streams=20, steps=200, seed=4).run()
    assert np.array_equal(a.price, b.price) and not np.array_equal(a.price, c.price)


def test_version1_is_a_single_heston_path():
    r = market.Market(version=1, steps=500, seed=1).run()
    assert r.streams.shape == (1, 501) and np.array_equal(r.price, r.streams[0])
    assert r.extras['variance'].min() >= 0 and r.price.min() > 0


def test_version2_average_loses_noise_unless_streams_share_a_factor():
    def vol(n, corr, seed):
        return market.Market(version=2, streams=n, steps=800, correlation=corr, seed=seed).run().facts()['annualised vol']
    indep_small = np.mean([vol(10, 0.0, s) for s in range(6)])
    indep_big = np.mean([vol(400, 0.0, s) for s in range(6)])
    assert indep_small / indep_big > 4.0                                  # ~ sqrt(400 / 10) = 6.3
    shared = np.mean([vol(400, 0.5, s) for s in range(6)])
    assert shared == pytest.approx(0.2 * math.sqrt(0.5), rel=0.25)        # sigma sqrt(rho)
    r = market.Market(version=2, streams=300, steps=300, seed=1).run()
    assert set(r.kinds) == set(market.KINDS)
    assert r.streams.shape == (300, 301)
    assert np.allclose(r.price, r.streams.mean(axis=0))
    only = market.Market(version=2, streams=30, steps=100, kinds=('gbm',), aggregate='geometric', seed=1).run()
    assert set(only.kinds) == {'gbm'} and np.allclose(only.price, np.exp(np.log(only.streams).mean(axis=0)))


def test_version3_traders_scatter_around_the_internal_value():
    m = market.Market(version=3, streams=200, steps=600, deviation=0.02, bias=0.0, seed=2)
    r = m.run()
    assert r.value is not None and r.events is None and r.streams.shape == (200, 601)
    gap = np.log(r.streams / r.value[None, :])
    assert gap[:, 100:].std() == pytest.approx(0.02, rel=0.2)             # stationary dispersion
    assert r.tracking()['corr of returns with value returns'] > 0.95
    follow = market.Market(version=3, streams=200, steps=600, tether='follow', seed=2).run()
    g2 = np.log(follow.streams / follow.value[None, :])
    assert g2[:, -1].std() > 2 * g2[:, 50].std()                          # untethered: dispersion keeps growing


def test_version3_value_kinds():
    for kind in ('gbm', 'abm', 'bridge', 'heston'):
        r = market.Market(version=3, value=kind, streams=10, steps=200, seed=1).run()
        assert np.isfinite(r.value).all() and np.isfinite(r.price).all()
    br = market.Market(version=3, value='bridge', value_target=130.0, streams=5, steps=200, seed=1).run()
    assert br.value[-1] == pytest.approx(130.0)


def test_version4_events_and_jumps():
    T, rate = 1.0, 30.0
    counts = [len(market.Market(version=4, steps=500, streams=5, event_rate=rate, seed=s).run().events) for s in range(80)]
    assert np.mean(counts) == pytest.approx(rate * T, rel=0.08)
    assert np.var(counts) == pytest.approx(rate * T, rel=0.4)             # Poisson: variance = mean
    hk = [len(market.Market(version=4, steps=500, streams=5, event_process='hawkes', event_rate=10, event_alpha=60,
                            event_beta=100, seed=s).run().events) for s in range(80)]
    assert np.mean(hk) == pytest.approx(10 / (1 - 0.6), rel=0.12)         # stationary rate mu / (1 - alpha/beta)
    assert np.var(hk) > 2.0 * np.mean(hk)                                 # overdispersed: clustered arrivals
    r = market.Market(version=4, steps=2000, streams=5, event_rate=60, jump_median=0.02, jump_cap=0.1,
                      p_up=1.0, seed=3).run()
    j = r.events['jump'].to_numpy()
    assert (j > 0).all() and j.max() <= 0.1 + 1e-12                       # p_up = 1: all positive, capped
    assert np.median(j) == pytest.approx(0.02, rel=0.3)
    s = market.Market(version=4, steps=2000, streams=5, event_rate=60, seed=3).run().events['jump']
    assert 0.3 < (s > 0).mean() < 0.7


def test_news_makes_fat_tails_and_underreaction_makes_drift():
    k3 = np.mean([market.Market(version=3, steps=1500, streams=50, seed=s).run().facts()['excess kurtosis'] for s in range(4)])
    k4 = np.mean([market.Market(version=4, steps=1500, streams=50, event_rate=30, seed=s).run().facts()['excess kurtosis'] for s in range(4)])
    assert k4 > k3 + 1.5

    # paired runs (same seed, same news, same trader noise): only the under-reaction differs
    first, after = [], []
    for sd in range(5):
        runs = {u: market.Market(version=4, steps=1500, streams=80, underreaction=u, event_rate=20, seed=sd).run()
                for u in (0.0, 0.6)}
        r0, r6 = (np.diff(np.log(runs[u].price)) for u in (0.0, 0.6))
        for st, jump in zip(runs[0.0].events['step'], runs[0.0].events['jump']):
            if abs(jump) > 0.015 and st + 12 < len(r0):
                sgn = np.sign(jump)
                first.append(sgn * (r6[st - 1] - r0[st - 1]) / abs(jump))                 # impact on the news step
                after.append(sgn * (r6[st:st + 12].sum() - r0[st:st + 12].sum()) / abs(jump))   # the 12 steps after
    assert np.mean(first) < -0.3          # under-reacting traders move less at once...
    assert np.mean(after) > 0.1           # ... and then keep moving in the direction of the news


def test_compare_versions_and_plots():
    import matplotlib
    matplotlib.use('Agg')
    cmp = market.compare_versions(streams=30, steps=300, seed=1)
    assert list(cmp.table.columns) == ['version 1', 'version 2', 'version 3', 'version 4']
    assert cmp.plot(show=False) is not None
    for v in (1, 2, 3, 4):
        assert cmp.results[v].plot(show=False) is not None
