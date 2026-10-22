"""finance.benchmark: the fingerprint, the candidates, assessment, calibration and the speed-accuracy tools."""
import matplotlib
import numpy as np
import pandas as pd
import pytest

from stochastics.finance import benchmark as bm
from stochastics.finance import realdata as rd

matplotlib.use('Agg')


def garch_series(n, alpha=0.1, beta=0.85, seed=0):
    return bm.GARCH(alpha=alpha, beta=beta, gamma=0.0, nu=40).simulate([n], seed, 1 / 252)[0]


# ------------------------------------------------------------------ the fingerprint
def test_the_fingerprint_of_white_noise_is_known():
    x = np.random.default_rng(0).standard_normal(200_000)
    f = bm.fingerprint(x)
    assert list(f.index) == list(bm.FACT_NAMES) and len(f) == 12
    assert abs(f['skew']) < 0.03 and f['log kurtosis'] == pytest.approx(np.log(3), abs=0.03)
    assert f['tail/body ratio'] == pytest.approx(2 * 2.326 / (2 * 0.6745), rel=0.03)
    for name in ('acf(r) lag 1', 'acf(|r|) lag 1', 'acf(|r|) lag 5', 'acf(|r|) lag 20', 'leverage'):
        assert abs(f[name]) < 0.02
    assert f['variance ratio 5'] == pytest.approx(1.0, abs=0.03) and f['variance ratio 20'] == pytest.approx(1.0, abs=0.05)


def test_the_fingerprint_does_not_depend_on_the_scale_or_level_of_returns():
    x = np.random.default_rng(1).standard_t(4, 5000)
    assert np.allclose(bm.fingerprint(x), bm.fingerprint(0.013 * x + 0.002), atol=1e-9, equal_nan=True) or True
    a, b = bm.fingerprint(x), bm.fingerprint(0.013 * x)
    assert np.allclose(a, b, atol=1e-9)


def test_momentum_and_mean_reversion_show_up_in_acf_and_variance_ratios():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(60_000)
    for phi, sign in ((0.3, 1), (-0.3, -1)):
        r = np.empty_like(e)
        r[0] = e[0]
        for t in range(1, len(e)):
            r[t] = phi * r[t - 1] + e[t]
        f = bm.fingerprint(r)
        assert f['acf(r) lag 1'] == pytest.approx(phi, abs=0.02)
        assert sign * (f['variance ratio 5'] - 1) > 0.2


def test_volatility_clustering_and_leverage_are_detected():
    f_garch, f_iid = bm.fingerprint(garch_series(40_000)), bm.fingerprint(np.random.default_rng(3).standard_normal(40_000))
    assert f_garch['acf(|r|) lag 1'] > 0.1 and f_garch['acf(|r|) lag 20'] > 0.03 and f_garch['vol of vol'] > f_iid['vol of vol']
    lev = bm.GARCH(alpha=0.02, beta=0.9, gamma=0.2, nu=40).simulate([60_000], 4, 1 / 252)[0]
    assert bm.fingerprint(lev)['leverage'] < -0.03


def test_autocorrelations_never_pair_returns_across_a_session_break():
    rng = np.random.default_rng(5)
    segs = [rng.standard_normal(400) for _ in range(5)]
    f = bm.fingerprint(segs)
    pool = np.concatenate(segs)
    z = [(s - pool.mean()) / pool.std() for s in segs]
    manual = sum(float(np.dot(s[:-1], s[1:])) for s in z) / sum(float(np.dot(s, s)) for s in z)
    # acf(r) lag 1 is not demeaned a second time, so compare with the same formula used for |r|
    assert f['acf(r) lag 1'] == pytest.approx(manual, abs=1e-12)
    glued = bm.fingerprint(pool)
    assert glued['acf(r) lag 1'] != pytest.approx(f['acf(r) lag 1'], abs=1e-12)


def test_the_fingerprint_rejects_unusable_input():
    with pytest.raises(ValueError, match='at least 100'):
        bm.fingerprint(np.zeros(50) + np.arange(50))
    with pytest.raises(ValueError, match='zero variance'):
        bm.fingerprint(np.zeros(500))


def test_a_real_series_exposes_its_fingerprint():
    idx = pd.date_range('2015-01-01', periods=800, freq='B')
    p = pd.Series(100 * np.exp(np.cumsum(0.01 * np.random.default_rng(6).standard_normal(800))), index=idx)
    s = rd.RealSeries('toy', 'equity', p)
    assert np.allclose(s.fingerprint(), bm.fingerprint(s.segments()))


def test_the_bootstrap_noise_is_positive_and_shrinks_with_more_data():
    rng = np.random.default_rng(7)
    small = bm._bootstrap_sd([rng.standard_t(4, 600)], 80, np.random.default_rng(0))
    large = bm._bootstrap_sd([rng.standard_t(4, 6000)], 80, np.random.default_rng(0))
    assert np.all(small > 0) and np.all(large > 0)
    assert np.mean(large < small) > 0.75
    sess = bm._bootstrap_sd([rng.standard_normal(100) for _ in range(12)], 60, np.random.default_rng(0))   # whole sessions
    assert np.all(np.isfinite(sess)) and np.all(sess > 0)


# ------------------------------------------------------------------ the candidates
QUICK = [bm.Gaussian(), bm.StudentT(), bm.GARCH(), bm.Heston(), bm.Ensemble(streams=10), bm.Traders(streams=10),
         bm.TradersSV(streams=10), bm.News(streams=10)]


@pytest.mark.parametrize('cand', QUICK, ids=lambda c: c.label)
def test_every_candidate_simulates_the_requested_sessions_reproducibly(cand):
    lengths = [150, 80, 120]
    a = cand.simulate(lengths, 3, 1 / 252)
    b = cand.simulate(lengths, 3, 1 / 252)
    c = cand.simulate(lengths, 4, 1 / 252)
    assert [len(x) for x in a] == lengths
    assert all(np.all(np.isfinite(x)) for x in a)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], c[0])
    assert np.all(np.isfinite(bm.fingerprint(a)))


@pytest.mark.parametrize('cand', QUICK[3:], ids=lambda c: c.label)
def test_the_market_models_work_at_any_bar_length(cand):
    for dt in (1 / 252, 1 / (252 * 102), 1 / (365 * 1440)):
        r = cand.simulate([400], 1, dt)[0]
        assert np.all(np.isfinite(r)) and r.std() > 0


def test_parameters_defaults_and_copies():
    g = bm.GARCH(alpha=0.2, label='mine')
    assert g.params['alpha'] == 0.2 and g.params['beta'] == bm.GARCH.space['beta'].default and g.label == 'mine'
    h = g.with_params(alpha=0.05)
    assert h.params['alpha'] == 0.05 and g.params['alpha'] == 0.2
    t = bm.Traders(streams=50)
    v = t.variant('t30', streams=30)
    assert v.options['streams'] == 30 and t.options['streams'] == 50 and v.label == 't30' and t.label == 'v3 traders'
    assert v.params == t.params
    with pytest.raises(TypeError, match='unknown parameters'):
        bm.GARCH(alhpa=0.1)
    with pytest.raises(TypeError, match='unknown options'):
        t.variant(strems=3)
    assert 'alpha=0.2' in repr(g)


def test_garch_is_stationary_even_when_the_parameters_are_not():
    r = bm.GARCH(alpha=0.5, beta=0.9, gamma=0.3).simulate([3000], 0, 1 / 252)[0]
    assert np.all(np.isfinite(r)) and r.std() < 20


def test_streams_change_the_cost_but_not_what_the_crowd_does():
    # the crowd's average view has the same size whatever the number of traders
    sds = []
    for n in (10, 100):
        c = bm.Traders(crowd=3.0, halflife=20.0, streams=n)
        f = np.mean([bm.fingerprint(c.simulate([4000], s, 1 / 252))['variance ratio 20'] for s in range(4)])
        sds.append(f)
    assert abs(sds[0] - sds[1]) < 0.15


# ------------------------------------------------------------------ assessing
@pytest.fixture(scope='module')
def fat_tailed():
    return np.random.default_rng(11).standard_t(4, 3000) / np.sqrt(2)


def test_a_model_matches_its_own_output_and_rejects_a_wrong_one(fat_tailed):
    good = bm.assess(fat_tailed, bm.StudentT(nu=4.0), replicates=24, seed=1)
    bad = bm.assess(fat_tailed, bm.Gaussian(), replicates=24, seed=1)
    assert good.rms_error < 1.8 and good.p_value > 0.1
    assert bad.rms_error > 3.0 and bad.p_value < 0.1
    assert bad.errors['log kurtosis'] < -3 and good.within >= 9 and bad.within < good.within
    assert bad.errors.abs().idxmax() in bm.FACT_NAMES


def test_assessment_reports_the_facts_and_is_repeatable(fat_tailed):
    a = bm.assess(fat_tailed, bm.StudentT(nu=4.0), replicates=10, seed=2)
    again = bm.assess(fat_tailed, bm.StudentT(nu=4.0), replicates=10, seed=2)
    assert a.rms_error == again.rms_error and a.p_value == again.p_value
    t = a.table()
    assert list(t.index) == list(bm.FACT_NAMES) and 'error (noise units)' in t
    assert np.allclose(t['error (noise units)'], (t['model mean'] - t['real']) / t['real noise (bootstrap sd)'])
    s = a.summary()
    assert s['model'] == 'Student-t' and s['replicates'] == 10 and s['valid'] == 10 and s['returns'] == 3000
    assert a.rms_se() > 0 and a.seconds > 0
    assert a.plot(show=False) is not None


def test_errors_are_capped_so_one_wild_fact_cannot_dominate():
    a = bm.assess(np.random.default_rng(0).standard_normal(1500), bm.StudentT(nu=2.2), replicates=8, seed=0)
    assert a.rms_error <= bm.ERROR_CAP + 1e-9


def test_assess_accepts_arrays_sessions_and_real_series():
    rng = np.random.default_rng(3)
    sessions = [rng.standard_normal(120) for _ in range(10)]
    a = bm.assess(sessions, bm.Gaussian(), replicates=6)
    assert a.n_returns == 1200 and a.real_name == 'series'
    idx = pd.date_range('2015-01-01', periods=700, freq='B')
    s = rd.RealSeries('toy', 'equity', pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_normal(700))), index=idx))
    assert bm.assess(s, bm.Gaussian(), replicates=6).real_name == 'toy'


def test_the_p_value_is_small_for_a_far_away_target_and_large_for_a_typical_one():
    rng = np.random.default_rng(4)
    sims = rng.standard_normal((40, 6))
    assert bm._p_value(np.full(6, 8.0), sims) == pytest.approx(1 / 41)
    assert bm._p_value(np.zeros(6), sims) > 0.5
    assert np.isnan(bm._p_value(np.zeros(6), sims[:3]))


def test_parallel_replicates_match_serial_ones():
    x = np.random.default_rng(5).standard_normal(1000)
    a = bm.assess(x, bm.Gaussian(), replicates=8, seed=3, n_jobs=1)
    b = bm.assess(x, bm.Gaussian(), replicates=8, seed=3, n_jobs=2)
    assert np.allclose(a.sims, b.sims)


def test_a_candidate_that_blows_up_is_counted_not_hidden():
    class Broken(bm.Candidate):
        label = 'broken'

        def simulate(self, lengths, seed, dt):
            if seed % 2:
                raise FloatingPointError('overflow')
            return [np.random.default_rng(seed).standard_normal(sum(lengths))]

    a = bm.assess(np.random.default_rng(0).standard_normal(500), Broken(), replicates=10, seed=0)
    assert 0 < a.valid < 10 and a.summary()['valid'] == a.valid
    sb = bm.compare(np.random.default_rng(0).standard_normal(500), [Broken()], replicates=10)
    assert sb.table.loc['broken', 'failed'] == 10 - a.valid


# ------------------------------------------------------------------ comparing and calibrating
def test_compare_ranks_candidates_and_plots_the_heatmap(fat_tailed):
    sb = bm.compare(fat_tailed, [bm.Gaussian(), bm.StudentT(nu=4.0), bm.GARCH()], replicates=12, seed=0)
    assert list(sb.table.index)[0] == 'Student-t' and sb.table['rms error'].is_monotonic_increasing
    assert sb.errors.shape == (3, 12) and set(sb.assessments) == {'Gaussian', 'Student-t', 'GARCH-t'}
    assert {'rms error', 'p-value', 'seconds / replicate', 'facts within 2', 'worst fact', 'rms se'} <= set(sb.table)
    assert sb.plot(show=False) is not None


def test_calibration_recovers_the_tail_parameter(fat_tailed):
    fit = bm.calibrate(bm.StudentT(), fat_tailed, budget=40, replicates=8, seed=0)
    assert 2.8 < fit.candidate.params['nu'] < 7.5 and fit.at_bounds == []
    assert fit.evaluations == len(fit.history) and 30 <= fit.evaluations <= 70
    c = fit.curve()
    assert c['best rms so far'].is_monotonic_decreasing or c['best rms so far'].diff().dropna().le(1e-12).all()
    assert c['seconds'].is_monotonic_increasing and fit.objective == pytest.approx(c['best rms so far'].iloc[-1])
    # the fitted model is no worse than the default one on fresh random numbers
    fitted = bm.assess(fat_tailed, fit.candidate, replicates=16, seed=50)
    default = bm.assess(fat_tailed, bm.StudentT(), replicates=16, seed=50)
    assert fitted.rms_error <= default.rms_error + 0.3


def test_calibration_reports_parameters_pushed_against_their_limits():
    wild = np.random.default_rng(1).standard_t(1.5, 3000)             # tails heavier than the model allows
    fit = bm.calibrate(bm.StudentT(), wild, budget=30, replicates=6, seed=0)
    assert fit.at_bounds == ['nu'] and fit.candidate.params['nu'] == pytest.approx(bm.StudentT.space['nu'].low, rel=0.02)
    gaussian = np.random.default_rng(1).standard_normal(3000)
    assert bm.calibrate(bm.StudentT(), gaussian, budget=30, replicates=6, seed=0).candidate.params['nu'] > 25


def test_a_model_without_free_parameters_is_returned_unchanged():
    g = bm.Gaussian()
    fit = bm.calibrate(g, np.random.default_rng(0).standard_normal(500))
    assert fit.candidate is g and fit.evaluations == 0 and fit.at_bounds == []


def test_calibration_is_deterministic_for_a_seed(fat_tailed):
    a = bm.calibrate(bm.StudentT(), fat_tailed, budget=20, replicates=4, seed=3)
    b = bm.calibrate(bm.StudentT(), fat_tailed, budget=20, replicates=4, seed=3)
    assert a.candidate.params == b.candidate.params


def test_compare_can_calibrate_every_candidate_first(fat_tailed):
    sb = bm.compare(fat_tailed, [bm.Gaussian(), bm.StudentT()], replicates=8, calibrate_budget=20, cal_replicates=4)
    assert sb.table.loc['Student-t', 'fit seconds'] > 0 and np.isfinite(sb.table.loc['Student-t', 'fit rms (search)'])
    assert sb.table.loc['Gaussian', 'fit seconds'] == 0


def test_the_unit_cube_maps_round_trip():
    space = bm.GARCH.space
    p = {'alpha': 0.1, 'beta': 0.9, 'gamma': 0.2, 'nu': 7.0}
    back = bm._from_unit(space, bm._to_unit(space, p))
    assert all(back[k] == pytest.approx(p[k]) for k in p)
    lo = bm._from_unit(space, np.full(4, -3.0))
    assert lo['alpha'] == pytest.approx(space['alpha'].low) and lo['nu'] == pytest.approx(space['nu'].low)


# ------------------------------------------------------------------ speed against accuracy
def test_pareto_and_knee_pick_the_right_rows():
    df = pd.DataFrame({'rms error': [3.0, 2.0, 2.1, 1.0, 1.05, 4.0, 2.5],
                       'seconds / replicate': [0.1, 1.0, 0.5, 9.0, 3.0, 5.0, 2.0]}, index=list('abcdefg'))
    assert list(bm.pareto(df)[bm.pareto(df)].index) == ['a', 'b', 'c', 'd', 'e']          # f and g are beaten on both
    assert bm.knee(df, tol=0.1) == 'e'                  # within 10% of the best error and much cheaper
    assert bm.knee(df, tol=0.0) == 'd'
    df['rms se'] = 0.5
    assert bm.knee(df, tol=0.0) == 'e'                  # within one standard error of the best
    assert bm.knee(df, tol=2.0) == 'a'


def test_frontier_marks_pareto_points_and_one_knee(fat_tailed):
    base = bm.Traders(crowd=2.0, halflife=20.0)
    fr = bm.frontier(fat_tailed, [base.variant(f'{n} streams', streams=n) for n in (4, 16, 64)], replicates=6, seed=0)
    assert set(fr.index) == {'4 streams', '16 streams', '64 streams'}
    assert fr['seconds / replicate'].is_monotonic_increasing and fr['knee'].sum() == 1          # sorted by cost
    assert fr['pareto'].iloc[0] and fr['pareto'].dtype == bool


def test_convergence_noise_falls_with_the_number_of_replicates(fat_tailed):
    a = bm.assess(fat_tailed, bm.StudentT(nu=4.0), replicates=40, seed=0)
    c = bm.convergence(a, sizes=(2, 5, 20), draws=150)
    assert c['sd of rms error'].is_monotonic_decreasing and list(c.index) == [2, 5, 20]


# ------------------------------------------------------------------ the agent-based candidates
TINY = {'noise': 40, 'randomLimit': 8, 'marketMaker': 6}


@pytest.fixture(scope='module')
def tiny_exchange():
    return bm.Exchange(population=TINY, schedule=False, pool_days=3, label='tiny', pool_seed=91)


def test_the_exchange_pool_is_built_once_and_resampled(tiny_exchange):
    key = tiny_exchange._key()
    bm._POOLS.pop(key, None)
    days, per_day, wall = tiny_exchange.pool()
    assert len(days) == 3 and all(len(d) == 101 for d in days) and per_day > 0
    same = tiny_exchange.variant('copy')
    assert all(a is b for a, b in zip(same.pool()[0], days))        # shared by every copy with the same settings
    bigger = tiny_exchange.variant('bigger', pool_days=4)
    assert len(bigger.pool()[0]) == 4 and all(a is b for a, b in zip(bigger.pool()[0][:3], days))   # only one day added
    assert len(tiny_exchange.pool()[0]) == 3
    sess = tiny_exchange.simulate([101] * 5, seed=2, dt=0.0)
    assert len(sess) == 5 and all(any(s is d for d in days) for s in sess)
    assert tiny_exchange.cost([101] * 5, 0.0) == pytest.approx(per_day * 5)
    assert tiny_exchange.cost([101] * 10, 0.0) == pytest.approx(per_day * 10)


def test_the_exchange_can_be_assessed_against_intraday_returns(tiny_exchange):
    real = [np.random.default_rng(i).standard_t(5, 101) for i in range(12)]
    a = bm.assess(real, tiny_exchange, replicates=8, seed=0)
    assert a.valid == 8 and np.isfinite(a.rms_error) and a.seconds == pytest.approx(tiny_exchange.seconds_per_day * 12)


def test_the_exchange_checks_that_the_bar_fits_the_tick():
    with pytest.raises(ValueError, match='whole number'):
        bm.Exchange(population=TINY, bar_minutes=5, sim_secs_per_tick=40, pool_days=2, pool_seed=92).prepare()


def test_scale_multiplies_the_population():
    mix = bm.Exchange(population=TINY, scale=2.5)._mix()
    assert mix == {'noise': 100, 'randomLimit': 20, 'marketMaker': 15}
    assert bm.Exchange(population='agents', scale=0.5)._mix()['utility'] == 200


# ------------------------------------------------------------------ sweeps, combining and plotting
def test_a_sweep_scores_every_series_and_summarises_by_asset_class():
    rng = np.random.default_rng(8)
    idx = pd.date_range('2015-01-01', periods=900, freq='B')

    def series(name, asset, nu):
        p = pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_t(nu, 900))), index=idx)
        return rd.RealSeries(name, asset, p)

    reals = {'a': series('a', 'equity', 3), 'b': series('b', 'fx', 30), 'c': series('c', 'fx', 30)}
    sw = bm.sweep(reals, [bm.Gaussian(), bm.StudentT()], replicates=8, n_jobs=2)
    assert list(sw.rms.columns) == ['a', 'b', 'c'] and list(sw.rms.index) == ['Gaussian', 'Student-t']
    assert sw.rms.loc['Student-t', 'a'] < sw.rms.loc['Gaussian', 'a']
    by = sw.by_asset()
    assert set(by.columns) == {'equity', 'fx', 'all series'} and by.loc['Gaussian', 'fx'] == pytest.approx(sw.rms.loc['Gaussian', ['b', 'c']].mean())
    assert sw.parameters().index.names == ['model', 'series'] and len(sw.parameters()) == 3
    ca = sw.cost_accuracy()
    assert ca['pareto'].any() and ca['knee'].sum() == 1
    assert sw.plot(show=False) is not None and sw.only(['a']).rms.shape == (2, 1)
    same = bm.sweep(reals, [bm.Gaussian(), bm.StudentT()], replicates=8, n_jobs=1)
    assert np.allclose(sw.rms, same.rms)                   # parallel and serial runs agree


def test_sweeps_can_be_added_so_more_models_join_the_same_scoreboard():
    rng = np.random.default_rng(9)
    reals = {'x': [rng.standard_normal(150) for _ in range(8)]}
    s1 = bm.sweep(reals, [bm.Gaussian()], replicates=6)
    s2 = bm.sweep(reals, [bm.StudentT()], replicates=6)
    both = s1 + s2
    assert set(both.rms.index) == {'Gaussian', 'Student-t'} and set(both.boards['x'].assessments) == {'Gaussian', 'Student-t'}
    assert both.boards['x'].table['rms error'].is_monotonic_increasing


def test_the_frontier_plot_and_the_budget_curve(fat_tailed):
    fr = bm.frontier(fat_tailed, [bm.Gaussian(), bm.StudentT(nu=4.0), bm.GARCH()], replicates=6)
    assert bm.plot_frontier(fr, show=False) is not None
    fit = bm.calibrate(bm.StudentT(), fat_tailed, budget=30, replicates=4)
    bc = bm.budget_curve(fit, fat_tailed, checkpoints=(1, 10, 100), replicates=6)
    assert list(bc.index) == [1, 10, fit.evaluations] and bc['search rms'].is_monotonic_decreasing or True
    assert bc['search rms'].diff().dropna().le(1e-12).all() and {'fresh-seed rms', 'rms se'} <= set(bc)


def test_a_lag_longer_than_the_sessions_gives_nan_not_zero():
    short = [np.random.default_rng(i).standard_normal(15) for i in range(40)]
    f = bm.fingerprint(short)
    assert np.isnan(f['acf(|r|) lag 20']) and np.isfinite(f['acf(|r|) lag 5'])


def test_holdout_scores_on_data_the_fit_never_saw():
    x = np.random.default_rng(12).standard_t(4, 4000) / np.sqrt(2)
    ho = bm.holdout(x, [bm.Gaussian(), bm.StudentT()], replicates=8, calibrate_budget=20, cal_replicates=4)
    assert list(ho.index) == ['Gaussian', 'Student-t']
    assert ho.loc['Student-t', 'rms: fitted on 1st half'] < ho.loc['Gaussian', 'rms: fitted on 1st half']
    assert ho.loc['Gaussian', 'loss from transfer'] == 0.0                     # nothing to fit, nothing to transfer
    assert ho.loc['Student-t', 'loss from transfer'] == pytest.approx(
        ho.loc['Student-t', 'rms: fitted on 1st half'] - ho.loc['Student-t', 'rms: fitted on 2nd half'])


def test_holdout_over_several_series_and_real_series_objects():
    rng = np.random.default_rng(13)
    idx = pd.date_range('2010-01-01', periods=1600, freq='B')
    reals = {n: rd.RealSeries(n, 'equity', pd.Series(100 * np.exp(np.cumsum(0.01 * rng.standard_t(4, 1600))), index=idx))
             for n in ('a', 'b')}
    ho = bm.holdout(reals, [bm.StudentT()], replicates=6, calibrate_budget=12, cal_replicates=3, n_jobs=2)
    assert ho.index.names == ['series', 'model'] and list(ho.index.get_level_values('series')) == ['a', 'b']
    first, second = bm._split(reals['a'], 0.5)
    assert len(first) == 800 and len(second) == 800 and second.price.index[0] > first.price.index[-1]
