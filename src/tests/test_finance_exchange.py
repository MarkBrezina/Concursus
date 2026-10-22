"""finance.agents / exchange / profiles: the agent-based market (versions 5-7)."""
import math

import numpy as np
import pandas as pd
import pytest

from stochastics.finance import agents as A
from stochastics.finance import exchange
from stochastics.finance.lob import BUY, SELL
from stochastics.finance.profiles import (FEATURE_NAMES, HYPOTHESES, Behavior, Profile, ProfiledTrader,
                                          sample_profiles)

QUICK = dict(population={'noise': 60, 'randomLimit': 10, 'marketMaker': 8}, schedule=False, record_trades=False)


def assert_conserved(res, tol=1e-6):
    c = res.conservation()
    assert c['net change in shares'] == 0
    assert abs(c['net change in cash']) < tol
    assert c['total pnl'] == pytest.approx(c['pnl from initial endowments'], abs=1e-6)


# ------------------------------------------------------------------ the exchange
def test_a_full_scheduled_day_conserves_cash_and_shares():
    res = exchange.simulate('game', ticks=700, seed=1, record_trades=True)
    assert_conserved(res)
    assert len(res.price) == 700 and np.all(np.isfinite(res.price)) and res.price.min() > 0
    assert res.trades is not None and res.trades['qty'].sum() == res.volume.sum()
    assert res.trades_n.sum() == len(res.trades)
    res.book.check()                                  # never crossed, indices consistent
    two_sided = np.isfinite(res.bid) & np.isfinite(res.ask)
    assert two_sided.mean() > 0.9 and np.all(res.bid[two_sided] < res.ask[two_sided])
    assert res.news, 'the schedule should log its events'


def test_same_seed_same_day_different_seed_different_day():
    a = exchange.simulate(seed=5, ticks=300, **QUICK)
    b = exchange.simulate(seed=5, ticks=300, **QUICK)
    c = exchange.simulate(seed=6, ticks=300, **QUICK)
    assert np.array_equal(a.price, b.price) and np.array_equal(a.volume, b.volume)
    assert a.agents['pnl'].equals(b.agents['pnl'])
    assert not np.array_equal(a.price, c.price)


def test_a_seed_repeats_across_python_sessions():
    """Same seed, same day in every new session (set iteration order depends on the hash seed)."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    code = ("from stochastics.finance import exchange; "
            "r = exchange.simulate('game', seed=1, ticks=250, record_trades=False); "
            "print(repr(float(r.price[-1])), int(r.volume.sum()), round(float(r.agents['pnl'].sum()), 6))")
    outs = {subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, cwd=root,
                           env={**os.environ, 'PYTHONHASHSEED': h, 'PYTHONPATH': str(root)}).stdout for h in ('1', '2', '3')}
    assert len(outs) == 1, outs


def test_every_registered_agent_type_runs_and_conserves():
    for kind in sorted(A.BOT_TYPES):
        pop = {'noise': 40, 'marketMaker': 6, kind: 5}
        res = exchange.simulate(pop, ticks=240, seed=2, schedule=False, record_trades=False)
        assert_conserved(res)
        assert kind in set(res.agents['type']) or res.agents['type'].str.startswith(kind[:3]).any(), kind


def test_unknown_agent_type_is_reported():
    with pytest.raises(KeyError, match='unknown agent types'):
        exchange.simulate({'noise': 10, 'nope': 3}, ticks=10)


def test_market_makers_earn_and_noise_traders_pay():
    # in the website's full mix (in a tiny population inventory risk swamps the spread income)
    res = exchange.simulate('game', seed=1, ticks=1000, record_trades=False)
    t = res.by_type()
    assert {'agents', 'mean pnl', 'total pnl', 'profitable', 'shares traded', 'volume share'} <= set(t.columns)
    assert t.loc['marketMaker', 'mean pnl'] > 0 and t.loc['marketMaker', 'profitable'] > 0.6
    assert t.loc['noise', 'total pnl'] < 0
    assert t['volume share'].sum() == pytest.approx(1.0)
    assert_conserved(res)


def test_static_population_keeps_its_size_and_the_schedule_changes_it():
    static = exchange.simulate(seed=1, ticks=250, **QUICK)
    assert len(set(static.n_active)) == 1 and static.n_active[0] == 78
    sched = exchange.simulate('game', seed=1, ticks=250, record_trades=False)
    assert sched.n_active.max() > sched.n_active.min() or sched.n_active[0] != 1000


def test_the_hidden_value_moves_and_makers_follow_it():
    res = exchange.simulate(seed=4, ticks=800, **QUICK)
    assert res.value.std() > 0 and np.all(res.value > 0)
    # makers centre their quotes on it, so the mid cannot wander far from it
    assert np.abs(np.log(res.price / res.value)).mean() < 0.01


def test_extra_agents_policy_slot_and_trait_effects():
    calls = []

    def buyer(agent, m):
        calls.append(m.tick)
        if m.tick == 50:
            agent.market(BUY, 40)
        if m.tick == 150:
            agent.flatten()

    extra = [(A.PolicyAgent, {'policy': buyer, 'params': {'size': 40}})]
    res = exchange.simulate(seed=2, ticks=200, extra=extra, **QUICK)
    row = res.agents[res.agents['type'] == 'policy'].iloc[0]
    assert row['shares'] == 0 and row['fills'] >= 2 and row['volume'] == 80 and len(calls) == 200
    assert_conserved(res)
    tr = exchange.simulate(seed=2, ticks=600, **QUICK).trait_effects('marketMaker')
    assert list(tr.columns) == ['spearman', 'p-value', 'agents'] and len(tr) >= 1
    assert tr['spearman'].abs().max() <= 1.0


def test_market_state_indicators_are_sane():
    res = exchange.simulate(seed=1, ticks=400, **QUICK)
    h = res.price
    m = exchange.MarketState(res.book, h, len(h), len(h))
    assert m.sma(20) == pytest.approx(h[-20:].mean())
    assert m.std(30) == pytest.approx(h[-30:].std(ddof=1))
    assert m.ret(10) == pytest.approx(h[-1] / h[-11] - 1)
    assert -1.0 <= m.imbalance <= 1.0 and -1.0 <= m.up_down_score(20) <= 1.0
    assert abs(m.dev_z(60)) < 10 and abs(m.move_z(10)) < 10
    assert 0.01 <= m.tick_sigma <= 0.003 * m.mid + 1e-12


def test_exchange_summary_facts_and_plot():
    import matplotlib
    matplotlib.use('Agg')
    res = exchange.simulate(seed=1, ticks=400, **QUICK)
    s = res.summary()
    assert s['trades'] == res.trades_n.sum() and s['two-sided quotes'] > 0.9
    assert 'excess kurtosis' in res.facts().index
    assert res.plot(show=False) is not None


# ------------------------------------------------------------------ the brief's personalities
def test_brief_population_runs_with_all_its_types():
    res = exchange.simulate('brief', ticks=400, seed=1, schedule=False, record_trades=False)
    types = set(res.agents['type'])
    assert {'valueNoise', 'noise', 'investor', 'marketMaker', 'meanReverter', 'whale', 'informedWhale',
            'trendFollower', 'volatility'} <= types
    assert_conserved(res)


def test_value_noise_traders_hold_a_distribution_of_private_values():
    res = exchange.simulate({'valueNoise': 120, 'marketMaker': 8, 'noise': 20}, ticks=200, seed=3, schedule=False,
                            record_trades=False)
    vn = [a for a in res.fleet.agents.values() if a.type == 'valueNoise']
    prices = np.array([getattr(a, 'view', np.nan) for a in vn])
    assert np.isfinite(prices).all() and prices.std() > 0.3


def test_utility_trader_scales_its_holding_with_risk_aversion():
    kw = dict(seed=1, ticks=500, **QUICK)
    lo = exchange.simulate(extra=[(A.UtilityTrader, {'risk_aversion': 0.5}) for _ in range(10)], **kw)
    hi = exchange.simulate(extra=[(A.UtilityTrader, {'risk_aversion': 8.0}) for _ in range(10)], **kw)
    f = lambda r: r.agents.loc[r.agents['type'] == 'utility', 'shares'].abs().mean()
    assert f(hi) <= f(lo) + 1e-9


# ------------------------------------------------------------------ version 7: profiles
def test_profile_validation_and_sampling():
    with pytest.raises(ValueError):
        Profile(hypothesis='astrology')
    with pytest.raises(ValueError):
        Profile(greed=1.5)
    with pytest.raises(ValueError):
        Profile(patience=0.0)
    a, b = sample_profiles(50, seed=3), sample_profiles(50, seed=3)
    assert a == b and sample_profiles(50, seed=4) != a
    assert {p.hypothesis for p in sample_profiles(200, seed=1)} == set(HYPOTHESES)
    caps = np.array([p.capital for p in sample_profiles(500, seed=2)])
    assert caps.max() > 20 * np.median(caps), 'a few traders should be whale-sized'


def test_behavior_mappings_move_the_right_way():
    b = Behavior()
    lo, hi = Profile(patience=0.5), Profile(patience=3.0)
    assert b.activity(lo) > b.activity(hi)
    assert b.min_hold(lo) < b.min_hold(hi)
    assert b.market_prob(lo) > b.market_prob(hi)
    assert b.take_profit(lo) < b.take_profit(hi)
    assert b.stop_loss(Profile(resilience=0.9)) > b.stop_loss(Profile(resilience=0.1))
    assert b.band(Profile(greed=0.9)) > b.band(Profile(greed=0.1))
    assert b.take_profit(Profile(greed=0.9)) > b.take_profit(Profile(greed=0.1))
    fragile, tough = Profile(resilience=0.1), Profile(resilience=0.9)
    assert b.flinch(fragile, -0.005) > b.flinch(tough, -0.005) > 0
    assert b.flinch(fragile, +0.01) == 0.0, 'only a losing position can make a trader flinch'
    assert b.stop_loss(Profile(stoploss=0.05)) == 0.05 and b.take_profit(Profile(takeprofit=0.07)) == 0.07
    big, small = Profile(capital=1e6), Profile(capital=1e4)

    class M:
        mid = 100.0
    assert b.size(big, M) > 50 * b.size(small, M) > 0
    assert Behavior().anchor(Profile(hypothesis='buy_hold')) == 1.0


def test_belief_blends_in_the_hidden_value_with_info():
    class M:
        mid, value = 100.0, 103.0
    b = Behavior()
    assert b.belief(Profile(internal_value=99.0, info=0.0), M, 100.0) == pytest.approx(99.0)
    assert b.belief(Profile(internal_value=99.0, info=1.0), M, 100.0) == pytest.approx(103.0)
    assert b.belief(Profile(internal_value=99.0, info=0.5), M, 100.0) == pytest.approx(101.0)
    assert b.belief(Profile(info=0.0), M, 97.5) == pytest.approx(97.5), 'default belief is the desired price'


def test_profiled_traders_trade_conserve_and_are_deterministic():
    extra = [(ProfiledTrader, {'profile': p}) for p in sample_profiles(120, seed=7)]
    a = exchange.simulate(seed=2, ticks=700, extra=extra, **QUICK)
    b = exchange.simulate(seed=2, ticks=700, extra=extra, **QUICK)
    assert_conserved(a)
    assert np.array_equal(a.price, b.price)
    pr = a.agents[a.agents['type'] == 'profiled']
    assert len(pr) == 120 and pr['fills'].sum() > 100 and (pr['volume'] > 0).mean() > 0.3
    assert set(pr['label']) <= {f'profiled:{h}' for h in HYPOTHESES}
    split = a.by_type(by='label')
    assert split.index.str.startswith('profiled:').sum() >= 3


def test_buy_and_hold_never_goes_short_and_noise_hypothesis_trades_both_ways():
    ps = [Profile(hypothesis='buy_hold', patience=0.7, capital=40_000, internal_value=101.0) for _ in range(30)]
    ns = [Profile(hypothesis='noise', patience=0.7, greed=0.1, capital=40_000) for _ in range(30)]
    extra = [(ProfiledTrader, {'profile': p}) for p in ps + ns]
    res = exchange.simulate(seed=1, ticks=900, extra=extra, **QUICK)
    seen = {a.profile.hypothesis: [] for a in res.fleet.agents.values() if a.type == 'profiled'}
    for a in res.fleet.agents.values():
        if a.type == 'profiled':
            seen[a.profile.hypothesis].append(a.shares)
    assert min(seen['buy_hold']) >= 0 and max(seen['buy_hold']) > 0
    assert max(seen['noise']) > 0 > min(seen['noise']), 'the noise hypothesis should be long and short'


def test_stop_loss_and_take_profit_close_positions():
    class Fixed(Behavior):
        def signal(self, p, m, desired, rng):               # always buy
            return 5.0

        def market_prob(self, p):
            return 1.0

        def activity(self, p):
            return 0.5

        def wants_exit(self, *a):
            return False

    prof = Profile(hypothesis='trend', stoploss=0.004, takeprofit=0.004, capital=20_000, patience=0.3)
    extra = [(ProfiledTrader, {'profile': prof, 'behavior': Fixed()}) for _ in range(10)]
    res = exchange.simulate(seed=1, ticks=900, extra=extra, **QUICK)
    pr = [a for a in res.fleet.agents.values() if a.type == 'profiled']
    # they re-enter after every exit, so exits show up as many round trips
    assert np.mean([a.n_fills for a in pr]) > 4
    # a closed round trip is never much worse than the stop (market orders can slip a little)
    closed = [a.realized / max(a.volume / 2, 1) for a in pr]
    assert min(closed) > -0.02 * 100


def test_a_learned_policy_can_replace_the_rule_engine():
    seen = []

    def policy(trader, m):
        x = trader.features(m)
        seen.append(x)
        if m.tick == 60:
            trader.market(BUY, 25)
        if m.tick == 160:
            trader.exit_position()

    extra = [(ProfiledTrader, {'profile': Profile(capital=30_000), 'policy': policy})]
    res = exchange.simulate(seed=1, ticks=200, extra=extra, **QUICK)
    row = res.agents[res.agents['type'] == 'profiled'].iloc[0]
    assert row['shares'] == 0 and row['volume'] == 50
    X = np.array(seen)
    assert X.shape == (200, len(FEATURE_NAMES)) and np.isfinite(X).all()
    assert X[:, FEATURE_NAMES.index('patience')].tolist() == [1.0] * 200


def test_profiles_population_preset_and_trait_effects():
    res = exchange.simulate('profiles', ticks=350, seed=1, schedule=False, record_trades=False)
    assert (res.agents['type'] == 'profiled').sum() == 400
    assert_conserved(res)
    te = res.trait_effects('profiled')
    assert {'patience', 'greed', 'resilience'} <= set(te.index)


def test_lazy_package_attributes():
    import stochastics.finance as f
    for name in f.__all__:
        assert getattr(f, name).__name__ == f'stochastics.finance.{name}'
    with pytest.raises(AttributeError):
        f.not_a_module
