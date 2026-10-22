"""Population models: ledgers close, conservation, reproducibility, expected behaviour."""
import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest

from stochastics.popdyn import (Consumption, Economy, Ecosystem, Epidemic, Investment,
                                KineticExchange, LuckEvents, RockPaperScissors, Schelling, Tax,
                                Trait, Wages, YardSale, gini, lorenz, top_share)


@pytest.fixture(scope='module')
def eco():
    return Ecosystem(width=50, height=50, seed=3).run(400, record_every=5)


def test_ecosystem_energy_ledger_closes(eco):
    stocks = eco.ledger[['grass_energy', 'rabbit_energy', 'wolf_energy']].sum(axis=1)
    assert abs(eco.ledger.residual).max() < 1e-8 * max(1.0, stocks.max())


def test_ecosystem_bookkeeping(eco):
    a = eco.agents
    tot = eco.totals()['total']
    assert int((a.cause == 'eaten').sum()) == int(tot['rabbits eaten by wolves (items)'])
    born_r = int(((a.species == 'rabbit') & (a.born > 0)).sum())
    assert born_r == int(tot['rabbit births'])
    alive = a[a.alive]
    assert int((alive.species == 'rabbit').sum()) == int(eco.counts['rabbit'].iloc[-1])
    assert (a.loc[a.parent >= 0, 'generation'].to_numpy()
            == a.loc[a.loc[a.parent >= 0, 'parent'], 'generation'].to_numpy() + 1).all()


def test_ecosystem_reproducible():
    a = Ecosystem(width=30, height=30, seed=9).run(100).counts
    b = Ecosystem(width=30, height=30, seed=9).run(100).counts
    assert a.equals(b)


def test_ecosystem_traits_stay_within_ceilings(eco):
    r = eco.agents[eco.agents.species == 'rabbit']
    assert r.speed.max() <= eco.eco.prey.max_speed + 1e-12
    assert r.fertility.between(0, 1).all()
    eff = eco.trait_effects('rabbit')
    assert set(eff.index) == {'speed', 'vision', 'fertility'}


def test_epidemic_conserves_people_and_tree():
    r = Epidemic(n=1500, seed=2, sigma=0.5, fatality=0.05).run(40)
    c = r.counts
    assert (c[['S', 'E', 'I', 'R', 'D']].sum(axis=1) == 1500).all()
    p = r.people
    caused = p.secondary.sum()
    infected_by_someone = (p.infected_by >= 0).sum()
    assert caused == infected_by_someone
    assert (p.loc[p.infected_by >= 0, 'infected_at'] > 0).all()


def test_epidemic_mobility_speeds_spread():
    def reach(mobility):
        return Epidemic(n=1500, mobility=mobility, seed=4).run(40).people.infected_at.notna().mean()
    assert reach(0.2) > reach(0.005) + 0.2           # mixing turns local clusters into an epidemic


def test_epidemic_mean_field_r0():
    m = Epidemic(n=2000, radius=0.01, beta=1.0, gamma=0.5)
    ode = m.mean_field(50)
    assert abs(ode.attrs['R0'] - 1.0 * (1999 * np.pi * 1e-4) / 0.5) < 0.05
    assert np.allclose(ode.sum(axis=1), 2000)


def test_rock_paper_scissors_densities_and_extinction():
    r = RockPaperScissors(size=50, mobility=2e-2, seed=3).run(1500, frame_every=500)
    fr = r.counts
    assert np.allclose(fr.sum(axis=1), 1.0)
    assert r.summary().loc['species surviving', 'value'].startswith('1')
    r2 = RockPaperScissors(size=80, mobility=5e-5, seed=3).run(300, frame_every=300)
    assert r2.summary().loc['species surviving', 'value'].startswith('3')


@pytest.mark.parametrize('rules', [[YardSale()], [KineticExchange()], [YardSale(), Tax(0.05)]])
def test_exchange_rules_conserve_wealth(rules):
    r = Economy(n=3000, rules=rules, seed=1).run(60)
    total = r.ledger.total
    assert np.allclose(total, total.iloc[0], rtol=1e-12)


def test_wealth_ledger_closes_with_creation():
    r = Economy(n=3000, rules=[Wages(), Investment(), LuckEvents(), Consumption(), Tax(0.02)],
                seed=2).run(60)
    assert abs(r.ledger.residual).max() < 1e-8 * r.ledger.total.max()
    assert (r.agents.final >= 0).all()


def test_luck_beats_talent():
    r = Economy(n=10_000, rules=[LuckEvents()], seed=3).run(80)
    eff = r.trait_effects()
    assert eff.loc['luck', 'spearman_rho'] > eff.loc['talent', 'spearman_rho'] > 0
    assert r.counts['gini'].iloc[-1] > 0.6


def test_custom_traits_and_rule():
    class Inheritance:
        def apply(self, m):
            gift = 0.01 * m.wealth * (m.traits['family'] > 0.9)
            m.wealth += gift
            return {'created': gift.sum()}

    traits = [Trait('family', 'uniform', 0, 1), Trait('talent', 'normal', 0.6, 0.1, clip=(0, 1))]
    r = Economy(n=2000, traits=traits, rules=[Inheritance(), LuckEvents()], seed=4).run(40)
    assert 'family' in r.agents and abs(r.ledger.residual).max() < 1e-8 * r.ledger.total.max()


def test_inequality_measures():
    assert gini(np.ones(100)) == pytest.approx(0, abs=1e-12)
    x = np.zeros(1000)
    x[0] = 1
    assert gini(x) == pytest.approx(0.999, abs=1e-3)
    assert top_share(np.arange(1, 101), 0.01) == pytest.approx(100 / 5050)
    px, py = lorenz(np.ones(10))
    assert np.allclose(px, py)


def test_schelling_segregates():
    r = Schelling(size=40, tolerance=0.35, seed=1).run(40)
    assert r.counts['similarity'].iloc[-1] > r.counts['similarity'].iloc[0] + 0.15


def test_dashboards_and_animations(eco):
    assert eco.plot(show=False) is not None
    assert 'img' in eco.animate(max_frames=3).data
    ep = Epidemic(n=500, seed=1).run(10)
    assert ep.plot(show=False) is not None and 'img' in ep.animate(max_frames=3).data
    rps = RockPaperScissors(size=40, seed=1).run(20)
    assert rps.plot(show=False) is not None and 'img' in rps.animate(max_frames=3).data
    ec = Economy(n=1000, seed=1).run(20)
    assert ec.plot(show=False) is not None
    sg = Schelling(size=30, seed=1).run(10)
    assert sg.plot(show=False) is not None and 'img' in sg.animate(max_frames=3).data
