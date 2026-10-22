"""Invariants and equivalences for stochastics.physics.

Run with:  pytest -q
"""
import matplotlib

matplotlib.use('Agg')

import numpy as np
import pytest

from stochastics.physics import (Cluster, Collisions, CompositionAffinity, ContactPolicy, Coulomb,
                                 Drag, Environment, Gravity, MergeWithinDistance, NoForce,
                                 Particle, ParticleSystem, SpaceExpansion, emit, simulate)
from stochastics.physics.policies import (ElectromagneticPolicy, ExternalPolicy,
                                          GravitationalPolicy, InternalPolicy, OppositesAttract,
                                          ThermalEmission, polarity_sequence)


def residual(result):
    d = result.diagnostics
    return float(d.accounted_energy.iloc[-1] - d.accounted_energy.iloc[0])


def two_body(comp_a, comp_b, speed=0.4, mass=1.0, radius=0.1):
    return ParticleSystem.from_particles([
        Particle(mass=mass, position=[-0.5, 0, 0], velocity=[speed, 0, 0], radius=radius,
                 composition=comp_a),
        Particle(mass=mass, position=[0.5, 0, 0], velocity=[-speed, 0, 0], radius=radius,
                 composition=comp_b),
    ])


# ------------------------------------------------------------------ clocks
def test_polarity_sequence_matches_the_brief():
    seq = polarity_sequence(polarity=-1, spin=0.5, steps=10, dt=1.0)
    assert np.allclose(seq, [-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5, 0])


def test_clock_runs_inside_the_engine():
    p = ParticleSystem.from_particles([Particle(spin=0.5, polarity=-1, position=[0, 0])])
    r = simulate(p, internal=InternalPolicy(), steps=8, dt=1.0)
    assert np.allclose(r.history.polarity_now, [-1, -0.5, 0, 0.5, 1, 0.5, 0, -0.5, -1])


# ------------------------------------------------------------ force laws
def _reference_gravity(state, G=0.5, soft=0.15):
    a = np.zeros_like(state.position)
    ids = state.live
    x, m = state.position[ids], state.mass[ids]
    delta = x[None, :, :] - x[:, None, :]
    inv3 = ((delta ** 2).sum(-1) + soft ** 2) ** -1.5
    np.fill_diagonal(inv3, 0)
    a[ids] = G * np.einsum('ijk,ij,j->ik', delta, inv3, m)
    return a


def _reference_coulomb(state, k=0.2, soft=0.15):
    a = np.zeros_like(state.position)
    ids = state.live
    x, m, q = state.position[ids], state.mass[ids], state.charge[ids]
    delta = x[None, :, :] - x[:, None, :]
    inv3 = ((delta ** 2).sum(-1) + soft ** 2) ** -1.5
    np.fill_diagonal(inv3, 0)
    a[ids] = -k * (q / m)[:, None] * np.einsum('ijk,ij,j->ik', delta, inv3, q)
    return a


def test_force_laws_match_reference_implementation():
    s = ParticleSystem.random(n=150, preset='em', dimension=3, seed=3)
    assert np.allclose(Gravity()(s, 0.0), _reference_gravity(s), atol=1e-12)
    assert np.allclose(Coulomb()(s, 0.0), _reference_coulomb(s), atol=1e-12)
    combined = Environment(law=[Gravity(), Coulomb()]).law
    assert np.allclose(combined(s, 0.0), _reference_gravity(s) + _reference_coulomb(s), atol=1e-12)


def test_force_is_minus_gradient_of_potential():
    s = ParticleSystem.random(n=12, preset='em', dimension=3, seed=8)
    law = Environment(law=[Gravity(G=0.3), Coulomb(k=0.4)]).law
    acc = law(s, 0.0)
    h = 1e-6
    for i, d in [(0, 0), (3, 1), (7, 2)]:
        s.position[i, d] += h
        up = law.potential(s)
        s.position[i, d] -= 2 * h
        down = law.potential(s)
        s.position[i, d] += h
        assert np.isclose(-(up - down) / (2 * h) / s.mass[i], acc[i, d], rtol=1e-5, atol=1e-8)


def test_policies_reproduce_known_physics():
    base = ParticleSystem.random(n=50, preset='em', dimension=3, seed=5)
    cols = ['x', 'y', 'z']
    a = simulate(base.copy(), law=Coulomb(k=0.2), steps=200, dt=0.01, record_every=50)
    b = simulate(base.copy(), external=ElectromagneticPolicy(k=0.2), steps=200, dt=0.01,
                 record_every=50)
    assert np.allclose(a.history[cols], b.history[cols], atol=1e-12)
    a = simulate(base.copy(), law=Gravity(G=0.5), steps=200, dt=0.01, record_every=50)
    b = simulate(base.copy(), external=GravitationalPolicy(G=0.5), steps=200, dt=0.01,
                 record_every=50)
    assert np.allclose(a.history[cols], b.history[cols], atol=1e-12)


# ------------------------------------------------------------ conservation
def test_pure_mergers_conserve_everything():
    world = ParticleSystem.random(n=80, dimension=2, seed=12)
    world.charge[:40], world.charge[40:] = 1.0, -1.0
    r = simulate(world, law=NoForce(),
                 rules=[MergeWithinDistance(distance=0.23, radiated_fraction=0.3)],
                 steps=600, dt=0.025)
    d = r.diagnostics
    assert len(r.events) > 10
    for col in ('mass', 'charge', 'p_x', 'p_y', 'accounted_energy'):
        assert np.allclose(d[col], d[col].iloc[0], atol=1e-10), col


def test_merger_energy_is_accounted_up_to_integrator_error():
    def run(dt):
        s = ParticleSystem.random(n=40, dimension=2, seed=4, position_std=1.0, velocity_std=0.2)
        s.radius[:] = 0.06
        return simulate(s, law=Gravity(G=0.05, softening=0.1), rules=[Collisions()],
                        steps=int(4 / dt), dt=dt, record_every=20)
    coarse, fine = run(0.01), run(0.005)
    assert len(fine.events) > 10
    assert abs(residual(fine)) < abs(residual(coarse)) / 2.5   # ~second order


def test_momentum_mass_charge_exact_with_collisions_3d():
    s = ParticleSystem.random(n=200, preset='em', seed=2, clusters=[
        Cluster(mean=[0, 0, 0], std=0.5), Cluster(mean=[1.5, 0, 0], std=0.5, velocity=[-0.5, 0, 0])])
    r = simulate(s, Environment(law=[Gravity(G=0.05), Coulomb(k=0.02)], rules=[Collisions()]),
                 steps=200, dt=0.01, record_every=20)
    d = r.diagnostics
    assert len(r.events) > 20
    assert np.allclose(d.mass, d.mass.iloc[0], rtol=1e-12)
    assert np.allclose(d.charge, d.charge.iloc[0], atol=1e-10)
    for c in 'xyz':
        assert np.allclose(d[f'p_{c}'], d[f'p_{c}'].iloc[0], atol=1e-9)
    start = r.initial_state.composition[r.initial_state.live].sum(axis=0)
    end = r.final_state.composition[r.final_state.live].sum(axis=0)
    assert np.array_equal(start, end)                 # constituents are conserved
    assert (r.final_state.composition >= 0).all()


def test_drag_and_expansion_are_booked():
    s = ParticleSystem.random(n=60, dimension=3, seed=9, velocity_std=0.3)
    r = simulate(s, law=Gravity(G=0.02), rules=[Drag(gamma=0.5)],
                 expansion=SpaceExpansion(rate=0.05), steps=400, dt=0.005, record_every=40)
    d = r.diagnostics
    assert d.dissipated.iloc[-1] > 0 and d.expansion.iloc[-1] != 0
    scale = abs(d.kinetic.iloc[0]) + abs(d.potential.iloc[0])
    assert abs(residual(r)) / scale < 1e-3


def test_expansion_scales_separations_exponentially():
    s = ParticleSystem.random(n=20, dimension=2, seed=1)
    s.velocity[:] = 0
    d0 = np.linalg.norm(s.position[0] - s.position[1])
    simulate(s, law=NoForce(), expansion=SpaceExpansion(rate=0.2), steps=500, dt=0.01)
    assert np.isclose(np.linalg.norm(s.position[0] - s.position[1]), d0 * np.exp(0.2 * 5.0))


# ------------------------------------------------------------- collisions
def test_elastic_bounce_swaps_velocities_and_keeps_energy():
    s = two_body([1, 0], [1, 0])
    r = simulate(s, law=NoForce(), rules=[Collisions(ContactPolicy())], steps=300, dt=0.01)
    assert list(r.events.event) == ['elastic']
    assert np.allclose(s.velocity[:, 0], [-0.4, 0.4])
    d = r.diagnostics
    assert np.isclose(d.kinetic.iloc[-1], d.kinetic.iloc[0])


@pytest.mark.parametrize('comp_a, comp_b, speed, expected', [
    ([5, 0], [5, 0], 0.4, 'elastic'),     # every part repels every part
    ([3, 2], [3, 2], 0.4, 'inelastic'),   # mixed interiors: lossy bounce
    ([3, 2], [0, 5], 0.4, 'inelastic'),   # mostly opposite parts, fast: escapes
    ([3, 2], [0, 5], 0.1, 'merge'),       # ... same pair, slow: captured
    ([5, 0], [0, 5], 0.4, 'merge'),       # every part attracts: sticks
])
def test_composition_sets_the_collision_outcome(comp_a, comp_b, speed, expected):
    s = two_body(comp_a, comp_b, speed=speed)
    r = simulate(s, law=NoForce(), rules=[Collisions(CompositionAffinity())], steps=800, dt=0.01)
    assert r.events.event.iloc[0] == expected
    d = r.diagnostics
    assert np.isclose(d.accounted_energy.iloc[-1], d.accounted_energy.iloc[0])


def test_restitution_gradient_is_monotone():
    pol = CompositionAffinity()
    es = [pol.restitution(w) for w in np.linspace(-1, 1, 21)]
    assert np.isclose(es[0], 1.0) and np.isclose(es[-1], pol.e_min)
    assert np.all(np.diff(es) <= 0)


def test_continuous_detection_prevents_tunnelling():
    def fast_pair():
        return ParticleSystem.from_particles([
            Particle(position=[-0.3, 0], velocity=[60.0, 0], radius=0.05, composition=[1, 0]),
            Particle(position=[0.3, 0], velocity=[0.0, 0], radius=0.05, composition=[1, 0])])
    swept = simulate(fast_pair(), law=NoForce(), rules=[Collisions(ContactPolicy())], steps=1,
                     dt=0.02)
    naive = simulate(fast_pair(), law=NoForce(),
                     rules=[Collisions(ContactPolicy(), continuous=False)], steps=1, dt=0.02)
    assert len(swept.events) == 1 and len(naive.events) == 0


def test_emission_conserves_mass_momentum_and_energy():
    s = ParticleSystem.from_particles([Particle(mass=2.0, charge=1.0, energy=1.0, radius=0.1,
                                                velocity=[0.3, -0.2, 0.1], position=[0, 0, 0],
                                                composition=[3, 1])])
    p0 = s.mass[0] * s.velocity[0]
    e0 = 0.5 * s.mass[0] * s.velocity[0] @ s.velocity[0] + s.internal_energy[0]
    emit(s, 0, energy=0.6, direction=[1, 1, 0], time=0.0, parts=np.array([0, 1]))
    live = s.live
    assert np.isclose(s.mass[live].sum(), 2.0) and np.isclose(s.charge[live].sum(), 1.0)
    assert np.allclose((s.mass[live, None] * s.velocity[live]).sum(axis=0), p0)
    e1 = 0.5 * np.sum(s.mass[live] * (s.velocity[live] ** 2).sum(axis=1)) + s.internal_energy[live].sum()
    assert np.isclose(e1, e0)
    assert np.array_equal(s.composition[live].sum(axis=0), [3, 1])


def test_thermal_emission_cycle_keeps_the_books():
    p = ParticleSystem.random(n=60, preset='em', dimension=2, seed=7)
    r = simulate(p, external=OppositesAttract(strength=0.05, softening=0.1),
                 internal=ThermalEmission(threshold=0.01), steps=600, dt=0.01, record_every=30)
    d = r.diagnostics
    assert {'emit'} <= set(r.events.event)
    assert np.allclose(d.mass, d.mass.iloc[0]) and np.allclose(d.charge, d.charge.iloc[0])
    scale = abs(d.kinetic.iloc[0]) + abs(d.potential.iloc[0])
    assert abs(residual(r)) / scale < 1e-3


# ----------------------------------------------------------------- policies
def test_non_reciprocal_policy_runs_without_potential():
    class Chase(ExternalPolicy):
        def affinity(self, a, b):
            return np.where(a.cluster_id[:, None] == 0, 1.0, -1.0) * np.ones((a.n, b.n))

    p = ParticleSystem.random(n=30, dimension=2, seed=3, n_clusters=2)
    r = simulate(p, external=Chase(strength=0.01, reciprocal=False), steps=50, dt=0.01)
    assert np.isnan(r.diagnostics.potential).all()


def test_opposites_attract_with_clocks_is_energy_accounted():
    p = ParticleSystem.random(n=80, preset='em', dimension=3, seed=11)
    r = simulate(p, external=OppositesAttract(strength=0.05), internal=InternalPolicy(),
                 steps=500, dt=0.01, record_every=50)
    d = r.diagnostics
    assert d.drive.abs().iloc[-1] > 0
    scale = abs(d.kinetic.iloc[0]) + abs(d.potential.iloc[0])
    assert abs(residual(r)) / scale < 1e-3


# ------------------------------------------------------------------- state
def test_random_with_explicit_clusters():
    p = ParticleSystem.random(n=500, seed=210, clusters=[
        Cluster(mean=[0, 0, 0], std=100), Cluster(mean=[100, -100, 0], std=150, weight=3)])
    assert p.dimension == 3 and len(p) == 500
    counts = np.bincount(p.cluster_id)
    assert 80 < counts[0] < 170 and counts[0] + counts[1] == 500
    assert np.allclose(p.position[p.cluster_id == 1].mean(axis=0), [100, -100, 0], atol=30)
    with pytest.raises(ValueError):
        ParticleSystem.random(n=10, dimension=2, clusters=[Cluster(mean=[0, 0, 0])])


def test_particle_views_write_through_and_build_from_particles():
    p = ParticleSystem.from_particles([
        Particle(mass=0.005, charge=1, position=[1, 2, 3], radius=0.01, spin=0.5, polarity=0.5,
                 energy=1, velocity=[0.1, 0, 0]),
        Particle(mass=2.0, position=[0, 0, 0])])
    assert p[0].mass == 0.005 and p[0].energy == 1 and list(p[0].position) == [1, 2, 3]
    p[0].mass = 3.0
    p[1].position[:] = [9, 9, 9]
    assert p.mass[0] == 3.0 and np.allclose(p.position[1], 9)
    new_id = p.add(Particle(mass=1.0, position=[5, 5, 5]))
    assert len(p) == 3 and p.get(new_id).mass == 1.0
    assert 'polarity_now' in repr(p[0])


def test_state_time_continues_between_runs():
    p = ParticleSystem.random(n=5, seed=1)
    simulate(p, law=NoForce(), steps=10, dt=0.1)
    r = simulate(p, law=NoForce(), steps=10, dt=0.1)
    assert np.isclose(r.diagnostics.time.iloc[0], 1.0) and np.isclose(p.time, 2.0)


# ------------------------------------------------------------------ visuals
def test_plots_and_animations_render():
    from stochastics.physics import (animate_2d, animate_3d, plot_collision_regimes,
                                     plot_dashboard, plot_initial)
    p = ParticleSystem.random(n=40, preset='em', seed=4, clusters=[
        Cluster(mean=[0, 0, 0], std=0.5), Cluster(mean=[2, 0, 0], std=0.5)])
    assert plot_initial(p, show=False) is not None
    r = simulate(p, Environment(law=Gravity(G=0.05), rules=[Collisions()]), steps=60, dt=0.01,
                 record_every=10)
    assert plot_dashboard(r, show=False) is not None
    assert plot_collision_regimes(show=False) is not None
    assert 'img' in animate_2d(r, max_frames=4).data
    assert 'img' in animate_3d(r, max_frames=4, trail=2).data
