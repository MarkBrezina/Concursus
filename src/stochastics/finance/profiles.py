"""Version 7 (first draft): traders described by psychology, not by a strategy class.

    from stochastics.finance import exchange
    from stochastics.finance.profiles import Profile, ProfiledTrader, sample_profiles

    me = Profile(resilience=0.5, greed=0.5, patience=1.0, hypothesis='mean_reversion', desired_price=100)
    res = exchange.simulate('brief', extra=[(ProfiledTrader, {'profile': p}) for p in sample_profiles(300, seed=1)])
    res.by_type(by='label')                       # one row per hypothesis
    res.trait_effects('profiled')                 # which traits make money

A :class:`Profile` is just numbers.  The hard part, as the brief says, is the *functions* that turn those numbers
into trading, so every one of them is a separate, replaceable method of :class:`Behavior`: change one, or
subclass ``Behavior`` and pass ``behavior=``, and the rest keeps working.  The defaults below are a first
reading of the brief, written to be argued with, not a finished theory.

=============  ================================================================================================
trait          what it does in the default mapping
=============  ================================================================================================
resilience     0..1  stop-loss distance grows with it; ``1 - resilience`` is the chance per tick of dropping a
               position that is going against the trader, scaled by how far it has gone
greed          0..1  how far (in standard deviations) the signal must stretch before entering, and how large a
               profit the trader waits for
patience       >0 (1 = typical)  scales how often it looks at the market (``1/patience``), how long it must hold
               before a signal exit is allowed, the share of orders sent as market orders (impatient: market),
               and how much it leans on its desired price rather than on its hypothesis
hypothesis     ``'mean_reversion'`` buys below / sells above the moving average; ``'trend'`` follows the move;
               ``'buy_hold'`` buys below its belief and waits for stop or profit; ``'noise'`` acts on a coin flip
desired_price  the price the trader trades around (default: the price when it joins)
internal_value the trader's perceived fundamental value (default: ``desired_price``)
info           0..1  how much of the market's hidden fair value leaks into its belief (0 = fixed belief,
               1 = knows the truth): the "informed trader is a big trader with better behaviour"
capital        sets order size: ``capital * 5% * (0.5 + resilience) / price`` shares - the bigger it is, the
               closer it is to a whale
stoploss /     override the derived distances (fractions, e.g. 0.01)
takeprofit
=============  ================================================================================================

Note on ``internal_value``: your example lists it next to the desired price, but later says "the larger
internal_value becomes, the closer to whale the trader becomes".  Here those are two fields, ``internal_value``
(a belief, in price units) and ``capital`` (size).  If you meant one number for both, merge them in
``Behavior.size`` and ``Behavior.belief``.

A learned policy (e.g. a GRU) plugs in through ``policy=``: it replaces the rule engine, receives the trader and
the market state, and can use ``trader.features(m)`` (a fixed-length vector of market features and the trader's
own traits) as its observation.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field, fields
from typing import Callable

import numpy as np

from .agents import BOT_TYPES, Agent, _clamp, register
from .lob import BUY, SELL

__all__ = ['Profile', 'Behavior', 'ProfiledTrader', 'sample_profiles', 'HYPOTHESES', 'FEATURE_NAMES']

HYPOTHESES = ('mean_reversion', 'trend', 'buy_hold', 'noise')
FEATURE_NAMES = ('dev_z_60', 'move_z_10', 'move_z_30', 'imbalance', 'vol_rank', 'heat', 'belief_gap_pct',
                 'position_value / capital', 'pnl_pct', 'holding_ticks / 100',
                 'resilience', 'greed', 'patience', 'info')


@dataclass
class Profile:
    """A trader's psychology as numbers (see the module docstring for what each does)."""
    resilience: float = 0.5
    greed: float = 0.5
    patience: float = 1.0
    hypothesis: str = 'mean_reversion'
    desired_price: float | None = None
    internal_value: float | None = None
    capital: float = 50_000.0
    info: float = 0.0
    stoploss: float | None = None
    takeprofit: float | None = None

    def __post_init__(self):
        if self.hypothesis not in HYPOTHESES:
            raise ValueError(f'hypothesis must be one of {HYPOTHESES}')
        if self.patience <= 0 or self.capital <= 0:
            raise ValueError('patience and capital must be positive')
        for k in ('resilience', 'greed', 'info'):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError(f'{k} must be in [0, 1]')

    @classmethod
    def random(cls, rng: random.Random, price: float = 100.0, hypotheses=HYPOTHESES, weights=None,
               capital_median: float = 50_000.0, capital_sigma: float = 1.2) -> 'Profile':
        """Draw one trader.  Resilience ~ Beta(2, 2), greed ~ Beta(2, 3), patience lognormal around 1, capital
        lognormal (a few whales), info ~ Beta(0.6, 4) pushed up for big traders."""
        z = rng.gauss(0, 1)
        cap = capital_median * math.exp(capital_sigma * z)
        info = _clamp(rng.betavariate(0.6, 4.0) * (1 + 0.5 * z), 0.0, 1.0)
        desired = price * math.exp(0.01 * rng.gauss(0, 1))
        hyp = rng.choices(hypotheses, weights=weights)[0]
        return cls(resilience=rng.betavariate(2, 2), greed=rng.betavariate(2, 3), patience=math.exp(0.6 * rng.gauss(0, 1)),
                   hypothesis=hyp, desired_price=desired, internal_value=desired * math.exp(0.004 * rng.gauss(0, 1)),
                   capital=cap, info=info)


def sample_profiles(n: int, seed: int | None = None, price: float = 100.0, **kw) -> list[Profile]:
    """``n`` random profiles (see :meth:`Profile.random`)."""
    rng = random.Random(seed)
    return [Profile.random(rng, price, **kw) for _ in range(n)]


class Behavior:
    """Turns a :class:`Profile` into decisions.  Every method is replaceable on its own."""

    def activity(self, p: Profile) -> float:
        """Chance per tick that the trader looks at the market (before the attention multiplier)."""
        return _clamp(0.04 / p.patience, 0.002, 0.5)

    def band(self, p: Profile) -> float:
        """How stretched the signal (in standard deviations) must be to enter."""
        return 0.3 + 1.2 * p.greed

    def stop_loss(self, p: Profile) -> float:
        return p.stoploss if p.stoploss is not None else 0.003 + 0.02 * p.resilience

    def take_profit(self, p: Profile) -> float:
        return p.takeprofit if p.takeprofit is not None else (0.003 + 0.012 * p.greed) * math.sqrt(p.patience)

    def flinch(self, p: Profile, pnl_pct: float) -> float:
        """Chance per tick of dropping the position because it is going against the trader."""
        if pnl_pct >= 0:
            return 0.0
        return _clamp((1.0 - p.resilience) * 0.25 * (-pnl_pct) / self.stop_loss(p), 0.0, 1.0)

    def min_hold(self, p: Profile) -> int:
        """Ticks a position is held before a *signal* may close it (stops and profit targets always apply)."""
        return int(round(20 * p.patience))

    def market_prob(self, p: Profile) -> float:
        """Share of entries sent as market orders: the impatient pay the spread, the patient rest limit orders."""
        return _clamp(1.2 - 0.7 * p.patience, 0.05, 1.0)

    def anchor(self, p: Profile) -> float:
        """Weight on the desired price against the hypothesis: patient traders lean on their own valuation."""
        return 0.5 * p.patience / (1.0 + p.patience) if p.hypothesis != 'buy_hold' else 1.0

    def belief(self, p: Profile, m, desired: float) -> float:
        """Perceived value: the trader's own view, with some of the hidden fair value leaking in."""
        own = p.internal_value if p.internal_value is not None else desired
        return (1.0 - p.info) * own + p.info * m.value

    def hypothesis_signal(self, p: Profile, m, rng) -> float:
        """Standardised signal from the trader's hypothesis; positive means buy."""
        if p.hypothesis == 'mean_reversion':
            return -m.dev_z(60)
        if p.hypothesis == 'trend':
            return m.move_z(30)
        if p.hypothesis == 'noise':
            return 1.5 * rng.gauss(0, 1)
        return 0.0

    def anchor_signal(self, p: Profile, m, desired: float) -> float:
        """How far below the believed value the price sits (net of half the spread), in standard deviations of recent prices."""
        gap = self.belief(p, m, desired) - m.mid
        half = 0.5 * (m.book.spread() or 0.0)                       # a gap inside the spread cannot pay
        gap = math.copysign(max(0.0, abs(gap) - half), gap)
        return gap / max(m.std(60), 0.0005 * m.mid)

    def signal(self, p: Profile, m, desired: float, rng) -> float:
        a = self.anchor(p)
        s_h = self.hypothesis_signal(p, m, rng) if a < 1.0 else 0.0
        return (1.0 - a) * s_h + a * self.anchor_signal(p, m, desired)

    def wants_exit(self, p: Profile, m, direction: int, s: float, rng) -> bool:
        """Signal-based exit (``s`` is the current signal; ``direction`` is +1 long, -1 short)."""
        if p.hypothesis == 'buy_hold':
            return False
        aligned = direction * s
        if p.hypothesis == 'mean_reversion':
            return aligned < 0.2                           # back at the mean (or beyond)
        if p.hypothesis == 'trend':
            return aligned < -0.25 * self.band(p)          # the trend has turned
        return rng.random() < 0.02                         # noise: leaves at random

    def size(self, p: Profile, m) -> int:
        """Shares per entry: a fraction of capital, so big traders are whales."""
        return max(1, int(p.capital * 0.05 * (0.5 + p.resilience) / m.mid))


class ProfiledTrader(Agent):
    """A trader driven by a :class:`Profile` through a :class:`Behavior` (or by a learned ``policy``)."""
    TYPE = 'profiled'
    TRAITS = ('resilience', 'greed', 'patience', 'log_capital', 'info', 'belief_gap_pct')

    def __init__(self, id, fleet, profile: Profile | None = None, behavior: Behavior | None = None,
                 policy: Callable | None = None, **opts):
        super().__init__(id, fleet, **opts)
        price = fleet.book.mid()
        self.profile = profile or Profile.random(self.rng, price)
        p = self.profile
        self.behavior = behavior or Behavior()
        self.policy = policy
        self.desired_price = p.desired_price if p.desired_price is not None else price
        self.internal_value = p.internal_value if p.internal_value is not None else self.desired_price
        self.resilience, self.greed, self.patience, self.info = p.resilience, p.greed, p.patience, p.info
        self.log_capital = math.log10(p.capital)
        self.belief_gap_pct = 100.0 * (self.internal_value / price - 1.0)
        self.label = f'profiled:{p.hypothesis}'
        self.entry_tick = 0
        self.max_position = max(1, int(0.5 * p.capital / price))

    # ---- observation for learned policies
    def features(self, m) -> np.ndarray:
        """Fixed-length observation (see ``FEATURE_NAMES``): market features and the trader's own traits."""
        p = self.profile
        gap = self.behavior.belief(p, m, self.desired_price) / m.mid - 1.0
        held = (self.fleet.tick - self.entry_tick) / 100.0 if self.shares else 0.0
        return np.array([m.dev_z(60), m.move_z(10), m.move_z(30), m.imbalance,
                         0.5 if m.vol_rank is None else m.vol_rank, m.heat, 100.0 * gap,
                         self.shares * m.mid / p.capital, self.pnl_pct(m.mid), held,
                         p.resilience, p.greed, p.patience, p.info], dtype=np.float32)

    def on_fill(self, side, price, qty):
        was_flat = self.shares == 0
        super().on_fill(side, price, qty)
        if was_flat and self.shares != 0:
            self.entry_tick = self.fleet.tick

    def exit_position(self):
        self.cancel_all()
        self.flatten()

    def on_tick(self, m):
        if self.policy is not None:
            self.policy(self, m)
            return
        p, b = self.profile, self.behavior
        if self.shares != 0:
            pnl = self.pnl_pct(m.mid)
            if pnl >= b.take_profit(p) or pnl <= -b.stop_loss(p) or self.chance(b.flinch(p, pnl)):
                self.exit_position()
            elif self.fleet.tick - self.entry_tick >= b.min_hold(p) and m.length >= 60:
                s = b.signal(p, m, self.desired_price, self.rng)
                if b.wants_exit(p, m, 1 if self.shares > 0 else -1, s, self.rng):
                    self.exit_position()
            return
        if m.length < 60 or not self.chance(b.activity(p) * m.attention):
            return
        self.prune_orders()
        if self.orders:
            return                                       # an entry order is still resting
        s = b.signal(p, m, self.desired_price, self.rng)
        if abs(s) < b.band(p):
            return
        side = BUY if s > 0 else SELL
        if p.hypothesis == 'buy_hold' and side == SELL:
            return                                       # buy and hold only ever buys
        qty = self.capped(side, b.size(p, m))
        if qty <= 0 or not self.allows(side):
            return
        if self.chance(b.market_prob(p)):
            self.market(side, qty)
        else:
            off = self.rand(0.0, 1.0) * m.tick_sigma
            self.limit(side, m.mid - off if side == BUY else m.mid + off, qty, 8)


register('profiled', ProfiledTrader)
