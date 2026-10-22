"""Trading agents for the order-book market (versions 5 and 6).

Every agent reads a shared ``MarketState`` ``m`` (built once per tick by the
fleet) and trades through the helpers on :class:`Agent`.  Positions, cash and
average entry are tracked from actual fills, so exit rules act on what really
executed.

Ported from ``bots.js`` (same class names and type keys, so a population dict
copied from the website works here): ``noise`` ``marketMaker`` ``whale`` ``buyer``
``seller`` ``limitLadder`` ``std`` ``trend1`` ``trend2`` ``trend3`` ``mr1`` ``mr2``
``mr3`` ``jump`` ``volatility`` ``randomLimit`` ``randomBuyer`` ``reverter``.

Added from the version-5/6 brief:

* ``valueNoise``     retail buyer with a private idea of fair price: buys below it, sells above
* ``investor``       follows a running mean, watches its own profit, applies stop-loss / take-profit
* ``meanReverter``   20/40 moving-average crossing, or a band of k standard deviations round the 60 average
* ``informedWhale``  large orders when the price is far from its perceived value (which may leak the truth)
* ``trendFollower``  buys an x% rise, takes profit after y% more, stops out z% below the entry
* ``utility``        wealth, risk aversion, a desired price that adapts at a learning rate (version 5)
* ``policy``         an empty slot: pass any ``policy(agent, m)`` function (version 5)

Version 7's psychology-profile trader lives in ``profiles.py``.

To write your own, subclass :class:`Agent`, override ``on_tick(m)`` and register it::

    class Dip(Agent):
        TYPE = 'dip'
        def on_tick(self, m):
            if m.ret(5) < -0.003 and self.shares == 0:
                self.market('BUY', 20)
    BOT_TYPES['dip'] = Dip
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from .lob import BUY, SELL

__all__ = ['TUNING', 'Agent', 'BaseBot', 'BOT_TYPES', 'register', 'ROUND_LOTS']

# ===================================================================
# TUNING: the handful of numbers that set how the market behaves (see TUNING in bots.js).
# Prices are fractions of the current price (0.001 = 0.1%).  Mutate in place to experiment:
#     agents.TUNING.value_vol = 0.0005
# ===================================================================
TUNING = SimpleNamespace(
    noise_sigma_floor=0.0008,      # noise limits sit 1-3 sigma away; sigma never below this ...
    noise_sigma_cap=0.003,         # ... or above this, so liquidity can't flee in a panic
    noise_ttl=(6, 30),             # noise limit orders rest 1.5-7.5 sim minutes
    noise_activity=(0.02, 0.05),
    maker_size=(6, 15),            # shares per level unit (levels are 1x, 2x, 3x)
    maker_half_spread=(0.0001, 0.0003),
    maker_max_half_spread=0.0008,
    maker_max_inventory=(300, 800),
    maker_pickoff=0.5,             # makers take orders this many half-spreads through fair value
    maker_lean=4,                  # at the limit, quotes shift this many half-spreads against inventory
    value_vol=0.0003,              # per-step volatility of the hidden fair value ("news")
    news_vol_of_vol=0.6,           # how much news intensity swings between quiet and busy stretches
    news_regime_minutes=45,        # how long a quiet or busy stretch typically lasts
    news_max_scale=3,              # news is never more than this many times normal
    value_pull=0.02,               # fair value drifts this much toward where the market trades
    maker_value_weight=(0.8, 1.0), # how closely each maker centres its quotes on fair value
    value_event_drift=0.0002,      # per-step fair-value drift at full event strength (US open)
    trend2_activity=(0.01, 0.03),
    heat_power=0.5,                # attention: noise-type traders act heat**this times as often
    random_limit_activity=(0.03, 0.08),
    random_limit_tail=1.2,         # power-law tail of their distance from the best price
    random_buyer_activity=(0.002, 0.006),
    reverter_size=(40, 200),
)

ROUND_LOTS = (5, 10, 10, 20, 25, 25, 50, 50, 100, 100, 200)


def _round2(x: float) -> float:
    return math.floor(x * 100 + 0.5) / 100


def _floor2(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def _ceil2(x: float) -> float:
    return math.ceil(x * 100 - 1e-9) / 100


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ===================================================================
# BASE AGENT
# ===================================================================
class Agent:
    """Base class: position and cash bookkeeping, order helpers, a delayed-action slot."""
    TYPE = 'agent'
    TRAITS: tuple = ()          # numeric attributes listed in results (``trait_effects``)

    def __init__(self, id, fleet, **opts):
        self.id = id
        self.fleet = fleet
        self.book = fleet.book
        self.rng = fleet.rng
        self.cohort = opts.get('cohort', 'base')
        self.bias_fn = opts.get('bias_fn')            # event cohorts lean one way: returns -1..1
        self.shares = 0
        self.avg_entry = 0.0
        self.realized = 0.0
        self.cash = 0.0
        self.volume = 0
        self.n_fills = 0
        self.orders: dict = {}                        # resting order ids (a dict keeps insertion order, so runs repeat)
        self.pending = None                           # delayed reaction (action, at)
        self.max_position = math.inf                  # risk limit for agents that only ever add
        self.shares0 = 0                              # holdings and price when the agent joined
        self.price0 = fleet.book.mid()

    # ---- randomness (one shared, seedable source)
    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def rand(self, a: float, b: float) -> float:
        return a + self.rng.random() * (b - a)

    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)

    def pick(self, seq):
        return seq[int(self.rng.random() * len(seq))]

    def gauss(self) -> float:
        return self.rng.gauss(0.0, 1.0)

    # ---- bookkeeping
    @property
    def type(self) -> str:
        return self.TYPE

    @property
    def tilt(self) -> float:
        """Directional tilt from an event (0 for ordinary traders)."""
        return self.bias_fn() if self.bias_fn else 0.0

    def pick_side(self) -> str:
        return BUY if self.rng.random() < 0.5 + self.tilt / 2 else SELL

    def allows(self, side: str) -> bool:
        """Event cohorts mostly skip trades that go against their direction."""
        b = self.tilt
        if not b:
            return True
        return (side == BUY) == (b > 0) or self.rng.random() > abs(b)

    def capped(self, side: str, qty: float) -> float:
        """Shares that may still be added on ``side`` without breaching ``max_position``."""
        room = self.max_position - self.shares if side == BUY else self.max_position + self.shares
        return max(0, min(qty, room))

    def wealth(self, mid: float) -> float:
        return self.cash + self.shares * mid

    def pnl(self, mid: float) -> float:
        """Profit since the agent joined, marking holdings to ``mid``."""
        return self.cash + self.shares * mid - self.shares0 * self.price0

    def pnl_pct(self, mid: float) -> float:
        """Return on the current holding, e.g. -0.01 = down 1%."""
        if self.shares == 0 or not self.avg_entry:
            return 0.0
        if self.shares > 0:
            return (mid - self.avg_entry) / self.avg_entry
        return (self.avg_entry - mid) / self.avg_entry

    def traits(self) -> dict:
        return {k: float(getattr(self, k)) for k in self.TRAITS}

    # ---- orders
    def market(self, side: str, qty: float):
        qty = int(qty)
        if qty <= 0:
            return None
        return self.book.submit(self.id, side, qty)

    def limit(self, side: str, price: float, qty: float, ttl: int | None = None):
        price = _round2(price)
        qty = int(qty)
        if not price > 0 or qty <= 0:
            return None
        res = self.book.submit(self.id, side, qty, price, self.fleet.tick + ttl if ttl else None)
        if res is not None and res.resting > 0:
            self.orders[res.id] = None
            if len(self.orders) > 24:
                self.prune_orders()
        return res

    def prune_orders(self) -> None:
        self.orders = {i: None for i in self.orders if self.book.has(i)}

    def cancel_all(self) -> None:
        for i in self.orders:
            self.book.cancel(i, self.id)
        self.orders.clear()

    def flatten(self) -> None:
        if self.shares > 0:
            self.market(SELL, self.shares)
        elif self.shares < 0:
            self.market(BUY, -self.shares)

    def on_fill(self, side: str, price: float, qty: int) -> None:
        self.volume += qty
        self.n_fills += 1
        if side == BUY:
            self.cash -= price * qty
            if self.shares < 0:
                c = min(qty, -self.shares)
                self.realized += (self.avg_entry - price) * c
                self.shares += c
                qty -= c
                if self.shares == 0:
                    self.avg_entry = 0.0
            if qty > 0:
                self.avg_entry = (self.shares * self.avg_entry + qty * price) / (self.shares + qty)
                self.shares += qty
        else:
            self.cash += price * qty
            if self.shares > 0:
                c = min(qty, self.shares)
                self.realized += (price - self.avg_entry) * c
                self.shares -= c
                qty -= c
                if self.shares == 0:
                    self.avg_entry = 0.0
            if qty > 0:
                s = -self.shares
                self.avg_entry = (s * self.avg_entry + qty * price) / (s + qty)
                self.shares -= qty

    # ---- time
    def later(self, delay: int, action) -> None:
        """People don't all react on the same tick: act ``delay`` ticks from now."""
        self.pending = (action, self.fleet.tick + delay)

    def tick(self, m) -> None:
        if self.pending and self.fleet.tick >= self.pending[1]:
            action = self.pending[0]
            self.pending = None
            action(m)
        self.on_tick(m)

    def on_tick(self, m) -> None:
        pass

    def on_leave(self) -> None:
        self.cancel_all()
        self.pending = None


BaseBot = Agent


def _human_size(a: Agent, median: float, maximum: int = 1000) -> int:
    """Order size the way people pick them: mostly small, now and then large, big ones often rounded."""
    q = int(_clamp(math.floor(median * math.exp(0.9 * a.gauss()) + 0.5), 1, maximum))
    if q < 20 or a.chance(0.4):
        return q
    step = 100 if q >= 200 else 50 if q >= 100 else 25 if q >= 50 else 10
    return max(step, int(math.floor(q / step + 0.5)) * step)


# ===================================================================
# 1. NOISE: random side, random market/limit; limits 1-3 sigma from price. Buy limits rest below the
# price and sell limits above, so noise traders also build the deeper book. They trade more when
# the market is moving (attention).
# ===================================================================
class NoiseBot(Agent):
    TYPE = 'noise'
    TRAITS = ('activity',)

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        a = opts.get('activity')
        self.activity = a if a is not None else self.rand(*TUNING.noise_activity)

    def on_tick(self, m):
        if not self.chance(self.activity * m.attention):
            return
        side = self.pick_side()
        qty = self.randint(1, 50)
        if self.chance(0.5):
            self.market(side, qty)
            return
        k = self.rand(1, 3) * m.sigma
        b = self.tilt
        aggressive = b != 0 and (side == BUY) == (b > 0) and self.chance(abs(b))
        price = m.mid - k if (side == BUY) != aggressive else m.mid + k
        self.limit(side, price, qty, self.randint(*TUNING.noise_ttl))

    def opening_order(self, m):
        side = BUY if self.chance(0.5) else SELL
        k = self.rand(1, 3) * m.sigma
        self.limit(side, m.mid - k if side == BUY else m.mid + k, self.randint(1, 50), self.randint(*TUNING.noise_ttl))


# ===================================================================
# 2. MARKET MAKER: quotes both sides on 3 levels around its estimate of fair value (the market price
# blended with a hidden "news" random walk). Spread widens when the market is jumpy; quotes lean
# against inventory so makers work back toward flat.
# ===================================================================
class MarketMakerBot(Agent):
    TYPE = 'marketMaker'
    TRAITS = ('size', 'base_half', 'max_inventory', 'value_weight')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.levels = 3
        self.size = self.randint(*TUNING.maker_size)
        self.base_half = self.rand(*TUNING.maker_half_spread)
        self.max_inventory = self.randint(*TUNING.maker_max_inventory)
        self.value_weight = self.rand(*TUNING.maker_value_weight)

    def on_tick(self, m):
        # Read the market as it stands (including our own quotes) before replacing them: reading it
        # after cancelling would re-centre on older orders further out and pull the price back.
        mid = self.book.mid()
        self.cancel_all()
        half = _clamp(self.base_half * mid + 0.5 * m.tick_sigma, 0.01, TUNING.maker_max_half_spread * mid)
        # A long maker shades both quotes down (short: up) so it drifts back toward flat. Too strong a
        # lean turns every imbalance into a slow, self-feeding trend; the hard limit is a backstop.
        lean = -TUNING.maker_lean * _clamp(self.shares / self.max_inventory, -1, 1) * half
        centre = mid + self.value_weight * (m.value - mid) + self.tilt * half * 4 + lean
        full = abs(self.shares) >= self.max_inventory
        # Stale orders: an offer below the maker's fair price (or a bid above it) is free money, so it
        # takes it. This keeps the market efficient. Picking off never takes a maker past half its limit.
        edge = TUNING.maker_pickoff * half
        self.pick_off(BUY, centre - edge, self.max_inventory / 2 - self.shares)
        self.pick_off(SELL, centre + edge, self.max_inventory / 2 + self.shares)
        best_bid, best_ask = self.book.best_bid(), self.book.best_ask()
        for i in range(self.levels):
            off = half * (1 + i)
            bid = _floor2(centre - off)
            ask = _ceil2(centre + off)
            if best_ask is not None:
                bid = min(bid, _round2(best_ask - 0.01))
            if best_bid is not None:
                ask = max(ask, _round2(best_bid + 0.01))
            q = self.size * (i + 1)
            if not (full and self.shares > 0):
                self.limit(BUY, bid, q)
            if not (full and self.shares < 0):
                self.limit(SELL, ask, q)

    def pick_off(self, side, price, room):
        """Take whatever rests at or better than ``price`` on the other side, up to 3 quote sizes."""
        qty = 0
        for p, q in self.book.levels(SELL if side == BUY else BUY, 5):
            if (p > price) if side == BUY else (p < price):
                break
            qty += q
        qty = min(qty, self.size * 3, room)
        if qty > 0:
            self.market(side, qty)


# ===================================================================
# 3. WHALE: shows up at random and moves 100-1000 shares in one go, as a market order or as a limit
# at the opposite best price (takes the touch, the rest rests there).
# ===================================================================
class WhaleBot(Agent):
    TYPE = 'whale'
    TRAITS = ('activity',)

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        a = opts.get('activity')
        self.activity = a if a is not None else self.rand(0.002, 0.005)

    def on_tick(self, m):
        b = self.tilt
        if not self.chance(self.activity + abs(b) * 0.04):
            return
        side = (BUY if b > 0 else SELL) if b else self.pick_side()
        qty = self.randint(100, 1000)
        touch = self.book.best_ask() if side == BUY else self.book.best_bid()
        if self.chance(0.5) or touch is None:
            self.market(side, qty)
        else:
            self.limit(side, touch, qty, 20)


# ===================================================================
# 4. THE BUYER: buys 100 every 2 hours from activation. Sells everything at -1% or +1.5% on the holding.
# 5. THE SELLER: the mirror image (shorts 100, covers at -1% or +1.5% on the short).
# ===================================================================
class BuyerBot(Agent):
    TYPE = 'buyer'
    SIDE = BUY

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.next_trade = fleet.tick + opts.get('delay', 0)
        self.interval = fleet.ticks_for(120)

    def on_tick(self, m):
        held = self.shares > 0 if self.SIDE == BUY else self.shares < 0
        if held:
            r = self.pnl_pct(m.mid)
            if r <= -0.01 or r >= 0.015:
                self.flatten()
        if self.fleet.tick >= self.next_trade:
            self.market(self.SIDE, 100)
            self.next_trade = self.fleet.tick + self.interval


class SellerBot(BuyerBot):
    TYPE = 'seller'
    SIDE = SELL


# ===================================================================
# 6. THE LIMIT PLAYER: a ladder around the price when it starts: buys at -1/-2/-3/-4% (10/20/30/50),
# sells at 0/+1/+2/+3% (10/20/30/50). Closes out at -1.5% or +1.5% on the position, then
# re-ladders around the new price after a short break.
# ===================================================================
LADDER_BUYS = ((0.99, 10), (0.98, 20), (0.97, 30), (0.96, 50))
LADDER_SELLS = ((1.00, 10), (1.01, 20), (1.02, 30), (1.03, 50))


class LimitLadderBot(Agent):
    TYPE = 'limitLadder'

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.deployed = False
        self.resume_at = fleet.tick + opts.get('delay', 0)

    def on_tick(self, m):
        if self.fleet.tick < self.resume_at:
            return
        if not self.deployed:
            anchor = m.mid
            for f, q in LADDER_BUYS:
                self.limit(BUY, _floor2(anchor * f), q)
            for f, q in LADDER_SELLS:
                self.limit(SELL, _ceil2(anchor * f), q)
            self.deployed = True
            return
        if self.shares != 0:
            r = self.pnl_pct(m.mid)
            if r <= -0.015 or r >= 0.015:
                self.cancel_all()
                self.flatten()
                self.deployed = False
                self.resume_at = self.fleet.tick + self.fleet.ticks_for(10)


# ===================================================================
# 7. STD BOT: price 2-3 sigma below its 30-step mean -> buy; as far above -> sell (and always at least
# 0.1-0.2% away). Takes profit when price is back at the mean.
# ===================================================================
class StdBot(Agent):
    TYPE = 'std'
    TRAITS = ('size', 'activity', 'z', 'min_dev')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.size = self.randint(10, 60)
        self.activity = self.rand(0.15, 0.4)
        self.z = self.rand(2, 3)
        self.min_dev = self.rand(0.001, 0.002)

    def on_tick(self, m):
        if m.length < 30 or not self.chance(self.activity):
            return
        z = m.dev_z(30)
        far = abs(z) >= self.z and abs(m.mid / m.sma(30) - 1) >= self.min_dev
        if far and z < 0 and self.shares <= 0 and self.allows(BUY):
            self.market(BUY, self.size - self.shares)
        elif far and z > 0 and self.shares >= 0 and self.allows(SELL):
            self.market(SELL, self.size + self.shares)
        elif self.shares != 0 and abs(z) < 0.3:
            self.flatten()


# ===================================================================
# 8. TREND BOT 1: last 10 minutes of returns mostly positive -> long, mostly negative -> short; closes
# when returns turn mostly flat.
# ===================================================================
class TrendBot1(Agent):
    TYPE = 'trend1'
    TRAITS = ('enter', 'flat', 'size', 'activity')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.window = fleet.ticks_for(10)
        self.enter = self.rand(0.2, 0.35)       # (ups - downs) / n needed to call it a trend
        self.flat = self.rand(0.04, 0.1)        # below this it's "mostly flat"
        self.size = self.randint(10, 80)
        self.activity = self.rand(0.15, 0.4)

    def on_tick(self, m):
        if m.length <= self.window or not self.chance(self.activity):
            return
        score = m.up_down_score(self.window)
        if self.shares == 0:
            if score >= self.enter and self.allows(BUY):
                self.market(BUY, self.size)
            elif score <= -self.enter and self.allows(SELL):
                self.market(SELL, self.size)
        elif (abs(score) <= self.flat or (self.shares > 0 and score <= -self.enter)
              or (self.shares < 0 and score >= self.enter)):
            self.flatten()


# ===================================================================
# 9. TREND BOT 2: more buyers than sellers resting in the book -> buy limit at the 10th ask level
# (sweeps up to it); more sellers -> sell limit at the 10th bid level. 10-200 shares.
# ===================================================================
class TrendBot2(Agent):
    TYPE = 'trend2'
    TRAITS = ('threshold', 'activity')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.threshold = self.rand(0.15, 0.35)  # book imbalance needed
        self.activity = self.rand(*TUNING.trend2_activity)
        self.max_position = 150

    def on_tick(self, m):
        if not self.chance(self.activity):
            return
        imb = m.imbalance
        qty = self.randint(10, 200)
        if imb > self.threshold and self.allows(BUY):
            lv = m.ask_levels
            if lv:
                self.limit(BUY, lv[-1][0], self.capped(BUY, qty), 8)
        elif imb < -self.threshold and self.allows(SELL):
            lv = m.bid_levels
            if lv:
                self.limit(SELL, lv[-1][0], self.capped(SELL, qty), 8)


# ===================================================================
# 10. TREND BOT 3 (the original): price moved 2-3 sigma over its 3-8 step lookback (and at least
# 0.08%) -> market order with the move. Short cooldown after trading.
# ===================================================================
class TrendBot3(Agent):
    TYPE = 'trend3'
    TRAITS = ('lookback', 'z')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.lookback = self.randint(3, 8)
        self.z = self.rand(2, 3)
        self.cooldown_until = 0
        self.max_position = 50

    def on_tick(self, m):
        if m.length <= self.lookback or self.fleet.tick < self.cooldown_until:
            return
        move = m.ret(self.lookback)
        if abs(m.move_z(self.lookback)) < self.z or abs(move) < 0.0008:
            return
        side = BUY if move > 0 else SELL
        if not self.allows(side):
            return
        self.market(side, self.capped(side, self.randint(4, 15)))
        self.cooldown_until = self.fleet.tick + self.lookback


# ===================================================================
# 11. MR BOT 1: 20/60/120-step moving averages. When they line up downward (20 < 60 < 120) it buys,
# expecting a bounce; when they line up upward it sells. Holds until the opposite signal.
# ===================================================================
class MRBot1(Agent):
    TYPE = 'mr1'
    TRAITS = ('size',)

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.size = self.randint(10, 60)
        self.state = None

    def on_tick(self, m):
        if m.length < 120:
            return
        s20, s60, s120 = m.sma(20), m.sma(60), m.sma(120)
        state = 'down' if s20 < s60 < s120 else 'up' if s20 > s60 > s120 else 'mixed'
        if state == self.state:
            return
        prev, self.state = self.state, state
        if prev is None:
            return                                   # no trade on the first reading
        delay = self.randint(0, 8)
        if state == 'down':
            self.later(delay, lambda m_: self.shares <= 0 and self.allows(BUY) and self.market(BUY, self.size - self.shares))
        elif state == 'up':
            self.later(delay, lambda m_: self.shares >= 0 and self.allows(SELL) and self.market(SELL, self.size + self.shares))


# ===================================================================
# 12. MR BOT 2: order-book imbalance. Sellers outweigh buyers -> short; buyers outweigh sellers -> buy.
# Closes (market) once the book is back in balance.
# ===================================================================
class MRBot2(Agent):
    TYPE = 'mr2'
    TRAITS = ('enter', 'exit', 'size', 'activity')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.enter = self.rand(0.25, 0.45)
        self.exit = self.rand(0.03, 0.08)
        self.size = self.randint(10, 100)
        self.activity = self.rand(0.15, 0.4)

    def on_tick(self, m):
        if not self.chance(self.activity):
            return
        imb = m.imbalance
        if self.shares == 0:
            if imb <= -self.enter and self.allows(SELL):
                self.market(SELL, self.size)
            elif imb >= self.enter and self.allows(BUY):
                self.market(BUY, self.size)
        elif abs(imb) <= self.exit:
            self.flatten()


# ===================================================================
# 13. MR BOT 3 (the original): price 2-3 sigma away from its 10-step mean (and at least 0.08%) ->
# market order back toward it. Short cooldown after trading.
# ===================================================================
class MRBot3(Agent):
    TYPE = 'mr3'
    TRAITS = ('z',)

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.z = self.rand(2, 3)
        self.cooldown_until = 0
        self.max_position = 150

    def on_tick(self, m):
        if m.length < 10 or self.fleet.tick < self.cooldown_until:
            return
        z = m.dev_z(10)
        if abs(z) < self.z or abs(m.mid / m.sma(10) - 1) < 0.0008:
            return
        if z > 0 and self.allows(SELL):
            self.market(SELL, self.capped(SELL, 10))
        elif z < 0 and self.allows(BUY):
            self.market(BUY, self.capped(BUY, 10))
        else:
            return
        self.cooldown_until = self.fleet.tick + self.randint(2, 6)


# ===================================================================
# 14. JUMP TRADER: a jump down -> buy, sell 10 steps later; a jump up -> sell, buy back 10 steps later.
# ===================================================================
class JumpTrader(Agent):
    TYPE = 'jump'
    TRAITS = ('threshold',)

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.threshold = self.rand(2.5, 3.5)    # jump = this many sigma of 1-step returns (and at least 0.05%)
        self.exit_at = None

    def on_tick(self, m):
        if self.exit_at is not None and self.fleet.tick >= self.exit_at:
            self.flatten()
            self.exit_at = None
        if self.exit_at is not None or self.shares != 0 or m.length < 30:
            return
        r = m.ret(1)
        if abs(r) < max(self.threshold * m.ret_std(60), 0.0005):
            return
        side = BUY if r < 0 else SELL
        if not self.allows(side):
            return
        self.market(side, self.randint(10, 150))
        self.exit_at = self.fleet.tick + 10


# ===================================================================
# 15. VOLATILITY TRADER: when recent volatility ranks high against the last hour it sells at random;
# when it ranks low, it buys at random. Ranking (not a fixed multiple) keeps "high" and "low"
# equally common, so the bot doesn't drift one way all day.
# ===================================================================
class VolatilityTrader(Agent):
    TYPE = 'volatility'
    TRAITS = ('activity', 'high', 'low')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.activity = self.rand(0.02, 0.05)
        self.high = self.rand(0.7, 0.85)        # percentile that counts as "high"
        self.low = self.rand(0.15, 0.3)         # percentile that counts as "low"
        self.max_position = 200

    def on_tick(self, m):
        if m.vol_rank is None or not self.chance(self.activity):
            return
        qty = self.randint(10, 80)
        if m.vol_rank >= self.high and self.allows(SELL):
            self.market(SELL, self.capped(SELL, qty))
        elif m.vol_rank <= self.low and self.allows(BUY):
            self.market(BUY, self.capped(BUY, qty))


# ===================================================================
# 16. RANDOM LIMIT GIVER: drops limit orders at random prices and forgets about them. Most rest a few
# cents behind the best price, some far back (power-law distance, as in real books), about one in
# eight improves on the best price inside the spread, and the odd one is priced through the market.
# ===================================================================
class RandomLimitBot(Agent):
    TYPE = 'randomLimit'
    TRAITS = ('activity', 'cancel_rate')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        a = opts.get('activity')
        self.activity = a if a is not None else self.rand(*TUNING.random_limit_activity)
        self.cancel_rate = self.rand(0.005, 0.02)   # chance per step of pulling one resting order
        self.ttl = (fleet.ticks_for(2), fleet.ticks_for(60))

    def on_tick(self, m):
        if self.orders and self.chance(self.cancel_rate):
            self.cancel_one()
        if self.chance(self.activity * m.attention):
            self.place_order(m)

    def opening_order(self, m):
        self.place_order(m)

    def place_order(self, m):
        side = self.pick_side()
        own = self.book.best_bid() if side == BUY else self.book.best_ask()
        other = self.book.best_ask() if side == BUY else self.book.best_bid()
        back = -1 if side == BUY else 1             # direction away from the market
        u = self.rng.random()
        if own is None or other is None:
            price = m.mid + back * self.rand(1, 3) * m.sigma
        elif u < 0.03:
            price = other - back * 0.01 * self.randint(0, 3)             # through the market
        elif u < 0.15 and abs(other - own) > 0.015:
            inside = int(round(abs(other - own) * 100)) - 1
            price = own - back * 0.01 * self.randint(1, max(1, inside))  # improves the best price
        else:
            d0 = max(0.01, 0.2 * m.sigma)
            dist = min(0.05 * m.mid, d0 * (max(self.rng.random(), 1e-12) ** (-1 / TUNING.random_limit_tail) - 1))
            price = own + back * dist
        self.limit(side, price, _human_size(self, 15, 500), self.randint(*self.ttl))

    def cancel_one(self):
        i = self.randint(0, len(self.orders) - 1)
        oid = list(self.orders)[i]
        self.book.cancel(oid, self.id)
        self.orders.pop(oid, None)


# ===================================================================
# 17. RANDOM BUYER: an everyday investor. Buys a round lot at random moments with a market order, more
# often when the stock is moving, and sells the way people really do (the disposition effect):
# quick to bank a small win, slow to accept a loss, until the loss hurts enough to panic out. Now
# and then one sells for no reason at all. Half of them start the day already holding, mostly at a
# loss: yesterday they sold their winners and kept their losers.
# ===================================================================
class RandomBuyer(Agent):
    TYPE = 'randomBuyer'
    TRAITS = ('activity', 'lot', 'max_lots', 'take_profit', 'panic_at', 'boredom')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        a = opts.get('activity')
        self.activity = a if a is not None else self.rand(*TUNING.random_buyer_activity)
        self.lot = self.pick(ROUND_LOTS)
        self.max_lots = self.randint(1, 4)
        self.take_profit = self.rand(0.003, 0.012)
        self.panic_at = self.rand(0.012, 0.03)
        self.boredom = 1 / fleet.ticks_for(self.rand(60, 240))    # no-reason sale: about once per 1-4 hours held
        if self.cohort == 'base' and self.chance(0.5):
            self.shares = self.lot * self.randint(1, self.max_lots)
            self.avg_entry = fleet.book.mid() * (1 + self.rand(-0.8 * self.take_profit, 0.8 * self.panic_at))
            self.shares0 = self.shares

    def on_tick(self, m):
        if self.shares > 0:
            r = self.pnl_pct(m.mid)
            if (r >= self.take_profit and self.chance(0.2)) or r <= -self.panic_at or self.chance(self.boredom):
                self.market(SELL, self.shares)
                return
        if (self.shares < self.lot * self.max_lots and self.chance(self.activity * m.attention)
                and self.allows(BUY)):
            self.market(BUY, self.lot)


# ===================================================================
# 18. AGGRESSIVE REVERTER: watches how far the price has run over its window (5-30 minutes). When the
# run is big (2-3 sigma for that window, and at least 0.15-0.4%), it reacts hard:
#   push   piles in with the move (FOMO); likelier on moderate moves
#   revert slams it back toward where it came from; likelier the more stretched the move is
#   hold   defends the new level: stacks bids just under it and offers just over it, refilling them,
#          so the price stalls there. If the level breaks anyway, it dumps everything and walks away.
# Each reaction ends at its target (half the move again, in its favour), its stop (half the move
# against it) or after 10-30 minutes.
# ===================================================================
REVERTER_WINDOWS = (5, 10, 15, 20, 30)   # minutes


class AggressiveReverter(Agent):
    TYPE = 'reverter'
    TRAITS = ('trigger', 'min_move', 'size', 'window')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.window = fleet.ticks_for(self.pick(REVERTER_WINDOWS))
        self.trigger = self.rand(2, 3)
        self.min_move = self.rand(0.0015, 0.004)
        self.temper = {'push': self.rand(0.2, 1), 'revert': self.rand(0.2, 1), 'hold': self.rand(0.2, 1)}
        self.size = self.randint(*TUNING.reverter_size)
        self.play = None
        self.ready_at = fleet.tick + self.randint(0, self.window)

    def on_tick(self, m):
        if self.play:
            self.manage(m)
            return
        if self.fleet.tick < self.ready_at or m.length <= self.window:
            return
        move = m.ret(self.window)
        z = m.move_z(self.window)
        if abs(z) >= self.trigger and abs(move) >= self.min_move:
            self.react(m, move, abs(z) / self.trigger)

    def react(self, m, move, stretch):
        w = {'push': self.temper['push'] / stretch, 'revert': self.temper['revert'] * stretch, 'hold': self.temper['hold']}
        r = self.rng.random() * (w['push'] + w['revert'] + w['hold'])
        r -= w['push']
        if r < 0:
            mode = 'push'
        else:
            r -= w['revert']
            mode = 'revert' if r < 0 else 'hold'
        direction = 1 if move > 0 else -1
        side = BUY if (direction > 0) == (mode != 'revert') else SELL
        if mode != 'hold' and not self.allows(side):
            return
        self.play = {'mode': mode, 'dir': direction, 'side': side, 'level': m.mid,
                     'dist': abs(m.mid - m.mid / (1 + move)),           # size of the run, $
                     'to_trade': 0 if mode == 'hold' else self.size, 'refill_at': 0,
                     'until': self.fleet.tick + self.fleet.ticks_for(self.randint(10, 30))}

    def manage(self, m):
        p = self.play
        t = self.fleet.tick
        x = p['dir'] * (m.mid - p['level']) / p['dist']        # progress past the level, in runs
        if p['mode'] == 'hold':
            if abs(x) > 0.5 or t >= p['until']:
                self.end_play()
            elif t >= p['refill_at']:
                self.defend(m)
            return
        if p['to_trade'] > 0:
            q = min(p['to_trade'], math.ceil(self.size / self.randint(2, 4)))
            self.market(p['side'], q)
            p['to_trade'] -= q
        gain = x if p['mode'] == 'push' else -x
        if t >= p['until'] or abs(gain) >= 0.5:
            self.end_play()

    def defend(self, m):
        p = self.play
        self.cancel_all()
        gap = max(0.01, m.tick_sigma)
        q = math.ceil(self.size / 4)
        for i in (1, 2):
            if self.shares < self.size:
                self.limit(BUY, _floor2(p['level'] - gap * i), q)
            if self.shares > -self.size:
                self.limit(SELL, _ceil2(p['level'] + gap * i), q)
        p['refill_at'] = self.fleet.tick + self.randint(2, 4)

    def end_play(self):
        self.cancel_all()
        self.flatten()
        self.play = None
        self.ready_at = self.fleet.tick + self.window       # don't react to the same run twice

    def on_leave(self):
        super().on_leave()
        self.play = None


# ===================================================================
# VERSION 6 PERSONALITIES FROM THE BRIEF
# ===================================================================
class ValueNoise(Agent):
    """Retail buyer with a private idea of the fair price.

    It buys at random moments when the market is below its idea (by at least ``buy_below``) and sells at
    random moments when the market is above it (by ``sell_above``).  The idea is drawn once around the
    price when the trader joins, so a crowd of them holds a *distribution* of private values.
    """
    TYPE = 'valueNoise'
    TRAITS = ('view', 'buy_below', 'sell_above', 'activity', 'lot')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        spread = opts.get('view_sd', 0.01)
        self.view = opts.get('view') or fleet.book.mid() * math.exp(spread * self.gauss())
        self.buy_below = self.rand(0.001, 0.008)
        self.sell_above = self.rand(0.0, 0.008)
        self.activity = opts.get('activity') or self.rand(0.01, 0.04)
        self.lot = self.pick(ROUND_LOTS)
        self.max_lots = self.randint(1, 3)

    def on_tick(self, m):
        if not self.chance(self.activity * m.attention):
            return
        if self.shares > 0 and m.mid >= self.view * (1 + self.sell_above):
            self.market(SELL, min(self.shares, self.lot))
        elif self.shares < self.lot * self.max_lots and m.mid <= self.view * (1 - self.buy_below):
            self.market(BUY, self.lot)


class Investor(Agent):
    """Genuine investor: follows a running mean, watches its own profit, applies stop-loss and take-profit.

    Buys when the price is ``entry_dev`` below its running mean (window ``window`` ticks), then sells the
    whole position when the return on it reaches ``take_profit`` or falls to ``-stop_loss``.
    """
    TYPE = 'investor'
    TRAITS = ('window', 'entry_dev', 'take_profit', 'stop_loss', 'activity', 'lot')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.window = opts.get('window') or self.pick((30, 60, 120))
        self.entry_dev = self.rand(0.0005, 0.003)
        self.take_profit = self.rand(0.003, 0.012)
        self.stop_loss = self.rand(0.004, 0.02)
        self.activity = opts.get('activity') or self.rand(0.03, 0.12)
        self.lot = self.pick(ROUND_LOTS)

    def on_tick(self, m):
        if self.shares > 0:
            r = self.pnl_pct(m.mid)
            if r >= self.take_profit or r <= -self.stop_loss:
                self.market(SELL, self.shares)
            return
        if m.length < self.window or not self.chance(self.activity * m.attention):
            return
        if m.mid <= m.sma(self.window) * (1 - self.entry_dev) and self.allows(BUY):
            self.market(BUY, self.lot)


class MeanReverter(Agent):
    """Mean reversion from a moving-average crossing or a standard-deviation band.

    ``mode='cross'``: the ``fast`` (20) average crossing above the ``slow`` (40) one means the price has
    run up, so it sells; crossing below, it buys.  ``mode='band'``: it buys when the price is more than
    ``k`` (1.0) standard deviations below the ``window`` (60) moving average and sells when it is more
    than ``k`` above, closing when the price is back at the average.
    """
    TYPE = 'meanReverter'
    TRAITS = ('k', 'size', 'activity')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.mode = opts.get('mode') or self.pick(('cross', 'band'))
        self.fast, self.slow, self.window = opts.get('fast', 20), opts.get('slow', 40), opts.get('window', 60)
        self.k = opts.get('k') or self.rand(0.8, 1.5)
        self.size = self.randint(10, 60)
        self.activity = self.rand(0.1, 0.4)
        self.sign = None
        self.max_position = 4 * self.size

    def on_tick(self, m):
        if self.mode == 'cross':
            if m.length < self.slow:
                return
            sign = 1 if m.sma(self.fast) > m.sma(self.slow) else -1
            prev, self.sign = self.sign, sign
            if prev is None or sign == prev:
                return
            if sign > 0 and self.allows(SELL):
                self.market(SELL, self.size + max(self.shares, 0))
            elif sign < 0 and self.allows(BUY):
                self.market(BUY, self.size - min(self.shares, 0))
            return
        if m.length < self.window or not self.chance(self.activity):
            return
        mean, sd = m.sma(self.window), m.std(self.window)
        if sd <= 0:
            return
        z = (m.mid - mean) / sd
        if z <= -self.k and self.shares <= 0 and self.allows(BUY):
            self.market(BUY, self.capped(BUY, self.size))
        elif z >= self.k and self.shares >= 0 and self.allows(SELL):
            self.market(SELL, self.capped(SELL, self.size))
        elif self.shares != 0 and abs(z) < 0.15:
            self.flatten()


class InformedWhale(Agent):
    """Whale that trades on a perceived value: buys big below it, sells big above it.

    ``perceived`` is fixed before the run (default: the starting price times ``exp(N(0, perceived_sd))``).
    ``info`` in [0, 1] blends in the market's hidden fair value (the "news" walk): 0 trades on its fixed
    belief only, 1 knows the true value at every tick.
    """
    TYPE = 'informedWhale'
    TRAITS = ('perceived', 'info', 'edge', 'activity')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.perceived = opts.get('perceived') or fleet.book.mid() * math.exp(opts.get('perceived_sd', 0.01) * self.gauss())
        self.info = opts.get('info', 0.0)
        self.edge = opts.get('edge') or self.rand(0.001, 0.004)
        self.activity = opts.get('activity') or self.rand(0.01, 0.03)
        self.size = opts.get('size') or self.randint(200, 1000)
        self.max_position = opts.get('max_position', 6 * self.size)

    def belief(self, m) -> float:
        return (1 - self.info) * self.perceived + self.info * m.value

    def on_tick(self, m):
        gap = self.belief(m) / m.mid - 1                        # > 0: the price is too low
        if abs(gap) < self.edge or not self.chance(self.activity):
            return
        side = BUY if gap > 0 else SELL
        qty = self.capped(side, int(self.size * min(1.0, abs(gap) / (4 * self.edge)) + 1))
        touch = self.book.best_ask() if side == BUY else self.book.best_bid()
        if self.chance(0.5) or touch is None:
            self.market(side, qty)
        else:
            self.limit(side, touch, qty, 20)


class TrendFollower(Agent):
    """Buys after an ``x`` rise, sells after ``y`` more, stops out ``z`` below the entry.

    Watches the return over ``window`` ticks.  When it reaches ``+x`` the trader buys (entry level
    ``P0``); it sells everything when the price reaches ``P0 (1 + y)`` (take profit) or falls to
    ``P0 (1 - z)`` (the move failed).  Falls of ``x`` are traded the same way on the short side if
    ``short`` is true.
    """
    TYPE = 'trendFollower'
    TRAITS = ('x', 'y', 'z', 'window', 'size')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.x = opts.get('x') or self.rand(0.002, 0.006)
        self.y = opts.get('y') or self.rand(0.003, 0.010)
        self.z = opts.get('z') or self.rand(0.002, 0.006)
        self.window = opts.get('window') or self.randint(20, 60)
        self.size = opts.get('size') or self.randint(20, 100)
        self.short = opts.get('short', True)
        self.entry = None
        self.cooldown_until = 0

    def on_tick(self, m):
        if self.shares != 0 and self.entry is not None:
            direction = 1 if self.shares > 0 else -1
            move = direction * (m.mid / self.entry - 1)
            if move >= self.y or move <= -self.z:
                self.flatten()
                self.entry = None
                self.cooldown_until = self.fleet.tick + self.window
            return
        if self.shares != 0 or m.length <= self.window or self.fleet.tick < self.cooldown_until:
            return
        r = m.ret(self.window)
        if r >= self.x and self.allows(BUY):
            self.market(BUY, self.size)
            self.entry = m.mid
        elif self.short and r <= -self.x and self.allows(SELL):
            self.market(SELL, self.size)
            self.entry = m.mid


# ===================================================================
# VERSION 5: AGENTS DESCRIBED BY WEALTH, BELIEFS AND RISK
# ===================================================================
class UtilityTrader(Agent):
    """Mean-variance trader with a desired price that it learns.

    * ``desired`` is its belief about fair value; each tick it moves a fraction ``learning_rate`` of
      the way toward the market price (adaptation: 0 never learns, 1 simply copies the price);
    * expected return over a ``horizon`` is ``mu = desired / price - 1``, risk ``s`` the recent return
      volatility scaled to that horizon; it wants to hold the fraction ``(mu / s) / risk_aversion`` of its
      wealth in the stock (a Kelly-style rule), at most ``leverage`` times its wealth;
    * it only trades when the gap to that target is more than ``band`` of its wealth;
    * a stop-loss and take-profit on the current holding override everything.
    """
    TYPE = 'utility'
    TRAITS = ('capital', 'risk_aversion', 'learning_rate', 'stop_loss', 'take_profit', 'activity', 'desired0')

    def __init__(self, id, fleet, **opts):
        super().__init__(id, fleet, **opts)
        self.capital = opts.get('capital') or 20_000.0
        self.risk_aversion = opts.get('risk_aversion') or self.rand(1.0, 8.0)
        self.desired = opts.get('desired_price') or fleet.book.mid() * math.exp(0.01 * self.gauss())
        self.desired0 = self.desired
        self.learning_rate = opts.get('learning_rate') if opts.get('learning_rate') is not None else self.rand(0.0005, 0.01)
        self.stop_loss = opts.get('stop_loss') or self.rand(0.005, 0.02)
        self.take_profit = opts.get('take_profit') or self.rand(0.005, 0.03)
        self.activity = opts.get('activity') or self.rand(0.05, 0.25)
        self.horizon = opts.get('horizon', 60)
        self.leverage = opts.get('leverage', 1.0)
        self.band = opts.get('band', 0.05)
        self.cooldown_until = 0

    def on_tick(self, m):
        self.desired += self.learning_rate * (m.mid - self.desired)
        if self.shares != 0:
            r = self.pnl_pct(m.mid)
            if r <= -self.stop_loss or r >= self.take_profit:
                self.flatten()
                self.cooldown_until = self.fleet.tick + 20
                return
        if m.length < 30 or self.fleet.tick < self.cooldown_until or not self.chance(self.activity * m.attention):
            return
        wealth = max(self.capital + self.wealth(m.mid), 1.0)
        s = max(m.ret_std(60), 1e-5) * math.sqrt(self.horizon)
        mu = self.desired / m.mid - 1
        frac = _clamp((mu / s) / self.risk_aversion, -1.0, 1.0) * self.leverage
        target = int(frac * wealth / m.mid)
        delta = target - self.shares
        if abs(delta) * m.mid < self.band * wealth:
            return
        side = BUY if delta > 0 else SELL
        if self.allows(side):
            self.market(side, min(abs(delta), 300))


class PolicyAgent(Agent):
    """An empty slot for experiments: ``policy(agent, m)`` is called every tick.

    Inside the policy use ``agent.market(side, qty)``, ``agent.limit(side, price, qty)``, ``agent.flatten()``,
    ``agent.cancel_all()``, ``agent.shares``, ``agent.cash``, ``agent.wealth(m.mid)`` and ``agent.params``
    (a dict for the agent's own traits: stop-loss, desired price, anything).
    """
    TYPE = 'policy'

    def __init__(self, id, fleet, policy=None, params=None, **opts):
        super().__init__(id, fleet, **opts)
        self.policy = policy
        self.params = dict(params or {})

    def traits(self) -> dict:
        return {k: float(v) for k, v in self.params.items() if isinstance(v, (int, float))}

    def on_tick(self, m):
        if self.policy is not None:
            self.policy(self, m)


BOT_TYPES: dict[str, type] = {
    'noise': NoiseBot, 'marketMaker': MarketMakerBot, 'whale': WhaleBot, 'buyer': BuyerBot,
    'seller': SellerBot, 'limitLadder': LimitLadderBot, 'std': StdBot, 'trend1': TrendBot1,
    'trend2': TrendBot2, 'trend3': TrendBot3, 'mr1': MRBot1, 'mr2': MRBot2, 'mr3': MRBot3,
    'jump': JumpTrader, 'volatility': VolatilityTrader, 'randomLimit': RandomLimitBot,
    'randomBuyer': RandomBuyer, 'reverter': AggressiveReverter,
    # brief
    'valueNoise': ValueNoise, 'investor': Investor, 'meanReverter': MeanReverter,
    'informedWhale': InformedWhale, 'trendFollower': TrendFollower,
    'utility': UtilityTrader, 'policy': PolicyAgent,
}


def register(name: str, cls: type) -> None:
    """Add an agent class under a type key (so ``population={'mine': 20}`` works)."""
    BOT_TYPES[name] = cls
