"""The exchange: market state, a fleet of agents with a trading day, and the simulation loop.

    from stochastics.finance import exchange

    res = exchange.simulate(population='game', seed=1)         # the website's 1,000 agents, one day
    res.summary()                                              # prices, volume, spread, volatility
    res.by_type()                                              # who made money, who traded
    res.facts()                                                # tails and volatility clustering
    res.plot()

    res = exchange.simulate(population={'noise': 400, 'marketMaker': 20, 'trendFollower': 60}, schedule=False)

A port of ``BotFleet.js`` plus the loop that drives it.  Each tick (15 simulated seconds, 09:30-18:00 is
2,040 ticks) every active agent acts once in random order; the price history is the book's mid price at the
end of each tick.  The fleet also keeps a hidden fair value ("news" random walk with calm and busy
stretches) that market makers lean on, an attention measure ("heat") that makes noise traders busier when
the price is moving, and the day's schedule: an opening crowd, hourly arrivals and departures, a
directional crowd at the 14:30 US open and the opening crowd's return for the close.

Random numbers come from Python's ``random.Random(seed)``: the same seed gives the same day.  It is not
bit-identical to the JavaScript generator.  The reinforcement-learning trader (``ai/RLTrader.js``) was not
part of the files and is not ported; ``PolicyAgent`` and ``profiles.ProfiledTrader`` are the places to
plug learned policies in.
"""
from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .. import _viz as V
from . import agents as A
from .agents import BOT_TYPES, TUNING, Agent
from .lob import BUY, SELL, OrderBook
from .market import _acf, stylized_facts
from . import profiles as _profiles  # noqa: F401  (registers the 'profiled' agent type)

__all__ = ['MarketState', 'Fleet', 'ExchangeResult', 'simulate', 'POPULATIONS', 'BASE_MIX']

# ---- population presets -----------------------------------------------------------------
# The website's base population (1,000 agents): 90% noise traders of four kinds, 10% spread over the
# other types.  Hourly joiners are drawn with the same weights, so the mix stays 90/10.
BASE_MIX = {
    'noise': 700, 'randomLimit': 100, 'randomBuyer': 60, 'reverter': 40,
    'marketMaker': 30, 'whale': 3, 'buyer': 3, 'seller': 3, 'limitLadder': 5, 'std': 6,
    'trend1': 6, 'trend2': 6, 'trend3': 6, 'mr1': 6, 'mr2': 6, 'mr3': 6, 'jump': 6, 'volatility': 8,
}
# The version-6 brief: noise traders split 20% value-anchored / 50% random / 30% genuine investors,
# market makers, mean reverters, whales, informed whales, trend followers and volatility traders.
BRIEF_MIX = {
    'valueNoise': 160, 'noise': 450, 'investor': 240,
    'marketMaker': 30, 'meanReverter': 60, 'whale': 4, 'informedWhale': 4, 'trendFollower': 40, 'volatility': 12,
}
POPULATIONS = {
    'game': BASE_MIX,
    'brief': BRIEF_MIX,
    'minimal': {'noise': 150, 'randomLimit': 30, 'marketMaker': 10},
    # version 5: the same background with 400 wealth-aware utility traders
    'agents': {'noise': 350, 'randomLimit': 50, 'marketMaker': 30, 'whale': 3, 'utility': 400},
    # version 7: a noise/maker background plus 400 psychology-profile traders with random profiles
    'profiles': {'noise': 350, 'randomLimit': 50, 'marketMaker': 30, 'whale': 3, 'profiled': 400},
}

MIN_MARKET_MAKERS = 12          # hourly churn never removes makers below this
OPENING_CROWD = {'whale': 3, 'noise': 97}
US_OPEN_SIZE = 200
US_OPEN_FADE_MINUTES = 60       # the push decays with this time constant
US_OPEN_SMALL = (0.18, 0.28)    # lean of a small event (0 = none, 1 = one-sided)
US_OPEN_BIG = (0.34, 0.46)
Z_WINDOW = 480                  # z-scores compare against the last 2 hours of steps
Z_MAX_K = 30                    # longest move measured directly (longer ones are scaled)
Z_FLOOR = 0.00005               # 0.5 bp: z-scores never divide by less than this
HEAT_FAST_MINUTES = 3           # "recent" volatility for the heat reading ...
HEAT_SLOW_MINUTES = 60          # ... against this longer baseline
HEAT_RANGE = (0.5, 3.0)
TREND_TYPES = ('trend1', 'trend2', 'trend3')
MR_TYPES = ('mr1', 'mr2', 'mr3')


# ===================================================================
# MARKET STATE: one snapshot per tick, shared by every agent. Indicators are computed on first
# use and cached, so a thousand agents asking for the 30-step mean cost one calculation.
# ===================================================================
class MarketState:
    def __init__(self, book: OrderBook, hist: np.ndarray, n: int, tick: int):
        self.book = book
        self.h = hist[:n]                        # price history (mid at the end of every previous tick)
        self.length = n
        self.tick = tick
        self.mid = book.mid()
        self.vol_rank = None                     # 0..1, set by the fleet once there is enough history
        self.value = self.mid                    # hidden fair value, set by the fleet
        self.heat = 1.0                          # attention: recent volatility vs the last hour
        self._c: dict = {}

    def _memo(self, key, fn):
        v = self._c.get(key)
        if v is None:
            v = self._c[key] = fn()
        return v

    # ---- price statistics
    def sma(self, n: int) -> float:
        """Simple moving average of the last ``n`` recorded prices."""
        def f():
            k = min(n, self.length)
            return float(self.h[-k:].mean()) if k else self.mid
        return self._memo(('sma', n), f)

    def std(self, n: int) -> float:
        """Standard deviation of the last ``n`` prices ($)."""
        def f():
            k = min(n, self.length)
            return float(self.h[-k:].std(ddof=1)) if k >= 2 else 0.0
        return self._memo(('std', n), f)

    def ret(self, k: int) -> float:
        """Return over the last ``k`` steps (0.01 = +1%)."""
        if self.length <= k:
            return 0.0
        return float(self.h[-1] / self.h[-1 - k] - 1.0)

    def ret_std(self, n: int) -> float:
        """Standard deviation of the last ``n`` one-step returns."""
        def f():
            k = min(n, self.length - 1)
            if k < 2:
                return 0.0
            r = self.h[-k:] / self.h[-k - 1:-1] - 1.0
            return math.sqrt(max(0.0, float((r * r).mean() - r.mean() ** 2)))
        return self._memo(('rstd', n), f)

    # ---- z-scores: how unusual a reading is against the same reading over the last Z_WINDOW steps
    # (+-2 is unusual, +-3 is rare). Thresholds in these units adapt to calm and busy days alike.
    def k_std(self, k: int) -> float:
        """Typical size of a ``k``-step return lately (root mean square); scaled like a random walk beyond Z_MAX_K."""
        def f():
            if k > Z_MAX_K:
                return self.k_std(Z_MAX_K) * math.sqrt(k / Z_MAX_K)
            n = min(Z_WINDOW, self.length - 1 - k)
            floor = Z_FLOOR * math.sqrt(k)
            if n >= 20:
                a, b = self.h[self.length - n:], self.h[self.length - n - k:self.length - k]
                r = a / b - 1.0
                return max(floor, math.sqrt(float((r * r).sum()) / n))
            return max(floor, TUNING.value_vol * math.sqrt(k))
        return self._memo(('ks', k), f)

    def move_z(self, k: int) -> float:
        """The move over the last ``k`` steps as a z-score."""
        return self._memo(('mz', k), lambda: self.ret(k) / self.k_std(k) if self.length > k else 0.0)

    def dev_z(self, n: int) -> float:
        """The gap between the price and its ``n``-step mean as a z-score."""
        def f():
            L = self.length
            if L < n + 20:
                return 0.0
            start = max(n - 1, L - Z_WINDOW)
            c = np.concatenate(([0.0], np.cumsum(self.h)))
            i = np.arange(start, L)
            sma = (c[i + 1] - c[i + 1 - n]) / n
            r = self.h[start:] / sma - 1.0
            rms = max(Z_FLOOR, math.sqrt(float((r * r).sum()) / (L - start)))
            return (self.mid / self.sma(n) - 1.0) / rms
        return self._memo(('dz', n), f)

    def up_down_score(self, n: int) -> float:
        """(ups - downs) / n over the last ``n`` one-step changes: +1 all up, -1 all down."""
        def f():
            k = min(n, self.length - 1)
            if k < 1:
                return 0.0
            d = np.diff(self.h[-k - 1:])
            return float(np.count_nonzero(d > 1e-9) - np.count_nonzero(d < -1e-9)) / k
        return self._memo(('ud', n), f)

    # ---- scales
    @property
    def sigma(self) -> float:
        """Price scale for noise-trader limits: std of the last 30 prices, kept within a band of the price."""
        def f():
            s = self.std(30) if self.length >= 5 else 0.0
            return min(TUNING.noise_sigma_cap * self.mid, max(TUNING.noise_sigma_floor * self.mid, s))
        return self._memo('sigma', f)

    @property
    def tick_sigma(self) -> float:
        """Typical one-step price change ($): at least one cent, at most 0.3% of the price."""
        return self._memo('tick_sigma', lambda: min(0.003 * self.mid, max(0.01, self.ret_std(30) * self.mid)))

    @property
    def attention(self) -> float:
        """Participation multiplier for attention-driven traders."""
        return self._memo('attention', lambda: self.heat ** TUNING.heat_power)

    # ---- the book
    @property
    def bid_levels(self):
        return self._memo('bid_lv', lambda: self.book.levels(BUY, 10))

    @property
    def ask_levels(self):
        return self._memo('ask_lv', lambda: self.book.levels(SELL, 10))

    @property
    def imbalance(self) -> float:
        """Resting buy vs sell quantity over the top 10 levels: +1 all buyers, -1 all sellers."""
        def f():
            b = sum(q for _, q in self.bid_levels)
            a = sum(q for _, q in self.ask_levels)
            return (b - a) / (b + a) if b + a > 0 else 0.0
        return self._memo('imb', f)


def _weighted_pick(rng, mix: dict) -> str:
    total = sum(mix.values())
    r = rng.random() * total
    for k, w in mix.items():
        r -= w
        if r < 0:
            return k
    return list(mix)[-1]


# ===================================================================
# THE FLEET
# ===================================================================
class Fleet:
    """The trading population, its hidden fair value and the day's schedule."""

    def __init__(self, book: OrderBook, rng: random.Random, population: dict, *, sim_secs_per_tick: float = 15,
                 open_secs: int = 34200, close_secs: int = 64800, events=('open', 'churn', 'us_open', 'close'),
                 on_news=None):
        self.book, self.rng = book, rng
        self.mix = dict(population)
        self.ticks_per_minute = 60.0 / sim_secs_per_tick
        self.open_secs, self.close_secs = open_secs, close_secs
        self.event_names = set(events)
        self.on_news = on_news
        self.tick = 0
        self.sim_seconds = open_secs
        self.active: dict[str, Agent] = {}
        self.agents: dict[str, Agent] = {}       # everyone who ever traded (fills are routed to all)
        self.roster: list[Agent] = []
        self.roster_dirty = True
        self.seq = 0
        self.events: deque = deque()
        self.opening_crowd: list[Agent] = []
        self.last_us_open = None
        self.vol_series: deque = deque(maxlen=self.ticks_for(60))
        self.value = None                        # hidden fair value: a random walk ("news") makers lean on
        self.event_bias = None                   # the US open is news too: it pushes fair value while it lasts
        self.fast_vol = self.slow_vol = 0.8 * TUNING.value_vol
        self.news_log = 0.0
        self.log: list[tuple[int, str]] = []
        self.flow_buy = self.flow_sell = 0
        self.last_flow = 0.0
        self.flow_ema = 0.0
        self.volume_ema = 0.0
        self.trader_history: deque = deque(maxlen=self.ticks_for(60))
        self.vwap_num = self.vwap_den = 0.0
        self.day_open = None
        self.day_high, self.day_low = -math.inf, math.inf
        self.heat = 1.0
        book.on_trade(self._on_trade)

    # ---- helpers
    @property
    def size(self) -> int:
        return len(self.active)

    def ticks_for(self, minutes: float) -> int:
        return max(1, int(math.floor(minutes * self.ticks_per_minute + 0.5)))

    def count_by_type(self) -> dict:
        out: dict = {}
        for a in self.active.values():
            out[a.type] = out.get(a.type, 0) + 1
        return out

    def news(self, text: str) -> None:
        self.log.append((self.tick, text))
        if self.on_news:
            self.on_news(text)

    def _on_trade(self, t) -> None:
        if t.aggressor == BUY:
            self.flow_buy += t.qty
        else:
            self.flow_sell += t.qty
        self.vwap_num += t.price * t.qty
        self.vwap_den += t.qty
        buyer = self.agents.get(t.buyer)
        if buyer is not None:
            buyer.on_fill(BUY, t.price, t.qty)
        seller = self.agents.get(t.seller)
        if seller is not None:
            seller.on_fill(SELL, t.price, t.qty)

    # ---- population
    def spawn(self, kind, **opts) -> Agent:
        cls = BOT_TYPES[kind] if isinstance(kind, str) else kind
        key = kind if isinstance(kind, str) else getattr(cls, 'TYPE', cls.__name__)
        self.seq += 1
        opts.setdefault('cohort', 'base')
        agent = cls(f'bot:{key}_{self.seq}', self, **opts)
        self.active[agent.id] = agent
        self.agents[agent.id] = agent
        self.roster_dirty = True
        return agent

    def retire(self, agent: Agent) -> None:
        agent.on_leave()
        self.active.pop(agent.id, None)
        self.roster_dirty = True

    def readmit(self, agent: Agent) -> None:
        self.active[agent.id] = agent
        self.roster_dirty = True

    # ---- the day
    def start(self, hist: np.ndarray) -> None:
        # Base population. Buyers, sellers and ladders start at staggered times in the first two hours
        # so their 2-hour cycles don't all line up.
        stagger = self.ticks_for(120)
        for kind, n in self.mix.items():
            for _ in range(int(n)):
                self.spawn(kind, cohort='base', delay=self.rng.randint(0, stagger))

        def at(h, m=0):
            return h * 3600 + m * 60
        ev = []
        names = self.event_names
        if 'open' in names:
            ev += [(self.open_secs, lambda: self.opening_crowd_arrives(True)),
                   (self.open_secs + 3600, self.opening_crowd_leaves)]
        if 'us_open' in names:
            ev.append((at(14, 30), self.us_open))
        if 'close' in names and 'open' in names:
            ev.append((self.close_secs - 3600, lambda: self.opening_crowd_arrives(False)))
        if 'churn' in names:
            h = math.ceil(self.open_secs / 3600)
            while h * 3600 < self.close_secs:
                if h * 3600 > self.open_secs:
                    ev.append((at(h), self.hourly_churn))
                h += 1
        ev.sort(key=lambda e: e[0])
        self.events = deque(ev)
        # Opening book: market makers quote and some noise traders rest orders before the first trade,
        # so the day doesn't open on an empty book.
        m = MarketState(self.book, hist, 0, 0)
        self.value = m.mid
        self.day_open = m.mid
        for a in list(self.active.values()):
            if a.type == 'marketMaker':
                a.on_tick(m)
        for a in list(self.active.values()):
            if hasattr(a, 'opening_order') and self.rng.random() < 0.5:
                a.opening_order(m)

    def news_intensity(self) -> float:
        """News doesn't arrive at an even pace: quiet and busy stretches of about ``news_regime_minutes``.

        Returns a multiplier for the size of this step's news (1 on average, at most ``news_max_scale``);
        this is what gives the price fat tails and volatility that clusters.
        """
        v = TUNING.news_vol_of_vol
        tau = self.ticks_for(TUNING.news_regime_minutes)
        self.news_log += -self.news_log / tau + v * math.sqrt(2 / tau) * self.rng.gauss(0, 1)
        return min(TUNING.news_max_scale, math.exp(self.news_log - v * v / 2))

    def snapshot(self, hist, n, tick) -> MarketState:
        m = MarketState(self.book, hist, n, tick)
        vol = self.flow_buy + self.flow_sell
        self.last_flow = (self.flow_buy - self.flow_sell) / vol if vol > 0 else 0.0
        self.flow_ema = 0.7 * self.flow_ema + 0.3 * self.last_flow
        self.volume_ema = 0.97 * self.volume_ema + 0.03 * vol if self.volume_ema else vol
        self.trader_history.append(len(self.active))
        self.flow_buy = self.flow_sell = 0
        self.day_high, self.day_low = max(self.day_high, m.mid), min(self.day_low, m.mid)
        m.flow, m.flow_ema, m.last_volume, m.volume_ema = self.last_flow, self.flow_ema, vol, self.volume_ema
        m.traders_change = len(self.active) - self.trader_history[0]
        m.day_open, m.day_high, m.day_low = self.day_open if self.day_open is not None else m.mid, self.day_high, self.day_low
        m.vwap = self.vwap_num / self.vwap_den if self.vwap_den > 0 else m.mid
        m.traders = len(self.active)
        return m

    def on_tick(self, tick: int, sim_seconds: float, hist: np.ndarray, n: int) -> None:
        self.tick, self.sim_seconds = tick, sim_seconds
        while self.events and self.events[0][0] <= sim_seconds:
            self.events.popleft()[1]()
        m = self.snapshot(hist, n, tick)
        # Fair value takes a random step (news) and drifts a little toward where the market actually
        # trades, so big trades leave a lasting mark.
        if self.value is None:
            self.value = m.mid
        drift = self.event_bias() * TUNING.value_event_drift if self.event_bias else 0.0
        self.value *= math.exp(TUNING.value_vol * self.news_intensity() * self.rng.gauss(0, 1) + drift)
        self.value += TUNING.value_pull * (m.mid - self.value)
        m.value = self.value
        # Attention ("heat"): how busy the last few minutes were against the last hour.
        if n > 1:
            r = abs(math.log(hist[n - 1] / hist[n - 2]))
            self.fast_vol += (r - self.fast_vol) / self.ticks_for(HEAT_FAST_MINUTES)
            self.slow_vol += (r - self.slow_vol) / self.ticks_for(HEAT_SLOW_MINUTES)
        self.heat = m.heat = min(HEAT_RANGE[1], max(HEAT_RANGE[0], self.fast_vol / self.slow_vol))
        # Where does current volatility rank against the last ~hour of readings?
        if n > 21:
            v = m.ret_std(20)
            self.vol_series.append(v)
            if len(self.vol_series) >= 40:
                m.vol_rank = sum(1 for x in self.vol_series if x < v) / len(self.vol_series)
        if self.roster_dirty:
            self.roster = list(self.active.values())
            self.roster_dirty = False
        self.rng.shuffle(self.roster)           # nobody gets to act first every tick
        for a in self.roster:
            a.tick(m)

    # ---- scheduled events
    def opening_crowd_arrives(self, first: bool) -> None:
        if first:
            for kind, n in OPENING_CROWD.items():
                for _ in range(n):
                    act = self.rng.uniform(0.12, 0.25) if kind == 'noise' else self.rng.uniform(0.008, 0.015)
                    self.opening_crowd.append(self.spawn(kind, cohort='open', activity=act))
            self.news('Opening hour: 100 extra traders are in the market until 10:30.')
        else:
            for a in self.opening_crowd:
                self.readmit(a)
            self.news('Final hour: the opening crowd is back for the close.')

    def opening_crowd_leaves(self) -> None:
        for a in self.opening_crowd:
            if a.id in self.active:
                self.retire(a)
        self.news('The opening rush is over: 100 traders left the market.')

    def hourly_churn(self) -> None:
        n = self.rng.randint(10, 100)
        if self.rng.random() < 0.5:
            for _ in range(n):
                self.spawn(_weighted_pick(self.rng, self.mix), cohort='churn')
            self.news(f'{n} new traders joined the market.')
            return
        makers = sum(1 for b in self.active.values() if b.type == 'marketMaker')
        candidates = [b for b in self.active.values() if b.cohort in ('base', 'churn')]
        self.rng.shuffle(candidates)
        removed = 0
        for a in candidates:
            if removed >= n:
                break
            if a.type == 'marketMaker':
                if makers <= MIN_MARKET_MAKERS:
                    continue
                makers -= 1
            self.retire(a)
            removed += 1
        self.news(f'{removed} traders left the market.')

    def us_open(self) -> None:
        rng = self.rng
        direction = 1 if rng.random() < 0.5 else -1
        big = rng.random() < 0.5
        lo, hi = US_OPEN_BIG if big else US_OPEN_SMALL
        strength = rng.uniform(lo, hi)
        start = self.tick
        fade = self.ticks_for(US_OPEN_FADE_MINUTES)

        def bias():
            return direction * strength * math.exp(-(self.tick - start) / fade)
        self.last_us_open = {'direction': direction, 'strength': strength, 'big': big, 'start_tick': start}
        self.event_bias = bias
        opts = {'cohort': 'us', 'bias_fn': bias}
        self.spawn('whale', **opts)
        self.spawn('marketMaker', **opts)
        self.spawn('marketMaker', **opts)
        for _ in range(3, US_OPEN_SIZE):
            category = rng.choice(('noise', 'trend', 'mr'))
            kind = 'noise' if category == 'noise' else rng.choice(TREND_TYPES if category == 'trend' else MR_TYPES)
            # The US open is a volume surge: arriving noise traders are busier than usual
            self.spawn(kind, **opts, **({'activity': rng.uniform(0.08, 0.15)} if kind == 'noise' else {}))
        self.news('14:30 US markets are open: 200 new traders joined.')


# ===================================================================
# RESULT
# ===================================================================
@dataclass
class ExchangeResult:
    """One simulated trading day."""
    tick_seconds: float
    seconds: np.ndarray            # simulated clock at every tick
    price: np.ndarray              # mid at the end of every tick
    last: np.ndarray               # last trade price
    bid: np.ndarray
    ask: np.ndarray
    volume: np.ndarray             # shares traded in the tick
    trades_n: np.ndarray           # trades in the tick
    n_active: np.ndarray
    value: np.ndarray              # hidden fair value
    heat: np.ndarray
    agents: pd.DataFrame
    traits: pd.DataFrame
    trades: pd.DataFrame | None
    news: list
    fleet: Fleet = field(repr=False)
    book: OrderBook = field(repr=False)
    seed: int | None = None
    price0: float = 100.0
    runtime: float = 0.0

    @property
    def dt(self) -> float:
        """One tick as a fraction of a (252-day) trading year."""
        return 1.0 / (252.0 * len(self.price))

    @property
    def returns(self) -> np.ndarray:
        return np.diff(np.log(self.price))

    @property
    def clock(self) -> np.ndarray:
        """Time of day in hours."""
        return self.seconds / 3600.0

    def facts(self) -> pd.Series:
        return stylized_facts(self.price, self.dt)

    def summary(self) -> pd.Series:
        spread = (self.ask - self.bid)
        two_sided = np.isfinite(spread)
        r = self.returns
        return pd.Series({
            'ticks': len(self.price), 'agents ever': len(self.agents), 'active at close': int(self.n_active[-1]),
            'trades': int(self.trades_n.sum()), 'shares traded': int(self.volume.sum()),
            'open': self.price[0], 'close': self.price[-1], 'high': self.price.max(), 'low': self.price.min(),
            'day return': self.price[-1] / self.price0 - 1,
            'annualised vol': r.std(ddof=1) / math.sqrt(self.dt),
            'mean spread (cents)': 100 * float(np.nanmean(spread[two_sided])) if two_sided.any() else np.nan,
            'two-sided quotes': float(two_sided.mean()),
            'seconds to simulate': round(self.runtime, 1),
        }, name='exchange day')

    def by_type(self, by: str = 'type') -> pd.DataFrame:
        """Profit and trading per agent type (profit marks holdings at the closing price).

        ``by='label'`` splits types that carry a finer label, e.g. the hypotheses of ``'profiled'`` traders.
        """
        g = self.agents.groupby(by)
        out = pd.DataFrame({
            'agents': g.size(), 'mean pnl': g['pnl'].mean(), 'total pnl': g['pnl'].sum(), 'sd pnl': g['pnl'].std(),
            'profitable': g['pnl'].apply(lambda s: float((s > 0).mean())),
            'shares traded': g['volume'].sum(), 'mean |position|': g['shares'].apply(lambda s: float(s.abs().mean())),
        })
        out['volume share'] = out['shares traded'] / out['shares traded'].sum()
        return out.sort_values('total pnl', ascending=False)

    def conservation(self) -> pd.Series:
        """Every trade has a buyer and a seller: cash and shares must net to zero across all agents."""
        d_shares = float((self.agents['shares'] - self.agents['shares0']).sum())
        endowment = float((self.agents['shares0'] * (self.price[-1] - self.agents['price0'])).sum())
        return pd.Series({'net change in shares': d_shares, 'net change in cash': float(self.agents['cash'].sum()),
                          'total pnl': float(self.agents['pnl'].sum()), 'pnl from initial endowments': endowment})

    def trait_effects(self, kind: str, target: str = 'pnl') -> pd.DataFrame:
        """Rank correlation of each trait of one agent type with profit (or any numeric agent column)."""
        ids = self.agents.index[self.agents['type'] == kind]
        t = self.traits.loc[self.traits.index.intersection(ids)].dropna(axis=1, how='all')
        y = self.agents.loc[t.index, target]
        rows = []
        for col in t.columns:
            x = t[col]
            if x.nunique() > 1 and y.nunique() > 1:
                rho, p = stats.spearmanr(x, y)
                rows.append({'trait': col, 'spearman': rho, 'p-value': p, 'agents': len(x)})
        return pd.DataFrame(rows).set_index('trait').sort_values('spearman', key=abs, ascending=False) if rows \
            else pd.DataFrame(columns=['spearman', 'p-value', 'agents'])

    def plot(self, show: bool = True):
        import matplotlib.pyplot as plt
        hrs = self.clock
        with V.themed():
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            ax = axes[0, 0]
            ax.plot(hrs, self.price, color=V.BLUE, lw=1.2, label='mid price')
            ax.plot(hrs, self.value, color=V.ORANGE, lw=1.0, ls='--', label='hidden fair value')
            for tick, text in self.news:
                if text.startswith('14:30'):
                    ax.axvline(hrs[min(tick, len(hrs) - 1)], color=V.RED, lw=0.9, alpha=0.7)
                    ax.text(hrs[min(tick, len(hrs) - 1)], ax.get_ylim()[1], ' US open', color=V.RED, fontsize=7.5, va='top')
            lo, hi = ax.get_ylim()
            ax2 = ax.twinx()
            ax2.bar(hrs, self.volume, width=hrs[1] - hrs[0], color=V.OTHER, alpha=0.5, lw=0)
            ax2.set_ylim(0, max(self.volume.max(), 1) * 5)
            ax2.set_yticks([])
            ax2.grid(False)
            ax.set_zorder(ax2.get_zorder() + 1)
            ax.patch.set_visible(False)
            ax.set_title('Price, fair value and volume')
            ax.set_xlabel('hour of day')
            ax.legend(loc='upper left')
            ax = axes[0, 1]
            spread = (self.ask - self.bid) * 100
            ax.plot(hrs, spread, color=V.BLUE, lw=0.7, alpha=0.8)
            w = max(5, len(hrs) // 60)
            ax.plot(hrs, pd.Series(spread).rolling(w, min_periods=1).mean(), color=V.ORANGE, lw=1.5, label=f'{w}-tick mean')
            ax.set_ylim(0, np.nanpercentile(spread, 99.5) * 1.2 if np.isfinite(spread).any() else 5)
            ax.set_title('Bid-ask spread (cents)')
            ax.set_xlabel('hour of day')
            ax.legend()
            ax = axes[0, 2]
            bp, bq = self.book.depth(BUY, 30)
            ap, aq = self.book.depth(SELL, 30)
            if bp.size:
                ax.step(bp, bq, where='post', color=V.BLUE, label='bids')
                ax.fill_between(bp, bq, step='post', color=V.BLUE, alpha=0.15)
            if ap.size:
                ax.step(ap, aq, where='post', color=V.ORANGE, label='asks')
                ax.fill_between(ap, aq, step='post', color=V.ORANGE, alpha=0.15)
            ax.set_title('Order book at the close (cumulative depth)')
            ax.set_xlabel('price')
            ax.legend()
            ax = axes[1, 0]
            r = self.returns
            z = (r - r.mean()) / r.std(ddof=1)
            ax.hist(z, bins=np.linspace(-7, 7, 71), density=True, color=V.BLUE, alpha=0.45, label='returns')
            xs = np.linspace(-7, 7, 300)
            ax.plot(xs, stats.norm.pdf(xs), color=V.TEXT, lw=1.1, label='normal')
            ax.set_yscale('log')
            ax.set_ylim(1e-4, 1)
            ax.set_title(f'Tick returns: excess kurtosis {stats.kurtosis(r):.1f}')
            ax.set_xlabel('standardised return')
            ax.legend()
            ax = axes[1, 1]
            bt = self.by_type()
            order = bt.sort_values('mean pnl')
            colors = [V.BLUE if v >= 0 else V.RED for v in order['mean pnl']]
            ax.barh(order.index, order['mean pnl'], color=colors)
            ax.axvline(0, color=V.TEXT_2, lw=0.8)
            ax.tick_params(axis='y', labelsize=7)
            ax.set_title('Mean profit per agent, by type')
            ax.set_xlabel('profit marked at the close')
            ax = axes[1, 2]
            lags = np.arange(1, 41)
            band = 1.96 / math.sqrt(len(r))
            ax.axhspan(-band, band, color=V.OTHER, alpha=0.25, lw=0)
            ax.plot(lags, _acf(r, lags), 'o-', ms=3, color=V.BLUE, label='returns')
            ax.plot(lags, _acf(np.abs(r), lags), 'o-', ms=3, color=V.ORANGE, label='absolute returns')
            ax.set_title('Autocorrelation')
            ax.set_xlabel('lag (ticks)')
            ax.legend()
            V.suptitle(fig, f'Exchange: {len(self.agents)} agents, {int(self.trades_n.sum()):,} trades, '
                            f'{self.price[0]:.2f} -> {self.price[-1]:.2f}')
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            return V.finish(fig, show)


# ===================================================================
# THE LOOP
# ===================================================================
def simulate(population='game', ticks: int | None = None, seed: int | None = None, price0: float = 100.0,
             schedule: bool | tuple = True, extra=(), sim_secs_per_tick: float = 15, open_secs: int = 34200,
             close_secs: int = 64800, record_trades: bool = True, on_news=None, tick: float = 0.01) -> ExchangeResult:
    """Simulate one trading day.

    ``population``: a preset name (``'game'``, ``'brief'``, ``'minimal'``) or a dict ``{type: count}``.
    ``schedule``: ``True`` for the full day (opening crowd, hourly churn, US open, closing crowd), ``False``
    for a static population, or a tuple of the events to keep from ``('open', 'churn', 'us_open', 'close')``.
    ``extra``: additional agents, each a type key, a class or ``(type or class, options)``,
    e.g. ``extra=[(PolicyAgent, {'policy': my_policy})]``.
    """
    import time
    t0 = time.perf_counter()
    mix = dict(POPULATIONS[population]) if isinstance(population, str) else dict(population)
    unknown = [k for k in mix if k not in BOT_TYPES]
    if unknown:
        raise KeyError(f'unknown agent types {unknown}; known: {sorted(BOT_TYPES)}')
    events = () if schedule is False else (('open', 'churn', 'us_open', 'close') if schedule is True else tuple(schedule))
    rng = random.Random(seed)
    book = OrderBook(tick=tick, last_price=price0)
    fleet = Fleet(book, rng, mix, sim_secs_per_tick=sim_secs_per_tick, open_secs=open_secs, close_secs=close_secs,
                  events=events, on_news=on_news)
    n_ticks = int(ticks) if ticks else int((close_secs - open_secs) / sim_secs_per_tick)
    hist = np.empty(n_ticks)
    log = [] if record_trades else None
    if record_trades:
        book.on_trade(lambda t: log.append((fleet.tick, t.price, t.qty, t.buyer, t.seller, t.aggressor)))
    fleet.start(hist)
    for item in extra:
        kind, opts = (item, {}) if isinstance(item, (str, type)) else item
        fleet.spawn(kind, **{'cohort': 'extra', **opts})
    out = {k: np.empty(n_ticks) for k in ('last', 'bid', 'ask', 'volume', 'trades', 'active', 'value', 'heat')}
    prev_vol, prev_n = 0, 0
    for t in range(n_ticks):
        secs = open_secs + t * sim_secs_per_tick
        book.expire(t)
        fleet.on_tick(t, secs, hist, t)
        mid = book.mid()
        hist[t] = mid
        bb, ba = book.best_bid(), book.best_ask()
        out['last'][t], out['bid'][t], out['ask'][t] = book.last_price, np.nan if bb is None else bb, np.nan if ba is None else ba
        out['volume'][t], out['trades'][t] = book.volume - prev_vol, book.n_trades - prev_n
        prev_vol, prev_n = book.volume, book.n_trades
        out['active'][t], out['value'][t], out['heat'][t] = len(fleet.active), fleet.value, fleet.heat
    final = hist[-1]
    rows, trait_rows = [], {}
    for a in fleet.agents.values():
        rows.append({'id': a.id, 'type': a.type, 'label': getattr(a, 'label', a.type), 'cohort': a.cohort, 'shares': a.shares, 'shares0': a.shares0,
                     'price0': a.price0, 'cash': a.cash, 'realized': a.realized, 'pnl': a.pnl(final),
                     'volume': a.volume, 'fills': a.n_fills})
        tr = a.traits()
        if tr:
            trait_rows[a.id] = tr
    agents = pd.DataFrame(rows).set_index('id')
    trades = pd.DataFrame(log, columns=['tick', 'price', 'qty', 'buyer', 'seller', 'aggressor']) if record_trades else None
    return ExchangeResult(
        tick_seconds=sim_secs_per_tick, seconds=open_secs + np.arange(n_ticks) * sim_secs_per_tick, price=hist,
        last=out['last'], bid=out['bid'], ask=out['ask'], volume=out['volume'], trades_n=out['trades'],
        n_active=out['active'], value=out['value'], heat=out['heat'], agents=agents,
        traits=pd.DataFrame.from_dict(trait_rows, orient='index'), trades=trades, news=list(fleet.log),
        fleet=fleet, book=book, seed=seed, price0=price0, runtime=time.perf_counter() - t0)
