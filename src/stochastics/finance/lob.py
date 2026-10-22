"""Limit order book with price-time priority (a Python port of ``OrderBook.js``).

    from stochastics.finance.lob import OrderBook

    book = OrderBook()
    book.submit('maker', 'SELL', 50, price=100.05)
    book.submit('maker', 'BUY', 50, price=99.95)
    fill = book.submit('taker', 'BUY', 20)             # no price: a market order
    fill                                               # Fill(id=..., filled=20, avg_price=100.05, resting=0)
    book.best_bid(), book.best_ask(), book.mid()
    book.levels('SELL', 5)                             # [(price, qty), ...] best first
    book.on_trade(print)                               # called with a Trade for every fill

Same behaviour as the JavaScript book: limit orders match whatever crosses and
the rest rests; market orders never rest; an order matched against the book trades
at the *resting* order's price; within a price level the oldest order goes first;
cancels go through an id index; orders may expire at a given tick.  Differences:

* prices are integers of ``tick`` internally (no floating-point price keys), and
  ``tick`` is configurable (default 0.01);
* every :class:`Trade` carries the ``aggressor`` side (the JS book needs the
  "AI update" for that), and the book counts ``n_trades`` and ``volume``;
* nothing prevents an owner from trading with its own resting orders, as in the JS book.
"""
from __future__ import annotations

import heapq
import math
from bisect import bisect_left, insort
from collections import deque
from typing import Callable, NamedTuple

import numpy as np

__all__ = ['BUY', 'SELL', 'Order', 'Trade', 'Fill', 'OrderBook']

BUY, SELL = 'BUY', 'SELL'


class Trade(NamedTuple):
    price: float
    qty: int
    buyer: str
    seller: str
    buy_order: str
    sell_order: str
    aggressor: str          # side of the incoming order


class Fill(NamedTuple):
    """Result of submitting an order."""
    id: str
    filled: int
    avg_price: float
    resting: int            # quantity left on the book (always 0 for market orders)


class Order:
    """A resting order (``price`` in ticks)."""
    __slots__ = ('id', 'owner', 'side', 'price', 'qty', 'expires')

    def __init__(self, id, owner, side, price, qty, expires=None):
        self.id, self.owner, self.side, self.price, self.qty, self.expires = id, owner, side, price, qty, expires

    def __repr__(self):
        return f'Order({self.id}, {self.owner}, {self.side} {self.qty} @ {self.price} ticks)'


class _Level:
    __slots__ = ('orders', 'qty')

    def __init__(self):
        self.orders: deque[Order] = deque()
        self.qty = 0


class OrderBook:
    def __init__(self, tick: float = 0.01, last_price: float = 100.0, on_trade: Callable[[Trade], None] | None = None):
        self.tick = float(tick)
        self._nd = max(0, int(round(-math.log10(self.tick)))) + 2
        self._levels = {BUY: {}, SELL: {}}          # price in ticks -> _Level
        self._keys = {BUY: [], SELL: []}            # sorted ascending, best last: ticks (bids) or -ticks (asks)
        self._index: dict[str, Order] = {}
        self._expiry: list = []                     # heap of (tick, seq, order id)
        self._seq = 0
        self.last_price = float(last_price)
        self.listeners: list[Callable[[Trade], None]] = [on_trade] if on_trade else []
        self.n_trades = 0
        self.volume = 0

    # ----------------------------------------------------------------- units
    def to_ticks(self, price: float) -> int:
        return int(round(price / self.tick))

    def to_price(self, ticks: int) -> float:
        return round(ticks * self.tick, self._nd)

    def next_id(self, prefix: str = 'o') -> str:
        self._seq += 1
        return f'{prefix}{self._seq}'

    def on_trade(self, fn: Callable[[Trade], None]) -> None:
        self.listeners.append(fn)

    # ---------------------------------------------------------------- reading
    def best_bid(self) -> float | None:
        k = self._keys[BUY]
        return self.to_price(k[-1]) if k else None

    def best_ask(self) -> float | None:
        k = self._keys[SELL]
        return self.to_price(-k[-1]) if k else None

    def mid(self) -> float:
        """Mid of best bid and ask; the last trade price when one side is empty (as in the JS book)."""
        b, a = self._keys[BUY], self._keys[SELL]
        if b and a:
            return 0.5 * (self.to_price(b[-1]) + self.to_price(-a[-1]))
        return self.last_price or 100.0

    def spread(self) -> float | None:
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else round(a - b, self._nd)

    def levels(self, side: str, depth: int = 10) -> list[tuple[float, int]]:
        """Aggregated ``(price, qty)`` levels, best first."""
        keys, lv = self._keys[side], self._levels[side]
        out = []
        for key in reversed(keys[-depth:]):
            t = key if side == BUY else -key
            out.append((self.to_price(t), lv[t].qty))
        return out

    def depth(self, side: str, levels: int = 25):
        """``(prices, cumulative quantity)`` arrays for a depth chart."""
        lv = self.levels(side, levels)
        if not lv:
            return np.empty(0), np.empty(0)
        p, q = zip(*lv)
        return np.array(p), np.cumsum(q)

    def has(self, order_id: str) -> bool:
        return order_id in self._index

    def find(self, order_id: str) -> Order | None:
        return self._index.get(order_id)

    def open_orders(self, owner: str) -> list[Order]:
        return [o for o in self._index.values() if o.owner == owner]

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------ maintenance
    def _insert(self, o: Order) -> None:
        lv = self._levels[o.side]
        level = lv.get(o.price)
        if level is None:
            level = lv[o.price] = _Level()
            insort(self._keys[o.side], o.price if o.side == BUY else -o.price)
        level.orders.append(o)
        level.qty += o.qty
        self._index[o.id] = o
        if o.expires is not None:
            self._seq += 1
            heapq.heappush(self._expiry, (o.expires, self._seq, o.id))

    def cancel(self, order_id: str, owner: str | None = None) -> bool:
        """Remove one resting order; if ``owner`` is given it must own the order."""
        o = self._index.get(order_id)
        if o is None or (owner is not None and o.owner != owner):
            return False
        lv = self._levels[o.side]
        level = lv[o.price]
        level.orders.remove(o)
        level.qty -= o.qty
        if not level.orders:
            del lv[o.price]
            keys = self._keys[o.side]
            del keys[bisect_left(keys, o.price if o.side == BUY else -o.price)]
        del self._index[order_id]
        return True

    def cancel_owner(self, owner: str) -> int:
        """Cancel every resting order of ``owner`` (slow: scans the whole book)."""
        ids = [i for i, o in self._index.items() if o.owner == owner]
        for i in ids:
            self.cancel(i)
        return len(ids)

    def expire(self, tick: int) -> int:
        """Remove orders whose expiry tick has passed."""
        n = 0
        h = self._expiry
        while h and h[0][0] <= tick:
            exp, _, oid = heapq.heappop(h)
            o = self._index.get(oid)
            if o is not None and o.expires is not None and o.expires <= tick:
                self.cancel(oid)
                n += 1
        return n

    # ---------------------------------------------------------------- matching
    def submit(self, owner: str, side: str, qty: float, price: float | None = None,
               expires: int | None = None, id: str | None = None) -> Fill | None:
        """Send an order: ``price=None`` is a market order, otherwise a limit order.

        Returns ``None`` for an invalid order (non-positive quantity or price).
        """
        qty = int(qty)
        if qty <= 0:
            return None
        oid = id or self.next_id('b')
        if price is None:
            filled, avg = self._match(owner, side, qty, None, oid)
            return Fill(oid, filled, avg, 0)
        t = self.to_ticks(price)
        if t <= 0:
            return None
        filled, avg = self._match(owner, side, qty, t, oid)
        rest = qty - filled
        if rest > 0:
            self._insert(Order(oid, owner, side, t, rest, expires))
        return Fill(oid, filled, avg, rest)

    def market(self, owner: str, side: str, qty: float) -> Fill | None:
        return self.submit(owner, side, qty)

    def limit(self, owner: str, side: str, price: float, qty: float, expires: int | None = None) -> Fill | None:
        return self.submit(owner, side, qty, price, expires)

    def _match(self, owner, side, qty, limit, oid):
        opp = SELL if side == BUY else BUY
        keys, levels = self._keys[opp], self._levels[opp]
        remaining, notional = qty, 0.0
        buying = side == BUY
        while remaining > 0 and keys:
            key = keys[-1]
            t = -key if buying else key                 # resting asks are stored as -ticks
            if limit is not None and (t > limit if buying else t < limit):
                break
            level = levels[t]
            orders = level.orders
            price = self.to_price(t)
            while remaining > 0 and orders:
                top = orders[0]
                q = top.qty if top.qty < remaining else remaining
                remaining -= q
                notional += q * price
                top.qty -= q
                level.qty -= q
                if top.qty == 0:
                    orders.popleft()
                    del self._index[top.id]
                self.last_price = price
                self.n_trades += 1
                self.volume += q
                if self.listeners:
                    trade = (Trade(price, q, owner, top.owner, oid, top.id, side) if buying
                             else Trade(price, q, top.owner, owner, top.id, oid, side))
                    for fn in self.listeners:
                        fn(trade)
            if not orders:
                del levels[t]
                keys.pop()
        filled = qty - remaining
        return filled, (notional / filled if filled else 0.0)

    # ------------------------------------------------------------------- checks
    def check(self) -> None:
        """Assert the book's internal invariants (used by the tests)."""
        n = 0
        for side in (BUY, SELL):
            keys, lv = self._keys[side], self._levels[side]
            assert keys == sorted(keys) and len(keys) == len(lv) == len(set(keys)), 'price keys out of order'
            for key in keys:
                t = key if side == BUY else -key
                level = lv[t]
                assert level.orders, 'empty level kept'
                assert level.qty == sum(o.qty for o in level.orders) > 0, 'level quantity mismatch'
                for o in level.orders:
                    assert o.price == t and o.side == side and o.qty > 0 and self._index[o.id] is o
                n += len(level.orders)
        assert n == len(self._index), 'index and levels disagree'
        b, a = self._keys[BUY], self._keys[SELL]
        assert not (b and a) or b[-1] < -a[-1], 'crossed book'
