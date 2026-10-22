"""finance.lob: price-time priority order book."""
import random

import pytest

from stochastics.finance.lob import BUY, SELL, OrderBook


def test_resting_orders_and_best_prices():
    b = OrderBook()
    assert b.best_bid() is None and b.best_ask() is None and b.mid() == 100.0 and b.spread() is None
    b.limit('a', BUY, 99.95, 10)
    b.limit('b', BUY, 99.90, 20)
    b.limit('c', SELL, 100.05, 30)
    b.limit('d', SELL, 100.10, 40)
    assert (b.best_bid(), b.best_ask()) == (99.95, 100.05)
    assert b.mid() == pytest.approx(100.0) and b.spread() == pytest.approx(0.10)
    assert b.levels(BUY, 5) == [(99.95, 10), (99.90, 20)]
    assert b.levels(SELL, 1) == [(100.05, 30)]
    b.check()


def test_time_priority_within_a_level_and_partial_fills():
    b = OrderBook()
    trades = []
    b.on_trade(trades.append)
    b.limit('first', SELL, 100.00, 10)
    b.limit('second', SELL, 100.00, 10)
    fill = b.market('t', BUY, 15)
    assert (fill.filled, fill.avg_price, fill.resting) == (15, 100.0, 0)
    assert [(t.seller, t.qty) for t in trades] == [('first', 10), ('second', 5)]
    assert b.levels(SELL, 3) == [(100.0, 5)]                        # second keeps its place with 5 left
    b.check()


def test_market_order_sweeps_levels_at_resting_prices_and_never_rests():
    b = OrderBook()
    b.limit('m', SELL, 100.01, 5)
    b.limit('m', SELL, 100.03, 5)
    fill = b.market('t', BUY, 12)
    assert fill.filled == 10 and fill.resting == 0                  # book exhausted, remainder discarded
    assert fill.avg_price == pytest.approx((5 * 100.01 + 5 * 100.03) / 10)
    assert b.best_ask() is None and len(b) == 0 and b.last_price == 100.03


def test_limit_order_crosses_then_rests():
    b = OrderBook()
    b.limit('m', SELL, 100.00, 10)
    fill = b.limit('t', BUY, 100.02, 25)                            # takes 10 at 100.00, rests 15 at 100.02
    assert (fill.filled, fill.resting, fill.avg_price) == (10, 15, 100.0)
    assert b.best_bid() == 100.02 and b.best_ask() is None
    b.check()
    again = b.limit('x', SELL, 100.02, 5)                           # trades at the resting bid's price
    assert again.avg_price == 100.02 and b.levels(BUY, 1) == [(100.02, 10)]


def test_trade_events_carry_both_sides_and_aggressor():
    b = OrderBook()
    seen = []
    b.on_trade(seen.append)
    b.limit('bidder', BUY, 99.99, 5)
    sell = b.market('seller', SELL, 5)
    b.limit('asker', SELL, 100.02, 3)
    buy = b.market('buyer', BUY, 3)
    t1, t2 = seen
    assert (t1.buyer, t1.seller, t1.aggressor, t1.price, t1.qty) == ('bidder', 'seller', SELL, 99.99, 5)
    assert t1.sell_order == sell.id
    assert (t2.buyer, t2.seller, t2.aggressor) == ('buyer', 'asker', BUY) and t2.buy_order == buy.id
    assert b.n_trades == 2 and b.volume == 8


def test_cancel_by_id_and_by_owner():
    b = OrderBook()
    a = b.limit('a', BUY, 99.0, 10)
    b.limit('a', BUY, 99.0, 7)
    c = b.limit('c', BUY, 98.0, 3)
    assert not b.cancel(a.id, owner='mallory') and b.has(a.id)       # wrong owner
    assert b.cancel(a.id, owner='a') and not b.has(a.id)
    assert b.levels(BUY, 5) == [(99.0, 7), (98.0, 3)]
    assert b.cancel(c.id) and b.levels(BUY, 5) == [(99.0, 7)]       # empty level disappears
    assert not b.cancel('nope')
    assert b.cancel_owner('a') == 1 and len(b) == 0
    b.check()


def test_expiry_and_partial_fill_keeps_expiry():
    b = OrderBook()
    keep = b.limit('a', SELL, 101.0, 10)
    gone = b.limit('a', BUY, 99.0, 10, expires=5)
    part = b.limit('a', BUY, 98.0, 10, expires=8)
    b.market('t', SELL, 4)                                          # part of the 99.0 bid trades
    assert b.expire(4) == 0 and b.has(gone.id)
    assert b.expire(5) == 1 and not b.has(gone.id)
    b.market('t', SELL, 3)                                          # 98.0 bid is partially filled...
    assert b.expire(8) == 1 and not b.has(part.id)                 # ... and still expires on schedule
    assert b.has(keep.id)
    b.check()


def test_invalid_orders_and_rounding():
    b = OrderBook()
    assert b.limit('a', BUY, 100.0, 0) is None
    assert b.limit('a', BUY, 0.0, 5) is None
    assert b.limit('a', BUY, -1.0, 5) is None
    assert b.market('a', BUY, 0.4) is None                          # floors to 0
    b.limit('a', SELL, 100.004, 3)                                  # rounds to a tick
    assert b.best_ask() == 100.0
    b.limit('a', SELL, 100.006, 3)
    assert b.levels(SELL, 3) == [(100.0, 3), (100.01, 3)]
    assert OrderBook(tick=0.05).to_price(OrderBook(tick=0.05).to_ticks(10.07)) == 10.05


def test_mid_falls_back_to_last_trade_when_a_side_is_empty():
    b = OrderBook(last_price=50.0)
    b.limit('a', BUY, 49.0, 1)
    assert b.mid() == 50.0
    b.limit('b', SELL, 49.0, 1)                                     # crosses and trades at 49
    assert b.mid() == 49.0 and b.last_price == 49.0


def test_random_operations_keep_the_book_consistent():
    rng = random.Random(7)
    b = OrderBook()
    ids = []
    volume = 0

    def count(t):
        nonlocal volume
        volume += t.qty
    b.on_trade(count)
    for step in range(4000):
        r = rng.random()
        side = rng.choice((BUY, SELL))
        if r < 0.55:
            px = round(100 + rng.gauss(0, 0.3) + (0.05 if side == SELL else -0.05), 2)
            f = b.limit(f'o{rng.randrange(30)}', side, px, rng.randint(1, 50), expires=step + rng.randint(1, 80) if rng.random() < 0.5 else None)
            if f and f.resting:
                ids.append(f.id)
        elif r < 0.7:
            b.market(f'o{rng.randrange(30)}', side, rng.randint(1, 80))
        elif r < 0.95 and ids:
            b.cancel(ids.pop(rng.randrange(len(ids))))
        b.expire(step)
        if step % 50 == 0:
            b.check()
    b.check()
    assert volume == b.volume and b.n_trades > 100
