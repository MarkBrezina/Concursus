"""finance.realdata: loaders, sessions, resampling and the sample registry.  No test touches the network."""
import gzip
import sys
import types

import numpy as np
import pandas as pd
import pytest

from stochastics.finance import realdata as rd


def write_daily(path, n=200, seed=0, start='2020-01-01', price_col='Adj Close', freq='B'):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq)
    df = pd.DataFrame({'Date': idx.strftime('%m/%d/%Y'), price_col: 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))})
    df.to_csv(path, index=False)
    return df


def write_intraday(path, days=3, bars=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(days):
        day = pd.Timestamp('2021-03-01') + pd.Timedelta(days=d)
        for b in range(bars):
            t = day + pd.Timedelta(hours=9) + pd.Timedelta(minutes=5 * b)
            rows.append((t.strftime('%Y-%m-%d'), t.strftime('%H:%M:%S')))
    df = pd.DataFrame(rows, columns=['Date', 'Time'])
    df['Close'] = 100 * np.exp(np.cumsum(0.001 * rng.standard_normal(len(df))))
    df.to_csv(path, index=False)
    return df


# ------------------------------------------------------------------ loading
def test_a_daily_csv_is_read_with_guessed_columns(tmp_path):
    write_daily(tmp_path / 'a.csv')
    s = rd.load_csv(tmp_path / 'a.csv', asset='equity')
    assert s.kind == 'daily' and s.bar == '1D' and s.periods_per_year == 252.0
    assert len(s) == 200 and s.name == 'a'
    assert s.dt == pytest.approx(1 / 252)
    assert len(s.returns()) == 199 and len(s.segments()) == 1


def test_the_year_length_follows_the_asset_class(tmp_path):
    write_daily(tmp_path / 'a.csv')
    assert rd.load_csv(tmp_path / 'a.csv', asset='crypto').periods_per_year == 365.0
    assert rd.load_csv(tmp_path / 'a.csv', asset='fx').periods_per_year == 260.0
    assert rd.load_csv(tmp_path / 'a.csv', periods_per_year=12.0).periods_per_year == 12.0


def test_date_and_time_columns_make_an_intraday_series_cut_into_days(tmp_path):
    write_intraday(tmp_path / "i.csv", days=3, bars=30)
    s = rd.load_csv(tmp_path / 'i.csv', time=('Date', 'Time'), asset='equity')
    assert s.kind == 'intraday' and s.bar == '5min'
    assert s.periods_per_year == pytest.approx(252 * 30)
    segs = s.segments()
    assert [len(x) for x in segs] == [29, 29, 29]         # no return jumps over a night


def test_sessions_can_be_cut_every_n_bars_instead_of_by_date(tmp_path):
    write_intraday(tmp_path / "i.csv", days=3, bars=30)
    s = rd.load_csv(tmp_path / 'i.csv', time=('Date', 'Time'), asset='equity').sessions(10)
    assert all(len(x) <= 10 for x in s.segments())
    assert sum(len(x) for x in s.segments()) == 87


def test_epoch_timestamps_in_seconds_and_milliseconds_are_detected(tmp_path):
    t = 1_700_000_000 + 60 * np.arange(120)
    price = 100 + np.cumsum(np.random.default_rng(1).standard_normal(120)) * 0.01 + 50
    for unit, mult in (('s', 1), ('ms', 1000)):
        pd.DataFrame({'timestamp': t * mult, 'close': price}).to_csv(tmp_path / f'{unit}.csv', index=False)
        s = rd.load_csv(tmp_path / f'{unit}.csv', asset='crypto')
        assert s.price.index[0].year == 2023 and s.bar == '1min' and s.kind == 'intraday'


def test_where_keeps_one_series_of_a_long_format_file(tmp_path):
    rng = np.random.default_rng(2)
    idx = pd.date_range('2020-01-01', periods=60, freq='B')
    df = pd.concat([pd.DataFrame({'Date': idx, 'Country': c, 'Exchange rate': r * np.exp(np.cumsum(0.005 * rng.standard_normal(60)))})
                    for c, r in (('Euro', 1.1), ('Japan', 110.0))])
    df.to_csv(tmp_path / 'fx.csv', index=False)
    s = rd.load_csv(tmp_path / 'fx.csv', time='Date', price='Exchange rate', where={'Country': 'Japan'}, asset='fx')
    assert len(s) == 60 and s.price.iloc[0] == pytest.approx(110.0, rel=0.1)


def test_gzipped_files_and_missing_prices_are_handled(tmp_path):
    df = write_daily(tmp_path / 'a.csv')
    raw = (tmp_path / 'a.csv').read_bytes()
    with gzip.open(tmp_path / 'a.csv.gz', 'wb') as f:
        f.write(raw)
    df2 = df.copy()
    df2.loc[5, 'Adj Close'] = np.nan
    df2.to_csv(tmp_path / 'gap.csv', index=False)
    assert len(rd.load_csv(tmp_path / 'a.csv.gz')) == 200
    assert len(rd.load_csv(tmp_path / 'gap.csv')) == 199


def test_a_missing_price_column_gives_a_helpful_error(tmp_path):
    pd.DataFrame({'Date': pd.date_range('2020', periods=50), 'foo': np.arange(1, 51)}).to_csv(tmp_path / 'x.csv', index=False)
    with pytest.raises(ValueError, match='price column'):
        rd.load_csv(tmp_path / 'x.csv')


# ------------------------------------------------------------------ the container
def test_bad_prices_and_duplicates_are_cleaned_and_short_series_rejected():
    idx = pd.date_range('2020-01-01', periods=60, freq='D')
    p = pd.Series(np.linspace(1, 2, 60), index=idx)
    p.iloc[3] = -1.0
    p.iloc[4] = 0.0
    s = rd.RealSeries('x', 'equity', pd.concat([p, p.iloc[10:12]]))
    assert len(s) == 58
    assert s.price.index.is_monotonic_increasing and not s.price.index.has_duplicates
    with pytest.raises(ValueError, match='usable prices'):
        rd.RealSeries('tiny', 'equity', p.iloc[:10])


def test_a_gap_of_more_than_a_week_splits_a_daily_series():
    idx = list(pd.date_range('2020-01-01', periods=40, freq='D')) + list(pd.date_range('2020-03-01', periods=40, freq='D'))
    p = pd.Series(100 * np.exp(np.cumsum(0.01 * np.random.default_rng(3).standard_normal(80))), index=idx)
    segs = rd.RealSeries('g', 'equity', p).segments()
    assert [len(x) for x in segs] == [39, 39]


def test_resampling_minutes_to_five_minutes_and_to_days():
    idx = pd.date_range('2025-01-06 00:00', periods=40 * 1440, freq='1min')
    p = pd.Series(100 * np.exp(np.cumsum(0.0005 * np.random.default_rng(4).standard_normal(len(idx)))), index=idx)
    s = rd.RealSeries('btc', 'crypto', p, 'intraday', '1min', 365 * 1440, 'day')
    five = s.resample('5min')
    assert five.kind == 'intraday' and five.bar == '5min' and len(five) == 40 * 288
    assert five.periods_per_year == pytest.approx(365 * 288)
    assert five.price.iloc[0] == pytest.approx(p.iloc[4])        # last price in the first bar
    day = s.resample('1D')
    assert day.kind == 'daily' and len(day) == 40 and day.periods_per_year == 365.0


def test_window_and_head():
    idx = pd.date_range('2020-01-01', periods=100, freq='B')
    s = rd.RealSeries('w', 'equity', pd.Series(np.linspace(100, 120, 100), index=idx))
    assert len(s.window(last=30)) == 30
    assert s.window(start='2020-03-01').price.index[0] >= pd.Timestamp('2020-03-01')
    assert len(s.window(end='2020-03-15')) < 100
    assert len(s.head(40)) == 40


def test_the_summary_and_fingerprint_methods(tmp_path):
    write_daily(tmp_path / 'a.csv', n=600)
    s = rd.load_csv(tmp_path / 'a.csv', asset='equity')
    summ = s.summary()
    assert summ['returns used'] == 599 and summ['sessions'] == 1
    assert summ['annualised vol'] == pytest.approx(0.01 * np.sqrt(252), rel=0.15)
    f = s.fingerprint()
    assert len(f) == 12 and np.all(np.isfinite(f))


# ------------------------------------------------------------------ downloading (against file:// URLs)
def test_fetch_downloads_once_and_drops_a_cut_last_line(tmp_path):
    src = tmp_path / 'src.csv'
    src.write_text('a,b\n' + ''.join(f'{i},{i * 2}\n' for i in range(1000)))
    dest = tmp_path / 'cache' / 'out.csv'
    dest.parent.mkdir()
    rd.fetch(src.as_uri(), dest, max_bytes=500)
    text = dest.read_text()
    assert text.endswith('\n') and len(text) <= 500
    assert all(len(line.split(',')) == 2 for line in text.strip().split('\n'))
    first = dest.read_text()
    src.write_text('changed\n')
    rd.fetch(src.as_uri(), dest, max_bytes=500)                 # cached: not fetched again
    assert dest.read_text() == first
    rd.fetch(src.as_uri(), dest, refresh=True)
    assert dest.read_text() == 'changed\n'


def test_a_failed_download_explains_how_to_load_the_file_by_hand(tmp_path):
    with pytest.raises(RuntimeError, match='load_csv'):
        rd.fetch((tmp_path / 'nope.csv').as_uri(), tmp_path / 'out.csv')
    assert not (tmp_path / 'out.csv').exists()


def test_sample_downloads_into_the_cache_and_parses(tmp_path, monkeypatch):
    src = tmp_path / 'toy.csv'
    write_daily(src, n=120)
    monkeypatch.setitem(rd.SAMPLES, 'toy', rd.Sample(src.as_uri(), 'equity', 'daily', 'a toy', 'nobody',
                                                      dict(time='Date', price='Adj Close')))
    s = rd.sample('toy', cache=tmp_path / 'cache')
    assert len(s) == 120 and s.asset == 'equity' and (tmp_path / 'cache' / 'toy.csv').exists()
    assert len(rd.sample('toy', cache=tmp_path / 'cache', last=50)) == 50
    with pytest.raises(KeyError, match='unknown sample'):
        rd.sample('nothing')


def test_the_cache_directory_follows_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv('STOCHASTICS_DATA', str(tmp_path / 'envcache'))
    assert rd.cache_dir() == tmp_path / 'envcache' and (tmp_path / 'envcache').is_dir()


def test_the_registry_describes_every_sample():
    tab = rd.samples()
    assert {'sp500', 'eur_d', 'btc_d', 'index_5m', 'btc_1m', 'eurusd_1h'} <= set(tab.index)
    assert set(tab['asset']) <= {'equity', 'fx', 'crypto'} and set(tab['kind']) == {'daily', 'intraday'}
    assert all(u.startswith('https://') for u in tab['url'])
    assert {'equity', 'fx', 'crypto'} == set(tab['asset'])


# ------------------------------------------------------------------ yahoo (with a stand-in for yfinance)
def test_yahoo_cleans_the_frame_and_guesses_the_asset(monkeypatch):
    idx = pd.date_range('2024-01-02 14:30', periods=80, freq='1h', tz='America/New_York')
    cols = pd.MultiIndex.from_product([['Close', 'Open'], ['EURUSD=X']])
    frame = pd.DataFrame(np.column_stack([1.1 + 0.001 * np.arange(80), np.ones(80)]), index=idx, columns=cols)
    calls = {}

    def download(symbol, **kw):
        calls['kw'] = kw
        return frame

    monkeypatch.setitem(sys.modules, 'yfinance', types.SimpleNamespace(download=download))
    s = rd.yahoo('EURUSD=X', start='2024-01-01', interval='1h')
    assert s.asset == 'fx' and s.kind == 'intraday' and s.price.index.tz is None and len(s) == 80
    assert calls['kw']['interval'] == '1h' and calls['kw']['start'] == '2024-01-01'
    assert rd.yahoo('BTC-USD', start='2024-01-01').asset == 'crypto'
    assert rd.yahoo('SPY', start='2024-01-01').asset == 'equity'
    monkeypatch.setitem(sys.modules, 'yfinance', types.SimpleNamespace(download=lambda *a, **k: pd.DataFrame()))
    with pytest.raises(ValueError, match='no data'):
        rd.yahoo('SPY')
