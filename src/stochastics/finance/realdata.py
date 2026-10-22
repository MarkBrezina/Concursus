"""Real market data: loaders, a :class:`RealSeries` container and a few public sample sets.

    from stochastics.finance import realdata

    spx = realdata.sample('sp500')                       # S&P 500, daily (downloads once, then cached)
    btc = realdata.sample('btc_1m').resample('5min')     # Bitstamp BTC/USD, 1-minute bars -> 5-minute bars
    mine = realdata.load_csv('my_file.csv', name='SPY', asset='equity')
    yf = realdata.yahoo('EURUSD=X', interval='1h', start='2024-01-01')      # needs `pip install yfinance`
    realdata.samples()                                   # the list of built-in sample sets

A :class:`RealSeries` is a price series plus what the benchmark needs to know about it: the asset class, whether
it is daily or intraday, how many bars make a year, and where it breaks into sessions.  ``segments()`` returns the
log returns cut at those breaks (overnight gaps, weekends, missing data), so that autocorrelations are never
computed across a gap.

Nothing here is bundled with the package.  ``sample()`` downloads public files from GitHub the first time it is
used and keeps them in ``~/.cache/stochastics/data`` (override with the ``STOCHASTICS_DATA`` environment
variable or the ``cache`` argument).  The sources and their owners are listed by :func:`samples`; check each
dataset's own licence before redistributing it.
"""
from __future__ import annotations

import io
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ['RealSeries', 'load_csv', 'from_frame', 'yahoo', 'sample', 'samples', 'fetch', 'SAMPLES', 'cache_dir']

DAYS_PER_YEAR = {'equity': 252.0, 'fx': 260.0, 'crypto': 365.0}
TIME_NAMES = ('time', 'timestamp', 'datetime', 'date', 'open_time', 'open time')
PRICE_NAMES = ('adj close', 'adj_close', 'close', 'price', 'priceusd', 'last', 'mid')


# ====================================================================== the container
@dataclass
class RealSeries:
    """A price series with the metadata the benchmark needs.

    ``price``: positive prices on a ``DatetimeIndex``.  ``asset``: ``'equity'``, ``'fx'``, ``'crypto'`` (or
    anything else).  ``kind``: ``'daily'`` or ``'intraday'``.  ``bar``: the bar length, e.g. ``'5min'``.
    ``periods_per_year``: bars per year (sets the annualisation).  ``session``: how an intraday series is cut into
    sessions, ``'day'`` (calendar date) or a number of bars, e.g. ``102``.
    """
    name: str
    asset: str
    price: pd.Series
    kind: str = 'daily'
    bar: str = '1D'
    periods_per_year: float = 252.0
    session: object = 'day'
    source: str = ''

    def __post_init__(self):
        p = self.price.astype(float)
        p = p[~p.index.duplicated(keep='last')].sort_index()
        p = p[np.isfinite(p) & (p > 0)]
        if len(p) < 30:
            raise ValueError(f'{self.name}: only {len(p)} usable prices')
        self.price = p

    def __len__(self) -> int:
        return len(self.price)

    def __repr__(self) -> str:
        a, b = self.price.index[0], self.price.index[-1]
        return (f'RealSeries({self.name!r}, {self.asset}, {self.kind} {self.bar}, {len(self)} prices, '
                f'{a:%Y-%m-%d} to {b:%Y-%m-%d})')

    # ---- returns and sessions
    def returns(self) -> pd.Series:
        """Log returns on the price index (the first price has none)."""
        return np.log(self.price).diff().dropna()

    def segments(self) -> list[np.ndarray]:
        """Log returns cut into sessions: no pair of consecutive returns spans a break, and the returns that
        jump over a break (overnight, weekend, missing data) are dropped.

        Intraday data break at calendar-date changes (``session='day'``) or every ``session`` bars, and wherever
        the time gap is more than 5 median bars.  Daily data break only at gaps of more than 7 days.
        """
        r = self.returns()
        if r.empty:
            return []
        vals, idx = r.to_numpy(), r.index
        gaps = np.diff(idx.values).astype('timedelta64[s]').astype(float)
        if self.kind == 'daily':
            cut = np.r_[False, gaps > 7 * 86400.0]
        else:
            med = float(np.median(gaps)) if len(gaps) else 1.0
            cut = np.r_[False, gaps > 5 * med]
            if self.session == 'day':
                days = idx.normalize().values
                cut |= np.r_[False, days[1:] != days[:-1]]
        ids = np.cumsum(cut)
        segs = [vals[(ids == g) & ~cut] for g in np.unique(ids)]
        if self.kind != 'daily' and self.session != 'day':
            n = int(self.session)
            segs = [c for s in segs for c in (s[i:i + n] for i in range(0, len(s), n)) if len(c) >= max(5, n // 2)]
        return [s for s in segs if len(s) >= 5]

    def sessions(self, bars: int | str = 'day') -> 'RealSeries':
        """The same series cut into sessions of ``bars`` returns (or calendar days)."""
        return RealSeries(self.name, self.asset, self.price, self.kind, self.bar, self.periods_per_year, bars,
                          self.source)

    @property
    def dt(self) -> float:
        """One bar as a fraction of a year."""
        return 1.0 / self.periods_per_year

    def summary(self) -> pd.Series:
        r = self.returns().to_numpy()
        segs = self.segments()
        return pd.Series({
            'name': self.name, 'asset': self.asset, 'kind': f'{self.kind} ({self.bar})',
            'start': self.price.index[0].strftime('%Y-%m-%d'), 'end': self.price.index[-1].strftime('%Y-%m-%d'),
            'returns used': int(sum(len(s) for s in segs)), 'sessions': len(segs),
            'annualised vol': float(r.std(ddof=1) * math.sqrt(self.periods_per_year)),
            'share of zero returns': float(np.mean(r == 0)),
        }, name=self.name)

    # ---- transformations
    def resample(self, rule: str) -> 'RealSeries':
        """Last price in each bar of ``rule`` (e.g. ``'5min'``, ``'1h'``, ``'1D'``)."""
        p = self.price.resample(rule).last().dropna()
        per_day = max(1.0, len(p) / max(1, p.index.normalize().nunique()))
        daily = pd.Timedelta(rule) >= pd.Timedelta('20h') if _is_fixed(rule) else True
        kind = 'daily' if daily else 'intraday'
        ppy = DAYS_PER_YEAR.get(self.asset, 252.0) * (1.0 if daily else per_day)
        return RealSeries(self.name, self.asset, p, kind, rule, ppy, self.session, self.source)

    def window(self, start=None, end=None, last: int | None = None) -> 'RealSeries':
        """A slice by dates, or the last ``last`` prices."""
        p = self.price.loc[start:end] if (start is not None or end is not None) else self.price
        if last is not None:
            p = p.iloc[-int(last):]
        return RealSeries(self.name, self.asset, p, self.kind, self.bar, self.periods_per_year, self.session, self.source)

    def head(self, n: int) -> 'RealSeries':
        return RealSeries(self.name, self.asset, self.price.iloc[:int(n)], self.kind, self.bar,
                          self.periods_per_year, self.session, self.source)

    def fingerprint(self) -> pd.Series:
        """Scale-free stylised facts of this series (see ``benchmark.fingerprint``)."""
        from .benchmark import fingerprint
        return fingerprint(self.segments())


def _is_fixed(rule: str) -> bool:
    try:
        pd.Timedelta(rule)
        return True
    except (ValueError, TypeError):
        return False


# ====================================================================== building a series
def _periods_per_year(asset: str, kind: str, index: pd.DatetimeIndex) -> float:
    days = DAYS_PER_YEAR.get(asset, 252.0)
    if kind == 'daily':
        return days
    per_day = len(index) / max(1, index.normalize().nunique())
    return days * per_day


def _infer_kind(index: pd.DatetimeIndex) -> tuple[str, str]:
    gaps = np.diff(index.values).astype('timedelta64[s]').astype(float)
    med = float(np.median(gaps)) if len(gaps) else 86400.0
    kind = 'daily' if med >= 20 * 3600 else 'intraday'
    if med >= 20 * 3600:
        bar = '1D'
    elif med % 3600 == 0:
        bar = f'{int(med // 3600)}h'
    elif med % 60 == 0:
        bar = f'{int(med // 60)}min'
    else:
        bar = f'{int(med)}s'
    return kind, bar


def from_frame(df: pd.DataFrame, *, name: str = 'series', asset: str = 'other', price: str | None = None,
               kind: str | None = None, bar: str | None = None, periods_per_year: float | None = None,
               session='day', source: str = '') -> RealSeries:
    """A :class:`RealSeries` from a DataFrame with a ``DatetimeIndex`` and a price column."""
    if price is None:
        low = {str(c).lower(): c for c in df.columns}
        price = next((low[n] for n in PRICE_NAMES if n in low), None)
        if price is None:
            raise ValueError(f'cannot find a price column in {list(df.columns)}; pass price=...')
    p = pd.to_numeric(df[price], errors='coerce').dropna()
    k, b = _infer_kind(pd.DatetimeIndex(p.index))
    kind = kind or k
    ppy = periods_per_year or _periods_per_year(asset, kind, pd.DatetimeIndex(p.index))
    return RealSeries(name, asset, p, kind, bar or b, ppy, session, source)


def load_csv(path, *, time=None, price: str | None = None, time_format: str | None = None, unit: str | None = None,
             where: dict | None = None, name: str | None = None, asset: str = 'other', kind: str | None = None,
             bar: str | None = None, periods_per_year: float | None = None, session='day', nrows: int | None = None,
             **read_csv_kw) -> RealSeries:
    """Read a price file (CSV, optionally gzipped) into a :class:`RealSeries`.

    ``time``: the time column, or a tuple of two columns (e.g. ``('Date', 'Time')``); guessed if omitted.
    ``price``: the price column; guessed from ``Adj Close, Close, price, ...``.  ``unit='s'`` or ``'ms'`` for epoch
    timestamps (detected from the size of the numbers if omitted).  ``where={'Country': 'Euro'}`` keeps the rows of a
    long-format file that match.
    """
    df = pd.read_csv(path, nrows=nrows, **read_csv_kw)
    if where:
        for col, val in where.items():
            df = df[df[col] == val]
    cols = {str(c).lower(): c for c in df.columns}
    if time is None:
        time = next((cols[n] for n in TIME_NAMES if n in cols), df.columns[0])
    if isinstance(time, (tuple, list)):
        t = pd.to_datetime(df[list(time)].astype(str).agg(' '.join, axis=1), format=time_format, errors='coerce')
    else:
        raw = df[time]
        if pd.api.types.is_numeric_dtype(raw):
            u = unit or ('ms' if raw.iloc[0] > 1e11 else 's')
            t = pd.to_datetime(raw, unit=u, errors='coerce')
        else:
            t = pd.to_datetime(raw, format=time_format, errors='coerce')
    df = df.assign(__t=t).dropna(subset=['__t']).set_index('__t')
    df.index.name = None
    return from_frame(df, name=name or Path(str(path)).stem, asset=asset, price=price, kind=kind, bar=bar,
                      periods_per_year=periods_per_year, session=session, source=str(path))


def yahoo(symbol: str, start: str | None = None, end: str | None = None, interval: str = '1d',
          asset: str | None = None, name: str | None = None, period: str | None = None) -> RealSeries:
    """Download one symbol with the optional ``yfinance`` package (``pip install yfinance``).

    Yahoo limits intraday history (about 7 days of 1-minute, 60 days of 5-minute and 730 days of hourly bars).
    The asset class is guessed from the symbol (``=X`` FX, ``-USD`` crypto) unless given.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on the user's environment
        raise ImportError('yahoo() needs the optional yfinance package: pip install yfinance') from exc
    kw = dict(interval=interval, auto_adjust=True, progress=False)
    df = yf.download(symbol, period=period, **kw) if period else yf.download(symbol, start=start, end=end, **kw)
    if df is None or len(df) == 0:
        raise ValueError(f'Yahoo returned no data for {symbol!r} ({interval})')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    idx = pd.DatetimeIndex(df.index)
    df.index = idx.tz_convert('UTC').tz_localize(None) if idx.tz is not None else idx
    asset = asset or ('fx' if symbol.endswith('=X') else 'crypto' if symbol.upper().endswith('-USD') else 'equity')
    return from_frame(df, name=name or symbol, asset=asset, price='Close', source=f'yahoo:{symbol}:{interval}')


# ====================================================================== public samples
@dataclass(frozen=True)
class Sample:
    url: str
    asset: str
    kind: str
    description: str
    owner: str
    read: dict = field(default_factory=dict)       # keyword arguments for load_csv
    max_bytes: int | None = None                   # fetch only the first bytes (large files)
    bar: str | None = None


_ARCH = 'https://raw.githubusercontent.com/bashtage/arch/main/arch/data/{0}/{0}.csv.gz'
_BT = 'https://raw.githubusercontent.com/kernc/backtesting.py/master/backtesting/test/{0}.csv'
_FED = 'https://raw.githubusercontent.com/datasets/exchange-rates/main/data/daily.csv'
_CM = 'https://raw.githubusercontent.com/coinmetrics/data/master/csv/{0}.csv'

SAMPLES: dict[str, Sample] = {
    'sp500': Sample(_ARCH.format('sp500'), 'equity', 'daily', 'S&P 500 index, daily, 1999-2018', 'bashtage/arch',
                    dict(time='Date', price='Adj Close', time_format='%m/%d/%Y')),
    'nasdaq': Sample(_ARCH.format('nasdaq'), 'equity', 'daily', 'Nasdaq composite, daily, 1999-2018',
                     'bashtage/arch', dict(time='Date', price='Adj Close', time_format='%m/%d/%Y')),
    'goog': Sample(_BT.format('GOOG'), 'equity', 'daily', 'Google stock, daily, 2004-2013', 'kernc/backtesting.py',
                   dict(price='Close')),
    'index_5m': Sample('https://raw.githubusercontent.com/mementum/backtrader/master/datas/2006-min-005.txt',
                       'equity', 'intraday', 'equity index, 5-minute bars, January 2006 (21 sessions)',
                       'mementum/backtrader', dict(time=('Date', 'Time'), price='Close'), bar='5min'),
    'eur_d': Sample(_FED, 'fx', 'daily', 'USD per euro, daily, Fed H.10 noon rates, 1999-', 'datasets/exchange-rates',
                    dict(time='Date', price='Exchange rate', where={'Country': 'Euro'})),
    'jpy_d': Sample(_FED, 'fx', 'daily', 'JPY per USD, daily, Fed H.10 noon rates, 1971-', 'datasets/exchange-rates',
                    dict(time='Date', price='Exchange rate', where={'Country': 'Japan'})),
    'gbp_d': Sample(_FED, 'fx', 'daily', 'USD per GBP, daily, Fed H.10 noon rates, 1971-', 'datasets/exchange-rates',
                    dict(time='Date', price='Exchange rate', where={'Country': 'United Kingdom'})),
    'eurusd_1h': Sample(_BT.format('EURUSD'), 'fx', 'intraday', 'EUR/USD, hourly bars, Apr 2017 - Feb 2018',
                        'kernc/backtesting.py', dict(price='Close'), bar='1h'),
    'btc_d': Sample(_CM.format('btc'), 'crypto', 'daily', 'Bitcoin reference price, daily, 2010-', 'coinmetrics/data',
                    dict(time='time', price='PriceUSD')),
    'eth_d': Sample(_CM.format('eth'), 'crypto', 'daily', 'Ether reference price, daily, 2015-', 'coinmetrics/data',
                    dict(time='time', price='PriceUSD')),
    'btc_1m': Sample('https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/data/updates/'
                     'btcusd_bitstamp_1min_latest.csv', 'crypto', 'intraday',
                     'Bitstamp BTC/USD, 1-minute bars, from Jan 2025 (first ~8 MB: about 80 days)',
                     'ff137/bitstamp-btcusd-minute-data', dict(time='timestamp', price='close', unit='s'),
                     max_bytes=8_000_000, bar='1min'),
}


def cache_dir(path=None) -> Path:
    p = Path(path or os.environ.get('STOCHASTICS_DATA') or Path.home() / '.cache' / 'stochastics' / 'data')
    p.mkdir(parents=True, exist_ok=True)
    return p


def samples() -> pd.DataFrame:
    """The built-in sample sets: name, asset class, frequency, description and where each comes from."""
    return pd.DataFrame([{'name': k, 'asset': s.asset, 'kind': s.kind, 'description': s.description,
                          'source': s.owner, 'url': s.url} for k, s in SAMPLES.items()]).set_index('name')


def fetch(url: str, dest, max_bytes: int | None = None, timeout: float = 60.0, refresh: bool = False) -> Path:
    """Download ``url`` to ``dest`` (once).  ``max_bytes`` fetches only the start of a large file and drops the
    last, possibly cut, line."""
    dest = Path(dest)
    if dest.exists() and not refresh:
        return dest
    req = urllib.request.Request(url, headers={'User-Agent': 'stochastics-lab'})
    if max_bytes:
        req.add_header('Range', f'bytes=0-{int(max_bytes) - 1}')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(int(max_bytes)) if max_bytes else resp.read()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f'could not download {url} ({type(exc).__name__}: {exc}). Download it by hand and either put it at '
            f'{dest} or read it with realdata.load_csv(...).') from exc
    if max_bytes and b'\n' in data:
        data = data[:data.rindex(b'\n') + 1]
    tmp = dest.with_suffix(dest.suffix + '.part')
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest


def sample(name: str, cache=None, refresh: bool = False, start=None, end=None, last: int | None = None) -> RealSeries:
    """One of the built-in public sample sets as a :class:`RealSeries` (see :func:`samples`)."""
    if name not in SAMPLES:
        raise KeyError(f'unknown sample {name!r}; choose from {sorted(SAMPLES)}')
    s = SAMPLES[name]
    ext = '.csv.gz' if s.url.endswith('.gz') else '.csv'
    path = fetch(s.url, cache_dir(cache) / f'{name}{ext}', s.max_bytes, refresh=refresh)
    series = load_csv(path, name=name, asset=s.asset, kind=s.kind, bar=s.bar, **s.read)
    series.source = f'{s.owner}: {s.url}'
    if series.kind == 'intraday' and s.bar and series.bar != s.bar:
        series.bar = s.bar
    if start is not None or end is not None or last is not None:
        series = series.window(start, end, last)
    return series
