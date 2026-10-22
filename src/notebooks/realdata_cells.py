"""Real-data sections appended to notebooks 06 (daily data) and 07 (intraday data and the agent-based versions).

Imported by build_notebooks.py.  The narrative cells that quote results are in RESULTS_06 / RESULTS_07 at the bottom
(they are written after the notebooks have been executed and the numbers read).
"""
import nbformat as nbf


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


# =====================================================================================================
# notebook 06: daily data, versions 1-4 and three standard baselines
# =====================================================================================================
def daily_cells(results):
    return [
md(r"""
# Real data: how closely do these models behave like real markets?

Up to here the models were judged by eye. This section scores them against real price histories: equities, FX and
crypto. The question is not "does the path look plausible" but "does the model reproduce the *statistical
behaviour* of a real market, and what does each extra piece of realism cost in run time?"

**Method.**

1. **Fingerprint.** Every series, real or simulated, is reduced to twelve scale-free numbers (returns are
   standardised first, so the level of volatility never matters): skewness, kurtosis, the Hill tail index and a
   tail-to-body quantile ratio (fat tails); `acf(r)` at lag 1 (momentum or reversal); `acf(|r|)` at lags 1, 5 and 20
   (volatility clustering and how slowly it fades); the leverage correlation between a return and the next squared
   return; the dispersion of log realised volatility (vol of vol); and variance ratios at 5 and 20 bars
   (< 1 means mean-reverting, > 1 trending). Sessions (overnight and weekend gaps) are respected: nothing is paired
   across a gap.
2. **Error in noise units.** For each fact: (model mean over many simulated series of the same length − real) ÷ the
   real value's own sampling noise (a block bootstrap of the real series). An error of 1 means "one noise unit
   away", and **below about 2 means the model cannot be told apart from the data on that fact**. `rms error` is the
   root mean square of the twelve (each capped at 5). The same yardstick applies to every model, so a vague model
   is not rewarded for being vague.
3. **p-value.** Is the real fingerprint a plausible draw from the model? (rank of the real series' squared
   z-distance among the replicates'). With 24 replicates the smallest possible value is 1/25 = 0.04, so "0.04"
   means "rejected as far as this test can say".
4. **Calibration.** Every model with free parameters is *tuned to each series* by simulated method of moments
   (random starts, then Nelder–Mead on the `rms error`, with the same random numbers at every step so the objective
   is smooth), and then **re-scored on fresh random numbers**, so the score is not flattered by the search.
   Models are compared at their best, not at arbitrary defaults. Parameters are scale-free (half-lives in bars,
   jump sizes in units of one bar's volatility), so one model runs at daily and at 5-minute resolution.

Eight candidates: three standard baselines (**Gaussian**, **Student-t** and **GJR-GARCH(1,1)-t**, the practitioner's
benchmark), versions 1–4 of this notebook, and one extra: **version 3 with a stochastic-volatility internal value**
(`TradersSV`), which asks whether version 3 fails because of its traders or because of its Gaussian value. Versions 5–7 (the agent-based exchange) are scored in notebook 07,
because they need intraday data.
"""),
code(r"""
import time
from stochastics.finance import realdata, benchmark as bm

QUICK = False                      # True: a few-minute version with small budgets (rougher numbers)
REPS, BUDGET, CAL_REPS = (8, 24, 4) if QUICK else (24, 70, 6)
N_JOBS = 2

realdata.samples().loc[['sp500', 'goog', 'eur_d', 'jpy_d', 'btc_d', 'eth_d'], ['asset', 'description', 'source']]
"""),
md(r"""
The sample sets are public files on GitHub, downloaded on first use and cached (`~/.cache/stochastics/data`, or
the folder in the `STOCHASTICS_DATA` environment variable). Nothing is bundled with the package, and each file
keeps its own licence. They are small, free samples chosen because they are reachable without an account: not
the best data money can buy. To use your own data: `realdata.load_csv('file.csv', asset='equity')`, or
`realdata.yahoo('SPY', start='2015-01-01')` (needs `pip install yfinance`; not used here).
"""),
code(r"""
series = {
    'sp500': realdata.sample('sp500'),
    'goog':  realdata.sample('goog'),
    'eur_d': realdata.sample('eur_d').window(last=5000),
    'jpy_d': realdata.sample('jpy_d').window(last=5000),
    'btc_d': realdata.sample('btc_d').window(last=3500),
    'eth_d': realdata.sample('eth_d').window(last=3500),
}
pd.DataFrame({n: s.summary() for n, s in series.items()}).T
"""),
md(r"""
### What real markets look like (the fingerprints)
"""),
code(r"""
real_facts = pd.DataFrame({n: s.fingerprint() for n, s in series.items()})
real_facts.round(2)
"""),
md(r"""
### The eight candidates and the parameters the search may tune

Each model gets a handful of free parameters with bounds. Bounds matter: a fit that ends *on* a bound is telling you
that the model wants more than it is allowed (the tables below flag these).
"""),
code(r"""
models = [bm.Gaussian(), bm.StudentT(), bm.GARCH(), bm.Heston(), bm.Ensemble(), bm.Traders(), bm.TradersSV(), bm.News()]
pd.DataFrame({m.label: {k: f'{p.low:g} to {p.high:g}' + (' (log)' if p.log else '') for k, p in m.space.items()}
              for m in models}).T.fillna('none')
"""),
md(r"""
### Step 1 — one series in detail: S&P 500, defaults against tuned

`bm.compare(series, models)` scores the models as they come; `calibrate_budget=...` tunes each one first.
"""),
code(r"""
sp = series['sp500']
t0 = time.time()
default = bm.compare(sp, models, replicates=REPS)
tuned = bm.compare(sp, models, replicates=REPS, calibrate_budget=BUDGET, cal_replicates=CAL_REPS)
print(f'{time.time() - t0:.0f} s for both scoreboards')
cols = ['rms error', 'facts within 2', 'p-value', 'seconds / replicate', 'worst fact']
print('Defaults:')
display(default.table[cols].round(3))
print('Tuned to the S&P 500 (scored on fresh random numbers):')
display(tuned.table[cols + ['fit seconds', 'fit rms (search)', 'at bounds']].round(3))
"""),
code(r"""
tuned.plot()
"""),
md(r"""
Fact by fact for the best tuned model: the dots are the individual simulated series, the diamond their mean; the
grey band is ±1 noise unit.
"""),
code(r"""
best = tuned.table.index[0]
tuned.assessments[best].plot()
tuned.assessments[best].table().round(3)
"""),
md(r"""
### Step 2 — all six series, all eight candidates, each tuned to each series
"""),
code(r"""
t0 = time.time()
sw = bm.sweep(series, models, n_jobs=N_JOBS, replicates=REPS, calibrate_budget=BUDGET, cal_replicates=CAL_REPS)
print(f'{time.time() - t0:.0f} s')
sw.plot()
"""),
code(r"""
print('rms error by series:')
display(sw.rms.round(2))
print('mean rms error by asset class:')
display(sw.by_asset().round(2))
print('facts matched to within 2 noise units (of 12):')
display(sw.grid('facts within 2').astype(int))
print('p-value that the real series is a draw from the tuned model (0.04 = rejected):')
display(sw.p_values.round(2))
"""),
md(r"""
Which facts are hard? Average signed error over the six series (red: the model produces more of it than the data;
blue: less).
"""),
code(r"""
sw.plot_facts(signed=True)
"""),
md(r"""
What the fits chose, and where the search hit the edge of the allowed range:
"""),
code(r"""
display(sw.parameters().round(2))
print('parameters that ended on a bound:')
display(sw.grid('at bounds').replace('', '-'))
"""),
md(r"""
### Step 3 — a test the fit cannot flatter: tune on the first half, score on the second

Everything above tunes each model to a series and scores it on *the same series* (with fresh random numbers). A
model with more knobs has more ways to fit the period it was tuned on. The held-out test removes that advantage:
tune on the first half of the history, then score on the second half, which the search never saw. One series per
asset class; the S&P 500's first half contains 1999–2008 (the dot-com bust and the financial crisis), its second
half the long recovery.
"""),
code(r"""
t0 = time.time()
ho = bm.holdout({k: series[k] for k in ('sp500', 'eur_d', 'btc_d')}, models, n_jobs=N_JOBS, replicates=REPS,
                calibrate_budget=BUDGET, cal_replicates=CAL_REPS)
print(f'{time.time() - t0:.0f} s')
display(ho.round(2))
print('mean over the three series (rms on the second half):')
display(ho.groupby(level='model')[['rms: fitted on 1st half', 'rms: fitted on 2nd half', 'loss from transfer']].mean()
        .reindex([m.label for m in models]).round(2))
"""),
md(results['daily_holdout']),
md(results['daily_findings']),
md(r"""
## Speed against accuracy

Three separate questions: which *model* gives the most accuracy per second; which *fidelity setting* of a model
(how many traders) is enough; and how much *fitting* and how many *replicates* are worth paying for.
Cost is wall-clock seconds per simulated series of the real series' length on one core (the cost of fitting
is reported separately). The knee is the cheapest setting within 15% (or one standard error) of the best error.

### 1. Across the eight candidates
"""),
code(r"""
ca = sw.cost_accuracy()
display(ca.round(3))
bm.plot_frontier(ca, title='Eight ways to simulate a market: mean over the six real series')
"""),
md(r"""
### 2. How many traders? (`streams` in versions 3 and 4)

The tuned S&P 500 parameters, with the number of traders varied. Because the benchmark parametrises the *crowd's*
average deviation (each trader's own deviation is scaled by √streams), more traders only samples the same crowd
more finely: it should cost more and change little.
"""),
code(r"""
frs = {}
for base in (tuned.assessments['v3 traders'].candidate, tuned.assessments['v4 news'].candidate):
    fr = bm.frontier(sp, [base.variant(f'{base.label}: {n} traders', streams=n) for n in (10, 30, 100, 300, 1000)],
                     replicates=REPS)
    frs[base.label] = fr
    display(fr.round(3))
    bm.plot_frontier(fr, title=f'{base.label}: number of traders, S&P 500')
"""),
md(r"""
### 3. How much fitting is enough?

Each model is tuned for a budget of up to 160 simulated-moment evaluations. For several budgets the best
parameters found so far are re-scored on fresh random numbers (solid), next to the error the search itself saw
(dashed): the gap between the two is the search fitting its own noise.
"""),
code(r"""
BIG = 40 if QUICK else 160
curves = {}
for m in (bm.StudentT(), bm.GARCH(), bm.Heston(), bm.Ensemble(), bm.Traders(), bm.TradersSV(), bm.News()):
    fit = bm.calibrate(m, sp, budget=BIG, replicates=CAL_REPS, seed=11)
    curves[m.label] = bm.budget_curve(fit, sp, checkpoints=(1, 5, 10, 20, 40, 80, 160), replicates=REPS)
bm.plot_budget(curves)
pd.concat(curves, names=['model']).round(2)
"""),
md(r"""
### 4. How many replicates?

The `rms error` of a model is an average over simulated series, so it has its own simulation noise (`rms se`). How
fast does it shrink?
"""),
code(r"""
conv, secs = {}, {}
for name in ('GARCH-t', 'v1 Heston', 'v4 news'):
    a = bm.assess(sp, tuned.assessments[name].candidate, replicates=60, seed=5)
    conv[name] = bm.convergence(a, sizes=(2, 4, 8, 16, 32, 60))['sd of rms error']
    secs[name] = a.seconds
display(pd.DataFrame(conv).round(3))
print('seconds per replicate:', {k: round(v, 3) for k, v in secs.items()})
"""),
md(results['daily_speed']),
]


# =====================================================================================================
# notebook 07: intraday data and the agent-based versions
# =====================================================================================================
def intraday_cells(results):
    return [
md(r"""
# Real intraday data: how close are versions 5–7 to a real trading day?

Notebook 06 scored versions 1–4, a stochastic-volatility variant of version 3 and three standard baselines against daily data. Here the same benchmark
(twelve scale-free facts; error in units of the real series' own sampling noise; see notebook 06 for the method) is
applied to intraday data, where the agent-based exchange can be compared with real bars.

* **Real data**: a 5-minute equity index (21 January-2006 sessions of 102 bars), 5-minute Bitstamp BTC/USD bars
  (10 days resampled from 1-minute data), and hourly EUR/USD (April 2017 – February 2018). There is no free
  minute-level FX in the sample sets, so **FX is scored for versions 1–4 and the baselines only**; the exchange has
  an 8.5-hour day and is compared with the equity index and with crypto, which trades round the clock.
* **Exchange candidates**: `'agents'` (version 5: noise, random-limit, maker and whale background plus 400 utility
  traders), `'brief'` (version 6), `'profiles'` (version 7: 400 psychology-profile traders) and the website's
  `'game'` population. They are assessed **as designed, not tuned**: they have too many parameters and a day costs
  seconds. The quick models are tuned, so the comparison favours them. Keep that in mind when reading it.
* A simulated day costs seconds, so each exchange candidate is simulated once into a pool of independent days, and
  replicates are drawn from the pool by resampling days. This understates the model's own spread a little (the
  p-values are slightly optimistic about rejecting), but the cost per replicate is reported as the real cost of the
  days a replicate needs.
"""),
code(r"""
import time
from stochastics.finance import realdata, benchmark as bm

QUICK = False                         # True: a few-minute version with small budgets (rougher numbers)
REPS, BUDGET, CAL_REPS = (8, 24, 4) if QUICK else (24, 70, 6)
POOL = 6 if QUICK else 24             # simulated days per exchange candidate
N_JOBS = 2

ix = realdata.sample('index_5m')
btc = realdata.sample('btc_1m').resample('5min').window(last=288 * 10)
btc.name = 'btc_5m'
fx = realdata.sample('eurusd_1h').sessions(120)       # weekly stretches cut into sessions of 120 hours
intraday = {'index_5m': ix, 'btc_5m': btc, 'eurusd_1h': fx}
pd.DataFrame({n: s.summary() for n, s in intraday.items()}).T
"""),
md(r"""
### Intraday fingerprints next to their daily counterparts

Intraday returns are more clustered and heavier-tailed than daily ones, and carry a time-of-day pattern in volatility
(the opening and closing hours are busier) that the fingerprint does not remove.
"""),
code(r"""
daily = {'sp500 (daily)': realdata.sample('sp500'), 'btc_d (daily)': realdata.sample('btc_d').window(last=3500),
         'eur_d (daily)': realdata.sample('eur_d').window(last=5000)}
facts = pd.DataFrame({**{n: s.fingerprint() for n, s in intraday.items()}, **{n: s.fingerprint() for n, s in daily.items()}})
facts[['index_5m', 'sp500 (daily)', 'btc_5m', 'btc_d (daily)', 'eurusd_1h', 'eur_d (daily)']].round(2)
"""),
md(r"""
### Step 1 — baselines and versions 1–4, tuned to each intraday series
"""),
code(r"""
quick_models = [bm.Gaussian(), bm.StudentT(), bm.GARCH(), bm.Heston(), bm.Ensemble(), bm.Traders(), bm.TradersSV(), bm.News()]
t0 = time.time()
sw_q = bm.sweep(intraday, quick_models, n_jobs=N_JOBS, replicates=REPS, calibrate_budget=BUDGET, cal_replicates=CAL_REPS)
print(f'{time.time() - t0:.0f} s')
sw_q.plot()
display(sw_q.rms.round(2))
display(sw_q.parameters().round(2))
"""),
md(r"""
### Step 2 — the agent-based exchange: versions 5, 6, 7 and the web game

Each candidate is simulated once into a pool of independent trading days (5-minute bars from 15-second ticks), using
both cores. The table shows what a day costs.
"""),
code(r"""
exchange_models = [
    bm.Exchange(population='agents',   pool_days=POOL, label='v5 agents'),
    bm.Exchange(population='brief',    pool_days=POOL, label='v6 brief'),
    bm.Exchange(population='profiles', pool_days=POOL, label='v7 profiles'),
    bm.Exchange(population='game',     pool_days=POOL, label='web game'),
]
t0 = time.time()
for m in exchange_models:
    m.prepare(n_jobs=N_JOBS)
print(f'{time.time() - t0:.0f} s to simulate {POOL} days for each of the four populations')
pd.DataFrame({m.label: {'agents': sum(m._mix().values()), 'seconds per day (one core)': m.seconds_per_day,
                        'returns per day': len(m.pool()[0][0])} for m in exchange_models}).T.round(2)
"""),
code(r"""
sw_x = bm.sweep({'index_5m': ix, 'btc_5m': btc}, exchange_models, replicates=REPS)
sw_all = sw_q.only(['index_5m', 'btc_5m']) + sw_x            # twelve candidates on the two series both can face
sw_all.plot()
display(sw_all.rms.round(2))
cols = ['rms error', 'facts within 2', 'p-value', 'seconds / replicate', 'worst fact']
for n in ('index_5m', 'btc_5m'):
    print(n)
    display(sw_all.boards[n].table[cols].round(3))
"""),
code(r"""
sw_all.plot_facts(signed=True)
"""),
md(r"""
Fact by fact for the exchange populations against the equity index (dots: individual replicates; diamond: mean;
grey band ±1 noise unit):
"""),
code(r"""
sw_all.boards['index_5m'].assessments['v6 brief'].plot()
sw_all.boards['index_5m'].assessments['v6 brief'].table().round(3)
"""),
md(results['intraday_findings']),
md(r"""
## Speed against accuracy, intraday

### 1. All twelve candidates, two series
"""),
code(r"""
ca = sw_all.cost_accuracy()
display(ca.round(3))
bm.plot_frontier(ca, title='Twelve ways to simulate a market: mean over the equity index and BTC (5-minute bars)')
"""),
md(r"""
### 2. What does a bigger, finer exchange buy?

Version 6 (`'brief'`) with its population scaled by 0.25, 0.5 and 1, and with the engine's time step set to 15, 30
and 60 seconds (the 5-minute bars are the same). Scored against the equity index.
"""),
code(r"""
KPOOL = 4 if QUICK else 16
knob = [bm.Exchange(population='brief', scale=sc, sim_secs_per_tick=tk, pool_days=KPOOL, label=f'x{sc:g}, {tk}s tick')
        for sc in (0.25, 0.5, 1.0) for tk in (15, 30, 60)]
t0 = time.time()
for m in knob:
    m.prepare(n_jobs=N_JOBS)
print(f'{time.time() - t0:.0f} s to build the nine pools')
fr = bm.frontier(ix, knob, replicates=REPS)
display(fr.round(3))
bm.plot_frontier(fr, title='Version 6: population size and time step, against the equity index')
"""),
md(r"""
### 3. How many simulated days are enough?

The first `k` days of the version-6 pool, scored against the equity index (the pool's days are the same, so this
costs nothing extra).
"""),
code(r"""
base = exchange_models[1]
rows = {}
for k in (2, 4, 8, 16, POOL):
    a = bm.assess(ix, base.variant(pool_days=k), replicates=REPS)
    rows[k] = {'rms error': a.rms_error, 'rms se': a.rms_se(), 'p-value': a.p_value, 'facts within 2': a.within,
               'seconds to simulate': k * base.seconds_per_day}
pd.DataFrame(rows).T.rename_axis('days in pool').round(3)
"""),
md(results['intraday_speed']),
]
