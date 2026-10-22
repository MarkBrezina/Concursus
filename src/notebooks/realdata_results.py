"""Narrative cells for the real-data sections of notebooks 06 and 07.

They quote numbers from the executed notebooks (seeded, so a re-run reproduces the errors; run times vary with the
machine).  If you change settings (QUICK, REPS, BUDGET) or the data, re-read the tables before trusting the text.
"""
RESULTS = {}

RESULTS['daily_holdout'] = r"""
**Reading the held-out table.** The models that cannot produce volatility clustering (Gaussian, Student-t, version 2,
version 3) score 3–4 whether or not the fit saw the second half: tuning could not help them. The four that can
(GARCH-t, Heston, version 3 with a stochastic-volatility value, version 4) score about **1.9–2.1 on data they never
saw**, and **the differences among those four are inside the noise** (each entry has a simulation standard error of
0.1–0.2, and there are only three series). The full-sample ranking above put versions 3-SV and 4 ahead of Heston and
GARCH-t; under hold-out that lead shrinks to 0.1–0.2 and cannot be called.

The transfer loss says how much of a fit was the period rather than the market: version 4 loses almost nothing
(0.04), GARCH-t a little (0.15), Heston and version 3-SV about 0.5, flexible fits that partly describe the years they
were tuned on. Some single entries come out negative (a model tuned on the first half scoring better than one tuned on
the second): that is the search's own noise, and a reminder of how fine this test can resolve.
"""

RESULTS['daily_findings'] = r"""
## What the daily data say

**1. Two groups, and the line between them is volatility clustering.** Gaussian (mean rms error 3.86 over the six
series), Student-t (3.20), version 2 (3.40) and version 3 (3.75) have no volatility clustering: their errors on the
vol-of-vol and `acf(|r|)` facts are 4 to 10 noise units. GARCH-t (2.03), Heston (1.94), version 4 (1.61) and
version 3 with a stochastic-volatility value (1.58) have it. Heston, version 4 and version 3-SV match 7–12 of the twelve facts on every series; GARCH-t 4–11. The
split is the same in equities, FX and crypto, and it survives the hold-out. Fat tails alone are not enough: Student-t
fixes the tails and is still far away.

**2. Versions 2 and 3 fail for structural reasons, not because they were badly tuned.** No amount of fitting helps
them (see the budget curves below). Averaging independent streams (version 2) washes out the volatility structure: five of
six fits push the common-factor `correlation` to its upper bound, so the model is trying to turn itself back into a
single stream. Version 3 fails because its internal value is a Gaussian path: put a stochastic-volatility value
under the *same* traders and its error drops from 3.75 to 1.58, the best average of the eight. Whether the traders layer
adds anything beyond Heston is suggested (1.58 against 1.94 on the full series, 1.88 against 2.05 held-out) but not shown.

**3. Among the four that work there is no stable winner.** Version 4 is best on four of the six series, version 3-SV on
the S&P 500, and a plain GARCH-t on EUR/USD (1.06, 11 of 12 facts matched; the calmest series is the one that needs the
least machinery). On the full series version 4 and 3-SV lead Heston and GARCH-t by 0.3–0.5; under hold-out the four are tied.
Version 4 has five free parameters, GARCH-t four, Heston three, Student-t one, and the scoreboard does not
penalise parameters: read the hold-out, not the headline ranking, when choosing between them.

**4. What nobody gets right.** In the fact-by-fact heatmap every model produces too little negative skew (errors of +0.9 to +2.2 noise units), too
little leverage (a negative return followed by higher volatility; +0.2 to +2.5) and a lag-1 return autocorrelation that is a little too high
(+0.4 to +1.1): the models are too symmetric. The clustering models also get the *shape* of volatility memory wrong: `acf(|r|)` at lag 1 too
high (GARCH-t, Heston and 3-SV by 2–3 noise units) and at lag 20 too low (by 1–2): one exponential decay cannot match
what looks like a longer memory. Version 4's Hawkes clock is closer at lag 1 (+1.1) but still 1.4 too low at lag 20. Version 4's remaining
errors are on skew, leverage and the 20-bar variance ratio, which is a concrete place to add structure (for
example, a reaction to news that depends on the sign of the jump).

**5. No asset class is consistently harder.** The ranking of models is the same in equities, FX and crypto, and a model's
mean error differs by at most 0.4 between classes, less than the differences between models. The hardest single series is
the S&P 500 (best score 1.7): 1999–2018 contains the dot-com bust and the 2008 crisis, so it is a mixture of regimes.

**6. What the fits chose.** Version 4's fits agree: Hawkes branching of about 0.5, news roughly every 10–25 bars, jumps
of about four bar-volatilities, a crowd deviation of 3–5 bar-volatilities. On the S&P 500 and both crypto series the news rate ran to its upper limit (one event every ten bars), and on the S&P 500 the
jump size to its lower limit, so there the "news" is acting as a clustered-volatility generator rather than as discrete news. Take parameters that end on a bound as a statement about the model's structure,
not as estimates.

**Limits.** Six small public samples; each fit is tuned to twelve facts with about 70 evaluations; a model with more free
parameters can fit more. The p-value test is lenient to *wide* models: version 2 is not rejected on five of six series even
though its error is 3.4, because its tuned jumps make its own spread huge. That is why the error in noise units is the
main yardstick and the p-value a secondary one. At the 5% level (p = 0.04 is the smallest possible value here) version 4
and GARCH-t are each rejected on one series, version 3-SV on two (both crypto), Heston on three, and no model is accepted on every series.
"""

RESULTS['daily_speed'] = r"""
## What the speed–accuracy study says

**Across models.** At daily resolution accuracy is almost free: the slowest candidate needs 0.08 s for a 5,000-day
series. The Pareto-optimal models are Gaussian, Student-t, GARCH-t, version 4 and version 3-SV; Heston is beaten on both
cost and error by version 4. The knee is **version 4** (1.61 at 0.05 s): GARCH-t is ten times faster and 26% worse,
and version 3-SV is 2% better for 1.8 times the cost, which is inside the noise. In practice the cost that matters is
the one-off tuning, 2 s for GARCH-t and 20–40 s for versions 2, 3-SV and 4 per series.

**Number of traders.** More traders buy nothing. With the crowd's average deviation held fixed, the error of version 4 is
flat from 10 to 1,000 traders (1.92 to 2.04, noise about ±0.06) and version 3's is flat at 3.94–3.96, while run time grows
6–7 times (0.03 to 0.25 s). The knee is 30 traders. This is a statement about the *aggregate price*; it says nothing
about what individual traders would earn, where more of them would matter.

**Amount of fitting.** Error falls in steps, not smoothly. Heston, version 4 and version 3-SV need roughly 30–50 s (about 80
evaluations) and then plateau; more search is wasted. GARCH-t gains slowly and noisily (2.6 to 2.3 in about 5 s).
Student-t, version 2 and version 3 gain nothing at any budget, which is the budget-curve version of the structural failure in
finding 2. The search's own score and the fresh-seed score usually agree within noise, but at an unlucky checkpoint they
can differ by 0.6 (3-SV at 40 evaluations: 2.01 searched, 2.64 on fresh numbers). Taking the best of many noisy
evaluations flatters it, which is why every score in this notebook is re-computed on new random numbers.

**Number of replicates.** The standard error of the rms error shrinks like 1/√k: 16 replicates give about ±0.07 to
±0.10 for Heston and version 4 but ±0.23 for GARCH-t, whose fingerprint statistics are heavier-tailed; 60 give ±0.04 and ±0.11. So a
gap below about 0.2 between two models at 24 replicates is not a real difference: Heston against GARCH-t above is such a
case.

**Bottom line.** For matching the stylised facts of a daily price series: version 4 or version 3-SV with about 30
traders, about 80 fitting evaluations and at least 24 replicates (a minute or two per series), or GARCH-t when ten times
the speed matters more than a quarter of the error. Choosing among the four that work needs the hold-out test, not the
leaderboard.
"""

RESULTS['intraday_findings'] = r"""
## What the intraday data say

**1. The same two groups appear at 5-minute and hourly resolution.** Averaged over the three series, Gaussian (3.59),
Student-t (2.69), version 2 (3.52) and version 3 (3.51) have no volatility clustering and sit far from the data; Heston (1.80),
GARCH-t (1.84), version 4 (1.49) and version 3 with a stochastic-volatility value (1.44) have it. As on the daily data, the
Gaussian value inside version 3 is what fails: replace it and the error falls from 3.5 to 1.4. There is again no stable winner among the four that work:
version 4 is best on the equity index (1.17), version 3-SV on BTC (1.10), and a plain GARCH-t on hourly EUR/USD (1.12), the calmest series, where
the simplest model is enough. The series are short (21 sessions, 10 days, 42 weekly stretches) and nothing here was held out,
so differences of a few tenths among those four are not reliable.

**2. The exchange, untuned, lands about level with a tuned GARCH-t.** On the equity index versions 6 (2.25) and the web game (2.13)
are close to GARCH-t (2.34) and well ahead of the models with no volatility clustering (3.2–3.9); on BTC they are 2.11 and 1.58 against GARCH-t's 2.05.
They are behind tuned Heston, version 3-SV and version 4 on the index (1.2–1.5). On BTC the web game (1.58) is level with Heston and version 4 (1.70, 1.73) and version 6 is 0.4 behind them.
The catch is how much has been tuned: the quick models were fitted to each series with about 70 evaluations,
the exchange populations were run as designed. Nobody knows what tuning would do for the exchange, because one evaluation costs minutes (see the speed section).
Versions 5 and 7 are level with each other (2.7 and 2.6 on both series) and version 6 improves on them by 0.4–0.5, so the
step from the 'agents' population to the 'brief' one is a measurable gain, and the profile-based population of version 7, a first draft, has not yet
earned one. On the formal test no exchange version is accepted on either series (p = 0.04 is the smallest value possible here); on the index Heston,
3-SV and version 4 are accepted, and on BTC every model except GARCH-t is rejected, so the test cannot separate them there.

**3. Volatility clustering comes out of the trading rules.** None of it is put in as a volatility process, yet the exchange versions are within
about 2.4 noise units on the lag-1, lag-5 and lag-20 autocorrelation of absolute returns and on vol of vol (versions 6 and 7 and the web game; version 5's
lag 1 is 3.6 too high), where the models without clustering are 4.5–7.4 off on lags 1 and 5 and on vol of vol. It is a little weak at long
lags: `acf(|r|)` at lags 5 and 20 is up to 2.4 noise units too low in versions 5–7.

**4. Where the exchange differs, and the differences have different signs.** Averaged over the index and BTC, version 5 has a lag-1 return autocorrelation
that is far too *negative* (−9 noise units) and variance ratios that are too low (−6.4 at 5 bars): it over-reverts, the signature of price bouncing between bid and ask. Version 7 is the mirror image: lag-1 too *positive* (+5.1), variance ratios too high (+6.2 at 5 bars, +3.2 at 20): it
over-trends. Versions 6 and the web game over-trend at 20 bars (+3.0 and +4.3). For version 6 this is mostly the equity index, whose real 20-bar
variance ratio is 0.65 (+5.6 noise units on the index, about +0.4 on BTC, where the real value is 1.09): the exchange fits crypto's momentum better than
equities' mean reversion. Versions 6 and 7 and the web game have tails that are too thin by the Hill index (+3.1, +1.9, +1.5 noise units; version 5 does not) and version 6's
log kurtosis is 2.3 too low. These are concrete targets: the populations' mix of trend-following and mean-reverting rules is what to adjust,
and version 5 against version 7 shows the two directions it can be wrong in.

**Limits.** Three small public samples (one month of an equity index in 2006, ten days of BTC in 2025, hourly EUR/USD); no hold-out on
intraday data because the series are too short to split; the time-of-day pattern in volatility is not removed from the fingerprint (the exchange has an
intraday schedule, the quick models do not); the exchange's day is 8.5 hours while BTC trades around the clock; one pool of simulated days per population.
The `rms se` column only resamples days within the pool; the pool-size study below suggests that a pool's own sampling error is of the order of 0.1–0.2,
so exchange-to-exchange gaps below that, such as version 6 against the web game, cannot be called.
"""

RESULTS['intraday_speed'] = r"""
## What the intraday speed–accuracy study says

**Across models.** The quick models cost 0.002–0.05 s per replicate and the exchange 160–240 s (one core): **3,000 to 8,500 times more than versions 3-SV
and 4, and 27,000 to 39,000 times more than GARCH-t**. The Pareto-optimal candidates are Gaussian, Student-t, GARCH-t, version 3-SV and version 4, and the
knee is **version 4** (error 1.45 at 0.03 s, averaged over the index and BTC). No exchange population is on the frontier: the best of them (the web game, 1.85)
is worse than version 4 and 6,000 times as expensive, and version 6 (2.18) is level with GARCH-t (2.20) at about 27,000 times the cost. Counting the
tuning of the quick models does not change this (the whole sweep, eight models tuned on three series, took about four minutes). The exchange is not the efficient way to reproduce these
statistics; what it offers instead, individual traders with their own books, is something the quick models cannot say anything about.

**Time step.** A 60-second engine step is about 4 times cheaper than 15 seconds with no loss of accuracy: at the full population 2.27 against 2.33 (the difference is inside the noise), and
2.33 against 2.48 at half the population. The 5-minute bars are the same, only the number of trading rounds inside each changes.

**Population size.** Cost falls less than the population: 250 agents (scale 0.25) cost 2.3 to 2.6 times less per day than 1,000, not 4, so there is a fixed
cost per step. Accuracy falls slowly: at a 60-second step the errors are 2.53, 2.33 and 2.27 for 250, 500 and 1,000 agents. The **knee is
250 agents at a 60-second step** (2.53 at 14 s per replicate against 2.33 at 135 s for the default). Half the population at 60 s (2.33 at 21 s) matches the
default's accuracy at about a sixth of its cost. The settings are not independent: 250 agents at a 15-second step is an outlier (3.67), far worse than the same
population at 30 seconds (2.51). A six-day check shows why in the facts: its variance ratios are far too high (about 2.6 at 20 bars against 0.65 in the data), so it trends
too strongly. I did not look for the cause, so treat a knob that looks harmless as untested until its fingerprint has been looked at.

**Days in the pool.** The score is poor with few days and settles: 3.48 with 2 days, 2.86 with 4, 2.44 with 8, 2.33 with 16, 2.25 with 24 (and 3, 6, 8, 8, 8 of the twelve facts within 2 noise units).
**At least 8 days are needed, 16 is safer**: going from 8 to 24 days buys 0.19 for three times the cost, going from 2 to 8 buys 1.0. Small pools are bad because
a replicate built from a few days inherits their sampling noise. The 24-day pool behind the headline numbers is large enough, but not by a wide margin.

**Bottom line.** To match the statistics of a 5-minute series: tune version 4 or version 3-SV (a minute). To assess the exchange against real bars cheaply:
about half the default population, a 60-second step and 16 pooled days, 20 seconds of simulation per series in place of 2–3 minutes, with the caveat of one series and one pool seed.
Tuning the exchange itself with this method (70 evaluations of minutes each) would take hours per series, so it needs a cheaper surrogate before it is practical;
the calibration loop built for the quick models is the right tool once a day costs a second instead of six.
"""
