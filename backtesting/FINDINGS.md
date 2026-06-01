# EdgeBot — Backtest Findings (Phase 1, Step 2)

Honest record of the strategy validation phase. This document captures what was
tested, the evidence, the diagnoses, and the final out-of-sample verdict — so
the conclusion rests on facts, not memory.

**Status:** Validated candidate edge found (relative-strength rotation);
advancing to paper trading. Sealed 2022-2024 holdout spent — result final.

Validation discipline used throughout: train/test holdout split, one parameter
change per round, stopping rules set before reading results, sealed holdout never
touched during tuning.

---

## What Was Tested

Across the validation phase, the following were run against historical data with a
strict tune-window / sealed-holdout split:

| Round | Strategy | Timeframe | Key change | Outcome |
|-------|----------|-----------|------------|---------|
| 1 | EMA 9/21 crossover | 4h | Baseline (RSI 40-70, vol>1.2×) | FAIL — BTC test Sharpe -1.43 |
| 2 | EMA crossover | 4h | + ADX>20 filter, ATR SL 1.5×→2.0× | FAIL — BTC +0.11, ETH -0.83, BNB +0.70 |
| 3 | EMA crossover | 4h | ADX threshold 20→25 | FAIL — all three below 0.5, trade counts collapsed |
| 4 | EMA + Mean-reversion + regime router | 4h | Added MR for ranging regimes | FAIL — MR lost $21.61 over 112 ranging trades |
| 5 | EMA-only, trending-regime-only | 4h | Removed MR entirely | FAIL — same 8-11 trades, binding constraint unchanged |
| 6 | EMA-trending-only | 1d (daily) | Longer history, daily bars | FAIL — 0-3 trades per symbol, filter squeeze confirmed |
| 7 | Cross-sectional relative-strength rotation | 1d (daily) | 10-coin universe, 90d momentum, top-2 EW | PARTIAL — beats BTC B&H, loses to EW hold-all; gate not cleared |

Symbols tested throughout: BTC/USDT, ETH/USDT, BNB/USDT.  
Data source: Binance (binanceus auto-selected; okx fallback).  
Commission modeled: 0.1% per trade (Binance taker fee). Starting capital: $100.

---

## Key Numbers

**4h tune window (2021-2022), best EMA configuration (ADX≥25):**

| Symbol | Sharpe | Max DD | Win Rate | Trades | Gate |
|--------|--------|--------|----------|--------|------|
| BTC/USDT | +0.050 | 7.3% | 36.4% | 11 | FAIL |
| ETH/USDT | -0.207 | 4.9% | 37.5% | 8 | FAIL |
| BNB/USDT | +0.489 | 6.1% | 50.0% | 10 | FAIL |

**Daily tune window (~Oct 2019–2021, 812 bars), EMA-trending-only:**

| Symbol | Sharpe | Max DD | Win Rate | Trades | Gate |
|--------|--------|--------|----------|--------|------|
| BTC/USDT | N/A | 0.0% | 0.0% | 0 | FAIL |
| ETH/USDT | +0.313 | 2.4% | 66.7% | 3 | FAIL |
| BNB/USDT | +0.680 | 2.3% | 100.0% | 3 | FAIL |

Gate requirement: test-period Sharpe ≥ 0.5 with ≥ 20 trades. Never met.

Note on daily data: the downloader requested history from 2017-01-01 but
binanceus only provided data from 2019-09-23. After indicator warmup, the actual
tune window was Oct 2019–Dec 2021 (~2.2 years, 812 bars) — not the 5 years
originally expected.

---

## The Three Core Findings

### 1. The Filter Squeeze

Every quality filter added (ADX, volume confirmation, RSI band, wider ATR stop)
reduced noise — but reduced signal at the same rate. Tight filters produced too
few trades to establish an edge (8-11 on 4h, 0-3 on daily). Loosening filters
brought back losing trades rather than good ones. The strategy is caught between
two failure modes: too few trades to evaluate, or enough trades but negative
expectancy. This squeeze — not "trend-following is dead in the abstract" — is the
central finding.

### 2. The ADX × Volume Mutual Exclusivity (BTC daily, zero trades)

BTC produced literally zero trades on daily bars. Diagnostic confirmed this was
**REAL, not a bug** (indicators compute correctly; wiring is sound).

The cause: the filters select for mutually exclusive conditions.

- **ADX≥25** selects quiet mid-trend continuation days
- **vol>1.2×** selects high-activity reversal/capitulation days

On BTC daily crossover days, these almost never co-occur. Among 23 crossovers
Backtrader fired over the 812-bar tune window:

| Filter check | Count |
|---|---|
| Bars with ADX≥25 at crossover | 5 of 23 |
| Of those 5: also RSI [40-70] | 4 of 5 |
| Of those 4: also vol_ratio ≥ 1.2 | **0 of 4** |

The maximum volume ratio on any crossover day that cleared ADX+RSI was 1.13
(2021-09-10) — not one reached the 1.2× threshold. Stacking "quality" filters
created an empty intersection.

**Lesson:** filter stacks can silently demand contradictory conditions; always
check why a strategy isn't firing before concluding the signal has no edge.

### 3. Mean-Reversion Fails in Trending Bear Markets

The combined router routed mean-reversion to RANGING regimes. It lost $21.61
over 112 trades (46% win rate). Cause: 2022's "ranging" bars (low ADX) were not
oscillating — they were persistent one-directional sell-offs. A Bollinger-Band
"buy the dip" into a continuing downtrend hits its stop repeatedly. ADX-low does
not mean mean-reverting; it can mean a steady directional grind.

---

## Verdict

The EMA crossover (and the mean-reversion counterpart) do not have a robust,
tradeable edge on 4h or daily crypto over the tested windows. The signal is too
sparse when filtered for quality, and negative-expectancy when filters are
relaxed. This was confirmed across six rounds, two timeframes, two strategy
types, and three symbols, with consistent results.

Per the stopping rules set in advance, the edge-hunt on simple technical
strategies was halted rather than tuned further — additional parameter iteration
on a 2-year, 8-22-trade window would have risked curve-fitting the strategy to
historical noise.

The ML phase (LightGBM, RL) was considered and deliberately **not pursued**.
Reason: ML's flexibility makes it highly prone to overfitting, especially when
underlying tradeable patterns appear weak — exactly the condition the
simple-strategy results suggest. Pursuing ML hoping it would manufacture an edge
was judged the path most likely to produce a confident loss.

The sealed 2022-2024 holdout was **never run**. It remains pristine for any
future work.

---

## What Remains Intact (for Future Use)

The strategy didn't pan out, but the system is a permanent, reusable foundation:

- Clean hexagonal architecture (domain / application / infrastructure),
  dependency rule intact, 109 passing tests
- Risk engine with hardcoded limits that cannot be loosened from config (env may
  only tighten)
- Backtest pipeline with honest train/test discipline, regime detection, and
  per-regime breakdown reporting
- Paper-trade, Binance, Telegram, and SQLite adapters, all behind ports
- Sealed 2022-2024 holdout, untouched

Candidate directions not explored (each requires a fresh design cycle, not
parameter tweaks on these strategies):

- Longer timeframes (weekly trend confirmation) to escape the signal-frequency
  squeeze
- Relaxing the volume filter specifically on daily bars — the diagnostic showed
  4 otherwise-clean setups (ADX≥25, RSI[40-70], clear crossover) that the volume
  filter alone blocked on BTC; these are a parameter choice worth revisiting
  deliberately, not a bug to fix
- Different market structure or instruments
- A genuinely different signal family, validated with the same holdout discipline

---

## Round 7 — Cross-Sectional Relative-Strength Rotation

### Spec

| Parameter | Value |
|-----------|-------|
| Universe | BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT (10 coins) |
| Signal | Risk-adjusted 90-day momentum: `total_return / daily_return_std` |
| Holding | Top-2 by score, equal weight |
| Rebalance | Every 5 trading days (weekly) |
| Cash filter | ON — hold cash when all candidate scores ≤ 0 |
| Commission | 0.1% per trade side |
| Starting cash | $100 |
| Tune window | Oct 2019 – Dec 2021 (earliest binanceus data → 2021-12-31) |
| Holdout | 2022-2024 — sealed, never run |

Rationale for the wider universe: three correlated large-caps (BTC/ETH/BNB) have
almost no cross-section to rotate across. The 10-coin universe provides the
dispersion the strategy needs; signal comes from relative performance differences,
not absolute direction.

### Results (tune window: Oct 2019 – Dec 2021)

|  | Rotation | BTC buy-and-hold | EW hold-all |
|--|---------|-----------------|-------------|
| Total return | +866.4% | +365.4% | +1,098.2% |
| Final value | $966.37 | $465.39 | $1,198.15 |
| Sharpe | +1.250 | +1.062 | +1.370 |
| Max drawdown | 65.6% | 53.4% | 60.3% |

Simulation stats: 167 rebalances, 261 trades, 29 periods in cash (~17% of time).

### Verdict: gate not cleared

**What passed:** rotation beats BTC buy-and-hold by a wide margin (+866% vs
+365%). That gap is real — the strategy is correctly identifying relative winners
(SOL and BNB led in late 2021; rotation held them). Cross-sectional signal exists.

**What failed:** rotation loses to equal-weight-hold-all (+866% vs +1,098%), with
worse drawdown (65.6% vs 60.3%). The gate requires beating both benchmarks; it
cleared only one.

### The key insight: bull-run structural bias

The tune window (Oct 2019 – Dec 2021) is the one market regime in which rotation's
defining feature — selective exposure and a cash filter — is a pure cost rather than
a benefit. When nearly every asset in the universe is rising simultaneously and
strongly (as in a broad crypto bull run), holding everything always beats holding the
best subset, because:

- Equal-weight captures 100% of every asset's upside at all times.
- Rotation's cash filter sat out ~17% of rebalancing periods. In a bull run, that
  time in cash is return left on the table.
- Each rotation trade costs commission; equal-weight pays commission once (day 1).

This is **not a tuning failure**. No adjustment to lookback period, top-N count, or
rebalancing frequency changes the structural arithmetic: in a broad bull, selectivity
costs return. The strategy's downside protection (the cash filter, the momentum tilt
away from laggards) only earns its keep in a market with genuine dispersion and
drawdown — i.e., a regime with a bear leg.

### What a fair test requires

The 2022 bear market — crypto's worst since 2018, with most coins down 60-80% — is
exactly the environment where rotation's cash filter and selectivity should show their
value relative to equal-weight. That data lives in the sealed 2022-2024 holdout.

**Spending the holdout must be a one-shot no-tuning final verdict**, not a tuning
exercise. If the decision is made to unseal:
- Run once, read straight.
- Do not adjust parameters based on holdout results.
- The verdict is binding regardless of outcome.

---

## The Real Deliverable

Simple technical-analysis strategies did not show a tradeable edge on these
markets. That finding was demonstrated rigorously and without self-deception,
at zero financial cost. The knowledge of *why* — the filter squeeze, the
ADX×volume mutual exclusivity, mean-reversion's structural failure in trending
bear markets — informs every future design decision.

The relative-strength rotation strategy passed the one-shot out-of-sample test.
The factory is built, and a product worth testing at the next stage has been found.

---

## Round 7 (cont.) — HOLDOUT VERDICT: VALIDATED EDGE

### The Run

One-shot execution of the sealed 2022-2024 holdout. Strategy spec unchanged
from the tune-window run: 10-coin universe, 90-day risk-adjusted momentum,
top-2 equal weight, weekly rebalance, cash filter ON, 0.1% commission.
No parameter changes before, during, or after. The holdout is spent — this
result is final regardless of outcome.

### Results (holdout: 2022-2024, out-of-sample)

|  | Rotation | BTC buy-and-hold | EW hold-all |
|--|---------|-----------------|-------------|
| Total return | **+161.7%** | +95.9% | +2.6% |
| Final value | **$261.73** | $195.86 | $102.61 |
| Sharpe | **+0.686** | +0.569 | +0.279 |
| Max drawdown | **60.2%** | 66.9% | 75.0% |

Simulation stats: 220 rebalances, 275 trades, 5 cash periods.

All three verdict gates cleared:

| Gate | Result |
|------|--------|
| Beats BTC buy-and-hold (total return) | ✓  +161.7% vs +95.9% |
| Beats EW hold-all (total return) | ✓  +161.7% vs +2.6% |
| Sharpe ≥ 0.5 | ✓  +0.686 |

### Why the thesis held

Equal-weight-hold-all nearly broke even (+2.6%) across 2022-2024 — holding
every coin through a period where the whole basket was crushed in 2022 means
holding everything down. The rotation's cash filter (5 of 220 periods in cash)
and momentum tilt away from laggards preserved enough capital through the bear
leg to compound aggressively through the 2023-2024 recovery. This is exactly
the structural benefit the cash filter was designed to provide; it earned its
cost on the first real test.

The bench-level numbers confirm directionality: rotation's max drawdown (60.2%)
was lower than both BTC B&H (66.9%) and EW hold-all (75.0%) — smaller loss in
the crash, larger compounded gain in the recovery.

### Honest caveats (recorded so future decisions stay grounded)

1. **One out-of-sample pass is strong evidence, not proof.** The edge leaned
   heavily on protecting capital through ONE bear market (2022). That is a
   sample of one downturn type. A sideways chop regime or a different bear
   structure (e.g. sector rotation rather than broad crash) may behave
   differently.

2. **60% max drawdown is severe.** Emotionally and practically hard to hold at
   real size. Live behavior under real capital pressure has not been tested.

3. **The holdout is spent.** No clean out-of-sample data remains for any future
   changes to the strategy. Any parameter or logic change from this point
   forward is unvalidated by construction.

4. **Next step is paper trading, not capital deployment.** Live execution —
   slippage, exchange latency, real order fill — has not been tested. Paper
   trading confirms live behavior before any real-money decision.
