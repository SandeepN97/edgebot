# EdgeBot — Backtest Findings (Phase 1, Step 2)

Honest record of the strategy validation phase. The backtest gate did not pass.
No strategy reached a tradeable edge. This document captures what was tested,
the evidence, the diagnosis, and why the edge-hunt was stopped — so the
conclusion rests on facts, not memory.

**Status:** Step 2 (backtest) complete. Decision gate: **NOT PASSED** — edge-hunt
stopped deliberately. Project paused before paper trading (Step 3).

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

## The Real Deliverable

The product of this phase is not a profitable strategy — it is a validated,
honest finding: simple technical-analysis strategies did not show a tradeable
edge on these markets, demonstrated rigorously and without self-deception, at
zero financial cost.

The knowledge of *why* — the filter squeeze, the ADX×volume mutual exclusivity,
mean-reversion's structural failure in trending bear markets — is more valuable
than a strategy that happened to look good on past data, because it informs
every future design decision.

The factory is built. No product worth manufacturing was found yet.
That is a pause, not a failure.
