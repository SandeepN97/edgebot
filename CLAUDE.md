# CLAUDE.md — EdgeBot project rules for Claude Code

This file is read at the start of every session. It encodes the hard rules and working
discipline for this project. Follow these unless the user explicitly overrides them.

---

## What this project is

EdgeBot is a disciplined crypto trading research system built on Clean Architecture.
The goal is to find and validate genuine statistical edges through rigorous testing —
NOT to chase profit or build a prediction engine. Honesty over hope is the defining
principle. Most strategies fail validation, and that is recorded truthfully.

Current validated strategy: cross-sectional relative-strength rotation (10-coin
universe, 90-day risk-adjusted momentum, top-2 equal weight, weekly rebalance, cash
filter). Passed out-of-sample holdout validation. Now advancing to paper trading.

---

## Hard rules (do not violate)

### 1. Holdout discipline
- The sealed holdout window exists to give ONE honest out-of-sample verdict.
- NEVER run, print, reference, or tune against sealed holdout data while developing.
- A holdout verdict run is ONE shot, no tuning before/during/after, result is final.
- Once a holdout is spent, say so plainly. Do not silently reuse it.

### 2. Clean Architecture / dependency rule
- domain/ imports NOTHING from infrastructure/ or any external library (no ccxt,
  requests, sqlite3, telegram, lightgbm, alpaca, oanda, etc.). Pure Python only.
- application/ depends on domain/ and ports/, never on infrastructure/.
- New exchanges, strategies, or data sources enter via adapters behind existing ports.
- Before committing domain changes, verify no forbidden imports leaked in.

### 3. Risk limits cannot be loosened
- RiskEngine limits are hardcoded constants. Env overrides may only TIGHTEN them,
  never loosen. Never weaken a risk rule from config.

### 4. Live and backtest logic must stay identical
- Any strategy's live implementation and its backtest implementation must use the
  same rules. If they drift, the backtest is meaningless. Verify parity on changes.

### 5. One change per tuning round
- When iterating on a strategy, change ONE parameter or filter at a time, so its
  isolated effect is visible. No multi-knob thrashing.

### 6. Stopping rules and benchmarks
- Set a stopping rule before running a tuning experiment, and honor it even when
  inconvenient. Do not propose "one more tweak" past a stated stopping point.
- A strategy must beat relevant benchmarks (e.g. buy-and-hold), not just clear a flat
  metric. Beating Sharpe 0.5 while underperforming "just hold BTC" is no edge.
- Trade count must be high enough to be statistically meaningful before a result is
  trusted (a handful of trades is not a verdict).

### 7. No chasing ML on hope
- ML/RL is permitted only as a learning exercise with kill criteria set in advance,
  expecting it may also find no edge. Never frame ML as a way to manufacture an edge
  the simple strategies couldn't find — that framing leads to overfitting and loss.

---

## Documentation discipline (keep docs evolving)

At every validation verdict, phase change, or significant milestone — BEFORE committing:

1. **Append to backtesting/FINDINGS.md** — record what was tested, the numbers, the
   verdict, and honest caveats. FINDINGS.md is APPEND-ONLY: never delete or rewrite a
   past result, including failures. The failures are the credibility.

2. **Update the README status line** — the "Current status:" near the top must always
   reflect where the project actually stands. Update it when the phase or story changes.

3. **Docs follow reality** — if the README and the actual repo code disagree, fix the
   code/commit first, then the docs. Never let docs describe code that isn't there.

Do NOT update docs on every trivial commit — only at milestones (verdicts, phase
changes, validated results, concluded experiments).

---

## Working style

- Be honest about results, including bad ones. A clean "no edge" verdict is a success,
  not a failure to paper over.
- Do not suggest advancing a gate (e.g. to paper or live trading) unless the prior gate
  genuinely passed.
- Keep commits focused with clear messages describing what was tested or built.
- Run the full test suite before committing significant changes.
