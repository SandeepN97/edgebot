# EdgeBot

A disciplined, research-driven crypto trading system built on Clean Architecture.
Not a "money machine" — a rigorously tested framework for finding, validating, and
(eventually) trading statistical edges, with honesty and risk control as first principles.

**Current status:** One strategy (cross-sectional relative-strength rotation) has
passed out-of-sample validation. Advancing to paper trading. No real capital deployed.

---

## What this project actually is

EdgeBot started as a personal project to learn algorithmic trading on a $100 budget.
The real goal was never "get rich" — it was to build a professional-grade system and
learn, firsthand, what separates strategies that work from strategies that only *look*
like they work in a backtest.

The defining principle is **honesty over hope**. Every strategy must prove a real,
out-of-sample edge before it earns the right to trade. Most don't. This repo documents
that process truthfully — including six strategy variants that were tested and found to
have no edge, and one that survived validation.

If you take one thing from this project: the value is in the **discipline** (gated
validation, sealed holdout data, refusing to curve-fit, accepting "no edge" verdicts),
not in any single strategy.

---

## Goals

- Build a clean, testable, extensible trading system that can host many strategies
  without entangling business logic with exchange/infrastructure code.
- Find at least one strategy with a genuine, out-of-sample edge, validated with
  rigorous train/test discipline.
- Learn the full lifecycle — data, backtesting, risk management, paper trading,
  observability — on a small budget where mistakes are cheap.
- Scale only what's proven. Real capital follows validated edge, never hope.

The system is percentage-based, so the same code runs $100 or $100,000 — but the
amount is irrelevant until an edge is proven and confirmed live.

> **What this project is NOT:** a prediction engine, a get-rich scheme, or a product
> to sell others. It cannot tell you "what happens next" — no system reliably can. It
> finds and exploits small statistical edges with strict risk control. That distinction
> is the whole point.

---

## Architecture

EdgeBot follows Clean Architecture (Hexagonal / Ports & Adapters). The core rule:
source-code dependencies point only **inward**. The domain knows nothing about
exchanges, databases, or APIs.

```
edgebot/
├── domain/              # Pure business logic — zero infrastructure imports
│   ├── entities/        # Signal, Order, Position, Trade, MarketSnapshot
│   ├── strategy/        # Strategy logic, regime detection, relative-strength rotation
│   ├── risk/            # RiskEngine, PositionSizer, CircuitBreaker
│   └── portfolio/       # PortfolioState, PnLCalculator
│
├── application/         # Use cases + port interfaces
│   ├── ports/
│   │   ├── input/       # IMarketDataPort, ISignalPort
│   │   └── output/      # IOrderPort, INotifyPort, IMetricsPort
│   └── use_cases/       # RunStrategy, EvaluateRisk, RecordTrade
│
├── infrastructure/      # Adapters — all external I/O lives here
│   ├── adapters/
│   │   ├── market_data/ # Binance WebSocket (CCXT)
│   │   ├── execution/   # PaperTradeAdapter (live adapter when ready)
│   │   ├── notify/      # Telegram
│   │   └── persistence/ # SQLite
│   └── config/          # Dependency-injection container (composition root)
│
├── strategies/          # Strategy implementations (EMA crossover, mean-reversion)
├── backtesting/         # Backtest pipeline, regime detection, FINDINGS.md
└── tests/               # Unit + integration (130+ tests)
```

**Why this matters:** a new strategy, a new exchange (stocks, forex), or swapping paper
for live trading are all done by adding an adapter or strategy behind an existing
interface — the domain and risk logic never change. The architecture is built to scale
to multiple markets and strategies without rewrites.

---

## Risk management (non-negotiable)

Risk limits live in the domain as hardcoded constants. They are expressed as percentages
of NAV, so they scale to any account size. Critically, environment-variable overrides can
only make them **tighter**, never looser — you cannot weaken the seatbelt from config.

```
MAX_POSITION_SIZE_PCT  = 0.02   # max 2% of NAV per position
DAILY_LOSS_LIMIT_PCT   = 0.05   # circuit breaker halts trading at 5% daily loss
MAX_OPEN_POSITIONS     = 3      # (single-symbol strategies)
MIN_RISK_REWARD_RATIO  = 2.0    # minimum reward:risk on directional signals
FEE_BUFFER_PCT         = 0.002  # reserve for fees so they don't erode the edge
```

A circuit breaker halts all trading on a 5% daily drawdown and requires manual reset —
it never auto-restarts into a bad market.

---

## The validation discipline

Every strategy is tested the same way, and most fail. The process:

1. **Backtest on a tune window**, with a sealed holdout of later data that is never
   looked at during development.
2. **One change per round** — so the effect of each change is isolated.
3. **Stopping rules set in advance** — to prevent endless tuning (curve-fitting).
4. **Benchmark comparison** — a strategy must beat buy-and-hold, not just post a nice
   number. Beating a flat metric while underperforming "just hold Bitcoin" is no edge.
5. **One-shot holdout verdict** — when a strategy looks promising on the tune window,
   it gets ONE run on the sealed holdout, with no tuning before or after. That result
   is final, win or lose.

This discipline is what makes a backtest trustworthy instead of a self-deception.

---

## What's been tested

See [`backtesting/FINDINGS.md`](backtesting/FINDINGS.md) for full detail and diagnosis
of each round.

| Strategy | Timeframe | Verdict |
|----------|-----------|---------|
| EMA 9/21 crossover (+ filters, 5 rounds) | 4h | No edge — signal-frequency squeeze |
| Mean-reversion (Bollinger + RSI) | 4h | No edge — failed in trending bear market |
| EMA + mean-reversion + regime router | 4h | No edge — combining weak components doesn't help |
| EMA-trending-only | daily | No edge — filters select contradictory conditions |
| **Relative-strength rotation** | **daily** | **VALIDATED — beat both benchmarks out-of-sample** |

The honest headline: simple technical-analysis strategies (EMA, mean-reversion) did not
have a tradeable edge on crypto in the tested windows. Cross-sectional
relative-strength rotation DID pass out-of-sample validation, beating both
buy-and-hold-BTC and equal-weight-hold-all on the sealed 2022-2024 holdout, with lower
drawdown than either benchmark.

**Important caveat:** one out-of-sample pass is strong evidence, not proof. The edge
leaned heavily on protecting capital through one bear market (2022). It must now prove
itself in live paper trading before any real capital is considered.

---

## The validated strategy: relative-strength rotation

| Parameter | Value |
|-----------|-------|
| Universe | 10 liquid coins: BTC, ETH, BNB, SOL, XRP, ADA, DOGE, AVAX, LINK, DOT |
| Signal | Rank by 90-day risk-adjusted momentum (return / volatility) |
| Hold | Top 2 coins, equal weight |
| Rebalance | Every 5 trading days (weekly) |
| Cash filter | Sit out when all momentum scores ≤ 0 (downside protection) |
| Commission | 0.1% per side |

**Holdout results (2022-2024, out-of-sample):**

|  | Rotation | BTC B&H | EW Hold-All |
|--|---------|---------|------------|
| Total return | **+161.7%** | +95.9% | +2.6% |
| Sharpe | **+0.686** | +0.569 | +0.279 |
| Max drawdown | **60.2%** | 66.9% | 75.0% |

**Why it works (when it does):** it doesn't predict direction — it rides whatever is
already strongest, and steps aside in broad downturns. Its edge showed up precisely
where trend-timing strategies failed: protecting capital through the 2022 bear, then
compounding through the recovery.

---

## Roadmap

- [x] Phase 1 — Clean architecture scaffold, risk engine, backtest pipeline, 130+ tests
- [x] Step 2 — Strategy validation (6 failed, 1 validated out-of-sample)
- [ ] **Step 3 — Paper trading** *(current)*: run live against real data, execute
  nothing, confirm live behaviour matches backtest. Includes realistic per-coin
  slippage modeling (higher for less-liquid coins).
- [ ] Step 4 — Observability: Grafana dashboard, structured logging, trade journal review.
- [ ] Live trading — ONLY after paper trading confirms the edge survives real execution.
  Gradual capital, watching whether live tracks backtest.

Possible later directions (each requires fresh validation, not assumption):
- Additional uncorrelated validated strategies, combined via the regime router.
- Multi-market expansion (stocks via Alpaca, forex via OANDA) — the architecture
  supports it; each needs its own validated edge.

---

## Tech stack

| Concern | Tool |
|---------|------|
| Exchange connectivity | CCXT (Binance) |
| Backtesting | Backtrader + custom rotation engine |
| Data | Binance OHLCV, parquet feature store |
| Notifications | Telegram |
| Persistence | SQLite |
| Metrics (planned) | Prometheus + Grafana |
| Infrastructure | Docker |
| Language | Python 3.11+ |

---

## Quickstart

```bash
git clone https://github.com/SandeepN97/edgebot.git
cd edgebot
cp .env.example .env        # fill in API keys (use read-only keys, never withdrawal)
make install
make test                   # run the full suite (130+ tests)

# Paper trading — no real orders placed:
docker compose up -d
# or locally:
make paper
```

**Minimum `.env` to run paper trading:**
```
BINANCE_API_KEY=your_read_only_key
BINANCE_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
INITIAL_CASH=100
```

On first run the bot fetches 120 days of history, determines the current momentum
leaders, and sends a rebalance alert to Telegram. Subsequent alerts fire every 5
trading days.

---

## Honest disclaimer

This is an educational research project. Algorithmic trading carries significant
financial risk. A validated backtest is not a guarantee of future performance — markets
change, and an edge that worked out-of-sample can still fail live. Never trade with
money you cannot afford to lose. Nothing here is financial advice. The author is
learning, in public, on a small budget — and the most valuable output so far has been
the discipline and the lessons, not profit.
