"""DependencyContainer — wires all adapters and use cases together.

This is the composition root: the only place in the codebase where concrete
infrastructure classes are instantiated and injected into application-layer
use cases.  All other modules depend on interfaces (ports), never on this file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from application.use_cases.evaluate_risk import EvaluateRisk
from application.use_cases.record_trade import RecordTrade
from application.use_cases.run_strategy import RunStrategy
from domain.portfolio.portfolio_state import PortfolioState
from domain.risk.circuit_breaker import CircuitBreaker
from domain.risk.risk_engine import (
    DAILY_LOSS_LIMIT_PCT,
    FEE_BUFFER_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    MIN_RISK_REWARD_RATIO,
    RiskEngine,
)
from infrastructure.adapters.execution.paper_trade_adapter import PaperTradeAdapter
from infrastructure.adapters.market_data.binance_ws_adapter import BinanceWsAdapter
from infrastructure.adapters.notify.telegram_adapter import TelegramAdapter
from infrastructure.adapters.persistence.sqlite_adapter import SQLiteAdapter


@dataclass
class AppConfig:
    """Flat configuration bag populated from environment variables."""

    # Exchange
    binance_api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    binance_api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    binance_testnet: bool = field(
        default_factory=lambda: os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    )

    # Telegram
    telegram_token: str = field(default_factory=lambda: os.getenv("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))

    # Portfolio
    initial_cash: float = field(default_factory=lambda: float(os.getenv("INITIAL_CASH", "10000")))

    # Database
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", "data/edgebot.db"))

    # Risk overrides — env may only TIGHTEN limits, never loosen them.
    # Smaller value = tighter for size/loss/positions; larger = tighter for RR.
    max_position_size_pct: float = field(
        default_factory=lambda: min(
            float(os.getenv("MAX_POSITION_SIZE_PCT", str(MAX_POSITION_SIZE_PCT))),
            MAX_POSITION_SIZE_PCT,
        )
    )
    daily_loss_limit_pct: float = field(
        default_factory=lambda: min(
            float(os.getenv("DAILY_LOSS_LIMIT_PCT", str(DAILY_LOSS_LIMIT_PCT))),
            DAILY_LOSS_LIMIT_PCT,
        )
    )
    max_open_positions: int = field(
        default_factory=lambda: min(
            int(os.getenv("MAX_OPEN_POSITIONS", str(MAX_OPEN_POSITIONS))),
            MAX_OPEN_POSITIONS,
        )
    )
    min_risk_reward_ratio: float = field(
        default_factory=lambda: max(
            float(os.getenv("MIN_RISK_REWARD_RATIO", str(MIN_RISK_REWARD_RATIO))),
            MIN_RISK_REWARD_RATIO,
        )
    )

    # Execution
    paper_trade: bool = field(
        default_factory=lambda: os.getenv("PAPER_TRADE", "true").lower() == "true"
    )


class DependencyContainer:
    """Composition root that builds and exposes the wired application graph.

    Usage::

        container = DependencyContainer()
        await container.start()
        runner = container.run_strategy
        ...
        await container.stop()
    """

    def __init__(self, config: AppConfig | None = None) -> None:
        self.config = config or AppConfig()

        # Domain singletons
        self.portfolio = PortfolioState(initial_cash=self.config.initial_cash)
        self.circuit_breaker = CircuitBreaker(daily_loss_limit_pct=self.config.daily_loss_limit_pct)
        self.risk_engine = RiskEngine(
            max_position_size_pct=self.config.max_position_size_pct,
            daily_loss_limit_pct=self.config.daily_loss_limit_pct,
            max_open_positions=self.config.max_open_positions,
            min_risk_reward_ratio=self.config.min_risk_reward_ratio,
            fee_buffer_pct=FEE_BUFFER_PCT,
        )

        # Infrastructure adapters
        self.market_data = BinanceWsAdapter(
            api_key=self.config.binance_api_key,
            api_secret=self.config.binance_api_secret,
            testnet=self.config.binance_testnet,
        )
        self.order_port = PaperTradeAdapter() if self.config.paper_trade else None
        self.notify_port = TelegramAdapter(
            token=self.config.telegram_token,
            default_chat_id=self.config.telegram_chat_id,
        )
        self.db = SQLiteAdapter(db_path=self.config.db_path)

        # Application use cases
        self.evaluate_risk = EvaluateRisk(
            risk_engine=self.risk_engine,
            circuit_breaker=self.circuit_breaker,
        )
        self.record_trade = RecordTrade(
            trade_repo=self.db,
            portfolio=self.portfolio,
            circuit_breaker=self.circuit_breaker,
            notify_port=self.notify_port,
            metrics_port=_NoOpMetricsAdapter(),
        )

    async def start(self) -> None:
        """Connect all infrastructure adapters."""
        await self.db.connect()
        await self.market_data.connect()

    async def stop(self) -> None:
        """Gracefully disconnect all adapters."""
        await self.market_data.disconnect()
        await self.db.disconnect()

    def build_run_strategy(self, signal_port) -> RunStrategy:
        """Wire a concrete strategy into the RunStrategy use case."""
        assert self.order_port is not None, "order_port must be set before building RunStrategy"
        return RunStrategy(
            signal_port=signal_port,
            order_port=self.order_port,
            notify_port=self.notify_port,
            metrics_port=_NoOpMetricsAdapter(),
            evaluate_risk=self.evaluate_risk,
            portfolio=self.portfolio,
            circuit_breaker=self.circuit_breaker,
        )


class _NoOpMetricsAdapter:
    """Silent metrics adapter used when no metrics backend is configured."""

    async def record_trade(self, **kwargs) -> None: ...
    async def record_signal(self, **kwargs) -> None: ...
    async def record_portfolio_snapshot(self, **kwargs) -> None: ...
    async def record_order_latency(self, **kwargs) -> None: ...
    async def increment_counter(self, **kwargs) -> None: ...
    async def set_gauge(self, **kwargs) -> None: ...
