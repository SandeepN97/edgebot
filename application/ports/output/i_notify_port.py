"""INotifyPort — output port interface for sending trade alerts and notifications.

Any notification channel (Telegram, email, Slack) implements this interface so
the application layer can alert operators without importing any SDK directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum

from domain.entities.order import Order
from domain.entities.signal import Signal
from domain.entities.trade import Trade


class AlertLevel(Enum):
    """Severity level for notifications."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class INotifyPort(ABC):
    """Driven port: the application sends alerts through this interface."""

    @abstractmethod
    async def send_signal_alert(self, signal: Signal) -> None:
        """Notify operator that a new signal has been generated.

        Args:
            signal: The Signal that fired.
        """

    @abstractmethod
    async def send_order_update(self, order: Order) -> None:
        """Notify operator of an order status change (filled, cancelled, rejected).

        Args:
            order: The updated Order entity.
        """

    @abstractmethod
    async def send_trade_summary(self, trade: Trade) -> None:
        """Send a closed-trade summary with PnL details.

        Args:
            trade: The completed Trade entity.
        """

    @abstractmethod
    async def send_circuit_breaker_alert(self, reason: str) -> None:
        """Alert that the daily-loss circuit breaker has tripped.

        Args:
            reason: Human-readable description of why the breaker tripped.
        """

    @abstractmethod
    async def send_message(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        chat_id: str | None = None,
    ) -> None:
        """Send a free-form message at a given severity level.

        Args:
            message:  The text to send.
            level:    Severity of the message.
            chat_id:  Optional override for the destination channel.
        """
