"""TelegramAdapter — sends trade alerts via the Telegram Bot API.

Implements INotifyPort using the python-telegram-bot library.
Messages are formatted in Markdown and include emoji severity indicators.
"""

from __future__ import annotations

import logging
from datetime import date

from application.ports.output.i_notify_port import AlertLevel, INotifyPort
from domain.entities.order import Order
from domain.entities.signal import Signal
from domain.entities.trade import Trade

logger = logging.getLogger(__name__)

_LEVEL_EMOJI = {
    AlertLevel.INFO: "ℹ️",
    AlertLevel.WARNING: "⚠️",
    AlertLevel.ERROR: "🔴",
    AlertLevel.CRITICAL: "🚨",
}


class TelegramAdapter(INotifyPort):
    """Telegram Bot notification adapter.

    Args:
        token:           Telegram Bot API token from @BotFather.
        default_chat_id: Default chat/group ID to send messages to.
        parse_mode:      Telegram parse mode: ``"Markdown"`` or ``"HTML"``.
    """

    def __init__(
        self,
        token: str,
        default_chat_id: str,
        parse_mode: str = "Markdown",
    ) -> None:
        self._token = token
        self._default_chat_id = default_chat_id
        self._parse_mode = parse_mode
        self._bot = None

    def _get_bot(self):
        """Lazy-initialise the Bot instance to avoid import overhead at module load."""
        if self._bot is None:
            try:
                from telegram import Bot  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "python-telegram-bot is required: pip install python-telegram-bot"
                )
            self._bot = Bot(token=self._token)
        return self._bot

    # ------------------------------------------------------------------
    # Rotation-specific notifications
    # ------------------------------------------------------------------

    async def send_rebalance_alert(
        self,
        rebalance_date: date,
        basket_before: list[str],
        basket_after: list[str],
        scores: dict[str, float],
        portfolio_value: float,
        week_pnl: float,
        in_cash: bool = False,
    ) -> None:
        """Send a rotation rebalance summary to Telegram."""
        entered = [s.split("/")[0] for s in basket_after if s not in basket_before]
        exited = [s.split("/")[0] for s in basket_before if s not in basket_after]
        held = [s.split("/")[0] for s in basket_after if s in basket_before]

        lines = [f"🔄 *Rotation Rebalance — {rebalance_date}*\n"]

        if entered:
            lines.append(f"  ➕ Entered : `{', '.join(entered)}`")
        if exited:
            lines.append(f"  ➖ Exited  : `{', '.join(exited)}`")
        if held:
            lines.append(f"  ↔️  Held    : `{', '.join(held)}`")

        basket_str = ", ".join(s.split("/")[0] for s in basket_after) or "CASH"
        lines.append(f"\n  📦 Basket  : `{basket_str}`")
        lines.append(f"  💼 NAV     : `${portfolio_value:.2f}`")

        pnl_emoji = "📈" if week_pnl >= 0 else "📉"
        lines.append(f"  {pnl_emoji} Week PnL : `{week_pnl:+.2f} USDT`")

        if scores:
            top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            score_str = "  ".join(
                f"{s.split('/')[0]}={v:+.1f}" for s, v in top_scores
            )
            lines.append(f"\n  📊 Top scores: `{score_str}`")

        await self.send_message("\n".join(lines), AlertLevel.INFO)

    async def send_cash_filter_alert(
        self,
        rebalance_date: date,
        portfolio_value: float,
        scores: dict[str, float],
    ) -> None:
        """Alert that the cash filter activated — all momentum scores negative."""
        if scores:
            worst = sorted(scores.items(), key=lambda x: x[1])[:3]
            score_str = "  ".join(
                f"{s.split('/')[0]}={v:+.1f}" for s, v in worst
            )
        else:
            score_str = "no scores computed"

        message = (
            f"💸 *Cash Filter Activated — {rebalance_date}*\n\n"
            f"  Sitting in cash this week — top momentum negative.\n"
            f"  💼 NAV: `${portfolio_value:.2f}`\n"
            f"  📊 Worst scores: `{score_str}`"
        )
        await self.send_message(message, AlertLevel.WARNING)

    # ------------------------------------------------------------------
    # INotifyPort implementation
    # ------------------------------------------------------------------

    async def send_signal_alert(self, signal: Signal) -> None:
        direction_arrow = "📈" if signal.direction.value == "long" else "📉"
        rr = signal.risk_reward_ratio
        rr_str = f"{rr:.2f}" if rr is not None else "N/A"
        message = (
            f"{direction_arrow} *Signal: {signal.symbol}*\n"
            f"Direction: `{signal.direction.value.upper()}`\n"
            f"Confidence: `{signal.confidence:.0%}`\n"
            f"Entry: `{signal.entry_price or 'MARKET'}`\n"
            f"SL: `{signal.stop_loss}` | TP: `{signal.take_profit}`\n"
            f"R:R: `{rr_str}`\n"
            f"Reason: _{signal.reason}_\n"
            f"Strategy: `{signal.strategy_id}`"
        )
        await self.send_message(message, AlertLevel.INFO)

    async def send_order_update(self, order: Order) -> None:
        status_emoji = {"filled": "✅", "rejected": "❌", "cancelled": "🚫"}.get(
            order.status.value, "🔄"
        )
        message = (
            f"{status_emoji} *Order {order.status.value.upper()}*\n"
            f"Symbol: `{order.symbol}`\n"
            f"Side: `{order.side.value.upper()}`\n"
            f"Qty: `{order.quantity:.6f}`\n"
            f"Fill: `{order.avg_fill_price or 'pending'}`\n"
            f"Fee: `{order.fee:.4f}`"
        )
        await self.send_message(message, AlertLevel.INFO)

    async def send_trade_summary(self, trade: Trade) -> None:
        pnl_emoji = "💚" if trade.net_pnl >= 0 else "🔴"
        message = (
            f"{pnl_emoji} *Trade Closed: {trade.symbol}*\n"
            f"Direction: `{trade.direction.value.upper()}`\n"
            f"Entry: `{trade.entry_price:.4f}` → Exit: `{trade.exit_price:.4f}`\n"
            f"Net PnL: `{trade.net_pnl:+.4f}` (`{trade.return_pct:+.2%}`)\n"
            f"Fees: `{trade.total_fees:.4f}`\n"
            f"Duration: `{trade.duration_seconds / 3600:.1f}h`\n"
            f"Exit reason: _{trade.exit_reason}_"
        )
        level = AlertLevel.INFO if trade.net_pnl >= 0 else AlertLevel.WARNING
        await self.send_message(message, level)

    async def send_circuit_breaker_alert(self, reason: str) -> None:
        message = f"🚨 *CIRCUIT BREAKER TRIPPED*\n_{reason}_\n\nAll trading halted."
        await self.send_message(message, AlertLevel.CRITICAL)

    async def send_message(
        self,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        chat_id: str | None = None,
    ) -> None:
        target = chat_id or self._default_chat_id
        emoji = _LEVEL_EMOJI.get(level, "")
        full_message = f"{emoji} {message}" if emoji else message
        try:
            bot = self._get_bot()
            await bot.send_message(
                chat_id=target,
                text=full_message,
                parse_mode=self._parse_mode,
            )
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
