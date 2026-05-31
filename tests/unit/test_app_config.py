"""Unit tests for AppConfig risk-limit clamping.

Env vars may only TIGHTEN the hardcoded module constants — never loosen them.
These tests prove that a looser env value is silently clamped to the constant.
"""

from __future__ import annotations

import os

import pytest

from domain.risk.risk_engine import (
    DAILY_LOSS_LIMIT_PCT,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_SIZE_PCT,
    MIN_RISK_REWARD_RATIO,
)


def _make_config(**env_overrides):
    """Build an AppConfig with the given env vars patched in."""
    from infrastructure.config.container import AppConfig

    old = {}
    for k, v in env_overrides.items():
        old[k] = os.environ.get(k)
        os.environ[k] = str(v)
    try:
        return AppConfig()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ------------------------------------------------------------------
# daily_loss_limit_pct — smaller = tighter; looser env must be clamped
# ------------------------------------------------------------------


class TestDailyLossLimitClamping:
    def test_looser_env_is_clamped_to_constant(self) -> None:
        cfg = _make_config(DAILY_LOSS_LIMIT_PCT=0.50)
        assert cfg.daily_loss_limit_pct == pytest.approx(DAILY_LOSS_LIMIT_PCT)

    def test_tighter_env_is_accepted(self) -> None:
        tighter = DAILY_LOSS_LIMIT_PCT / 2
        cfg = _make_config(DAILY_LOSS_LIMIT_PCT=tighter)
        assert cfg.daily_loss_limit_pct == pytest.approx(tighter)

    def test_default_equals_constant(self) -> None:
        cfg = _make_config()
        assert cfg.daily_loss_limit_pct == pytest.approx(DAILY_LOSS_LIMIT_PCT)


# ------------------------------------------------------------------
# max_position_size_pct — smaller = tighter
# ------------------------------------------------------------------


class TestMaxPositionSizeClamping:
    def test_looser_env_is_clamped(self) -> None:
        cfg = _make_config(MAX_POSITION_SIZE_PCT=0.99)
        assert cfg.max_position_size_pct == pytest.approx(MAX_POSITION_SIZE_PCT)

    def test_tighter_env_is_accepted(self) -> None:
        tighter = MAX_POSITION_SIZE_PCT / 2
        cfg = _make_config(MAX_POSITION_SIZE_PCT=tighter)
        assert cfg.max_position_size_pct == pytest.approx(tighter)


# ------------------------------------------------------------------
# max_open_positions — smaller = tighter
# ------------------------------------------------------------------


class TestMaxOpenPositionsClamping:
    def test_looser_env_is_clamped(self) -> None:
        cfg = _make_config(MAX_OPEN_POSITIONS=100)
        assert cfg.max_open_positions == MAX_OPEN_POSITIONS

    def test_tighter_env_is_accepted(self) -> None:
        cfg = _make_config(MAX_OPEN_POSITIONS=1)
        assert cfg.max_open_positions == 1


# ------------------------------------------------------------------
# min_risk_reward_ratio — larger = tighter
# ------------------------------------------------------------------


class TestMinRiskRewardClamping:
    def test_looser_env_is_clamped(self) -> None:
        # 1.0 < 2.0 (the constant) → looser → must clamp to constant
        cfg = _make_config(MIN_RISK_REWARD_RATIO=1.0)
        assert cfg.min_risk_reward_ratio == pytest.approx(MIN_RISK_REWARD_RATIO)

    def test_tighter_env_is_accepted(self) -> None:
        tighter = MIN_RISK_REWARD_RATIO * 2
        cfg = _make_config(MIN_RISK_REWARD_RATIO=tighter)
        assert cfg.min_risk_reward_ratio == pytest.approx(tighter)

    def test_default_equals_constant(self) -> None:
        cfg = _make_config()
        assert cfg.min_risk_reward_ratio == pytest.approx(MIN_RISK_REWARD_RATIO)
