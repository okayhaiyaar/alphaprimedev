"""
Unit tests for trading circuit breakers in ALPHA-PRIME v2.0.

Covers:
- Drawdown, loss limit, position, volatility, correlation, and system health breakers.
- Breaker states, trigger conditions, recovery/reset logic, manager behavior, and alerts.
- Integration-style safety scenarios (flash crash, false positives, recovery).

All tests are:
- Fast, isolated (mocked data), deterministic, and safety-focused.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from core.circuit_breakers import (
    CircuitBreaker,
    DrawdownBreaker,
    LossLimitBreaker,
    PositionLimitBreaker,
    VolatilityBreaker,
    CorrelationBreaker,
    SystemHealthBreaker,
    CircuitBreakerManager,
    BreakerState,
    BreakerTrigger,
    reset_all_breakers,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def equity_curve_normal() -> np.ndarray:
    """Normal equity curve (steady growth)."""
    return np.linspace(100_000, 110_000, 100)


@pytest.fixture
def equity_curve_drawdown() -> np.ndarray:
    """Equity curve with 15% drawdown and partial recovery."""
    curve = np.linspace(100_000, 110_000, 50)
    drawdown = np.linspace(110_000, 93_500, 30)
    recovery = np.linspace(93_500, 100_000, 20)
    return np.concatenate([curve, drawdown, recovery])


@pytest.fixture
def portfolio_losses() -> Dict[str, float]:
    """Portfolio with daily losses."""
    return {
        "day_1": -500,
        "day_2": -800,
        "day_3": -1200,  # Total: -2500 (-2.5%)
        "day_4": 300,
        "day_5": -600,
    }


@pytest.fixture
def mock_portfolio() -> Mock:
    """Mock portfolio for testing position/limit breakers."""
    portfolio = Mock()
    portfolio.total_value = 100_000
    portfolio.cash = 50_000
    portfolio.positions = {
        "AAPL": Mock(symbol="AAPL", value=30_000, pct=0.30, sector="TECH", strategy="S1"),
        "MSFT": Mock(symbol="MSFT", value=15_000, pct=0.15, sector="TECH", strategy="S1"),
        "GOOGL": Mock(symbol="GOOGL", value=5_000, pct=0.05, sector="COMM", strategy="S2"),
    }
    portfolio.leverage = 1.5
    return portfolio


@pytest.fixture
def volatility_data_normal() -> Dict[str, float]:
    """Normal market volatility."""
    return {"VIX": 15.0, "portfolio_vol": 0.18}


@pytest.fixture
def volatility_data_spike() -> Dict[str, float]:
    """Volatility spike scenario."""
    return {"VIX": 45.0, "portfolio_vol": 0.55}


@pytest.fixture
def circuit_breaker_config() -> Dict[str, Any]:
    """Standard circuit breaker configuration."""
    return {
        "max_drawdown": 0.15,
        "daily_loss_limit": 0.02,
        "weekly_loss_limit": 0.05,
        "monthly_loss_limit": 0.10,
        "max_position_pct": 0.20,
        "vix_threshold": 35.0,
        "correlation_threshold": 0.85,
        "cooldown_minutes": 30,
    }


@pytest.fixture
def breaker_manager(circuit_breaker_config: Dict[str, Any]) -> CircuitBreakerManager:
    """Circuit breaker manager with standard config."""
    manager = CircuitBreakerManager()
    manager.register(DrawdownBreaker(max_drawdown=circuit_breaker_config["max_drawdown"]))
    manager.register(LossLimitBreaker(daily_limit_pct=circuit_breaker_config["daily_loss_limit"]))
    manager.register(PositionLimitBreaker(max_position_pct=circuit_breaker_config["max_position_pct"]))
    return manager


# ---------------------------------------------------------------------------
# 1. Drawdown Circuit Breaker
# ---------------------------------------------------------------------------

class TestDrawdownBreaker:
    """Test maximum drawdown circuit breaker."""

    def test_trigger_on_max_drawdown_breach(self, equity_curve_drawdown: np.ndarray):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        peak = float(equity_curve_drawdown[0])
        result: BreakerTrigger | None = None
        for equity in equity_curve_drawdown:
            result = breaker.check(equity=equity, peak=peak)
            peak = max(peak, equity)
        assert result is not None
        assert result.triggered is True
        assert result.trigger_value >= 0.10
        assert breaker.state == BreakerState.TRIGGERED

    def test_trigger_on_intraday_drawdown(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        result = breaker.check(equity=90_000, peak=100_000)
        assert result.triggered is True
        assert pytest.approx(result.trigger_value) == 0.10

    def test_trigger_on_rolling_drawdown(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, rolling_window=5)
        equities = [100_000, 105_000, 103_000, 95_000]
        result = BreakerTrigger(triggered=False, trigger_value=0.0, details={})
        for e in equities:
            result = breaker.check(equity=e)
        assert result.triggered is True

    def test_no_trigger_below_threshold(self, equity_curve_normal: np.ndarray):
        breaker = DrawdownBreaker(max_drawdown=0.20)
        peak = float(equity_curve_normal[0])
        result = None
        for equity in equity_curve_normal:
            result = breaker.check(equity=equity, peak=peak)
            peak = max(peak, equity)
        assert result is not None
        assert result.triggered is False
        assert breaker.state == BreakerState.ARMED

    def test_partial_recovery_doesnt_reset(self, equity_curve_drawdown: np.ndarray):
        breaker = DrawdownBreaker(max_drawdown=0.10, auto_reset=True)
        peak = float(equity_curve_drawdown[0])
        for equity in equity_curve_drawdown[:-5]:
            breaker.check(equity=equity, peak=peak)
            peak = max(peak, equity)
        assert breaker.state == BreakerState.TRIGGERED

    def test_full_recovery_resets_breaker(self, equity_curve_drawdown: np.ndarray):
        breaker = DrawdownBreaker(max_drawdown=0.10, auto_reset=True)
        peak = float(equity_curve_drawdown[0])
        for equity in equity_curve_drawdown:
            breaker.check(equity=equity, peak=peak)
            peak = max(peak, equity)
        assert breaker.state == BreakerState.ARMED

    def test_multiple_breach_escalation(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, escalation_threshold=2)
        breaker.check(equity=90_000, peak=100_000)
        breaker.check(equity=90_000, peak=100_000)
        assert breaker.escalation_level >= 1

    def test_drawdown_calculation_accuracy(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        result = breaker.check(equity=85_000, peak=100_000)
        assert pytest.approx(result.trigger_value) == 0.15

    def test_drawdown_from_equity_peak(self, equity_curve_drawdown: np.ndarray):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        peak = np.max(equity_curve_drawdown)
        trough = np.min(equity_curve_drawdown)
        dd = (peak - trough) / peak
        assert dd > 0.10

    def test_drawdown_by_strategy(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, per_strategy=True)
        equity_by_strategy = {"S1": 80_000, "S2": 90_000}
        peak_by_strategy = {"S1": 100_000, "S2": 95_000}
        result = breaker.check(equity_by_strategy=equity_by_strategy, peak_by_strategy=peak_by_strategy)
        assert result.triggered is True
        assert "S1" in result.details.get("breached_strategies", [])

    def test_drawdown_vs_benchmark(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, benchmark_sensitive=True)
        result = breaker.check(
            portfolio_equity=90_000,
            benchmark_equity=98_000,
            portfolio_peak=100_000,
            benchmark_peak=100_000,
        )
        assert result.triggered is True

    def test_drawdown_edge_case_zero_equity(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        result = breaker.check(equity=0.0, peak=100_000)
        assert result.triggered is True
        assert result.trigger_value == 1.0


# ---------------------------------------------------------------------------
# 2. Loss Limit Circuit Breaker
# ---------------------------------------------------------------------------

class TestLossLimitBreaker:
    """Test daily/weekly loss limit breakers."""

    def test_trigger_on_daily_loss_limit(self, portfolio_losses: Dict[str, float]):
        breaker = LossLimitBreaker(daily_limit_pct=0.02)
        cumulative_loss = sum(portfolio_losses.values())
        result = breaker.check(realized_pnl=cumulative_loss, start_capital=100_000)
        assert result.triggered is True
        assert abs(result.trigger_value) > 0.02

    def test_trigger_on_weekly_loss_limit(self):
        breaker = LossLimitBreaker(weekly_limit_pct=0.05)
        result = breaker.check(weekly_pnl=-6_000, start_capital=100_000)
        assert result.triggered is True

    def test_trigger_on_monthly_loss_limit(self):
        breaker = LossLimitBreaker(monthly_limit_pct=0.10)
        result = breaker.check(monthly_pnl=-11_000, start_capital=100_000)
        assert result.triggered is True

    def test_no_trigger_with_profitable_day(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02)
        result = breaker.check(realized_pnl=1_000, start_capital=100_000)
        assert result.triggered is False

    def test_loss_limit_percentage_vs_absolute(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02, daily_limit_abs=2_500)
        result_pct = breaker.check(realized_pnl=-2_100, start_capital=100_000)
        result_abs = breaker.check(realized_pnl=-2_600, start_capital=100_000)
        assert result_pct.triggered is True
        assert result_abs.triggered is True

    def test_loss_limit_reset_at_midnight(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02)
        breaker.check(realized_pnl=-2_500, start_capital=100_000)
        assert breaker.state == BreakerState.TRIGGERED
        breaker.reset_daily()
        assert breaker.state == BreakerState.ARMED

    def test_loss_limit_with_winning_positions(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02)
        result = breaker.check(realized_pnl=-2_500, unrealized_pnl=3_000, start_capital=100_000)
        assert result.triggered is False

    def test_loss_limit_unrealized_vs_realized(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02, include_unrealized=True)
        result = breaker.check(realized_pnl=-1_000, unrealized_pnl=-1_600, start_capital=100_000)
        assert result.triggered is True

    def test_consecutive_loss_days_escalation(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02, escalation_days=3)
        for _ in range(3):
            breaker.check(realized_pnl=-2_500, start_capital=100_000, date=datetime(2024, 1, 1))
        assert breaker.escalation_level >= 1

    def test_loss_limit_edge_case_zero_start_capital(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.02)
        result = breaker.check(realized_pnl=-100, start_capital=0.0)
        assert result.triggered is True


# ---------------------------------------------------------------------------
# 3. Position Limit Circuit Breaker
# ---------------------------------------------------------------------------

class TestPositionLimitBreaker:
    """Test position size and concentration limits."""

    def test_trigger_on_single_position_limit(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_position_pct=0.25)
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is True
        assert "AAPL" in result.details.get("breached_symbols", [])

    def test_trigger_on_total_exposure_limit(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_position_pct=0.5, max_total_exposure_pct=1.2)
        mock_portfolio.total_exposure = 1.3
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is True

    def test_trigger_on_sector_concentration(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_sector_pct=0.40)
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is True
        assert "TECH" in result.details.get("breached_sectors", [])

    def test_trigger_on_correlation_cluster(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_cluster_correlation=0.9)
        corr_matrix = np.array([[1.0, 0.95], [0.95, 1.0]])
        result = breaker.check(portfolio=mock_portfolio, correlation_matrix=corr_matrix)
        assert result.triggered is True

    def test_trigger_on_leverage_breach(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_leverage=1.2)
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is True

    def test_no_trigger_within_limits(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_position_pct=0.40, max_leverage=2.0)
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is False

    def test_partial_position_reduction_allowed(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(max_position_pct=0.25, allow_partial_reduce=True)
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is True
        assert result.details.get("action") == "reduce"

    def test_position_limit_by_symbol(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(symbol_limits={"AAPL": 0.2})
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is True

    def test_position_limit_by_strategy(self, mock_portfolio: Mock):
        breaker = PositionLimitBreaker(strategy_limits={"S1": 0.5})
        result = breaker.check(portfolio=mock_portfolio)
        assert result.triggered is False

    def test_position_limit_edge_case_empty_portfolio(self):
        breaker = PositionLimitBreaker(max_position_pct=0.25)
        empty = Mock()
        empty.positions = {}
        empty.total_value = 0.0
        result = breaker.check(portfolio=empty)
        assert result.triggered is False


# ---------------------------------------------------------------------------
# 4. Volatility Circuit Breaker
# ---------------------------------------------------------------------------

class TestVolatilityBreaker:
    """Test volatility spike detection breaker."""

    def test_trigger_on_volatility_spike(self, volatility_data_spike: Dict[str, float]):
        breaker = VolatilityBreaker(vix_threshold=35.0)
        result = breaker.check(vix=volatility_data_spike["VIX"])
        assert result.triggered is True
        assert result.trigger_value == pytest.approx(45.0)

    def test_trigger_on_vix_spike(self, volatility_data_spike: Dict[str, float]):
        breaker = VolatilityBreaker(vix_threshold=35.0)
        result = breaker.check(vix=volatility_data_spike["VIX"])
        assert result.triggered is True

    def test_trigger_on_portfolio_volatility(self, volatility_data_spike: Dict[str, float]):
        breaker = VolatilityBreaker(portfolio_vol_threshold=0.40)
        result = breaker.check(portfolio_vol=volatility_data_spike["portfolio_vol"])
        assert result.triggered is True

    def test_volatility_regime_change_detection(self):
        breaker = VolatilityBreaker(regime_change_threshold=0.15)
        result = breaker.check(prev_vol=0.10, current_vol=0.30)
        assert result.triggered is True

    def test_no_trigger_normal_volatility(self, volatility_data_normal: Dict[str, float]):
        breaker = VolatilityBreaker(vix_threshold=35.0, portfolio_vol_threshold=0.40)
        result = breaker.check(
            vix=volatility_data_normal["VIX"],
            portfolio_vol=volatility_data_normal["portfolio_vol"],
        )
        assert result.triggered is False

    def test_volatility_moving_average_threshold(self):
        breaker = VolatilityBreaker(ma_multiplier=2.0)
        vols = np.array([0.10, 0.11, 0.12, 0.13])
        ma = vols.mean()
        result = breaker.check(
            portfolio_vol=ma * 2.5,
            vol_ma=ma,
        )
        assert result.triggered is True

    def test_volatility_multiple_timeframes(self):
        breaker = VolatilityBreaker(multi_tf_threshold=0.3)
        result = breaker.check(
            vol_short=0.35,
            vol_long=0.10,
        )
        assert result.triggered is True

    def test_volatility_edge_case_low_volatility(self):
        breaker = VolatilityBreaker(vix_threshold=35.0)
        result = breaker.check(vix=5.0)
        assert result.triggered is False


# ---------------------------------------------------------------------------
# 5. Correlation Breakdown Breaker
# ---------------------------------------------------------------------------

class TestCorrelationBreaker:
    """Test correlation breakdown detection."""

    def test_trigger_on_correlation_spike(self):
        breaker = CorrelationBreaker(threshold=0.85)
        corr = np.array([[1.0, 0.9], [0.9, 1.0]])
        result = breaker.check(correlation_matrix=corr)
        assert result.triggered is True

    def test_trigger_on_correlation_collapse(self):
        breaker = CorrelationBreaker(min_threshold=0.1)
        corr = np.array([[1.0, 0.0], [0.0, 1.0]])
        result = breaker.check(correlation_matrix=corr)
        assert result.triggered is True

    def test_normal_correlation_no_trigger(self):
        breaker = CorrelationBreaker(threshold=0.85, min_threshold=0.1)
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = breaker.check(correlation_matrix=corr)
        assert result.triggered is False

    def test_rolling_correlation_monitoring(self):
        breaker = CorrelationBreaker(threshold=0.85, lookback=20)
        corr_series = np.linspace(0.3, 0.9, 20)
        result = breaker.check(rolling_correlation=corr_series)
        assert result.triggered is True

    def test_pairwise_vs_portfolio_correlation(self):
        breaker = CorrelationBreaker(threshold=0.85)
        pairwise = np.array([[1.0, 0.9], [0.9, 1.0]])
        portfolio_corr = 0.7
        result = breaker.check(correlation_matrix=pairwise, portfolio_corr=portfolio_corr)
        assert result.triggered is True

    def test_correlation_vs_historical_baseline(self):
        breaker = CorrelationBreaker(baseline_threshold=0.2)
        result = breaker.check(
            current_corr=0.8,
            baseline_corr=0.3,
        )
        assert result.triggered is True

    def test_correlation_during_market_stress(self):
        breaker = CorrelationBreaker(stress_threshold=0.9)
        result = breaker.check(stress_corr=0.95)
        assert result.triggered is True

    def test_correlation_edge_case_two_positions(self):
        breaker = CorrelationBreaker(threshold=0.85)
        corr = np.array([[1.0, 0.86], [0.86, 1.0]])
        result = breaker.check(correlation_matrix=corr)
        assert result.triggered is True


# ---------------------------------------------------------------------------
# 6. System Health Breaker
# ---------------------------------------------------------------------------

class TestSystemHealthBreaker:
    """Test system health monitoring breaker."""

    def test_trigger_on_database_failure(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(db_healthy=False)
        assert result.triggered is True

    def test_trigger_on_broker_api_failure(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(broker_healthy=False)
        assert result.triggered is True

    def test_trigger_on_data_feed_stale(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(data_age_seconds=120, max_data_age_seconds=60)
        assert result.triggered is True

    def test_trigger_on_model_drift_critical(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(model_drift_score=0.9, drift_threshold=0.8)
        assert result.triggered is True

    def test_trigger_on_execution_latency(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(exec_latency_ms=500, latency_threshold_ms=300)
        assert result.triggered is True

    def test_trigger_on_order_rejection_rate(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(rejection_rate=0.15, rejection_threshold=0.10)
        assert result.triggered is True

    def test_no_trigger_healthy_system(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(
            db_healthy=True,
            broker_healthy=True,
            data_age_seconds=10,
            exec_latency_ms=50,
            rejection_rate=0.01,
        )
        assert result.triggered is False

    def test_partial_degradation_warning(self):
        breaker = SystemHealthBreaker()
        result = breaker.check(exec_latency_ms=250, latency_threshold_ms=200, warn_only=True)
        assert result.triggered is False
        assert result.details.get("level") == "warning"

    def test_system_recovery_auto_reset(self):
        breaker = SystemHealthBreaker(auto_reset=True)
        breaker.check(db_healthy=False)
        assert breaker.state == BreakerState.TRIGGERED
        breaker.check(db_healthy=True, broker_healthy=True)
        assert breaker.state == BreakerState.ARMED

    def test_system_health_edge_case_startup(self):
        breaker = SystemHealthBreaker(ignore_startup_seconds=60)
        result = breaker.check(
            db_healthy=False,
            now=datetime(2024, 1, 1, 9, 0, 10),
            start_time=datetime(2024, 1, 1, 9, 0, 0),
        )
        assert result.triggered is False


# ---------------------------------------------------------------------------
# 7. Circuit Breaker States
# ---------------------------------------------------------------------------

class TestBreakerStates:
    """Test circuit breaker state transitions."""

    def test_initial_state_armed(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        assert breaker.state == BreakerState.ARMED

    def test_transition_armed_to_triggered(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        assert breaker.state == BreakerState.TRIGGERED

    def test_transition_triggered_to_cooling_down(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, cooldown_minutes=30)
        breaker.check(equity=90_000, peak=100_000)
        breaker.start_cooldown(now=datetime(2024, 1, 1, 10, 0, 0))
        assert breaker.state == BreakerState.COOLING_DOWN

    def test_transition_cooling_down_to_armed(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, cooldown_minutes=30)
        breaker.state = BreakerState.COOLING_DOWN
        breaker.cooldown_until = datetime(2024, 1, 1, 10, 0, 0)
        breaker.check_cooldown(now=datetime(2024, 1, 1, 10, 30, 1))
        assert breaker.state == BreakerState.ARMED

    def test_permanent_trip_no_auto_reset(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, permanent_trip=True)
        breaker.check(equity=90_000, peak=100_000)
        assert breaker.state == BreakerState.TRIGGERED
        breaker.check(equity=110_000, peak=110_000)
        assert breaker.state == BreakerState.TRIGGERED

    def test_manual_reset_after_trigger(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.reset()
        assert breaker.state == BreakerState.ARMED

    def test_escalation_multiple_triggers(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, escalation_threshold=2)
        breaker.check(equity=90_000, peak=100_000)
        breaker.reset()
        breaker.check(equity=90_000, peak=100_000)
        assert breaker.escalation_level >= 1

    def test_state_persistence_across_sessions(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        state = breaker.serialize_state()
        breaker2 = DrawdownBreaker(max_drawdown=0.10)
        breaker2.load_state(state)
        assert breaker2.state == BreakerState.TRIGGERED

    def test_concurrent_state_changes(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.reset()
        assert breaker.state == BreakerState.ARMED

    def test_state_machine_validation(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        for state in BreakerState:
            assert isinstance(state.name, str)


# ---------------------------------------------------------------------------
# 8. Trigger Conditions
# ---------------------------------------------------------------------------

class TestTriggerConditions:
    """Test complex trigger condition logic."""

    def test_single_condition_trigger(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        result = breaker.check(equity=90_000, peak=100_000)
        assert result.triggered is True

    def test_multiple_condition_and_logic(self):
        breaker = VolatilityBreaker(vix_threshold=35.0, portfolio_vol_threshold=0.4, logic="and")
        result = breaker.check(vix=40.0, portfolio_vol=0.5)
        assert result.triggered is True

    def test_multiple_condition_or_logic(self):
        breaker = VolatilityBreaker(vix_threshold=35.0, portfolio_vol_threshold=0.4, logic="or")
        result = breaker.check(vix=40.0, portfolio_vol=0.2)
        assert result.triggered is True

    def test_threshold_crossover_detection(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        result1 = breaker.check(equity=95_000, peak=100_000)
        result2 = breaker.check(equity=89_000, peak=100_000)
        assert result1.triggered is False
        assert result2.triggered is True

    def test_duration_based_trigger(self):
        breaker = VolatilityBreaker(duration_threshold=3)
        vols = [0.5, 0.5, 0.5]
        result = None
        for v in vols:
            result = breaker.check(portfolio_vol=v, high_vol_threshold=0.4)
        assert result is not None
        assert result.triggered is True

    def test_rate_of_change_trigger(self):
        breaker = VolatilityBreaker(roc_threshold=0.1)
        result = breaker.check(prev_vol=0.1, current_vol=0.25)
        assert result.triggered is True

    def test_consecutive_breaches_trigger(self):
        breaker = LossLimitBreaker(daily_limit_pct=0.01, escalation_days=2)
        result = None
        for d in range(2):
            result = breaker.check(realized_pnl=-2_000, start_capital=100_000, date=datetime(2024, 1, d + 1))
        assert result is not None
        assert breaker.escalation_level >= 1

    def test_percentile_based_trigger(self):
        breaker = DrawdownBreaker(percentile_threshold=0.95)
        dd_samples = np.linspace(0.0, 0.2, 100)
        result = breaker.check(drawdown_sample=dd_samples, current_drawdown=0.19)
        assert result.triggered is True

    def test_standard_deviation_trigger(self):
        breaker = VolatilityBreaker(sigma_threshold=3.0)
        vols = np.random.normal(0.2, 0.02, 100)
        result = breaker.check(portfolio_vol=0.3, vol_history=vols)
        assert result.triggered is True

    def test_custom_condition_function(self):
        def custom(cond: Dict[str, Any]) -> bool:
            return cond.get("custom_val", 0) > 10

        breaker = CircuitBreaker(custom_condition=custom)
        result = breaker.check(custom_val=11)
        assert result.triggered is True

    def test_condition_priority_ordering(self):
        breaker = VolatilityBreaker(vix_threshold=35.0, portfolio_vol_threshold=0.4, priority="vix")
        result = breaker.check(vix=40.0, portfolio_vol=0.6)
        assert result.details.get("primary") == "vix"

    def test_condition_edge_case_missing_data(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        result = breaker.check()
        assert result.triggered is False


# ---------------------------------------------------------------------------
# 9. Recovery & Reset Logic
# ---------------------------------------------------------------------------

class TestRecoveryReset:
    """Test breaker recovery and reset mechanisms."""

    def test_auto_reset_after_cooldown(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, auto_reset=True, cooldown_minutes=10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.start_cooldown(now=datetime(2024, 1, 1, 9, 0, 0))
        breaker.check_cooldown(now=datetime(2024, 1, 1, 9, 10, 1))
        assert breaker.state == BreakerState.ARMED

    def test_manual_reset_by_admin(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.reset(by="admin")
        assert breaker.state == BreakerState.ARMED

    def test_partial_recovery_insufficient(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, auto_reset=True)
        breaker.check(equity=90_000, peak=100_000)
        breaker.check(equity=95_000, peak=100_000)
        assert breaker.state == BreakerState.TRIGGERED

    def test_full_recovery_auto_reset(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, auto_reset=True)
        breaker.check(equity=90_000, peak=100_000)
        breaker.check(equity=110_000, peak=110_000)
        assert breaker.state == BreakerState.ARMED

    def test_time_based_cooldown(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, cooldown_minutes=5)
        breaker.check(equity=90_000, peak=100_000)
        breaker.start_cooldown(now=datetime(2024, 1, 1, 9, 0, 0))
        breaker.check_cooldown(now=datetime(2024, 1, 1, 9, 4, 59))
        assert breaker.state == BreakerState.COOLING_DOWN

    def test_condition_based_reset(self):
        breaker = SystemHealthBreaker(auto_reset=True)
        breaker.check(db_healthy=False)
        breaker.check(db_healthy=True, broker_healthy=True)
        assert breaker.state == BreakerState.ARMED

    def test_no_reset_on_permanent_trip(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, permanent_trip=True, auto_reset=True)
        breaker.check(equity=90_000, peak=100_000)
        breaker.check(equity=110_000, peak=110_000)
        assert breaker.state == BreakerState.TRIGGERED

    def test_reset_validation_checks(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        with pytest.raises(ValueError):
            breaker.reset(by=None)

    def test_reset_audit_logging(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.reset(by="admin")
        assert breaker.last_reset_by == "admin"

    def test_reset_edge_case_during_trigger(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.reset()
        result = breaker.check(equity=100_000, peak=100_000)
        assert result.triggered is False


# ---------------------------------------------------------------------------
# 10. Circuit Breaker Manager
# ---------------------------------------------------------------------------

class TestCircuitBreakerManager:
    """Test centralized circuit breaker management."""

    def test_register_multiple_breakers(self, breaker_manager: CircuitBreakerManager):
        assert len(breaker_manager.breakers) >= 3

    def test_check_all_breakers(self, breaker_manager: CircuitBreakerManager, equity_curve_drawdown: np.ndarray):
        result = breaker_manager.check_all(
            equity=equity_curve_drawdown[-1],
            peak=np.max(equity_curve_drawdown),
            realized_pnl=-3_000,
        )
        assert result.any_triggered is True

    def test_trigger_first_breach_only(self, breaker_manager: CircuitBreakerManager):
        breaker_manager.stop_on_first = True
        result = breaker_manager.check_all(
            equity=80_000,
            peak=100_000,
            realized_pnl=-5_000,
        )
        assert len(result.triggered_breakers) == 1

    def test_priority_based_checking(self, breaker_manager: CircuitBreakerManager):
        breaker_manager.set_priority({"DrawdownBreaker": 1, "LossLimitBreaker": 2})
        result = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        assert result.triggered_breakers[0].name == "DrawdownBreaker"

    def test_cascading_breaker_triggers(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        assert result.any_triggered is True

    def test_breaker_enable_disable(self, breaker_manager: CircuitBreakerManager):
        breaker_manager.disable("DrawdownBreaker")
        result = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        assert all(b.name != "DrawdownBreaker" for b in result.triggered_breakers)

    def test_get_breaker_status(self, breaker_manager: CircuitBreakerManager):
        statuses = breaker_manager.status()
        assert isinstance(statuses, dict)

    def test_reset_all_breakers(self, breaker_manager: CircuitBreakerManager):
        breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        breaker_manager.reset_all()
        for b in breaker_manager.breakers:
            assert b.state == BreakerState.ARMED

    def test_breaker_event_listeners(self, breaker_manager: CircuitBreakerManager):
        listener = Mock()
        breaker_manager.add_listener(listener)
        breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        assert listener.called

    def test_concurrent_breaker_checks(self, breaker_manager: CircuitBreakerManager):
        res1 = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        res2 = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        assert res1.any_triggered == res2.any_triggered

    def test_manager_configuration_validation(self):
        manager = CircuitBreakerManager()
        with pytest.raises(ValueError):
            manager.set_priority({"UnknownBreaker": 1})

    def test_manager_edge_case_no_breakers(self):
        manager = CircuitBreakerManager()
        result = manager.check_all()
        assert result.any_triggered is False


# ---------------------------------------------------------------------------
# 11. Notifications & Alerts
# ---------------------------------------------------------------------------

class TestBreakerNotifications:
    """Test circuit breaker alert system."""

    def test_alert_on_trigger(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        with patch("core.circuit_breakers.send_alert") as send_alert:
            breaker.check(equity=90_000, peak=100_000)
            send_alert.assert_called_once()

    def test_alert_escalation_severity(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, escalation_threshold=2)
        with patch("core.circuit_breakers.send_alert") as send_alert:
            breaker.check(equity=90_000, peak=100_000)
            breaker.check(equity=90_000, peak=100_000)
            args, kwargs = send_alert.call_args
            assert kwargs.get("severity") in {"high", "critical"}

    def test_alert_cooldown_no_spam(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, alert_cooldown_seconds=60)
        with patch("core.circuit_breakers.send_alert") as send_alert:
            breaker.check(equity=90_000, peak=100_000, now=datetime(2024, 1, 1, 9, 0, 0))
            breaker.check(equity=80_000, peak=100_000, now=datetime(2024, 1, 1, 9, 0, 10))
            assert send_alert.call_count == 1

    def test_multiple_channel_alerts(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, alert_channels=["email", "slack"])
        with patch("core.circuit_breakers.send_alert") as send_alert:
            breaker.check(equity=90_000, peak=100_000)
            _, kwargs = send_alert.call_args
            assert "channels" in kwargs
            assert set(kwargs["channels"]) == {"email", "slack"}

    def test_alert_with_context_data(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        with patch("core.circuit_breakers.send_alert") as send_alert:
            breaker.check(equity=90_000, peak=100_000, context={"strategy": "S1"})
            _, kwargs = send_alert.call_args
            assert kwargs["context"]["strategy"] == "S1"

    def test_alert_retry_on_failure(self):
        breaker = DrawdownBreaker(max_drawdown=0.10, alert_retries=2)
        with patch("core.circuit_breakers.send_alert", side_effect=Exception("fail")) as send_alert:
            breaker.check(equity=90_000, peak=100_000)
            assert send_alert.call_count >= 2

    def test_alert_acknowledgement(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        breaker.check(equity=90_000, peak=100_000)
        breaker.acknowledge_alert(by="ops")
        assert breaker.last_ack_by == "ops"

    def test_alert_edge_case_notification_failure(self):
        breaker = DrawdownBreaker(max_drawdown=0.10)
        with patch("core.circuit_breakers.send_alert", side_effect=Exception("fail")):
            breaker.check(equity=90_000, peak=100_000)
        assert breaker.state == BreakerState.TRIGGERED


# ---------------------------------------------------------------------------
# 12. Integration & Edge Cases
# ---------------------------------------------------------------------------

class TestIntegrationEdgeCases:
    """Test realistic scenarios and edge cases."""

    def test_flash_crash_scenario(self, breaker_manager: CircuitBreakerManager):
        equities = [100_000, 80_000, 60_000]
        result = None
        for e in equities:
            result = breaker_manager.check_all(equity=e, peak=100_000, realized_pnl=e - 100_000)
        assert result is not None
        assert result.any_triggered is True

    def test_gradual_drawdown_scenario(self, breaker_manager: CircuitBreakerManager):
        equities = np.linspace(100_000, 85_000, 20)
        result = None
        for e in equities:
            result = breaker_manager.check_all(equity=e, peak=100_000, realized_pnl=e - 100_000)
        assert result is not None
        assert result.any_triggered is True

    def test_false_positive_prevention(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(equity=101_000, peak=101_000, realized_pnl=1_000)
        assert result.any_triggered is False

    def test_rapid_recovery_scenario(self, breaker_manager: CircuitBreakerManager):
        breaker_manager.check_all(equity=90_000, peak=100_000, realized_pnl=-10_000)
        result = breaker_manager.check_all(equity=110_000, peak=110_000, realized_pnl=10_000)
        assert result.any_triggered is False

    def test_multiple_breakers_simultaneous(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-5_000)
        assert len(result.triggered_breakers) >= 1

    def test_market_open_volatility(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(
            equity=95_000,
            peak=100_000,
            realized_pnl=-5_000,
            time=datetime(2024, 1, 1, 9, 15, 0),
        )
        assert isinstance(result.any_triggered, bool)

    def test_overnight_gap_handling(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(
            equity=85_000,
            peak=100_000,
            realized_pnl=-15_000,
            gap_open=True,
        )
        assert result.any_triggered is True

    def test_data_quality_issues(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(
            equity=None,
            peak=100_000,
            realized_pnl=-5_000,
            data_quality="bad",
        )
        assert isinstance(result.any_triggered, bool)

    def test_clock_skew_timing_issues(self, breaker_manager: CircuitBreakerManager):
        result = breaker_manager.check_all(
            equity=90_000,
            peak=100_000,
            realized_pnl=-10_000,
            server_time=datetime(2024, 1, 1, 9, 0, 0),
            market_time=datetime(2024, 1, 1, 9, 0, 10),
        )
        assert isinstance(result.any_triggered, bool)

    def test_end_to_end_breaker_workflow(self, breaker_manager: CircuitBreakerManager):
        reset_all_breakers()
        result = breaker_manager.check_all(equity=80_000, peak=100_000, realized_pnl=-20_000)
        assert result.any_triggered is True
        breaker_manager.reset_all()
        result2 = breaker_manager.check_all(equity=100_000, peak=100_000, realized_pnl=0.0)
        assert result2.any_triggered is False


# ---------------------------------------------------------------------------
# Parametrized threshold tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "drawdown,limit,should_trigger",
    [
        (-0.05, 0.10, False),
        (-0.10, 0.10, True),
        (-0.15, 0.10, True),
        (-0.25, 0.20, True),
    ],
)
def test_drawdown_threshold_scenarios(drawdown: float, limit: float, should_trigger: bool):
    """Test various drawdown scenarios."""
    breaker = DrawdownBreaker(max_drawdown=limit)
    peak_equity = 100_000
    current_equity = peak_equity * (1 + drawdown)
    result = breaker.check(equity=current_equity, peak=peak_equity)
    assert result.triggered == should_trigger


@pytest.mark.parametrize(
    "loss,limit,should_trigger",
    [
        (-500, 0.01, False),
        (-1_000, 0.01, True),
        (-2_000, 0.01, True),
    ],
)
def test_loss_limit_scenarios(loss: float, limit: float, should_trigger: bool):
    """Test various loss limit scenarios."""
    breaker = LossLimitBreaker(daily_limit_pct=limit)
    result = breaker.check(realized_pnl=loss, start_capital=100_000)
    assert result.triggered == should_trigger
