"""
Unit tests for position sizing logic in ALPHA-PRIME v2.0.

Covers:
- Fixed fractional, Kelly, risk-based, volatility-adjusted, and Optimal F sizing.
- Portfolio-level constraints and multi-strategy allocation.
- Mathematical validation of sizing formulas and edge-case handling.

All tests are:
- Fast (<100ms), isolated (no APIs), deterministic (fixed seeds),
- And emphasize mathematical correctness.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Dict, Any, List

import numpy as np
import pytest

from core.position_sizer import (
    PositionSizer,
    FixedFractionalSizer,
    KellyCriterionSizer,
    RiskBasedSizer,
    VolatilityAdjustedSizer,
    OptimalFSizer,
    calculate_position_size,
    calculate_kelly_fraction,
    calculate_optimal_f,
    validate_position_size,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_portfolio() -> Dict[str, Any]:
    """Base portfolio for testing."""
    return {
        "capital": 100_000.0,
        "cash": 100_000.0,
        "positions": [],
    }


@pytest.fixture
def kelly_params() -> Dict[str, float]:
    """Sample Kelly criterion parameters."""
    return {
        "win_rate": 0.55,
        "avg_win": 1.5,
        "avg_loss": 1.0,
        "kelly_fraction": 0.25,  # Quarter Kelly
    }


@pytest.fixture
def risk_params() -> Dict[str, float]:
    """Sample risk-based sizing parameters."""
    return {
        "portfolio_risk_pct": 0.01,  # 1% per trade
        "entry_price": 150.0,
        "stop_loss": 145.0,  # 3.33% stop
        "max_position_pct": 0.05,  # 5% max
    }


@pytest.fixture
def volatility_data() -> Dict[str, float]:
    """Sample volatility data (annualized)."""
    return {
        "AAPL": 0.25,
        "MSFT": 0.20,
        "TSLA": 0.60,
    }


@pytest.fixture
def trade_history() -> np.ndarray:
    """Historical trade results for Kelly/Optimal F."""
    np.random.seed(42)
    wins = np.random.normal(0.02, 0.005, 60)
    losses = np.random.normal(-0.01, 0.003, 40)
    return np.concatenate([wins, losses])


@pytest.fixture
def position_sizer() -> PositionSizer:
    """Default PositionSizer instance."""
    return PositionSizer(
        max_position_pct=0.05,
        max_portfolio_risk=0.02,
        min_position_size=100,
    )


# ---------------------------------------------------------------------------
# 1. Fixed Fractional Sizing
# ---------------------------------------------------------------------------

class TestFixedFractionalSizing:
    """Test fixed fractional position sizing."""

    def test_fixed_fraction_basic_calculation(self, base_portfolio: Dict[str, Any]):
        """Test basic fixed fractional sizing."""
        sizer = FixedFractionalSizer(fraction=0.05)
        position_size = sizer.calculate(capital=base_portfolio["capital"], price=150.0)
        expected_shares = (100_000 * 0.05) / 150.0
        assert abs(position_size - int(expected_shares)) <= 1

    def test_fixed_fraction_with_max_position_limit(self):
        """Test max position limit enforcement."""
        sizer = FixedFractionalSizer(fraction=0.10, max_position_pct=0.05)
        position_size = sizer.calculate(capital=100_000, price=100.0)
        expected = (100_000 * 0.05) / 100.0
        assert position_size == int(expected)

    def test_fixed_fraction_with_portfolio_value(self):
        """Test sizing uses total portfolio value, not just cash."""
        sizer = FixedFractionalSizer(fraction=0.05)
        size = sizer.calculate(capital=200_000, price=100.0)
        assert size == int((200_000 * 0.05) / 100.0)

    def test_fixed_fraction_rounding(self):
        """Test rounding to whole shares."""
        sizer = FixedFractionalSizer(fraction=0.05)
        size = sizer.calculate(capital=10_000, price=333.33)
        assert isinstance(size, int)

    def test_fixed_fraction_minimum_position_size(self):
        """Test enforcing minimum position size."""
        sizer = FixedFractionalSizer(fraction=0.001, min_position_size=10)
        size = sizer.calculate(capital=10_000, price=100.0)
        assert size >= 10

    def test_fixed_fraction_multiple_positions(self, base_portfolio: Dict[str, Any]):
        """Test multiple positions do not exceed total exposure."""
        sizer = FixedFractionalSizer(fraction=0.05)
        s1 = sizer.calculate(capital=base_portfolio["capital"], price=100.0)
        s2 = sizer.calculate(capital=base_portfolio["capital"], price=200.0)
        notional = s1 * 100.0 + s2 * 200.0
        assert notional <= base_portfolio["capital"] * 0.10 + 1

    def test_fixed_fraction_cash_constraints(self, base_portfolio: Dict[str, Any]):
        """Test sizing respects available cash."""
        sizer = FixedFractionalSizer(fraction=0.5)
        size = sizer.calculate(capital=base_portfolio["cash"], price=1_000.0, available_cash=5_000.0)
        assert size <= 5_000.0 / 1_000.0

    def test_fixed_fraction_zero_portfolio_value(self):
        """Test zero capital returns zero size."""
        sizer = FixedFractionalSizer(fraction=0.05)
        size = sizer.calculate(capital=0.0, price=100.0)
        assert size == 0

    def test_fixed_fraction_negative_values_rejected(self):
        """Test negative capital or price rejected."""
        sizer = FixedFractionalSizer(fraction=0.05)
        with pytest.raises(ValueError):
            sizer.calculate(capital=-10_000, price=100.0)
        with pytest.raises(ValueError):
            sizer.calculate(capital=10_000, price=-100.0)

    def test_fixed_fraction_fraction_validation(self):
        """Test invalid fraction values rejected."""
        with pytest.raises(ValueError):
            FixedFractionalSizer(fraction=-0.1)
        with pytest.raises(ValueError):
            FixedFractionalSizer(fraction=1.5)


# ---------------------------------------------------------------------------
# 2. Kelly Criterion Sizing
# ---------------------------------------------------------------------------

class TestKellyCriterionSizing:
    """Test Kelly criterion optimal sizing."""

    def test_kelly_formula_basic(self, kelly_params: Dict[str, float]):
        """Test Kelly formula: f = (bp - q) / b."""
        f = calculate_kelly_fraction(
            win_rate=kelly_params["win_rate"],
            avg_win=kelly_params["avg_win"],
            avg_loss=kelly_params["avg_loss"],
        )
        assert abs(f - 0.25) < 0.01

    def test_kelly_with_win_rate_probability(self):
        f = calculate_kelly_fraction(win_rate=0.6, avg_win=2.0, avg_loss=1.0)
        assert f > 0.0

    def test_kelly_with_average_win_loss(self):
        f = calculate_kelly_fraction(win_rate=0.55, avg_win=3.0, avg_loss=1.0)
        assert f > 0.0

    def test_kelly_fractional_kelly_25_percent(self, kelly_params: Dict[str, float], base_portfolio: Dict[str, Any]):
        sizer = KellyCriterionSizer(kelly_fraction=0.25)
        size = sizer.calculate(
            capital=base_portfolio["capital"],
            price=100.0,
            win_rate=kelly_params["win_rate"],
            avg_win=kelly_params["avg_win"],
            avg_loss=kelly_params["avg_loss"],
        )
        full_kelly = calculate_kelly_fraction(
            kelly_params["win_rate"],
            kelly_params["avg_win"],
            kelly_params["avg_loss"],
        )
        frac = size * 100.0 / base_portfolio["capital"]
        assert abs(frac - full_kelly * 0.25) < 0.02

    def test_kelly_fractional_kelly_50_percent(self, kelly_params: Dict[str, float], base_portfolio: Dict[str, Any]):
        sizer = KellyCriterionSizer(kelly_fraction=0.5)
        size = sizer.calculate(
            capital=base_portfolio["capital"],
            price=100.0,
            win_rate=kelly_params["win_rate"],
            avg_win=kelly_params["avg_win"],
            avg_loss=kelly_params["avg_loss"],
        )
        assert size > 0

    def test_kelly_half_kelly_protection(self, kelly_params: Dict[str, float]):
        full = calculate_kelly_fraction(
            kelly_params["win_rate"],
            kelly_params["avg_win"],
            kelly_params["avg_loss"],
        )
        half = full * 0.5
        assert half < full

    def test_kelly_negative_expectancy_returns_zero(self):
        f = calculate_kelly_fraction(win_rate=0.4, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_kelly_edge_case_zero_probability(self):
        f = calculate_kelly_fraction(win_rate=0.0, avg_win=1.0, avg_loss=1.0)
        assert f == 0.0

    def test_kelly_edge_case_100_percent_win_rate(self):
        f = calculate_kelly_fraction(win_rate=1.0, avg_win=1.0, avg_loss=1.0)
        assert f <= 1.0

    def test_kelly_with_actual_trade_history(self, trade_history: np.ndarray):
        sizer = KellyCriterionSizer()
        f = sizer.from_trade_history(trade_history)
        assert 0.0 <= f <= 1.0

    def test_kelly_position_limit_capping(self, base_portfolio: Dict[str, Any]):
        sizer = KellyCriterionSizer(kelly_fraction=1.0, max_position_pct=0.10)
        size = sizer.calculate(
            capital=base_portfolio["capital"],
            price=100.0,
            win_rate=0.9,
            avg_win=2.0,
            avg_loss=1.0,
        )
        assert size <= (base_portfolio["capital"] * 0.10) / 100.0

    def test_kelly_with_asymmetric_payoffs(self):
        f = calculate_kelly_fraction(win_rate=0.4, avg_win=3.0, avg_loss=1.0)
        assert f > 0.0

    def test_kelly_mathematical_validation(self):
        p, b = 0.6, 2.0
        q = 1 - p
        manual = (p * b - q) / b
        f = calculate_kelly_fraction(win_rate=p, avg_win=b, avg_loss=1.0)
        assert abs(manual - f) < 1e-6

    def test_kelly_concurrent_positions(self, base_portfolio: Dict[str, Any]):
        sizer = KellyCriterionSizer(kelly_fraction=0.25, max_portfolio_risk=0.02)
        s1 = sizer.calculate(100_000, 100.0, 0.55, 1.5, 1.0)
        s2 = sizer.calculate(100_000, 100.0, 0.55, 1.5, 1.0)
        notional = (s1 + s2) * 100.0
        assert notional <= base_portfolio["capital"] * 0.02 * 2 + 1

    def test_kelly_leveraged_vs_unleveraged(self, base_portfolio: Dict[str, Any]):
        sizer = KellyCriterionSizer(kelly_fraction=0.25, allow_leverage=True)
        size_lev = sizer.calculate(100_000, 100.0, 0.6, 2.0, 1.0)
        sizer2 = KellyCriterionSizer(kelly_fraction=0.25, allow_leverage=False)
        size_unlev = sizer2.calculate(100_000, 100.0, 0.6, 2.0, 1.0)
        assert size_lev >= size_unlev


# ---------------------------------------------------------------------------
# 3. Risk-Based Sizing
# ---------------------------------------------------------------------------

class TestRiskBasedSizing:
    """Test risk-based position sizing (stop-loss based)."""

    def test_risk_based_with_stop_loss(self, risk_params: Dict[str, float], base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=risk_params["portfolio_risk_pct"])
        size = sizer.calculate(
            capital=base_portfolio["capital"],
            entry_price=risk_params["entry_price"],
            stop_loss=risk_params["stop_loss"],
        )
        assert size == 200  # 1000 / 5

    def test_risk_based_with_portfolio_risk_percentage(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.02)
        size = sizer.calculate(capital=base_portfolio["capital"], entry_price=100.0, stop_loss=95.0)
        assert size == 400  # 2000 / 5

    def test_risk_based_with_atr_stop(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01)
        size = sizer.calculate(capital=base_portfolio["capital"], entry_price=100.0, atr=2.0, atr_multiple=2.5)
        risk_per_share = 2.0 * 2.5
        assert size == int((100_000 * 0.01) / risk_per_share)

    def test_risk_based_with_percentage_stop(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01)
        size = sizer.calculate(capital=base_portfolio["capital"], entry_price=100.0, stop_pct=0.02)
        assert size == int((100_000 * 0.01) / (100.0 * 0.02))

    def test_risk_based_with_dollar_stop(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01)
        size = sizer.calculate(capital=base_portfolio["capital"], dollar_risk_per_share=5.0)
        assert size == int((100_000 * 0.01) / 5.0)

    def test_risk_based_position_limit_enforcement(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.5, max_position_pct=0.05)
        size = sizer.calculate(capital=base_portfolio["capital"], entry_price=100.0, stop_loss=90.0)
        assert size <= int((100_000 * 0.05) / 100.0)

    def test_risk_based_minimum_rr_ratio(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01, min_rr=2.0)
        size = sizer.calculate(
            capital=base_portfolio["capital"],
            entry_price=100.0,
            stop_loss=95.0,
            target_price=102.0,
        )
        assert size == 0

    def test_risk_based_invalid_stop_loss_rejected(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer()
        with pytest.raises(ValueError):
            sizer.calculate(capital=base_portfolio["capital"], entry_price=150.0, stop_loss=155.0)

    def test_risk_based_zero_risk_rejected(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer()
        with pytest.raises(ValueError):
            sizer.calculate(capital=base_portfolio["capital"], entry_price=100.0, stop_loss=100.0)

    def test_risk_based_correlated_positions(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01, portfolio_heat_limit=0.02)
        s1 = sizer.calculate(100_000, 100.0, 95.0, correlation=0.9)
        s2 = sizer.calculate(100_000, 100.0, 95.0, correlation=0.9, current_heat=0.02)
        assert s1 > 0
        assert s2 == 0

    def test_risk_based_portfolio_heat_limit(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01, portfolio_heat_limit=0.02)
        size = sizer.calculate(100_000, 100.0, 95.0, current_heat=0.02)
        assert size == 0

    def test_risk_based_volatility_scaling(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01, vol_scaling=True)
        size_low = sizer.calculate(100_000, 100.0, 95.0, volatility=0.1)
        size_high = sizer.calculate(100_000, 100.0, 95.0, volatility=0.4)
        assert size_low > size_high


# ---------------------------------------------------------------------------
# 4. Volatility-Adjusted Sizing
# ---------------------------------------------------------------------------

class TestVolatilityAdjustedSizing:
    """Test volatility-based position sizing."""

    def test_volatility_inverse_volatility_scaling(self, volatility_data: Dict[str, float], base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20)
        msft_size = sizer.calculate(100_000, 300.0, volatility_data["MSFT"])
        tsla_size = sizer.calculate(100_000, 200.0, volatility_data["TSLA"])
        assert msft_size > tsla_size * 2.5

    def test_volatility_target_volatility_sizing(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20)
        size = sizer.calculate(100_000, 100.0, 0.20)
        assert size > 0

    def test_volatility_with_historical_volatility(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20)
        np.random.seed(0)
        returns = np.random.normal(0.001, 0.02, 252)
        vol = returns.std() * np.sqrt(252)
        size = sizer.calculate(100_000, 100.0, vol)
        assert size > 0

    def test_volatility_with_ewma_volatility(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20, ewma_lambda=0.94)
        returns = np.full(252, 0.01)
        vol = sizer.estimate_ewma_volatility(returns)
        assert vol >= 0.0

    def test_volatility_normalized_position_size(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20)
        sizes = [
            sizer.calculate(100_000, 100.0, v) for v in (0.10, 0.20, 0.40)
        ]
        assert sizes[0] > sizes[1] > sizes[2]

    def test_volatility_extreme_volatility_capping(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20, max_volatility=0.50)
        size = sizer.calculate(100_000, 100.0, volatility=1.0)
        capped = sizer.calculate(100_000, 100.0, volatility=0.50)
        assert size == capped

    def test_volatility_low_volatility_floor(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20, min_volatility=0.10)
        size = sizer.calculate(100_000, 100.0, volatility=0.01)
        assert size == sizer.calculate(100_000, 100.0, volatility=0.10)

    def test_volatility_regime_adjusted_sizing(self, base_portfolio: Dict[str, Any]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20, regime_multiplier=0.5)
        size_low = sizer.calculate(100_000, 100.0, volatility=0.20, regime="low_vol")
        size_high = sizer.calculate(100_000, 100.0, volatility=0.20, regime="high_vol")
        assert size_low > size_high

    def test_volatility_multi_asset_normalization(self, volatility_data: Dict[str, float]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20)
        sizes = {
            sym: sizer.calculate(100_000, 100.0, vol)
            for sym, vol in volatility_data.items()
        }
        assert sizes["MSFT"] > sizes["TSLA"]

    def test_volatility_edge_case_zero_volatility(self):
        sizer = VolatilityAdjustedSizer(target_volatility=0.20)
        size = sizer.calculate(100_000, 100.0, volatility=0.0)
        assert size > 0  # treat as min_volatility


# ---------------------------------------------------------------------------
# 5. Optimal F Sizing
# ---------------------------------------------------------------------------

class TestOptimalFSizing:
    """Test Optimal F position sizing (Ralph Vince)."""

    def test_optimal_f_basic_calculation(self, trade_history: np.ndarray):
        f = calculate_optimal_f(trade_history)
        assert 0.0 <= f <= 1.0

    def test_optimal_f_from_trade_history(self, trade_history: np.ndarray):
        sizer = OptimalFSizer()
        f = sizer.from_trade_history(trade_history)
        assert f == calculate_optimal_f(trade_history)

    def test_optimal_f_geometric_mean_maximization(self, trade_history: np.ndarray):
        sizer = OptimalFSizer()
        f = sizer.from_trade_history(trade_history)
        fractions = np.linspace(0.1, 1.0, 10)
        equities = []
        for frac in fractions:
            eq = 100_000
            for r in trade_history:
                eq *= (1 + frac * r)
            equities.append(eq)
        best_idx = int(np.argmax(equities))
        assert abs(fractions[best_idx] - f) < 0.2

    def test_optimal_f_with_varying_position_sizes(self, trade_history: np.ndarray):
        f1 = calculate_optimal_f(trade_history)
        f2 = calculate_optimal_f(trade_history * 2.0)
        assert f1 == pytest.approx(f2)

    def test_optimal_f_fractional_optimal_f(self, trade_history: np.ndarray):
        sizer = OptimalFSizer(fraction=0.5)
        f = sizer.from_trade_history(trade_history)
        full = calculate_optimal_f(trade_history)
        assert f == pytest.approx(full * 0.5)

    def test_optimal_f_edge_case_all_wins(self):
        trades = np.full(50, 0.02)
        f = calculate_optimal_f(trades)
        assert f > 0.0

    def test_optimal_f_edge_case_all_losses(self):
        trades = np.full(50, -0.02)
        f = calculate_optimal_f(trades)
        assert f == 0.0

    def test_optimal_f_vs_kelly_comparison(self, trade_history: np.ndarray):
        f_opt = calculate_optimal_f(trade_history)
        f_kelly = calculate_kelly_fraction(0.6, 2.0, 1.0)
        assert abs(f_opt - f_kelly) < 0.5


# ---------------------------------------------------------------------------
# 6. Portfolio-Level Constraints
# ---------------------------------------------------------------------------

class TestPortfolioConstraints:
    """Test portfolio-level position sizing constraints."""

    def test_max_single_position_limit(self, position_sizer: PositionSizer):
        size = position_sizer.size_for_price(capital=100_000, price=100.0)
        assert size <= (100_000 * position_sizer.max_position_pct) / 100.0

    def test_max_total_exposure_limit(self, position_sizer: PositionSizer):
        sizes = [position_sizer.size_for_price(100_000, 100.0) for _ in range(10)]
        total_notional = sum(s * 100.0 for s in sizes)
        assert total_notional <= 100_000 * 1.0 + 1

    def test_max_sector_concentration(self, position_sizer: PositionSizer):
        heat = position_sizer.calculate_heat(sector_exposure={"tech": 0.7})
        assert heat <= 1.0

    def test_max_correlation_constraint(self, position_sizer: PositionSizer):
        allowed = position_sizer.check_correlation_limit(correlation=0.9, max_correlation=0.8)
        assert allowed is False

    def test_leverage_limits(self, position_sizer: PositionSizer):
        lev = position_sizer.calculate_leverage(exposure=200_000, equity=100_000)
        assert lev == 2.0

    def test_margin_requirements(self, position_sizer: PositionSizer):
        margin_ok = position_sizer.check_margin_requirements(exposure=50_000, margin_available=10_000, margin_rate=0.2)
        assert margin_ok is True

    def test_cash_reserve_requirements(self, position_sizer: PositionSizer):
        ok = position_sizer.check_cash_reserve(cash=20_000, equity=100_000, min_cash_pct=0.1)
        assert ok is True

    def test_position_count_limits(self, position_sizer: PositionSizer):
        allowed = position_sizer.check_position_count(current_count=20, max_positions=30)
        assert allowed is True

    def test_long_short_balance(self, position_sizer: PositionSizer):
        balance = position_sizer.long_short_balance(long_exposure=60_000, short_exposure=-40_000, equity=100_000)
        assert -1.0 <= balance <= 1.0

    def test_portfolio_heat_calculation(self, position_sizer: PositionSizer):
        heat = position_sizer.calculate_heat(position_risks=[0.01, 0.02, 0.005])
        assert heat >= 0.0

    def test_incremental_position_sizing(self, position_sizer: PositionSizer):
        inc = position_sizer.incremental_size(current_size=100, target_size=150)
        assert inc == 50

    def test_scale_in_scale_out_logic(self, position_sizer: PositionSizer):
        scale_in = position_sizer.scale_in(current_size=100, scale_step=0.5)
        scale_out = position_sizer.scale_out(current_size=100, scale_step=0.5)
        assert scale_in == 150
        assert scale_out == 50


# ---------------------------------------------------------------------------
# 7. Multi-Strategy Allocation
# ---------------------------------------------------------------------------

class TestMultiStrategyAllocation:
    """Test position sizing across multiple strategies."""

    def test_strategy_allocation_weights(self, position_sizer: PositionSizer):
        weights = position_sizer.allocate_to_strategies(
            capital=100_000,
            strategy_weights={"s1": 0.5, "s2": 0.5},
        )
        assert abs(sum(weights.values()) - 100_000) < 1e-6

    def test_strategy_risk_budgeting(self, position_sizer: PositionSizer):
        budgets = position_sizer.risk_budget(
            total_risk=0.02,
            strategy_risks={"s1": 1.0, "s2": 2.0},
        )
        assert budgets["s2"] < budgets["s1"]

    def test_strategy_capacity_constraints(self, position_sizer: PositionSizer):
        allocs = position_sizer.apply_capacity_constraints(
            allocations={"s1": 60_000, "s2": 60_000},
            capacities={"s1": 50_000, "s2": 100_000},
        )
        assert allocs["s1"] <= 50_000
        assert allocs["s2"] == 60_000

    def test_strategy_correlation_adjustment(self, position_sizer: PositionSizer):
        allocs = position_sizer.adjust_for_correlation(
            allocations={"s1": 50_000, "s2": 50_000},
            correlations={("s1", "s2"): 0.9},
        )
        assert allocs["s1"] < 50_000

    def test_strategy_performance_based_allocation(self, position_sizer: PositionSizer):
        allocs = position_sizer.performance_based_allocation(
            capital=100_000,
            sharpe_ratios={"s1": 1.0, "s2": 2.0},
        )
        assert allocs["s2"] > allocs["s1"]

    def test_strategy_equal_risk_contribution(self, position_sizer: PositionSizer):
        allocs = position_sizer.equal_risk_contribution(
            capital=100_000,
            volatilities={"s1": 0.2, "s2": 0.4},
        )
        assert allocs["s1"] < allocs["s2"]

    def test_strategy_max_drawdown_allocation(self, position_sizer: PositionSizer):
        allocs = position_sizer.max_dd_allocation(
            capital=100_000,
            max_drawdowns={"s1": 0.1, "s2": 0.3},
        )
        assert allocs["s1"] > allocs["s2"]

    def test_strategy_dynamic_reallocation(self, position_sizer: PositionSizer):
        allocs = position_sizer.dynamic_reallocation(
            capital=100_000,
            current_allocations={"s1": 50_000, "s2": 50_000},
            performance={"s1": -0.1, "s2": 0.1},
        )
        assert allocs["s2"] > allocs["s1"]


# ---------------------------------------------------------------------------
# 8. Mathematical Validation
# ---------------------------------------------------------------------------

class TestMathematicalValidation:
    """Test mathematical correctness of sizing formulas."""

    def test_kelly_formula_derivation(self):
        p, b = 0.6, 2.0
        q = 1 - p
        manual = (p * b - q) / b
        f = calculate_kelly_fraction(p, b, 1.0)
        assert abs(manual - f) < 1e-6

    def test_optimal_f_convergence(self, trade_history: np.ndarray):
        f1 = calculate_optimal_f(trade_history[:50])
        f2 = calculate_optimal_f(trade_history)
        assert abs(f1 - f2) < 0.2

    def test_risk_parity_allocation(self, position_sizer: PositionSizer):
        allocs = position_sizer.risk_parity(
            capital=100_000,
            volatilities={"A": 0.1, "B": 0.2},
        )
        assert allocs["A"] < allocs["B"]

    def test_mean_variance_optimization(self, position_sizer: PositionSizer):
        weights = position_sizer.mean_variance_opt(
            expected_returns=np.array([0.1, 0.15]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
            risk_aversion=3.0,
        )
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_sharpe_ratio_maximization(self, position_sizer: PositionSizer):
        weights = position_sizer.max_sharpe_portfolio(
            expected_returns=np.array([0.1, 0.15]),
            cov=np.array([[0.04, 0.01], [0.01, 0.09]]),
        )
        assert abs(weights.sum() - 1.0) < 1e-6

    def test_var_based_sizing_accuracy(self, position_sizer: PositionSizer):
        size = position_sizer.var_based_size(
            capital=100_000,
            var_95=0.02,
            price=100.0,
            risk_pct=0.01,
        )
        assert size == int((100_000 * 0.01) / (100.0 * 0.02))

    def test_monte_carlo_sizing_validation(self, position_sizer: PositionSizer):
        np.random.seed(0)
        rets = np.random.normal(0.001, 0.02, (1000, 50))
        size = position_sizer.monte_carlo_sizing(
            capital=100_000,
            price=100.0,
            return_paths=rets,
            risk_pct=0.01,
        )
        assert size > 0

    def test_geometric_vs_arithmetic_returns(self):
        rets = np.array([0.1, -0.05, 0.02])
        geo = np.prod(1 + rets) ** (1 / len(rets)) - 1
        arith = rets.mean()
        assert geo <= arith

    def test_compound_growth_rate(self, trade_history: np.ndarray):
        growth = np.prod(1 + trade_history) - 1
        assert isinstance(growth, float)

    def test_tail_risk_adjustments(self, position_sizer: PositionSizer):
        size = position_sizer.tail_risk_adjusted_size(
            capital=100_000,
            price=100.0,
            var_99=0.05,
            risk_pct=0.01,
        )
        assert size > 0


# ---------------------------------------------------------------------------
# 9. Edge Cases & Validation
# ---------------------------------------------------------------------------

class TestEdgeCasesValidation:
    """Test edge cases and input validation."""

    def test_zero_capital_rejected(self):
        with pytest.raises(ValueError):
            calculate_position_size(capital=0.0, price=100.0, method="fixed_fraction", fraction=0.05)

    def test_negative_capital_rejected(self):
        with pytest.raises(ValueError):
            calculate_position_size(capital=-1.0, price=100.0, method="fixed_fraction", fraction=0.05)

    def test_zero_price_rejected(self):
        with pytest.raises(ValueError):
            calculate_position_size(capital=10_000, price=0.0, method="fixed_fraction", fraction=0.05)

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            calculate_position_size(capital=10_000, price=-100.0, method="fixed_fraction", fraction=0.05)

    def test_invalid_win_rate_rejected(self):
        with pytest.raises(ValueError):
            calculate_kelly_fraction(win_rate=1.5, avg_win=1.0, avg_loss=1.0)

    def test_invalid_fraction_rejected(self):
        with pytest.raises(ValueError):
            validate_position_size(size=100, fraction=-0.1)

    def test_extreme_volatility_handling(self):
        sizer = VolatilityAdjustedSizer(target_volatility=0.2)
        size = sizer.calculate(100_000, 100.0, volatility=10.0)
        assert size > 0

    def test_float_precision_handling(self):
        size = calculate_position_size(capital=100_000.0, price=123.4567, method="fixed_fraction", fraction=0.05)
        assert isinstance(size, int)

    def test_integer_share_rounding(self):
        size = calculate_position_size(capital=10_000, price=333.33, method="fixed_fraction", fraction=0.1)
        assert isinstance(size, int)

    def test_minimum_lot_size_enforcement(self):
        size = calculate_position_size(
            capital=10_000,
            price=333.33,
            method="fixed_fraction",
            fraction=0.1,
            lot_size=10,
        )
        assert size % 10 == 0

    def test_currency_conversion_accuracy(self):
        usd_capital = 100_000.0
        eur_capital = usd_capital * 0.9
        size_usd = calculate_position_size(usd_capital, 100.0, "fixed_fraction", fraction=0.05)
        size_eur = calculate_position_size(eur_capital, 90.0, "fixed_fraction", fraction=0.05)
        assert abs(size_usd - size_eur) <= 1

    def test_concurrent_modification_safety(self, position_sizer: PositionSizer):
        size1 = position_sizer.size_for_price(100_000, 100.0)
        size2 = position_sizer.size_for_price(100_000, 100.0)
        assert size1 == size2


# ---------------------------------------------------------------------------
# 10. Integration & Scenarios
# ---------------------------------------------------------------------------

class TestIntegrationScenarios:
    """Test realistic trading scenarios."""

    def test_new_account_first_trade(self, base_portfolio: Dict[str, Any]):
        size = calculate_position_size(
            capital=base_portfolio["capital"],
            price=100.0,
            method="fixed_fraction",
            fraction=0.02,
        )
        assert size == int((100_000 * 0.02) / 100.0)

    def test_scaling_into_winning_position(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01)
        size1 = sizer.calculate(100_000, 100.0, stop_loss=95.0)
        size2 = sizer.calculate(100_000, 110.0, stop_loss=105.0)
        assert size2 <= size1

    def test_reducing_losing_position(self, base_portfolio: Dict[str, Any]):
        sizer = RiskBasedSizer(portfolio_risk_pct=0.01)
        size1 = sizer.calculate(100_000, 100.0, stop_loss=95.0)
        size2 = sizer.calculate(100_000, 90.0, stop_loss=85.0)
        assert size2 <= size1

    def test_portfolio_rebalancing_sizing(self, volatility_data: Dict[str, float]):
        sizer = VolatilityAdjustedSizer(target_volatility=0.2)
        sizes = {
            sym: sizer.calculate(100_000, 100.0, vol) for sym, vol in volatility_data.items()
        }
        assert sum(s * 100.0 for s in sizes.values()) <= 100_000 * 1.5

    def test_high_volatility_market_crash(self, trade_history: np.ndarray):
        sizer = KellyCriterionSizer(kelly_fraction=0.25)
        np.random.seed(1)
        crash = np.random.normal(-0.05, 0.03, 50)
        f = sizer.from_trade_history(crash)
        assert f < 0.1

    def test_low_volatility_grinding_market(self, trade_history: np.ndarray):
        sizer = KellyCriterionSizer(kelly_fraction=0.25)
        np.random.seed(2)
        grind = np.random.normal(0.002, 0.005, 50)
        f = sizer.from_trade_history(grind)
        assert f > 0.0

    def test_correlated_positions_sizing(self, position_sizer: PositionSizer):
        allowable = position_sizer.check_correlation_limit(0.9, 0.8)
        assert allowable is False

    def test_pairs_trading_sizing(self, position_sizer: PositionSizer):
        size1, size2 = position_sizer.pairs_trade_size(
            capital=100_000,
            price_long=100.0,
            price_short=100.0,
            vol_long=0.2,
            vol_short=0.2,
        )
        assert size1 == size2

    def test_portfolio_heat_breach_response(self, position_sizer: PositionSizer):
        heat = position_sizer.calculate_heat(position_risks=[0.01, 0.02, 0.03])
        assert heat > 0.05

    def test_end_to_end_sizing_workflow(self, base_portfolio: Dict[str, Any], trade_history: np.ndarray):
        f = calculate_optimal_f(trade_history)
        kelly = calculate_kelly_fraction(0.55, 1.5, 1.0)
        fraction = min(f, kelly, 0.05)
        size = calculate_position_size(
            capital=base_portfolio["capital"],
            price=100.0,
            method="fixed_fraction",
            fraction=fraction,
        )
        assert size > 0


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "win_rate,avg_win,avg_loss,expected_kelly",
    [
        (0.60, 2.0, 1.0, 0.40),
        (0.55, 1.5, 1.0, 0.25),
        (0.50, 1.0, 1.0, 0.00),
        (0.45, 1.0, 1.0, -0.10),
    ],
)
def test_kelly_various_scenarios(win_rate: float, avg_win: float, avg_loss: float, expected_kelly: float):
    """Test Kelly formula across various win rates and payoffs."""
    f = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
    if expected_kelly < 0:
        assert f == 0.0
    else:
        assert abs(f - expected_kelly) < 0.01


@pytest.mark.parametrize(
    "volatility,expected_relative_size",
    [
        (0.10, 2.0),
        (0.20, 1.0),
        (0.40, 0.5),
    ],
)
def test_volatility_scaling(volatility: float, expected_relative_size: float):
    """Test volatility-based scaling relative to target."""
    sizer = VolatilityAdjustedSizer(target_volatility=0.20)
    size = sizer.calculate(capital=100_000, price=100.0, volatility=volatility)
    target_size = sizer.calculate(capital=100_000, price=100.0, volatility=0.20)
    ratio = size / target_size
    assert abs(ratio - expected_relative_size) < 0.1
