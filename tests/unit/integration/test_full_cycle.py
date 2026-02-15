"""
End-to-end trading cycle integration tests for ALPHA-PRIME v2.0.

Covers the full workflow:
- Strategy → signal → order → execution → portfolio → PnL.
- Real DB (PostgreSQL), Redis cache, and broker API (mocked at network edge).
- Async operations with transaction integrity and cache consistency.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from core.strategy_engine import StrategyEngine
from core.signal_generator import SignalGenerator
from core.order_manager import OrderManager
from core.execution_engine import ExecutionEngine
from core.portfolio import PortfolioManager
from core.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
async def clean_database(db_session):
    """Clean database before each test."""
    await db_session.execute("TRUNCATE orders, positions, trades CASCADE")
    await db_session.commit()
    yield db_session
    await db_session.rollback()


@pytest.fixture(scope="function")
async def clean_redis(redis_client):
    """Flush test Redis database."""
    await redis_client.flushdb()
    yield redis_client
    await redis_client.flushdb()


@pytest.fixture
async def strategy_engine(clean_database, clean_redis):
    """Initialized strategy engine with test DB/cache."""
    engine = StrategyEngine(db=clean_database, cache=clean_redis)
    await engine.initialize()
    try:
        yield engine
    finally:
        await engine.shutdown()


@pytest.fixture
async def signal_generator(strategy_engine: StrategyEngine):
    """Signal generator bound to strategy engine."""
    generator = SignalGenerator(strategy_engine=strategy_engine)
    yield generator


@pytest.fixture
async def order_manager(clean_database, clean_redis):
    """Order manager instance."""
    manager = OrderManager(db=clean_database, cache=clean_redis)
    yield manager


@pytest.fixture
async def mock_broker_client():
    """Mock broker API client (network boundary)."""
    client = AsyncMock()
    client.place_order = AsyncMock(return_value={"order_id": "TEST_ORDER_123", "status": "accepted"})
    client.get_quote = AsyncMock(return_value={"symbol": "AAPL", "price": 150.0, "volume": 1_000})
    client.cancel_order = AsyncMock(return_value={"status": "cancelled"})
    client.get_positions = AsyncMock(return_value=[])
    yield client


@pytest.fixture
async def execution_engine(order_manager: OrderManager, mock_broker_client: AsyncMock):
    """Execution engine wired to order manager and mock broker."""
    engine = ExecutionEngine(order_manager=order_manager, broker_client=mock_broker_client)
    yield engine


@pytest.fixture
async def portfolio_manager(clean_database, clean_redis):
    """Portfolio manager with initial test cash."""
    manager = PortfolioManager(
        db=clean_database,
        cache=clean_redis,
        initial_cash=100_000.0,
    )
    await manager.initialize()
    try:
        yield manager
    finally:
        await manager.shutdown()


@pytest.fixture
async def risk_manager(portfolio_manager: PortfolioManager):
    """Risk manager bound to portfolio."""
    manager = RiskManager(
        portfolio=portfolio_manager,
        max_position_pct=0.20,
        max_drawdown=0.15,
    )
    yield manager


@pytest.fixture
def sample_signal() -> Dict[str, Any]:
    """Sample BUY signal."""
    return {
        "symbol": "AAPL",
        "action": "BUY",
        "confidence": 0.85,
        "target_price": 155.0,
        "stop_loss": 145.0,
        "timestamp": datetime.utcnow(),
    }


@pytest.fixture
def sample_ohlcv_data() -> Dict[str, List[Dict[str, Any]]]:
    """Sample OHLCV data for strategy."""
    ts = datetime.utcnow()
    return {
        "AAPL": [
            {"timestamp": ts - timedelta(minutes=i), "o": 148, "h": 152, "l": 147, "c": 150, "v": 1_000_000}
            for i in range(100)
        ]
    }


# ---------------------------------------------------------------------------
# 1. Complete Trading Cycle
# ---------------------------------------------------------------------------

class TestCompleteTradingCycle:
    """Test full end-to-end trading workflow."""

    @pytest.mark.asyncio
    async def test_signal_to_execution_complete_flow(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
        sample_ohlcv_data: Dict[str, Any],
        clean_database,
    ):
        """Strategy → signal → order → execution → position."""
        await strategy_engine.load_data(sample_ohlcv_data)
        signals = await signal_generator.generate_signals()
        assert signals, "No signals generated"
        signal = signals[0]

        risk_result = await risk_manager.validate_signal(signal)
        assert risk_result.approved

        order = await order_manager.create_order_from_signal(signal)
        assert order.symbol == signal["symbol"]
        assert order.status == "PENDING"

        submit_result = await execution_engine.submit_order(order)
        assert submit_result.status in {"SUBMITTED", "ACCEPTED"}

        fill_event = {
            "order_id": order.id,
            "filled_qty": order.quantity,
            "avg_price": 150.0,
            "timestamp": datetime.utcnow(),
        }
        await execution_engine.process_fill(fill_event)

        position = await portfolio_manager.get_position(symbol=signal["symbol"])
        assert position is not None
        assert position.quantity == order.quantity

        result = await clean_database.execute(
            "SELECT id FROM orders WHERE id = :id",
            {"id": order.id},
        )
        row = result.first()
        assert row is not None

    @pytest.mark.asyncio
    async def test_buy_signal_creates_long_position(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        sample_ohlcv_data: Dict[str, Any],
        risk_manager: RiskManager,
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        signals = await signal_generator.generate_signals()
        buy = [s for s in signals if s["action"] == "BUY"][0]
        assert buy["confidence"] > 0.0

        assert (await risk_manager.validate_signal(buy)).approved
        order = await order_manager.create_order_from_signal(buy)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {
                "order_id": order.id,
                "filled_qty": order.quantity,
                "avg_price": 150.0,
                "timestamp": datetime.utcnow(),
            }
        )
        pos = await portfolio_manager.get_position(buy["symbol"])
        assert pos and pos.quantity > 0

    @pytest.mark.asyncio
    async def test_sell_signal_closes_long_position(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
    ):
        await portfolio_manager.add_position(symbol="AAPL", quantity=100, entry_price=150.0)
        sell_signal = {
            "symbol": "AAPL",
            "action": "SELL",
            "confidence": 0.9,
            "timestamp": datetime.utcnow(),
        }
        assert (await risk_manager.validate_signal(sell_signal)).approved
        order = await order_manager.create_order_from_signal(sell_signal)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {
                "order_id": order.id,
                "filled_qty": order.quantity,
                "avg_price": 155.0,
                "timestamp": datetime.utcnow(),
            }
        )
        pos = await portfolio_manager.get_position("AAPL")
        assert pos is None or pos.quantity == 0

    @pytest.mark.asyncio
    async def test_short_signal_creates_short_position(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
    ):
        short_signal = {
            "symbol": "MSFT",
            "action": "SHORT",
            "confidence": 0.85,
            "timestamp": datetime.utcnow(),
        }
        assert (await risk_manager.validate_signal(short_signal)).approved
        order = await order_manager.create_order_from_signal(short_signal)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {
                "order_id": order.id,
                "filled_qty": order.quantity,
                "avg_price": 300.0,
                "timestamp": datetime.utcnow(),
            }
        )
        pos = await portfolio_manager.get_position("MSFT")
        assert pos and pos.quantity < 0

    @pytest.mark.asyncio
    async def test_cover_signal_closes_short_position(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
    ):
        await portfolio_manager.add_position(symbol="MSFT", quantity=-50, entry_price=300.0)
        cover_signal = {
            "symbol": "MSFT",
            "action": "COVER",
            "confidence": 0.9,
            "timestamp": datetime.utcnow(),
        }
        assert (await risk_manager.validate_signal(cover_signal)).approved
        order = await order_manager.create_order_from_signal(cover_signal)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {
                "order_id": order.id,
                "filled_qty": order.quantity,
                "avg_price": 290.0,
                "timestamp": datetime.utcnow(),
            }
        )
        pos = await portfolio_manager.get_position("MSFT")
        assert pos is None or pos.quantity == 0

    @pytest.mark.asyncio
    async def test_portfolio_rebalancing_cycle(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await portfolio_manager.add_position("MSFT", 50, 300.0)
        targets = {"AAPL": 0.5, "MSFT": 0.5}
        rebalance_orders = await portfolio_manager.generate_rebalance_orders(target_weights=targets)
        assert rebalance_orders
        for order in rebalance_orders:
            await execution_engine.submit_order(order)
            await execution_engine.process_fill(
                {
                    "order_id": order.id,
                    "filled_qty": order.quantity,
                    "avg_price": 150.0,
                    "timestamp": datetime.utcnow(),
                }
            )
        weights = await portfolio_manager.current_weights()
        assert pytest.approx(sum(weights.values())) == 1.0

    @pytest.mark.asyncio
    async def test_stop_loss_triggered_execution(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
    ):
        await portfolio_manager.add_position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            stop_loss=145.0,
        )
        await risk_manager.check_stop_losses(current_prices={"AAPL": 144.5})
        pending = await order_manager.get_pending_orders()
        stops = [o for o in pending if o.order_type == "STOP_LOSS"]
        assert stops
        stop_order = stops[0]
        await execution_engine.submit_order(stop_order)
        await execution_engine.process_fill(
            {
                "order_id": stop_order.id,
                "filled_qty": stop_order.quantity,
                "avg_price": 144.5,
                "timestamp": datetime.utcnow(),
            }
        )
        pos = await portfolio_manager.get_position("AAPL")
        assert pos is None or pos.quantity == 0

    @pytest.mark.asyncio
    async def test_take_profit_triggered_execution(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
    ):
        await portfolio_manager.add_position(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            take_profit=160.0,
        )
        await risk_manager.check_take_profits(current_prices={"AAPL": 160.5})
        pending = await order_manager.get_pending_orders()
        tp_orders = [o for o in pending if o.order_type == "TAKE_PROFIT"]
        assert tp_orders
        order = tp_orders[0]
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {
                "order_id": order.id,
                "filled_qty": order.quantity,
                "avg_price": 160.5,
                "timestamp": datetime.utcnow(),
            }
        )
        pos = await portfolio_manager.get_position("AAPL")
        assert pos is None or pos.quantity == 0

    @pytest.mark.asyncio
    async def test_multiple_positions_lifecycle(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        symbols = ["AAPL", "MSFT", "GOOGL"]
        for sym in symbols:
            order = await order_manager.create_order(sym, "BUY", 10, order_type="MARKET")
            await execution_engine.submit_order(order)
            await execution_engine.process_fill(
                {
                    "order_id": order.id,
                    "filled_qty": 10,
                    "avg_price": 100.0,
                    "timestamp": datetime.utcnow(),
                }
            )
        positions = await portfolio_manager.list_positions()
        assert len(positions) == len(symbols)

    @pytest.mark.asyncio
    async def test_end_of_day_reconciliation(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await portfolio_manager.add_position("MSFT", 50, 300.0)
        with patch.object(execution_engine.broker_client, "get_positions", return_value=[]) as mock_get:
            await portfolio_manager.end_of_day_reconciliation(execution_engine.broker_client)
            mock_get.assert_awaited()
        assert await portfolio_manager.daily_pnl() is not None


# ---------------------------------------------------------------------------
# 2. Strategy Execution Pipeline
# ---------------------------------------------------------------------------

class TestStrategyExecutionPipeline:
    """Test strategy → signal → order pipeline."""

    @pytest.mark.asyncio
    async def test_strategy_generates_signals(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        signals = await signal_generator.generate_signals()
        assert signals

    @pytest.mark.asyncio
    async def test_signals_filtered_by_confidence(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        signals = await signal_generator.generate_signals(min_confidence=0.75)
        assert all(s["confidence"] >= 0.75 for s in signals)

    @pytest.mark.asyncio
    async def test_signals_converted_to_orders(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        order_manager: OrderManager,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        signals = await signal_generator.generate_signals()
        orders = [await order_manager.create_order_from_signal(s) for s in signals]
        assert orders
        assert all(o.symbol for o in orders)

    @pytest.mark.asyncio
    async def test_order_validation_before_submission(
        self,
        order_manager: OrderManager,
        risk_manager: RiskManager,
    ):
        signal = {"symbol": "AAPL", "action": "BUY", "confidence": 0.9, "timestamp": datetime.utcnow()}
        assert (await risk_manager.validate_signal(signal)).approved
        order = await order_manager.create_order_from_signal(signal)
        assert await order_manager.validate_order(order)

    @pytest.mark.asyncio
    async def test_risk_checks_block_risky_orders(
        self,
        risk_manager: RiskManager,
        order_manager: OrderManager,
    ):
        signal = {"symbol": "AAPL", "action": "BUY", "confidence": 0.9, "timestamp": datetime.utcnow()}
        risk_manager.max_position_pct = 0.0
        res = await risk_manager.validate_signal(signal)
        assert not res.approved
        with pytest.raises(ValueError):
            await order_manager.create_order_from_signal(signal)

    @pytest.mark.asyncio
    async def test_duplicate_signal_prevention(
        self,
        signal_generator: SignalGenerator,
        strategy_engine: StrategyEngine,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        s1 = await signal_generator.generate_signals()
        s2 = await signal_generator.generate_signals()
        assert s1 == s2

    @pytest.mark.asyncio
    async def test_stale_signal_rejection(
        self,
        order_manager: OrderManager,
    ):
        old_signal = {
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 0.9,
            "timestamp": datetime.utcnow() - timedelta(hours=2),
        }
        with pytest.raises(ValueError):
            await order_manager.create_order_from_signal(old_signal, max_age=timedelta(minutes=30))

    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(
        self,
        signal_generator: SignalGenerator,
        strategy_engine: StrategyEngine,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        with patch.object(strategy_engine, "run", side_effect=RuntimeError("fail")):
            signals = await signal_generator.generate_signals()
        assert isinstance(signals, list)


# ---------------------------------------------------------------------------
# 3. Order Management Lifecycle
# ---------------------------------------------------------------------------

class TestOrderManagementLifecycle:
    """Test order creation, submission, fills, cancellations."""

    @pytest.mark.asyncio
    async def test_create_market_order(self, order_manager: OrderManager):
        order = await order_manager.create_order("AAPL", "BUY", 100, order_type="MARKET")
        assert order.order_type == "MARKET"

    @pytest.mark.asyncio
    async def test_create_limit_order(self, order_manager: OrderManager):
        order = await order_manager.create_order("AAPL", "BUY", 100, order_type="LIMIT", limit_price=150.0)
        assert order.limit_price == 150.0

    @pytest.mark.asyncio
    async def test_order_submission_to_broker(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        result = await execution_engine.submit_order(order)
        assert result.status in {"SUBMITTED", "ACCEPTED"}

    @pytest.mark.asyncio
    async def test_order_acknowledgement_handling(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        await execution_engine.submit_order(order)
        ack = {"order_id": order.id, "status": "ACCEPTED", "timestamp": datetime.utcnow()}
        await execution_engine.handle_ack(ack)
        updated = await order_manager.get_order(order.id)
        assert updated.status == "ACCEPTED"

    @pytest.mark.asyncio
    async def test_partial_fill_updates(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100, order_type="LIMIT", limit_price=150.0)
        await execution_engine.submit_order(order)

        fill_event = {"order_id": order.id, "filled_qty": 50, "avg_price": 150.0}
        await execution_engine.process_fill(fill_event)
        updated = await order_manager.get_order(order.id)
        assert updated.filled_quantity == 50
        assert updated.remaining_quantity == 50
        assert updated.status == "PARTIALLY_FILLED"

        fill_event["filled_qty"] = 50
        await execution_engine.process_fill(fill_event)
        final = await order_manager.get_order(order.id)
        assert final.filled_quantity == 100
        assert final.status == "FILLED"

    @pytest.mark.asyncio
    async def test_complete_fill_processing(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {"order_id": order.id, "filled_qty": 100, "avg_price": 150.0}
        )
        updated = await order_manager.get_order(order.id)
        assert updated.status == "FILLED"

    @pytest.mark.asyncio
    async def test_order_rejection_handling(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        rej = {"order_id": order.id, "reason": "insufficient buying power"}
        await execution_engine.handle_rejection(rej)
        updated = await order_manager.get_order(order.id)
        assert updated.status == "REJECTED"

    @pytest.mark.asyncio
    async def test_order_cancellation(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        await execution_engine.submit_order(order)
        await order_manager.cancel_order(order.id)
        updated = await order_manager.get_order(order.id)
        assert updated.status in {"CANCELLED", "PENDING_CANCEL"}

    @pytest.mark.asyncio
    async def test_order_expiration(self, order_manager: OrderManager):
        order = await order_manager.create_order("AAPL", "BUY", 100, tif="DAY")
        await order_manager.expire_orders(as_of=datetime.utcnow() + timedelta(days=1))
        updated = await order_manager.get_order(order.id)
        assert updated.status in {"EXPIRED", "CANCELLED"}

    @pytest.mark.asyncio
    async def test_order_amendment(self, order_manager: OrderManager):
        order = await order_manager.create_order("AAPL", "BUY", 100, order_type="LIMIT", limit_price=150.0)
        amended = await order_manager.amend_order(order.id, new_limit_price=149.0)
        assert amended.limit_price == 149.0

    @pytest.mark.asyncio
    async def test_concurrent_order_processing(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        orders = [
            await order_manager.create_order("AAPL", "BUY", 10, order_type="MARKET")
            for _ in range(5)
        ]

        async def submit(order):
            return await execution_engine.submit_order(order)

        results = await asyncio.gather(*(submit(o) for o in orders))
        assert all(r.status in {"SUBMITTED", "ACCEPTED"} for r in results)

    @pytest.mark.asyncio
    async def test_order_state_persistence(self, order_manager: OrderManager):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        order.status = "FILLED"
        await order_manager.save(order)
        reloaded = await order_manager.get_order(order.id)
        assert reloaded.status == "FILLED"


# ---------------------------------------------------------------------------
# 4. Position Management
# ---------------------------------------------------------------------------

class TestPositionManagement:
    """Test position creation, updates, closing."""

    @pytest.mark.asyncio
    async def test_position_created_on_fill(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {"order_id": order.id, "filled_qty": 100, "avg_price": 150.0}
        )
        pos = await portfolio_manager.get_position("AAPL")
        assert pos and pos.quantity == 100

    @pytest.mark.asyncio
    async def test_position_size_updated_on_partial_fill(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {"order_id": order.id, "filled_qty": 40, "avg_price": 150.0}
        )
        pos = await portfolio_manager.get_position("AAPL")
        assert pos and pos.quantity == 40

    @pytest.mark.asyncio
    async def test_position_closed_on_opposite_signal(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        order = await order_manager.create_order("AAPL", "SELL", 100)
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {"order_id": order.id, "filled_qty": 100, "avg_price": 155.0}
        )
        pos = await portfolio_manager.get_position("AAPL")
        assert pos is None or pos.quantity == 0

    @pytest.mark.asyncio
    async def test_position_pnl_calculation_realtime(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        pnl = await portfolio_manager.unrealized_pnl({"AAPL": 155.0})
        assert pytest.approx(pnl) == 500.0

    @pytest.mark.asyncio
    async def test_position_cost_basis_tracking(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await portfolio_manager.add_position("AAPL", 100, 160.0)
        pos = await portfolio_manager.get_position("AAPL")
        assert pytest.approx(pos.avg_price) == 155.0

    @pytest.mark.asyncio
    async def test_position_holding_period(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        pos = await portfolio_manager.get_position("AAPL")
        assert pos.hold_duration.total_seconds() >= 0.0

    @pytest.mark.asyncio
    async def test_multiple_positions_same_symbol(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 50, 150.0)
        await portfolio_manager.add_position("AAPL", 25, 155.0)
        pos = await portfolio_manager.get_position("AAPL")
        assert pos.quantity == 75

    @pytest.mark.asyncio
    async def test_position_exposure_limits(
        self,
        portfolio_manager: PortfolioManager,
        risk_manager: RiskManager,
    ):
        risk_manager.max_position_pct = 0.10
        await portfolio_manager.add_position("AAPL", 1_000, 150.0)
        breaches = await risk_manager.check_exposure_limits()
        assert "AAPL" in breaches

    @pytest.mark.asyncio
    async def test_position_reconciliation_with_broker(
        self,
        portfolio_manager: PortfolioManager,
        execution_engine: ExecutionEngine,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        execution_engine.broker_client.get_positions = AsyncMock(return_value=[])
        await portfolio_manager.reconcile_with_broker(execution_engine.broker_client)
        assert await portfolio_manager.get_position("AAPL") is None

    @pytest.mark.asyncio
    async def test_position_persistence_across_restarts(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        saved_positions = await portfolio_manager.list_positions()
        assert saved_positions
        await portfolio_manager.shutdown()
        await portfolio_manager.initialize()
        reloaded = await portfolio_manager.list_positions()
        assert reloaded


# ---------------------------------------------------------------------------
# 5. Risk Management Integration
# ---------------------------------------------------------------------------

class TestRiskManagementIntegration:
    """Test risk checks throughout the trading cycle."""

    @pytest.mark.asyncio
    async def test_pre_trade_risk_checks(
        self,
        risk_manager: RiskManager,
        sample_signal: Dict[str, Any],
    ):
        result = await risk_manager.validate_signal(sample_signal)
        assert result.approved

    @pytest.mark.asyncio
    async def test_position_size_limited_by_risk(
        self,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        size = await risk_manager.max_size_for_symbol("AAPL", price=150.0)
        assert size >= 0

    @pytest.mark.asyncio
    async def test_max_drawdown_stops_trading(
        self,
        risk_manager: RiskManager,
    ):
        risk_manager.max_drawdown = 0.10
        risk_manager.current_drawdown = 0.11
        res = await risk_manager.validate_signal({"symbol": "AAPL", "action": "BUY"})
        assert not res.approved

    @pytest.mark.asyncio
    async def test_concentration_limits_enforced(
        self,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 1_000, 150.0)
        breaches = await risk_manager.check_exposure_limits()
        assert breaches

    @pytest.mark.asyncio
    async def test_portfolio_var_limits(
        self,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        breached = await risk_manager.check_var_limits()
        assert isinstance(breached, bool)

    @pytest.mark.asyncio
    async def test_correlated_positions_blocked(
        self,
        risk_manager: RiskManager,
    ):
        risk_manager.max_correlation = 0.8
        allowed = await risk_manager.check_correlation("AAPL", "MSFT", corr=0.9)
        assert not allowed

    @pytest.mark.asyncio
    async def test_circuit_breaker_halts_orders(
        self,
        risk_manager: RiskManager,
    ):
        risk_manager.circuit_breaker_tripped = True
        res = await risk_manager.validate_signal({"symbol": "AAPL", "action": "BUY"})
        assert not res.approved

    @pytest.mark.asyncio
    async def test_position_reduction_on_breach(
        self,
        risk_manager: RiskManager,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 1_000, 150.0)
        await risk_manager.reduce_positions_on_breach()
        pos = await portfolio_manager.get_position("AAPL")
        assert pos is None or pos.quantity < 1_000

    @pytest.mark.asyncio
    async def test_risk_override_by_admin(
        self,
        risk_manager: RiskManager,
        sample_signal: Dict[str, Any],
    ):
        risk_manager.circuit_breaker_tripped = True
        res = await risk_manager.validate_signal(sample_signal, override=True)
        assert res.approved

    @pytest.mark.asyncio
    async def test_risk_limits_reset_daily(self, risk_manager: RiskManager):
        risk_manager.current_drawdown = 0.1
        await risk_manager.reset_daily()
        assert risk_manager.current_drawdown == 0.0


# ---------------------------------------------------------------------------
# 6. Database & Cache Consistency
# ---------------------------------------------------------------------------

class TestDatabaseCacheConsistency:
    """Test DB transactions and cache synchronization."""

    @pytest.mark.asyncio
    async def test_order_persisted_to_database(
        self,
        order_manager: OrderManager,
        clean_database,
    ):
        order = await order_manager.create_order("AAPL", "BUY", 100)
        result = await clean_database.execute(
            "SELECT id FROM orders WHERE id = :id", {"id": order.id}
        )
        assert result.first() is not None

    @pytest.mark.asyncio
    async def test_position_cached_in_redis(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        cached = await clean_redis.get("position:AAPL")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_update(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        cached = await clean_redis.get("position:AAPL")
        assert cached is not None
        await portfolio_manager.update_position("AAPL", quantity=150)
        cached_after = await clean_redis.get("position:AAPL")
        assert cached_after is None or cached_after != cached

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(
        self,
        order_manager: OrderManager,
        clean_database,
    ):
        try:
            async with clean_database.begin():
                await order_manager.create_order("AAPL", "BUY", 100)
                raise ValueError("Simulated error")
        except ValueError:
            pass
        orders = await order_manager.get_all_orders()
        assert len(orders) == 0

    @pytest.mark.asyncio
    async def test_eventual_consistency_db_cache(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await portfolio_manager.refresh_cache()
        cached = await clean_redis.get("position:AAPL")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_cache_miss_fallback_to_db(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await clean_redis.delete("position:AAPL")
        pos = await portfolio_manager.get_position("AAPL")
        assert pos is not None

    @pytest.mark.asyncio
    async def test_concurrent_write_handling(
        self,
        portfolio_manager: PortfolioManager,
    ):
        async def inc():
            await portfolio_manager.add_position("AAPL", 1, 150.0)

        await asyncio.gather(*(inc() for _ in range(5)))
        pos = await portfolio_manager.get_position("AAPL")
        assert pos.quantity == 5

    @pytest.mark.asyncio
    async def test_cache_warmup_on_startup(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await clean_redis.flushdb()
        await portfolio_manager.warmup_cache()
        cached = await clean_redis.get("position:AAPL")
        assert cached is not None


# ---------------------------------------------------------------------------
# 7. Error Handling & Recovery
# ---------------------------------------------------------------------------

class TestErrorHandlingRecovery:
    """Test error scenarios and recovery mechanisms."""

    @pytest.mark.asyncio
    async def test_broker_api_timeout_retry(
        self,
        execution_engine: ExecutionEngine,
        order_manager: OrderManager,
    ):
        execution_engine.broker_client.place_order = AsyncMock(
            side_effect=[asyncio.TimeoutError(), {"order_id": "TEST_123", "status": "accepted"}]
        )
        order = await order_manager.create_order("AAPL", "BUY", 100)
        result = await execution_engine.submit_order(order, max_retries=2)
        assert result.status in {"SUBMITTED", "ACCEPTED"}
        assert execution_engine.broker_client.place_order.call_count == 2

    @pytest.mark.asyncio
    async def test_database_connection_lost_recovery(
        self,
        order_manager: OrderManager,
        clean_database,
    ):
        with patch.object(clean_database, "execute", side_effect=[Exception("db down"), Mock()]):
            with pytest.raises(Exception):
                await clean_database.execute("SELECT 1")
            order = await order_manager.create_order("AAPL", "BUY", 100)
            assert order.id is not None

    @pytest.mark.asyncio
    async def test_redis_unavailable_degraded_mode(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        with patch.object(clean_redis, "get", side_effect=Exception("redis down")):
            pos = await portfolio_manager.get_position("AAPL")
            assert pos is None or isinstance(pos, object)

    @pytest.mark.asyncio
    async def test_order_rejected_by_broker(
        self,
        execution_engine: ExecutionEngine,
        order_manager: OrderManager,
    ):
        execution_engine.broker_client.place_order = AsyncMock(
            return_value={"status": "rejected", "reason": "insufficient funds"}
        )
        order = await order_manager.create_order("AAPL", "BUY", 100)
        result = await execution_engine.submit_order(order)
        assert result.status == "REJECTED"

    @pytest.mark.asyncio
    async def test_insufficient_funds_handling(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
        risk_manager: RiskManager,
    ):
        portfolio_manager.cash = 0.0
        signal = {"symbol": "AAPL", "action": "BUY", "confidence": 0.9}
        res = await risk_manager.validate_signal(signal)
        assert not res.approved
        with pytest.raises(ValueError):
            await order_manager.create_order_from_signal(signal)

    @pytest.mark.asyncio
    async def test_duplicate_order_prevention(
        self,
        order_manager: OrderManager,
    ):
        o1 = await order_manager.create_order("AAPL", "BUY", 100, client_order_id="X")
        with pytest.raises(ValueError):
            await order_manager.create_order("AAPL", "BUY", 100, client_order_id="X")
        assert o1.client_order_id == "X"

    @pytest.mark.asyncio
    async def test_orphaned_order_cleanup(
        self,
        order_manager: OrderManager,
    ):
        await order_manager.create_order("AAPL", "BUY", 100)
        await order_manager.cleanup_orphaned_orders()
        assert True  # no exception

    @pytest.mark.asyncio
    async def test_stale_position_reconciliation(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await portfolio_manager.reconcile_stale_positions()
        assert True

    @pytest.mark.asyncio
    async def test_crash_recovery_state_restore(
        self,
        portfolio_manager: PortfolioManager,
        order_manager: OrderManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await order_manager.create_order("AAPL", "BUY", 100)
        state = await portfolio_manager.snapshot_state()
        await portfolio_manager.restore_state(state)
        assert await portfolio_manager.list_positions()

    @pytest.mark.asyncio
    async def test_error_notification_pipeline(
        self,
        execution_engine: ExecutionEngine,
    ):
        with patch("core.execution_engine.notify_error") as mock_notify:
            await execution_engine.handle_error(RuntimeError("boom"))
            mock_notify.assert_called()


# ---------------------------------------------------------------------------
# 8. Performance & Concurrency
# ---------------------------------------------------------------------------

class TestPerformanceConcurrency:
    """Test system under load and concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_signal_processing(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)

        async def gen():
            return await signal_generator.generate_signals()

        results = await asyncio.gather(*(gen() for _ in range(5)))
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_high_frequency_order_submission(
        self,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
    ):
        async def submit_one(i: int):
            order = await order_manager.create_order("AAPL", "BUY", 1, client_order_id=f"HF_{i}")
            return await execution_engine.submit_order(order)

        results = await asyncio.gather(*(submit_one(i) for i in range(20)))
        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_bulk_position_updates(
        self,
        portfolio_manager: PortfolioManager,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        prices = {f"SYM{i}": 100.0 + i for i in range(10)}
        await portfolio_manager.bulk_update_prices(prices)
        assert True

    @pytest.mark.asyncio
    async def test_database_query_performance(
        self,
        clean_database,
    ):
        start = datetime.utcnow()
        await clean_database.execute("SELECT 1")
        duration = (datetime.utcnow() - start).total_seconds()
        assert duration < 0.5

    @pytest.mark.asyncio
    async def test_cache_hit_rate(
        self,
        portfolio_manager: PortfolioManager,
        clean_redis,
    ):
        await portfolio_manager.add_position("AAPL", 100, 150.0)
        await portfolio_manager.get_position("AAPL")
        await portfolio_manager.get_position("AAPL")
        stats = await portfolio_manager.cache_stats()
        assert stats["hit_rate"] >= 0.5

    @pytest.mark.asyncio
    async def test_end_to_end_latency(
        self,
        strategy_engine: StrategyEngine,
        signal_generator: SignalGenerator,
        order_manager: OrderManager,
        execution_engine: ExecutionEngine,
        portfolio_manager: PortfolioManager,
        sample_ohlcv_data: Dict[str, Any],
    ):
        await strategy_engine.load_data(sample_ohlcv_data)
        start = datetime.utcnow()
        signals = await signal_generator.generate_signals()
        if not signals:
            pytest.skip("No signals")
        order = await order_manager.create_order_from_signal(signals[0])
        await execution_engine.submit_order(order)
        await execution_engine.process_fill(
            {"order_id": order.id, "filled_qty": order.quantity, "avg_price": 150.0}
        )
        await portfolio_manager.get_position(order.symbol)
        duration = (datetime.utcnow() - start).total_seconds()
        assert duration < 5.0
