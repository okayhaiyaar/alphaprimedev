from __future__ import annotations

import importlib
import inspect
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MOCK_API_CALLS", "true")
    monkeypatch.setenv("PAPER_TRADING_ONLY", "true")
    env_path = Path(".env")
    if not env_path.exists():
        env_path.write_text("OPENAI_API_KEY=sk-test\nMOCK_API_CALLS=true\n", encoding="utf-8")

    import config

    config._SETTINGS = None


def test_imports_under_mock_mode(mock_env: None) -> None:
    modules = ["config", "brain", "data_engine", "portfolio", "scheduler", "app"]
    for module_name in modules:
        mod = importlib.import_module(module_name)
        assert mod is not None


def test_mock_mode_prevents_openai_call(mock_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import brain

    importlib.reload(brain)

    def _should_not_call(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("OpenAI call should not happen in MOCK mode")

    monkeypatch.setattr(brain, "call_gpt4o_oracle", _should_not_call)

    technicals = {
        "price_action": {"last_price": 100.0},
        "volatility": {"atr": 2.0, "bb_upper": 105.0, "bb_lower": 95.0},
        "trend": {},
        "momentum": {},
        "volume": {},
        "macd": {},
        "support_resistance": {},
        "data_quality": {},
    }
    intel = {
        "fundamentals": {"sources": [], "going_concern_flags": [], "debt_maturity_flags": []},
        "news_catalysts": {"sources": [], "headlines": [], "catalysts": []},
        "sentiment_score": {"sources": [], "hype_score": 0, "polarity": 0.0, "volume": 0},
    }

    decision = brain.consult_oracle("AAPL", intel=intel, technicals=technicals)
    assert decision.action == "WAIT"
    assert "mock_mode" in decision.tags


def test_data_engine_fallback_without_pandas_ta(mock_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import data_engine

    monkeypatch.setattr(data_engine, "HAS_PANDAS_TA", False)

    rows = 80
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    close = np.linspace(100, 120, rows)
    df = pd.DataFrame(
        {
            "Date": dates,
            "Open": close - 1,
            "High": close + 1,
            "Low": close - 2,
            "Close": close,
            "Volume": np.full(rows, 1_000_000),
        }
    )

    result = data_engine.calculate_hard_technicals(df, ticker="AAPL", timeframe="1d")
    assert result["ticker"] == "AAPL"
    assert "trend" in result
    assert "momentum" in result


def test_portfolio_lock_is_cross_platform(mock_env: None, tmp_path: Path) -> None:
    import portfolio

    source = inspect.getsource(portfolio)
    assert "fcntl" not in source

    target = tmp_path / "portfolio.json"
    target.write_text("{}", encoding="utf-8")

    with portfolio.file_lock(target):
        data = target.read_text(encoding="utf-8")
        assert data == "{}"


def test_scheduler_once_analysis_only_no_network(mock_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    import scheduler

    calls = {"execute": None}

    def _fake_scan(limit: int = 10, min_volume: int = 1_000_000):
        return ["AAPL"]

    def _fake_process(ticker: str, trader, execute_trades: bool = True):
        calls["execute"] = execute_trades
        return scheduler.ProcessResult(
            ticker=ticker,
            success=True,
            action_taken="WAIT",
            confidence=50,
            message="mocked",
            execution_time_seconds=0.01,
            error=None,
        )

    monkeypatch.setattr(scheduler, "scan_market_movers", _fake_scan)
    monkeypatch.setattr(scheduler, "process_ticker", _fake_process)

    result = scheduler.run_daily_cycle(limit=1, execute_trades=False)
    assert result.tickers_processed == 1
    assert calls["execute"] is False


def test_portfolio_lifecycle_persist_reload(mock_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import portfolio

    patched_settings = replace(
        portfolio.settings,
        trade_history_path=str(tmp_path / "trades.csv"),
        backup_dir=str(tmp_path / "backups"),
        starting_cash=10000.0,
    )
    monkeypatch.setattr(portfolio, "settings", patched_settings)

    portfolio_path = tmp_path / "portfolio.json"
    trader = portfolio.PaperTrader(portfolio_path=str(portfolio_path))
    res = trader.execute_trade(action="BUY", ticker="AAPL", price=100.0, quantity=1)
    assert res.success

    reloaded = portfolio.PaperTrader(portfolio_path=str(portfolio_path))
    state = reloaded.get_portfolio_state()
    assert "AAPL" in state.positions
    assert state.total_trades >= 1


def test_data_engine_empty_fallback_shape(mock_env: None) -> None:
    import data_engine

    df = data_engine._fallback_fetch("AAPL", "3mo", "1d")
    assert list(df.columns) == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert df.empty


def test_doctor_command_runs(mock_env: None) -> None:
    from alphaprime import cli

    code = cli.doctor()
    assert code == 0


def test_doctor_masks_secrets_and_includes_run_id(
    mock_env: None, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from alphaprime import cli

    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-1234")
    monkeypatch.setenv("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/123/very-secret")

    code = cli.doctor()
    out = capsys.readouterr().out

    assert code == 0
    assert "sk-super-secret-1234" not in out
    assert "very-secret" not in out
    assert "OPENAI_API_KEY:" in out
    assert "DISCORD_WEBHOOK_URL:" in out
    assert re.search(r"RUN_ID: [0-9a-f\-]{36}", out)


def test_doctor_does_not_crash_minimal_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from alphaprime import cli

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.setenv("MOCK_API_CALLS", "true")

    code = cli.doctor()
    assert code == 0
