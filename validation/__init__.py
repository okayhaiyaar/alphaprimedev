"""
ALPHA-PRIME Validation & Monitoring Submodule

Components:
- drift_monitor.py: Model drift detection
- backtest.py: Historical performance validation
- performance_tracker.py: Live performance monitoring

This package centralizes validation, backtesting and live
performance tracking utilities. All modules are optional;
import errors are suppressed to keep the core package usable.
"""

from __future__ import annotations

__all__: list[str] = []

# Drift monitoring
try:
    from .drift_monitor import (  # type: ignore[attr-defined]
        calculate_drift_metrics,
        check_drift,
        log_prediction,
    )

    __all__.extend(
        [
            "log_prediction",
            "check_drift",
            "calculate_drift_metrics",
        ]
    )
except ImportError:
    pass

# Backtesting
try:
    from .backtest import (  # type: ignore[attr-defined]
        BacktestResult,
        run_backtest,
    )

    __all__.extend(
        [
            "run_backtest",
            "BacktestResult",
        ]
    )
except ImportError:
    pass

# Live performance tracking
try:
    from .performance_tracker import (  # type: ignore[attr-defined]
        generate_performance_report,
        track_live_performance,
    )

    __all__.extend(
        [
            "track_live_performance",
            "generate_performance_report",
        ]
    )
except ImportError:
    pass
