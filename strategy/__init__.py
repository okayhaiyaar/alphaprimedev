"""
ALPHA-PRIME Strategy Submodule

Components:
- regime_detector.py: Bull/bear/sideways market classification
- event_calendar.py: Earnings, Fed meetings, economic data
- multi_timeframe.py: Confluence across timeframes

The strategy package provides higher-level market context
and signal confluence utilities. All submodules are optional;
missing pieces will not break imports.
"""

from __future__ import annotations

__all__: list[str] = []

# Regime detection
try:
    from .regime_detector import (  # type: ignore[attr-defined]
        RegimeType,
        detect_regime,
        get_current_regime,
    )

    __all__.extend(
        [
            "get_current_regime",
            "detect_regime",
            "RegimeType",
        ]
    )
except ImportError:
    pass

# Event calendar
try:
    from .event_calendar import (  # type: ignore[attr-defined]
        fetch_earnings_calendar,
        fetch_economic_calendar,
        get_upcoming_events,
    )

    __all__.extend(
        [
            "get_upcoming_events",
            "fetch_earnings_calendar",
            "fetch_economic_calendar",
        ]
    )
except ImportError:
    pass

# Multi-timeframe confluence
try:
    from .multi_timeframe import (  # type: ignore[attr-defined]
        analyze_multi_timeframe,
        check_timeframe_confluence,
    )

    __all__.extend(
        [
            "analyze_multi_timeframe",
            "check_timeframe_confluence",
        ]
    )
except ImportError:
    pass
