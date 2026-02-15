"""
============================================================
ALPHA-PRIME v2.0 - AI-Powered Trading System
============================================================

Modular Architecture:
1. config.py          - Configuration & logging
2. research_engine.py - Intelligence gathering (The Hunter)
3. data_engine.py     - Market data & technicals (The Mathematician)
4. brain.py           - AI decision engine (The Oracle)
5. app.py             - Streamlit dashboard (War Room)
6. alerts.py          - Multi-channel notifications
7. portfolio.py       - Paper trading ledger
8. scheduler.py       - Automated orchestration

Usage:
    from alpha_prime import consult_oracle, get_god_tier_intel
    from alpha_prime.portfolio import PaperTrader
    from alpha_prime.config import get_settings

Requirements:
    - Python 3.9+
    - OpenAI API key
    - See requirements.txt for dependencies

Author: ALPHA-PRIME Team
License: MIT
Version: 2.0.0
============================================================
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

__version__ = "2.0.0"
__author__ = "ALPHA-PRIME Team"
__license__ = "MIT"
__description__ = "AI-Powered Trading System with GPT-4o"

# ---------------------------------------------------------------------
# Version check (fail fast on unsupported Python versions)
# ---------------------------------------------------------------------
if sys.version_info < (3, 9):
    raise RuntimeError(
        f"ALPHA-PRIME requires Python 3.9+, "
        f"you have {sys.version_info.major}.{sys.version_info.minor}"
    )

# ---------------------------------------------------------------------
# Core imports (public API)
# ---------------------------------------------------------------------

from .config import (  # noqa: E402
    Settings,
    get_logger,
    get_settings,
    validate_environment,
)

from .research_engine import (  # noqa: E402
    analyze_fundamentals,
    fetch_news,
    fetch_social_sentiment,
    get_god_tier_intel,
)

from .data_engine import (  # noqa: E402
    calculate_hard_technicals,
    get_market_data,
    get_multi_timeframe_data,
    validate_data_quality,
)

from .brain import (  # noqa: E402
    OracleDecision,
    consult_oracle,
)

from .portfolio import (  # noqa: E402
    PerformanceMetrics,
    PaperTrader,
    Portfolio,
    Position,
    Trade,
)

from .alerts import (  # noqa: E402
    send_discord_alert,
    send_error_alert,
    send_multi_channel_alert,
)

from .scheduler import (  # noqa: E402
    process_ticker,
    run_daily_cycle,
    scan_market_movers,
    start_scheduler,
)

# Public export surface
__all__ = [
    # Version / meta
    "__version__",
    "__author__",
    "__license__",
    "__description__",
    # Config
    "get_settings",
    "get_logger",
    "validate_environment",
    "Settings",
    # Research (Module 1)
    "get_god_tier_intel",
    "analyze_fundamentals",
    "fetch_news",
    "fetch_social_sentiment",
    # Data (Module 2)
    "get_market_data",
    "calculate_hard_technicals",
    "get_multi_timeframe_data",
    "validate_data_quality",
    # Brain (Module 3)
    "consult_oracle",
    "OracleDecision",
    # Portfolio (Module 6)
    "PaperTrader",
    "Portfolio",
    "Position",
    "Trade",
    "PerformanceMetrics",
    # Alerts (Module 5)
    "send_discord_alert",
    "send_multi_channel_alert",
    "send_error_alert",
    # Scheduler (Module 7)
    "run_daily_cycle",
    "process_ticker",
    "scan_market_movers",
    "start_scheduler",
]

# ---------------------------------------------------------------------
# Initialization logging
# ---------------------------------------------------------------------

logger = logging.getLogger("alpha_prime")
if not logger.handlers:
    # If user did not configure logging yet, fall back to basicConfig.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
logger.info("ALPHA-PRIME v%s initialized", __version__)

# Optional type-checking-only imports for tooling (no runtime cost)
if TYPE_CHECKING:  # pragma: no cover
    from . import risk, strategy, validation  # noqa: F401
