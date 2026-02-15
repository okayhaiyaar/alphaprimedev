"""
ALPHA-PRIME Risk Management Submodule

Components:
- circuit_breakers.py: Trading halt conditions
- position_sizer.py: ATR/Kelly/Fixed position sizing
- correlation_check.py: Portfolio correlation limits

This package exposes a clean, optional risk management API.
Importing `alpha_prime.risk` will not fail if individual
risk components are missing; unavailable symbols are simply
omitted from __all__.
"""

from __future__ import annotations

__all__ = []  # Populated dynamically based on available submodules.

# Circuit breakers
try:
    from .circuit_breakers import (  # type: ignore[attr-defined]
        check_consecutive_losses,
        check_daily_loss_limit,
        check_trade_allowed,
        check_vix_threshold,
    )

    __all__.extend(
        [
            "check_trade_allowed",
            "check_consecutive_losses",
            "check_daily_loss_limit",
            "check_vix_threshold",
        ]
    )
except ImportError:
    # Optional module; risk API degrades gracefully.
    pass

# Position sizing
try:
    from .position_sizer import (  # type: ignore[attr-defined]
        calculate_kelly_fraction,
        calculate_position_size,
    )

    __all__.extend(
        [
            "calculate_position_size",
            "calculate_kelly_fraction",
        ]
    )
except ImportError:
    pass

# Correlation checks
try:
    from .correlation_check import (  # type: ignore[attr-defined]
        calculate_correlation_matrix,
        check_portfolio_correlation,
    )

    __all__.extend(
        [
            "check_portfolio_correlation",
            "calculate_correlation_matrix",
        ]
    )
except ImportError:
    pass
