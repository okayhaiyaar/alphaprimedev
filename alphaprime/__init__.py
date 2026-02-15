from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("alpha-prime")
except PackageNotFoundError:
    __version__ = "2.0.0"

__all__ = ["__version__"]
