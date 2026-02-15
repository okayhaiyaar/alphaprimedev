#!/usr/bin/env python
"""
============================================================
ALPHA-PRIME v2.0 - Setup Configuration
============================================================
Minimal setup.py that delegates to pyproject.toml (PEP 621).

This file exists primarily for backward compatibility.
Modern installation uses pyproject.toml directly.

Installation (modern):
    pip install -e .          # Editable install
    pip install -e .[dev]     # With dev dependencies

Legacy / compatibility:
    python setup.py develop   # Legacy editable install
    python setup.py sdist     # Legacy source distribution

Build (recommended):
    python -m build
============================================================
"""

from pathlib import Path

from setuptools import setup

# NOTE: This file is kept minimal for compatibility.
# Primary configuration is in pyproject.toml (PEP 621).
# Modern Python projects (setuptools 61.0+) can read all
# metadata from pyproject.toml, making setup.py optional.

# WARNING: Do NOT duplicate metadata here that exists in pyproject.toml.
# This causes version drift and maintenance burden.

# Read README for long_description (used by legacy tooling / PyPI)
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")


# ============================================================
# SETUP.PY vs PYPROJECT.TOML
# ============================================================
# Q: Do I need setup.py if I have pyproject.toml?
# A: No, not for Python 3.10+ and pip 21.3+.
#
# This file exists for:
# - Compatibility with older pip versions
# - Tools that don't yet support PEP 621 fully
# - A future place to hook custom build logic if required
#
# If you're targeting modern environments only, you can
# remove this file and rely solely on pyproject.toml.
# ============================================================

if __name__ == "__main__":
    # Delegate all metadata to pyproject.toml
    setup(
        long_description=long_description,
        long_description_content_type="text/markdown",
    )
