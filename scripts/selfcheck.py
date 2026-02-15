from __future__ import annotations

import argparse
import importlib
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class CheckItem:
    module: str
    required: bool
    reason: str


def _build_checklist(mode: str) -> list[CheckItem]:
    ui_core = [
        CheckItem("streamlit", True, "Required for `streamlit run app.py`"),
        CheckItem("pandas", True, "DataFrames used across app modules"),
        CheckItem("numpy", True, "Numerical computations in data/portfolio modules"),
        CheckItem("plotly", True, "Charts used by Streamlit dashboard"),
        CheckItem("yfinance", True, "Market data provider used at runtime"),
        CheckItem("openai", True, "Oracle client dependency"),
        CheckItem("pydantic", True, "Schemas/settings used by app modules"),
        CheckItem("diskcache", True, "Caching backend used by data/research modules"),
        CheckItem("tenacity", True, "Retry utilities used by API clients"),
    ]

    api_optional_in_ui = [
        CheckItem("fastapi", False, "Optional API service dependency"),
        CheckItem("uvicorn", False, "Optional ASGI server dependency"),
    ]

    api_required = [
        CheckItem("fastapi", True, "Required for FastAPI app factory"),
        CheckItem("uvicorn", True, "Required to run API service"),
        CheckItem("jwt", True, "JWT auth module imported by dashboard/app_v2.py"),
        CheckItem("prometheus_client", True, "Metrics endpoint dependency"),
    ]

    full_extra = [
        CheckItem("scipy", True, "Scientific stack"),
        CheckItem("statsmodels", True, "Statistical modeling"),
        CheckItem("sklearn", True, "ML utilities"),
        CheckItem("xgboost", True, "Optional model dependency listed in requirements"),
        CheckItem("evidently", True, "Drift monitoring dependency"),
        CheckItem("great_expectations", True, "Data quality dependency"),
        CheckItem("quantstats", True, "Performance analytics"),
        CheckItem("pypfopt", True, "Portfolio optimization dependency"),
    ]

    if mode == "ui":
        return ui_core + api_optional_in_ui
    if mode == "api":
        return ui_core + api_required
    return ui_core + api_required + full_extra


def _package_version() -> str:
    try:
        return version("alpha-prime")
    except PackageNotFoundError:
        return "not-installed"


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ALPHA-PRIME dependency self-check (Windows-friendly)."
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Require API dependencies (fastapi/uvicorn/etc).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Require full dependency set (UI + API + optional analytics stack).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"alpha-prime {_package_version()}",
    )
    return parser


def main() -> int:
    args = _arg_parser().parse_args()
    mode = "full" if args.full else "api" if args.api else "ui"

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Self-check mode: {mode}\n")

    hard_failures: list[str] = []
    optional_missing: list[str] = []

    for item in _build_checklist(mode):
        try:
            module = importlib.import_module(item.module)
            ver = getattr(module, "__version__", "unknown")
            print(f"[OK]   {item.module:<18} version={ver}")
        except Exception as exc:  # noqa: BLE001
            if item.required:
                hard_failures.append(item.module)
                print(f"[FAIL] {item.module:<18} {exc}")
            else:
                optional_missing.append(item.module)
                print(f"[WARN] {item.module:<18} optional in ui mode ({exc})")

    if optional_missing and mode == "ui":
        print("\nOptional API dependencies are missing in UI mode.")
        print("This does NOT block Streamlit startup.")
        print("To enable API locally:")
        print("  pip install -r requirements-api.txt")
        print("  python scripts/selfcheck.py --api")
        print("If pip fails with 403/proxy restrictions:")
        print("  .\\scripts\\install_offline.ps1 -Tier api")
        print("  See docs/RUN_WINDOWS.md (proxy + wheelhouse sections).")

    if hard_failures:
        print("\nSelf-check FAILED. Missing required modules for selected mode:")
        for module_name in hard_failures:
            print(f" - {module_name}")
        print("\nInstall guidance:")
        print("  Online:  pip install -r requirements.txt (ui) / requirements-api.txt (api) / requirements-full.txt (full)")
        print("  Offline: .\\scripts\\install_offline.ps1 -Tier core|api|full")
        print("  Docs:    docs/RUN_WINDOWS.md")
        return 1

    print("\nSelf-check PASSED.")
    if mode == "ui":
        print("Run API check when needed: python scripts/selfcheck.py --api")
        print("Run full check when needed: python scripts/selfcheck.py --full")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
