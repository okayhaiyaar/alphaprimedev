from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import __version__

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import get_run_id, get_settings, is_mock_mode, mask_openai_key, mask_webhook_url


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> int:
    return subprocess.call(cmd, cwd=str(_repo_root()))


def ui() -> int:
    parser = argparse.ArgumentParser(description="Run ALPHA-PRIME Streamlit UI")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--address", default="127.0.0.1")
    parser.add_argument("--version", action="version", version=f"alpha-prime {__version__}")
    args = parser.parse_args()
    return _run([
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        "--server.port",
        str(args.port),
        "--server.address",
        args.address,
    ])


def api() -> int:
    parser = argparse.ArgumentParser(description="Run ALPHA-PRIME FastAPI service")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--version", action="version", version=f"alpha-prime {__version__}")
    args = parser.parse_args()
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "dashboard.app_v2:create_app",
        "--factory",
        "--port",
        str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")
    return _run(cmd)


def scheduler() -> int:
    parser = argparse.ArgumentParser(description="Run ALPHA-PRIME scheduler")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to scheduler.py")
    parser.add_argument("--version", action="version", version=f"alpha-prime {__version__}")
    ns = parser.parse_args()
    sched_args = ns.args if ns.args else ["once"]
    return _run([sys.executable, "scheduler.py", *sched_args])


def _import_ok(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except Exception:
        return False


def collect_doctor_info() -> dict[str, Any]:
    root = _repo_root()
    wheelhouse = root / "wheelhouse"
    in_venv = bool(os.getenv("VIRTUAL_ENV")) or (
        hasattr(sys, "base_prefix") and sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )

    fastapi_ok = _import_ok("fastapi")
    uvicorn_ok = _import_ok("uvicorn")

    try:
        settings = get_settings()
        openai_display = mask_openai_key(settings.openai_api_key)
        webhook_display = mask_webhook_url(settings.discord_webhook_url)
    except Exception:  # noqa: BLE001
        openai_display = mask_openai_key(os.getenv("OPENAI_API_KEY"))
        webhook_display = mask_webhook_url(os.getenv("DISCORD_WEBHOOK_URL"))

    info: dict[str, Any] = {
        "version": __version__,
        "run_id": get_run_id(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "repo_root": str(root),
        "in_venv": in_venv,
        "mock_mode": is_mock_mode(),
        "wheelhouse_present": wheelhouse.exists(),
        "api_deps_installed": fastapi_ok and uvicorn_ok,
        "fastapi_installed": fastapi_ok,
        "uvicorn_installed": uvicorn_ok,
        "openai_api_key": openai_display,
        "discord_webhook": webhook_display,
        "recommended_commands": {
            "online_core": "pip install -r requirements.txt",
            "online_api": "pip install -r requirements-api.txt",
            "online_full": "pip install -r requirements-full.txt",
            "offline_core": ".\\scripts\\install_offline.ps1 -Tier core",
            "offline_api": ".\\scripts\\install_offline.ps1 -Tier api",
            "offline_full": ".\\scripts\\install_offline.ps1 -Tier full",
            "verify": "python scripts/selfcheck.py / --api / --full",
        },
    }

    if wheelhouse.exists():
        names = [p.name.lower() for p in wheelhouse.glob("*.whl")]
        info["wheelhouse_has_fastapi"] = any(n.startswith("fastapi-") for n in names)
        info["wheelhouse_has_uvicorn"] = any(n.startswith("uvicorn-") for n in names)

    return info


def _render_doctor(info: dict[str, Any]) -> list[str]:
    lines = [
        f"alpha-prime {info['version']}",
        f"RUN_ID: {info['run_id']}",
        f"Python: {info['python']}",
        f"Executable: {info['executable']}",
        f"Repo root: {info['repo_root']}",
        f"Virtual env active: {'yes' if info['in_venv'] else 'no'}",
        f"MOCK_API_CALLS: {'on' if info['mock_mode'] else 'off'}",
        f"OPENAI_API_KEY: {info['openai_api_key']}",
        f"DISCORD_WEBHOOK_URL: {info['discord_webhook']}",
        f"Wheelhouse present: {'yes' if info['wheelhouse_present'] else 'no'}",
        f"API deps installed: {'yes' if info['api_deps_installed'] else 'no'}",
    ]
    if "wheelhouse_has_fastapi" in info:
        has_api_wheels = info["wheelhouse_has_fastapi"] and info["wheelhouse_has_uvicorn"]
        lines.append(f"Wheelhouse API wheels: {'yes' if has_api_wheels else 'no'}")

    lines.extend(
        [
            "",
            "Recommended commands:",
            f"  Online core:   {info['recommended_commands']['online_core']}",
            f"  Online api:    {info['recommended_commands']['online_api']}",
            f"  Online full:   {info['recommended_commands']['online_full']}",
            f"  Offline core:  {info['recommended_commands']['offline_core']}",
            f"  Offline api:   {info['recommended_commands']['offline_api']}",
            f"  Offline full:  {info['recommended_commands']['offline_full']}",
            f"  Verify:        {info['recommended_commands']['verify']}",
        ]
    )
    if not info["api_deps_installed"]:
        lines.append(
            "\nTip: API deps missing. Install requirements-api.txt or use offline api tier."
        )
    lines.append(
        "If pip errors with 403/proxy restrictions, use wheelhouse workflow in docs/RUN_WINDOWS.md."
    )
    return lines


def doctor() -> int:
    try:
        info = collect_doctor_info()
        for line in _render_doctor(info):
            print(line)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"doctor failed: {exc}", file=sys.stderr)
        print("Try: python scripts/selfcheck.py and docs/RUN_WINDOWS.md", file=sys.stderr)
        return 1


def main_version() -> int:
    print(__version__)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ALPHA-PRIME command launcher")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Run environment diagnostics")
    sub.add_parser("version", help="Print version")
    args, _ = parser.parse_known_args()

    if args.command == "doctor":
        return doctor()
    return main_version()


if __name__ == "__main__":
    raise SystemExit(main())
