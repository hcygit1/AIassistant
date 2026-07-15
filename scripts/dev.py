#!/usr/bin/env python3
"""Unified local development launcher for PIPIXIA.

Usage:
  python scripts/dev.py
  python scripts/dev.py --backend-only
  python scripts/dev.py --frontend-only --frontend-port 3001
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"
DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8002",
    "http://127.0.0.1:8002",
)


def build_frontend_env(
    base_env: Mapping[str, str],
    *,
    backend_port: int,
    frontend_port: int,
) -> dict[str, str]:
    env = dict(base_env)
    env["PORT"] = str(frontend_port)
    env["NEXT_PUBLIC_API_URL"] = f"http://localhost:{backend_port}/api"
    return env


def build_backend_env(
    base_env: Mapping[str, str],
    *,
    frontend_port: int,
) -> dict[str, str]:
    env = dict(base_env)
    configured_origins = env.get("PIPIXIA_CORS_ORIGINS", "").strip()
    if configured_origins:
        origins = [
            origin.strip()
            for origin in configured_origins.split(",")
            if origin.strip()
        ]
    else:
        origins = list(DEFAULT_CORS_ORIGINS)

    for origin in (
        f"http://localhost:{frontend_port}",
        f"http://127.0.0.1:{frontend_port}",
    ):
        if origin not in origins:
            origins.append(origin)

    env["PIPIXIA_CORS_ORIGINS"] = ",".join(origins)
    return env


def _run(command: list[str], cwd: Path) -> int:
    return subprocess.run(command, cwd=str(cwd), check=False).returncode


def _start(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.Popen:
    return subprocess.Popen(command, cwd=str(cwd), env=env)


def _terminate_all(processes: list[subprocess.Popen]) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        if p.poll() is None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PIPIXIA one-command development launcher",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/dev.py\n"
            "  python scripts/dev.py --skip-install\n"
            "  python scripts/dev.py --backend-only\n"
            "  python scripts/dev.py --frontend-only --frontend-port 3001\n"
        ),
    )
    parser.add_argument("--backend-only", action="store_true", help="only run backend")
    parser.add_argument("--frontend-only", action="store_true", help="only run frontend")
    parser.add_argument("--skip-install", action="store_true", help="skip dependency install checks")
    parser.add_argument("--python", default=sys.executable, help="python executable for backend")
    parser.add_argument("--backend-port", type=int, default=8002, help="backend port")
    parser.add_argument("--frontend-port", type=int, default=3000, help="frontend port")
    args = parser.parse_args()

    if args.backend_only and args.frontend_only:
        print("[ERROR] --backend-only and --frontend-only cannot be used together.")
        return 2

    run_backend = not args.frontend_only
    run_frontend = not args.backend_only

    print("\n================================================")
    print("PIPIXIA Dev Launcher")
    print("================================================")
    print(f"[INFO] Root: {ROOT}")
    print(f"[INFO] Backend: {'ON' if run_backend else 'OFF'}")
    print(f"[INFO] Frontend: {'ON' if run_frontend else 'OFF'}")

    if not args.skip_install:
        if run_backend:
            print("[INFO] Installing backend dependencies...")
            code = _run([args.python, "-m", "pip", "install", "-r", "requirements.txt"], BACKEND_DIR)
            if code != 0:
                print("[ERROR] Backend dependency install failed.")
                return code
        if run_frontend:
            print("[INFO] Installing frontend dependencies...")
            code = _run(["npm", "install"], FRONTEND_DIR)
            if code != 0:
                print("[ERROR] Frontend dependency install failed.")
                return code

    processes: list[subprocess.Popen] = []
    try:
        if run_backend:
            print(f"[INFO] Starting backend on :{args.backend_port}")
            # Use serve (not start) to avoid interactive prompts mixing with frontend output.
            # First-time config: use Web UI at localhost or run `cd backend && python cli.py onboard` separately.
            backend_cmd = [args.python, "cli.py", "serve", "--port", str(args.backend_port)]
            backend_env = build_backend_env(
                os.environ,
                frontend_port=args.frontend_port,
            )
            processes.append(_start(backend_cmd, BACKEND_DIR, env=backend_env))

        if run_frontend:
            print(f"[INFO] Starting frontend on :{args.frontend_port}")
            frontend_env = build_frontend_env(
                os.environ,
                backend_port=args.backend_port,
                frontend_port=args.frontend_port,
            )
            processes.append(
                _start(["npm", "run", "dev"], FRONTEND_DIR, env=frontend_env)
            )

        print("[OK] Services are running. Press Ctrl+C to stop.")
        while True:
            for p in processes:
                code = p.poll()
                if code is not None:
                    print(f"[WARN] A process exited with code {code}, stopping all...")
                    _terminate_all(processes)
                    return code
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Stopping services...")
        _terminate_all(processes)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
