"""Process control for the dashboard: start/stop long-running services and run
one-shot pipeline stages.

State lives on disk (PID + metadata files under a runtime dir) so it survives
Streamlit reruns and even a dashboard restart. No streamlit dependency here, so
the logic is unit-testable on its own.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psutil
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

_DETACHED_PROCESS = 0x00000008  # Windows CreateProcess flag


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if psutil is not None:
        try:
            return psutil.pid_exists(int(pid))
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)  # POSIX liveness probe
        return True
    except OSError:
        return False
    except Exception:
        return False


class ProcessManager:
    """Manage background services and synchronous pipeline runs for one project."""

    def __init__(self, runtime_dir: Path, project_root: Path) -> None:
        self.dir = Path(runtime_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.root = Path(project_root)

    def _meta_path(self, name: str) -> Path:
        return self.dir / f"{name}.json"

    def _log_path(self, name: str) -> Path:
        return self.dir / f"{name}.log"

    def _read_meta(self, name: str) -> dict[str, Any] | None:
        p = self._meta_path(name)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    # --- long-running services ---------------------------------------------
    def is_running(self, name: str) -> bool:
        meta = self._read_meta(name)
        return bool(meta) and _pid_alive(meta.get("pid"))

    def status(self, name: str) -> dict[str, Any]:
        meta = self._read_meta(name) or {}
        running = self.is_running(name)
        started = meta.get("started_at")
        return {
            "name": name,
            "running": running,
            "pid": meta.get("pid") if running else None,
            "uptime_sec": (time.time() - started) if (running and started) else None,
        }

    def start(self, name: str, argv: list[str]) -> bool:
        """Spawn a detached background process. Returns False if already running."""
        if self.is_running(name):
            return False
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | _DETACHED_PROCESS  # type: ignore[attr-defined]
        else:
            kwargs["start_new_session"] = True
        # The child inherits its own dup of this handle, so we close ours after spawn.
        with self._log_path(name).open("a", encoding="utf-8") as log:
            log.write(f"\n=== start {time.strftime('%Y-%m-%d %H:%M:%S')} {argv} ===\n")
            log.flush()
            proc = subprocess.Popen(
                [sys.executable, *argv],
                cwd=str(self.root),
                stdout=log,
                stderr=subprocess.STDOUT,
                **kwargs,
            )
        self._meta_path(name).write_text(
            json.dumps({"pid": proc.pid, "argv": argv, "started_at": time.time()}),
            encoding="utf-8",
        )
        return True

    def stop(self, name: str) -> bool:
        meta = self._read_meta(name)
        if not meta:
            return False
        pid = meta.get("pid")
        if _pid_alive(pid):
            try:
                if psutil is not None:
                    p = psutil.Process(int(pid))
                    for child in p.children(recursive=True):
                        child.terminate()
                    p.terminate()
                    _gone, alive = psutil.wait_procs([p], timeout=5)
                    for a in alive:
                        a.kill()
                elif os.name == "nt":
                    subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], check=False)
                else:
                    os.killpg(os.getpgid(int(pid)), signal.SIGTERM)
            except Exception:
                pass
        try:
            self._meta_path(name).unlink()
        except OSError:
            pass
        return True

    def tail_log(self, name: str, lines: int = 40) -> str:
        p = self._log_path(name)
        if not p.exists():
            return ""
        return "\n".join(p.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])

    # --- one-shot pipeline stages ------------------------------------------
    def run_once(self, argv: list[str], timeout: int = 1800) -> tuple[int, str]:
        """Run a script synchronously; return (returncode, combined output)."""
        try:
            proc = subprocess.run(
                [sys.executable, *argv],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            return 124, f"timeout after {timeout}s\n{exc.stdout or ''}{exc.stderr or ''}"
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out


def set_symbols(project_root: Path, symbols: list[str]) -> list[str]:
    """Update the collected symbol universe.

    Written to `config/amber.local.yaml` (a gitignored override merged over the
    tracked `config/amber.yaml` at load time), so editing symbols never conflicts
    with `git pull` of the defaults.
    """
    import yaml

    cleaned = [s.strip().upper() for s in symbols if s.strip()]
    local_path = Path(project_root) / "config" / "amber.local.yaml"
    data: dict = {}
    if local_path.is_file():
        loaded = yaml.safe_load(local_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded
    data.setdefault("exchange", {}).setdefault("bybit", {})["symbols"] = cleaned
    local_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cleaned


# Command registry used by the dashboard control panel.
SERVICES: dict[str, dict[str, Any]] = {
    "ws_collector": {"label": "WS-коллектор (сбор сырых данных)", "argv": ["scripts/run_ws_collector.py"]},
    "pipeline": {"label": "Авто-конвейер (normalize + features)", "argv": ["scripts/run_pipeline_loop.py"]},
    "scanner": {"label": "Сканер (сигналы, loop)", "argv": ["scripts/run_scanner.py", "--loop"]},
}

PIPELINE_STEPS: list[dict[str, Any]] = [
    {"key": "backfill", "label": "REST-backfill истории", "argv": ["scripts/run_collectors.py"]},
    {"key": "normalize", "label": "Нормализация", "argv": ["scripts/run_normalize.py"]},
    {"key": "features", "label": "Фичи", "argv": ["scripts/run_features.py"]},
    {"key": "dataset", "label": "Датасет", "argv": ["scripts/build_dataset.py"]},
    {"key": "train", "label": "Обучение + калибровка + eval", "argv": ["scripts/train_model.py"]},
    {"key": "backtest", "label": "Бэктест", "argv": ["scripts/run_backtest.py"]},
    {"key": "cleanup", "label": "🧹 Очистить диск", "argv": ["scripts/cleanup.py"]},
]

# The one-click sequence. REST backfill runs first so a fresh install has enough
# history to clear the warm-up threshold; it is idempotent (watermark dedup), so
# re-running the cycle never duplicates data. Cleanup is a standalone button.
FULL_CYCLE: list[dict[str, Any]] = [s for s in PIPELINE_STEPS if s["key"] != "cleanup"]

# Exit code the train step uses to signal "not enough data yet" (informational,
# not a real failure).
NOT_ENOUGH_DATA_CODE = 2
