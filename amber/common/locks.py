"""Cross-platform single-instance file lock for pipeline stages.

Prevents two runs of the same stage (e.g. an overlapping normalize timer, or a
second scanner) from racing on shared offsets/watermarks. Dependency-free: an
exclusive-create lock file holds the owner PID; a stale lock (dead PID) is taken
over automatically.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class AlreadyRunning(RuntimeError):
    """Raised when another live instance already holds the lock."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # pragma: no cover - Windows-only path
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours
    except OSError:
        return False


class SingleInstanceLock:
    """Context manager acquiring an exclusive lock file for `name`.

    Raises AlreadyRunning if a live instance holds it. Stale locks (owner PID no
    longer alive) are reclaimed.
    """

    def __init__(self, lock_dir: Path, name: str) -> None:
        self.path = Path(lock_dir) / f"{name}.lock"
        self._acquired = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._create()
        except FileExistsError:
            owner = self._read_pid()
            if owner is not None and _pid_alive(owner):
                raise AlreadyRunning(f"another instance holds {self.path} (pid {owner})")
            logger.warning("reclaiming stale lock %s (pid %s not alive)", self.path, owner)
            self.path.unlink(missing_ok=True)
            self._create()

    def _create(self) -> None:
        fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        self._acquired = True

    def _read_pid(self) -> int | None:
        try:
            return int(self.path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            return None

    def release(self) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()
