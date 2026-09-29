"""Cross-platform single-instance file lock for pipeline stages.

Prevents two runs of the same stage (e.g. an overlapping normalize timer, or a
second scanner) from racing on shared offsets/watermarks. Dependency-free: a
kernel flock on POSIX (released automatically when the holder dies), an
exclusive-create PID file on Windows.
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


def _flock_available() -> bool:
    try:
        import fcntl  # noqa: F401
    except ImportError:  # pragma: no cover - Windows
        return False
    return True


class SingleInstanceLock:
    """Context manager acquiring an exclusive lock for `name`.

    Raises AlreadyRunning if a live instance holds it.

    On POSIX the lock is a kernel `flock` on the file, not the PID written in
    it: the kernel drops it the moment the holder dies, however it dies, so
    there is no stale lock to reclaim and no guessing from PIDs. The PID-only
    scheme this replaced trusted "a process with that PID exists", which after
    a reboot — lock files persist, PIDs restart from low numbers — could be any
    unrelated process, and the service would then refuse to start, silently,
    for as long as that process lived. The PID is still written, for display.

    Windows keeps the exclusive-create scheme with its PID liveness check.
    """

    def __init__(self, lock_dir: Path, name: str) -> None:
        self.path = Path(lock_dir) / f"{name}.lock"
        self._acquired = False
        self._fd: int | None = None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if _flock_available():
            self._acquire_flock()
            return
        try:  # pragma: no cover - Windows path
            self._create()
        except FileExistsError:  # pragma: no cover
            owner = self._read_pid()
            if owner is not None and _pid_alive(owner):
                raise AlreadyRunning(f"another instance holds {self.path} (pid {owner})")
            logger.warning("reclaiming stale lock %s (pid %s not alive)", self.path, owner)
            self.path.unlink(missing_ok=True)
            self._create()

    def _acquire_flock(self) -> None:
        import fcntl

        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            raise AlreadyRunning(f"another instance holds {self.path} (pid {self._read_pid()})") from None
        os.ftruncate(fd, 0)
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd
        self._acquired = True

    def _create(self) -> None:  # pragma: no cover - Windows path
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
        if not self._acquired:
            return
        if self._fd is not None:
            # Closing drops the flock. The file stays: unlinking it would let a
            # waiter lock the old inode while a newcomer locks a new one.
            os.close(self._fd)
            self._fd = None
        else:  # pragma: no cover - Windows path
            self.path.unlink(missing_ok=True)
        self._acquired = False

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def lock_holder(lock_dir: Path, name: str) -> int | None:
    """PID of the live process holding `name`, or None if nobody holds it."""
    path = Path(lock_dir) / f"{name}.lock"
    if not path.exists():
        return None
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        pid = None
    if not _flock_available():  # pragma: no cover - Windows
        return pid if pid is not None and _pid_alive(pid) else None
    import fcntl

    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except OSError:
        return pid if pid is not None else -1  # held; PID unreadable
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)
