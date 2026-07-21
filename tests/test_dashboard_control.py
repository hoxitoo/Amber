"""Tests for the dashboard process manager (no streamlit dependency)."""

import tempfile
import time
import unittest
from pathlib import Path

from amber.dashboard import control as C


class TestProcessManager(unittest.TestCase):
    def test_run_once_captures_output_and_returncode(self):
        with tempfile.TemporaryDirectory() as td:
            pm = C.ProcessManager(Path(td) / "procs", Path(td))
            rc, out = pm.run_once(["-c", "print('hello amber')"])
            self.assertEqual(rc, 0)
            self.assertIn("hello amber", out)

            rc2, _ = pm.run_once(["-c", "import sys; sys.exit(3)"])
            self.assertEqual(rc2, 3)

    def test_start_status_stop_lifecycle(self):
        with tempfile.TemporaryDirectory() as td:
            pm = C.ProcessManager(Path(td) / "procs", Path(td))
            self.assertFalse(pm.is_running("svc"))
            started = pm.start("svc", ["-c", "import time; time.sleep(30)"])
            self.assertTrue(started)
            time.sleep(0.5)
            self.assertTrue(pm.is_running("svc"))
            self.assertEqual(pm.status("svc")["running"], True)

            # starting again while running is a no-op
            self.assertFalse(pm.start("svc", ["-c", "import time; time.sleep(30)"]))

            self.assertTrue(pm.stop("svc"))
            time.sleep(0.5)
            self.assertFalse(pm.is_running("svc"))

    def test_tail_log_returns_process_output(self):
        with tempfile.TemporaryDirectory() as td:
            pm = C.ProcessManager(Path(td) / "procs", Path(td))
            pm.start("logger", ["-c", "print('marker-line'); import time; time.sleep(5)"])
            time.sleep(0.7)
            log = pm.tail_log("logger")
            pm.stop("logger")
            self.assertIn("marker-line", log)

    def test_stop_unknown_process_is_safe(self):
        with tempfile.TemporaryDirectory() as td:
            pm = C.ProcessManager(Path(td) / "procs", Path(td))
            self.assertFalse(pm.stop("nope"))


class TestSetSymbols(unittest.TestCase):
    def test_set_symbols_writes_config_and_backup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg = root / "config"
            cfg.mkdir()
            (cfg / "amber.yaml").write_text(
                "exchange:\n  bybit:\n    testnet: false\n    symbols:\n    - BTCUSDT\n",
                encoding="utf-8",
            )
            saved = C.set_symbols(root, ["btcusdt", " ethusdt ", "", "SOLUSDT"])
            self.assertEqual(saved, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])

            import yaml

            data = yaml.safe_load((cfg / "amber.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["exchange"]["bybit"]["symbols"], ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
            # untouched keys preserved
            self.assertEqual(data["exchange"]["bybit"]["testnet"], False)
            # backup exists
            self.assertTrue((cfg / "amber.yaml.bak").exists())
            self.assertIn("BTCUSDT", (cfg / "amber.yaml.bak").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
