"""One retrain, one dataset build, one of each service per box.

The dashboard starts processes itself while systemd runs the same ones on the
VPS. Without these locks its buttons launched second copies: two retrains at
~2.2 GB each on a 3.9 GB box, and two collectors double-counting every trade
and liquidation in the per-minute sums.
"""

import os
import tempfile
import unittest
from pathlib import Path

from amber.common.locks import AlreadyRunning, SingleInstanceLock
from amber.dashboard import control as C
from amber.datasets.build import build_dataset_from_config
from amber.pipeline.train_app import run_training


class TestRetrainAndBuildAreExclusive(unittest.TestCase):
    def test_a_second_retrain_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            models = Path(td) / "models"
            models.mkdir()
            with SingleInstanceLock(models, "train"):
                with self.assertRaises(AlreadyRunning):
                    run_training({}, Path(td) / "datasets", models, Path(td) / "logs")

    def test_a_second_dataset_build_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            datasets = Path(td) / "datasets"
            datasets.mkdir()
            cfg = {"storage": {"datasets_dir": str(datasets), "features_dir": str(Path(td) / "f")}}
            with SingleInstanceLock(datasets, "build"):
                with self.assertRaises(AlreadyRunning):
                    build_dataset_from_config(cfg)


class TestLocksSurviveAReboot(unittest.TestCase):
    def test_a_leftover_lock_naming_a_live_unrelated_pid_does_not_block(self):
        """After a reboot the lock file persists and its PID may belong to any
        new process. Only a held lock may block a start."""
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "stage.lock").write_text(str(os.getpid()))  # alive, but holds nothing
            with SingleInstanceLock(Path(td), "stage"):
                pass

    def test_a_holder_that_dies_releases_the_lock(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as td:
            code = (
                "import sys, os; sys.path.insert(0, %r)\n"
                "from amber.common.locks import SingleInstanceLock\n"
                "l = SingleInstanceLock(%r, 'stage'); l.acquire(); os._exit(9)\n"
            ) % (str(Path(__file__).resolve().parents[1]), td)
            subprocess.run([sys.executable, "-c", code], check=False)
            with SingleInstanceLock(Path(td), "stage"):  # killed without releasing
                pass


class TestDashboardSeesSystemdServices(unittest.TestCase):
    def test_a_service_it_did_not_start_shows_running_and_cannot_be_started_twice(self):
        with tempfile.TemporaryDirectory() as td:
            state = Path(td) / "state"
            pm = C.ProcessManager(state / "procs", Path(td))
            self.assertFalse(pm.status("pipeline")["running"])
            # What systemd's instance holds: the service lock.
            with SingleInstanceLock(state / "locks", "pipeline_loop"):
                st = pm.status("pipeline")
                self.assertTrue(st["running"])
                self.assertTrue(st["external"])
                self.assertEqual(st["pid"], os.getpid())
                self.assertFalse(pm.start("pipeline", ["-c", "pass"]), "started a second copy")
            self.assertFalse(pm.status("pipeline")["running"], "a released lock still reads as running")

    def test_every_service_names_the_lock_its_entry_point_takes(self):
        repo = Path(__file__).resolve().parents[1]
        for name, svc in C.SERVICES.items():
            script = (repo / svc["argv"][0]).read_text(encoding="utf-8")
            if name == "scanner":
                script += (repo / "amber" / "pipeline" / "scanner_app.py").read_text(encoding="utf-8")
            self.assertIn(f'"{svc["lock"]}"', script, f"{name}: entry point does not take lock {svc['lock']!r}")


if __name__ == "__main__":
    unittest.main()
