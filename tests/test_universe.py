import json
import tempfile
import unittest
from pathlib import Path

from amber.signals.universe import select_universe


class TestUniverse(unittest.TestCase):
    def test_select_top_k(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "features"
            for s, z in [("A", 0.1), ("B", 2.0), ("C", 1.0)]:
                d = root / "features" / s
                d.mkdir(parents=True, exist_ok=True)
                with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"symbol": s, "vol_z_20": z, "obs": 10}) + "\n")
            u = select_universe(root, top_k=2)
            self.assertEqual(len(u), 2)
            self.assertEqual(u[0], "B")

<<<<<<< HEAD
    def test_select_universe_applies_liquidity_floor(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "features"
            rows = [
                ("A", 3.0, 1000.0),
                ("B", 2.0, 25000.0),
                ("C", 1.0, 40000.0),
            ]
            for s, z, dv in rows:
                d = root / "features" / s
                d.mkdir(parents=True, exist_ok=True)
                with (d / "part-000.jsonl").open("w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"symbol": s, "vol_z_20": z, "obs": 10, "notional_volume_1m": dv}) + "\n")
            u = select_universe(root, top_k=3, min_dollar_volume=20_000)
            self.assertEqual(u, ["B", "C"])

=======
>>>>>>> origin/main

if __name__ == "__main__":
    unittest.main()
