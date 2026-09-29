"""read_tail reads from the end of the file; it must return exactly what the
old whole-file version did, for every shape of file the box produces."""

import json
import random
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path

import amber.common.jsonl as J


def _reference(path: Path, n: int) -> list:
    """The previous implementation: stream every line through a deque."""
    if n <= 0:
        return []
    with path.open("r", encoding="utf-8") as fh:
        lines = deque(fh, maxlen=n)
    out = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


class TestReadTail(unittest.TestCase):
    def setUp(self):
        self._block = J._BLOCK

    def tearDown(self):
        J._BLOCK = self._block

    def test_matches_the_whole_file_reader(self):
        rng = random.Random(0)
        f = Path(tempfile.mkdtemp()) / "f.jsonl"
        for trial in range(2000):
            # Tiny blocks force every boundary case: a block ending mid-line,
            # mid-character (é is two bytes), exactly on a newline.
            J._BLOCK = rng.choice([1, 2, 7, 64, 1 << 16])
            parts = []
            for i in range(rng.randint(0, 60)):
                k = rng.random()
                parts.append("" if k < .1 else "{bad" if k < .15 else json.dumps({"i": i, "s": "é" * rng.randint(0, 5)}))
            text = "\n".join(parts) + ("\n" if rng.random() < .7 and parts else "")
            f.write_text(text, encoding="utf-8")
            n = rng.randint(0, 70)
            self.assertEqual(J.read_tail(f, n), _reference(f, n), f"trial {trial}, n={n}")

    def test_cost_does_not_grow_with_the_file(self):
        f = Path(tempfile.mkdtemp()) / "big.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for i in range(300_000):
                fh.write(json.dumps({"ts": i, "close": 1.0}) + "\n")
        t = time.perf_counter()
        rows = J.read_tail(f, 100)
        elapsed = time.perf_counter() - t
        t = time.perf_counter()
        _reference(f, 100)
        whole_file = time.perf_counter() - t
        self.assertEqual([r["ts"] for r in rows], list(range(299_900, 300_000)))
        self.assertLess(elapsed * 5, whole_file, "tail read costs as much as scanning the file")


if __name__ == "__main__":
    unittest.main()
