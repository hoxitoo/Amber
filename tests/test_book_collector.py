"""Order book collector (roadmap D2): book upkeep, rows, files, and a full run
against a local stand-in for Bybit's public socket."""

import asyncio
import gzip
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from amber.exchange import orderbook as OB
from amber.exchange.streams import subscribe_requests
from amber.pipeline import book_app as BA


def _snap(bids, asks, u=100):
    return {"b": [[str(p), str(s)] for p, s in bids], "a": [[str(p), str(s)] for p, s in asks], "u": u}


class TestBook(unittest.TestCase):
    def test_snapshot_then_deltas_including_removals(self):
        b = OB.Book()
        b.apply("snapshot", _snap([(100, 1), (99, 2)], [(101, 1), (102, 3)]), 1)
        b.apply("delta", _snap([(99, 0), (100.5, 4)], [(101, 0)], u=101), 2)
        self.assertEqual(b.bids, {100.0: 1.0, 100.5: 4.0})
        self.assertEqual(b.asks, {102.0: 3.0})
        self.assertEqual(b.updated_ms, 2)

    def test_deltas_before_the_first_snapshot_are_ignored(self):
        b = OB.Book()
        b.apply("delta", _snap([(100, 1)], [(101, 1)]), 1)
        self.assertIsNone(b.features())

    def test_update_id_one_is_a_fresh_snapshot(self):
        b = OB.Book()
        b.apply("snapshot", _snap([(100, 1)], [(101, 1)]), 1)
        b.apply("delta", _snap([(50, 1)], [(51, 1)], u=1), 2)
        self.assertEqual((b.bids, b.asks), ({50.0: 1.0}, {51.0: 1.0}))

    def test_crossed_book_is_not_used(self):
        b = OB.Book()
        b.apply("snapshot", _snap([(101, 1)], [(100, 1)]), 1)
        self.assertIsNone(b.features())
        self.assertEqual(b.crossed, 1)

    def test_depth_bands(self):
        b = OB.Book()
        # mid 100: bids at 99.96 (4 bps), 99.8 (20 bps), 99.0 (100 bps); asks at 100.04, 100.3, 102
        b.apply("snapshot", _snap([(99.96, 1), (99.8, 1), (99.0, 1)], [(100.04, 1), (100.3, 1), (102, 1)]), 1)
        f = b.features()
        self.assertAlmostEqual(f["mid"], 100.0)
        self.assertAlmostEqual(f["spread_bps"], 8.0)
        self.assertAlmostEqual(f["bid5_usd"], 99.96)
        self.assertAlmostEqual(f["bid25_usd"], 99.96 + 99.8)
        self.assertAlmostEqual(f["bid100_usd"], 99.96 + 99.8 + 99.0)
        self.assertAlmostEqual(f["ask25_usd"], 100.04)
        self.assertAlmostEqual(f["ask50_usd"], 100.04 + 100.3)
        self.assertAlmostEqual(f["ask100_usd"], 100.04 + 100.3)
        self.assertAlmostEqual(f["ask_cov_bps"], 200.0)


class TestTrades(unittest.TestCase):
    def test_bucketed_by_trade_time_and_late_trades_counted(self):
        t = OB.TradeBuckets(5000)
        t.add("A", {"T": 10_001, "S": "Buy", "v": "2", "p": "10"})
        t.add("A", {"T": 14_999, "S": "Sell", "v": "1", "p": "10"})
        t.add("A", {"T": 15_000, "S": "Buy", "v": "1", "p": "10"})  # next interval
        self.assertEqual(t.take("A", 10_000), [20.0, 10.0, 1, 1, 20.0])
        t.add("A", {"T": 12_000, "S": "Buy", "v": "1", "p": "10"})  # its interval is written
        self.assertEqual(t.late, 1)
        self.assertEqual(t.take("A", 15_000), [10.0, 0.0, 1, 0, 10.0])

    def test_malformed_trades_are_skipped(self):
        t = OB.TradeBuckets(5000)
        t.add("A", {"T": "x", "S": "Buy", "v": "1", "p": "1"})
        t.add("A", {"S": "Buy"})
        self.assertEqual(t.take("A", 0), [0.0, 0.0, 0, 0, 0.0])


class TestRows(unittest.TestCase):
    def test_round_trip(self):
        b = OB.Book()
        b.apply("snapshot", _snap([(99.96, 1)], [(100.04, 1)]), 1)
        line = OB.format_row(5000, b.features(), [1.0, 2.0, 1, 1, 2.0], 300)
        r = OB.parse_row(line)
        self.assertEqual(r["ts"], 5000)
        self.assertAlmostEqual(r["mid"], 100.0)
        self.assertEqual(r["book_age_ms"], 300)
        self.assertEqual(len(line.split(",")), len(OB.COLUMNS))

    def test_no_book_still_writes_the_flow(self):
        r = OB.parse_row(OB.format_row(5000, None, [5.0, 0.0, 1, 0, 5.0], 99_000))
        self.assertIsNone(r["mid"])
        self.assertIsNone(r["book_age_ms"])
        self.assertEqual(r["buy_usd"], 5.0)

    def test_torn_and_header_lines_are_rejected(self):
        self.assertIsNone(OB.parse_row(",".join(OB.COLUMNS)))
        self.assertIsNone(OB.parse_row("5000,100.0,8"))


class TestWriter(unittest.TestCase):
    def test_header_once_and_append_after_a_restart(self):
        with tempfile.TemporaryDirectory() as td:
            w = BA.BookWriter(Path(td), 45)
            w.write(1_760_000_000_000, {"A": "x1"})
            w.close()
            w2 = BA.BookWriter(Path(td), 45)
            w2.write(1_760_000_005_000, {"A": "x2"})
            w2.close()
            lines = next(Path(td).glob("A/*.csv")).read_text().splitlines()
            self.assertEqual(lines, [",".join(OB.COLUMNS), "x1", "x2"])

    def test_finished_days_are_gzipped_and_old_days_deleted(self):
        day = 86_400_000
        t0 = 1_760_000_000_000 - 1_760_000_000_000 % day
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "A").mkdir()
            (root / "A" / "2020-01-01.csv.gz").write_bytes(b"old")
            w = BA.BookWriter(root, 45)
            w.write(t0, {"A": "d1"})
            w.write(t0 + day, {"A": "d2"})
            w.close()
            names = sorted(p.name for p in (root / "A").iterdir())
            d1, d2 = BA._day(t0), BA._day(t0 + day)
            self.assertEqual(names, [f"{d1}.csv.gz", f"{d2}.csv"])
            with gzip.open(root / "A" / f"{d1}.csv.gz", "rt") as fh:
                self.assertEqual(fh.read().splitlines()[1], "d1")

    def test_a_restart_on_a_later_day_compresses_the_leftover_day(self):
        day = 86_400_000
        t0 = 1_760_000_000_000 - 1_760_000_000_000 % day
        with tempfile.TemporaryDirectory() as td:
            w = BA.BookWriter(Path(td), 45)
            w.write(t0, {"A": "d1"})
            w.close()  # the process died before midnight
            w2 = BA.BookWriter(Path(td), 45)
            w2.write(t0 + 2 * day, {"A": "d3"})
            w2.close()
            self.assertTrue((Path(td) / "A" / f"{BA._day(t0)}.csv.gz").exists())

    def test_compressing_never_overwrites_an_existing_archive(self):
        day = 86_400_000
        t0 = 1_760_000_000_000 - 1_760_000_000_000 % day
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "A").mkdir()
            with gzip.open(root / "A" / f"{BA._day(t0)}.csv.gz", "wt") as fh:
                fh.write("header\nearlier\n")
            w = BA.BookWriter(root, 45)
            w.write(t0, {"A": "later"})  # the clock stepped back into that day
            w.write(t0 + day, {"A": "next"})
            w.close()
            with gzip.open(root / "A" / f"{BA._day(t0)}.csv.gz", "rt") as fh:
                text = fh.read()
            self.assertIn("earlier", text)
            self.assertIn("later", text)

    def test_nothing_is_written_on_a_full_disk(self):
        with tempfile.TemporaryDirectory() as td:
            w = BA.BookWriter(Path(td), 45)
            with mock.patch.object(BA, "free_bytes", return_value=10):
                w.write(1_760_000_000_000, {"A": "x"})
            self.assertFalse(list(Path(td).rglob("*.csv")))


class TestSampling(unittest.TestCase):
    def _collector(self):
        c = BA.BookCollector(["AUSDT"], Path(tempfile.mkdtemp()), interval_sec=5)
        c.books["AUSDT"].apply("snapshot", _snap([(99.9, 1)], [(100.1, 1)]), 1)
        return c

    def test_a_dead_socket_writes_flow_but_no_book(self):
        c = self._collector()
        c.last_msg_ms = 1_000_000
        live = OB.parse_row(c.sample(1_005_000, 1_005_500)["AUSDT"])
        dead = OB.parse_row(c.sample(1_030_000, 1_030_500)["AUSDT"])  # 30 s of silence
        self.assertAlmostEqual(live["mid"], 100.0)
        self.assertIsNone(dead["mid"])
        self.assertEqual(c.stale, 1)

    def test_a_clock_step_back_does_not_write_an_interval_twice(self):
        c = self._collector()
        c.last_msg_ms = 1_000_000
        self.assertTrue(c.sample(1_005_000, 1_005_100))
        self.assertEqual(c.sample(1_005_000, 1_005_200), {})
        self.assertEqual(c.sample(1_000_000, 1_005_300), {})

    def test_trades_of_a_skipped_interval_are_counted(self):
        c = self._collector()
        c.flow.add("AUSDT", {"T": 1_000_100, "S": "Buy", "v": "1", "p": "1"})
        c.last_msg_ms = 1_010_000
        c.sample(1_015_000, 1_015_100)  # interval 1_000_000 was never sampled
        self.assertEqual(c.flow.late, 1)


class TestTopics(unittest.TestCase):
    def test_book_and_trades_are_separate_subscribe_requests(self):
        reqs = subscribe_requests(BA.book_topics(["AUSDT", "BUSDT"], 200, True))
        self.assertEqual([r["req_id"] for r in reqs], ["orderbook", "publicTrade"])
        self.assertEqual(reqs[0]["args"], ["orderbook.200.AUSDT", "orderbook.200.BUSDT"])


class _FakeBybit:
    """Answers subscribes, sends a snapshot per book topic, then deltas and
    trades; drops the first connection after `drop_after` seconds."""

    def __init__(self, drop_after=None):
        self.connections = 0
        self.drop_after = drop_after

    async def handler(self, ws, *_):
        self.connections += 1
        first = self.connections == 1
        req = json.loads(await ws.recv())
        await ws.send(json.dumps({"success": True, "op": "subscribe", "req_id": req["req_id"]}))
        req2 = json.loads(await ws.recv())
        await ws.send(json.dumps({"success": True, "op": "subscribe", "req_id": req2["req_id"]}))
        books = [t for t in req["args"] + req2["args"] if t.startswith("orderbook")]
        trades = [t for t in req["args"] + req2["args"] if t.startswith("publicTrade")]
        now = int(time.time() * 1000)
        for t in books:
            await ws.send(json.dumps({"topic": t, "type": "snapshot", "ts": now,
                                      "data": _snap([(99.9, 1), (99.5, 2)], [(100.1, 1), (100.5, 2)], u=1)}))
        start = time.time()
        k = 0
        try:
            await self._stream(ws, books, trades, first, start, k)
        except Exception:  # the client went away at the end of the test
            return

    async def _stream(self, ws, books, trades, first, start, k):
        while True:
            if first and self.drop_after and time.time() - start > self.drop_after:
                await ws.close()
                return
            now = int(time.time() * 1000)
            for t in books:
                await ws.send(json.dumps({"topic": t, "type": "delta", "ts": now,
                                          "data": _snap([(99.9, 1 + k % 3)], [], u=2 + k)}))
            for t in trades:
                await ws.send(json.dumps({"topic": t, "type": "snapshot", "ts": now, "data": [
                    {"T": now, "S": "Buy" if k % 2 else "Sell", "v": "1", "p": "100", "s": t.split(".")[-1]}]}))
            k += 1
            await asyncio.sleep(0.05)


class TestEndToEnd(unittest.TestCase):
    def test_rows_flow_and_survive_a_reconnect(self):
        import websockets

        async def scenario(root):
            fake = _FakeBybit(drop_after=1.5)
            async with websockets.serve(fake.handler, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                cfg = {"exchange": {"bybit": {"symbols": ["AUSDT", "BUSDT"], "ws_url": "ws://x"}},
                       "storage": {"raw_dir": str(root)},
                       "book": {"interval_sec": 1, "depth": 200, "trades": True, "retention_days": 45}}
                task = asyncio.create_task(BA.run(cfg, ws_url=f"ws://127.0.0.1:{port}"))
                await asyncio.sleep(6.5)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            return fake.connections

        with tempfile.TemporaryDirectory() as td, mock.patch.object(BA, "SAMPLE_LAG_SEC", 0.2):
            connections = asyncio.run(scenario(Path(td)))
            self.assertGreaterEqual(connections, 2, "client did not reconnect")
            for sym in ("AUSDT", "BUSDT"):
                rows = [OB.parse_row(x) for x in next((Path(td) / "book" / sym).glob("*.csv")).read_text().splitlines()]
                rows = [r for r in rows if r]
                self.assertGreaterEqual(len(rows), 4)
                with_book = [r for r in rows if r["mid"] is not None]
                self.assertGreaterEqual(len(with_book), 3)
                self.assertAlmostEqual(with_book[-1]["mid"], 100.0)
                self.assertGreater(sum(r["n_buy"] + r["n_sell"] for r in rows), 10)
                ts = [r["ts"] for r in rows]
                self.assertEqual(ts, sorted(set(ts)), "duplicate or unordered rows")


class TestDisabled(unittest.TestCase):
    def test_disabled_exits_without_connecting(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "rbc", Path(__file__).resolve().parents[1] / "scripts" / "run_book_collector.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = {"book": {"enabled": False}, "storage": {"state_dir": "/nonexistent"}, "run": {}}
        with mock.patch.object(mod.ConfigLoader, "load_yaml", return_value=cfg), \
                mock.patch("amber.pipeline.book_app.run") as run:
            self.assertEqual(mod.main(), 0)
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
