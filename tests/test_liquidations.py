"""Forced liquidations, from the socket to the model's feature vector.

Every layer is tested, but three tests guard the ways this could fail silently
on the live box rather than loudly in development:

- **The side convention.** Bybit's `S` field names the side of the position
  that was closed, so `Buy` means a LONG was liquidated. Reading it as buying
  pressure inverts every feature built on the split, and nothing downstream
  would flag it. Pinned against the documentation's own example.
- **Buckets persisted by the previous version.** The first normalize run after
  a deploy reads minute buckets written before liquidations existed. They have
  no `liq_*` keys, and naive accumulation would raise KeyError on the first
  liquidation landing in a minute a trade had already opened.
- **A model trained before the change.** It scores feature rows that now carry
  three extra keys; it must keep using its own feature list.
"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from amber.exchange.normalizer import BybitNormalizer
from amber.exchange.schemas import Candle
from amber.features.online import FeatureEngine
from amber.features.spec import FEATURE_SPEC_VERSION
from amber.models.features import MODEL_FEATURES
from amber.pipeline.normalize_app import TRADE_BUCKETS_STATE_KEY, normalize_ws_raw
from amber.storage.state_store import StateStore

REPO = Path(__file__).resolve().parents[1]

# Verbatim from https://bybit-exchange.github.io/docs/v5/websocket/public/all-liquidation
DOC_EXAMPLE = {
    "topic": "allLiquidation.ROSEUSDT",
    "type": "snapshot",
    "ts": 1739502303204,
    "data": [{"T": 1739502302929, "s": "ROSEUSDT", "S": "Sell", "v": "20000", "p": "0.04499"}],
}


def _liq(symbol: str, ts: int, side: str, size: float, price: float) -> dict:
    return {
        "topic": f"allLiquidation.{symbol}",
        "type": "snapshot",
        "ts": ts,
        "data": [{"T": ts, "s": symbol, "S": side, "v": str(size), "p": str(price)}],
    }


def _kline(symbol: str, start: int, close: float = 100.0, volume: float = 50.0) -> dict:
    return {
        "topic": f"kline.1.{symbol}",
        "type": "snapshot",
        "ts": start + 30_000,
        "data": [{
            "start": start, "end": start + 60_000, "interval": "1",
            "open": str(close), "high": str(close * 1.001), "low": str(close * 0.999),
            "close": str(close), "volume": str(volume), "confirm": True,
            "timestamp": start + 30_000,
        }],
    }


def _trade(symbol: str, ts: int, side: str = "Buy", size: float = 1.0) -> dict:
    return {
        "topic": f"publicTrade.{symbol}",
        "type": "snapshot",
        "ts": ts,
        "data": [{"T": ts, "s": symbol, "S": side, "v": str(size), "p": "100"}],
    }


class TestParser(unittest.TestCase):
    def test_the_documentation_example_parses(self):
        out = BybitNormalizer.liquidations_from_ws(DOC_EXAMPLE)
        self.assertEqual(len(out), 1)
        symbol, ts, liquidated, usd = out[0]
        self.assertEqual((symbol, ts), ("ROSEUSDT", 1739502302929))
        self.assertAlmostEqual(usd, 20000 * 0.04499)

    def test_sell_means_a_short_was_liquidated(self):
        """Bybit docs: "Sell indicates a short position liquidation"."""
        self.assertEqual(BybitNormalizer.liquidations_from_ws(DOC_EXAMPLE)[0][2], "short")

    def test_buy_means_a_long_was_liquidated(self):
        """Bybit docs: "a Buy update indicates a long position liquidation".

        The counter-intuitive half. Taken at face value `Buy` reads as buying
        pressure, and every feature on the split would come out inverted.
        """
        out = BybitNormalizer.liquidations_from_ws(_liq("BTCUSDT", 1, "Buy", 1.0, 100.0))
        self.assertEqual(out[0][2], "long")

    def test_an_unknown_side_is_dropped_not_guessed(self):
        self.assertEqual(BybitNormalizer.liquidations_from_ws(_liq("BTCUSDT", 1, "Both", 1.0, 100.0)), [])

    def test_other_topics_are_ignored(self):
        self.assertEqual(BybitNormalizer.liquidations_from_ws(_trade("BTCUSDT", 1)), [])
        self.assertEqual(BybitNormalizer.liquidations_from_ws(_kline("BTCUSDT", 0)), [])

    def test_malformed_and_empty_items_are_skipped(self):
        payload = {
            "topic": "allLiquidation.BTCUSDT",
            "data": [
                {"T": 1, "s": "BTCUSDT", "S": "Buy", "v": "abc", "p": "100"},
                {"T": 1, "s": "BTCUSDT", "S": "Buy", "p": "100"},
                {"T": 1, "s": "BTCUSDT", "S": "Buy", "v": "0", "p": "100"},
                "not a dict",
                {"T": 2, "s": "BTCUSDT", "S": "Sell", "v": "2", "p": "50"},
            ],
        }
        out = BybitNormalizer.liquidations_from_ws(payload)
        self.assertEqual(out, [("BTCUSDT", 2, "short", 100.0)])


class TestNormalizedRow(unittest.TestCase):
    def _candle(self) -> Candle:
        return Candle(ts=0, symbol="BTCUSDT", tf="1m", open=1, high=1, low=1, close=1, volume=1)

    def test_liquidations_reach_the_row(self):
        row = BybitNormalizer().to_normalized(
            self._candle(),
            trades={"buy": 1, "sell": 2, "count": 3, "liq_long": 500.0, "liq_short": 250.0, "liq_count": 4},
        )
        self.assertEqual((row.liq_long_usd, row.liq_short_usd, row.liq_count), (500.0, 250.0, 4))

    def test_a_bucket_from_the_previous_version_yields_zeros(self):
        row = BybitNormalizer().to_normalized(self._candle(), trades={"buy": 1, "sell": 2, "count": 3})
        self.assertEqual((row.liq_long_usd, row.liq_short_usd, row.liq_count), (0.0, 0.0, 0))

    def test_old_rows_still_validate(self):
        """Normalized files written before this change have no liq fields."""
        from amber.exchange.schemas import NormalizedRow

        row = NormalizedRow(ts=0, symbol="X", tf="1m", open=1, high=1, low=1, close=1,
                            volume=1, bid=1, ask=1, oi=0, funding=0)
        self.assertEqual((row.liq_long_usd, row.liq_short_usd, row.liq_count), (0.0, 0.0, 0))


class TestNormalizeStage(unittest.TestCase):
    MINUTE = 1_700_000_040_000  # an exact minute boundary

    def _write(self, raw: Path, symbol: str, payloads: list[dict]) -> None:
        d = raw / "ws_raw" / symbol
        d.mkdir(parents=True, exist_ok=True)
        with (d / "part-000.jsonl").open("a", encoding="utf-8") as fh:
            for p in payloads:
                fh.write(json.dumps(p) + "\n")

    def _rows(self, raw: Path, symbol: str) -> list[dict]:
        path = raw / "normalized" / symbol / "part-000.jsonl"
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []

    def test_liquidations_land_on_their_minutes_candle(self):
        with tempfile.TemporaryDirectory() as td:
            raw, state = Path(td) / "raw", StateStore(Path(td) / "state")
            m = self.MINUTE
            self._write(raw, "BTCUSDT", [
                _liq("BTCUSDT", m + 5_000, "Buy", 2.0, 100.0),    # long, 200 USD
                _liq("BTCUSDT", m + 9_000, "Sell", 1.0, 100.0),   # short, 100 USD
                _liq("BTCUSDT", m + 70_000, "Sell", 3.0, 100.0),  # next minute
                _kline("BTCUSDT", m),
                _kline("BTCUSDT", m + 60_000),
            ])
            normalize_ws_raw(raw, state)

            rows = {r["ts"]: r for r in self._rows(raw, "BTCUSDT")}
            self.assertEqual((rows[m]["liq_long_usd"], rows[m]["liq_short_usd"], rows[m]["liq_count"]),
                             (200.0, 100.0, 2))
            self.assertEqual(rows[m + 60_000]["liq_short_usd"], 300.0)
            self.assertEqual(rows[m + 60_000]["liq_long_usd"], 0.0)

    def test_a_bucket_persisted_by_the_previous_version_does_not_crash(self):
        """The first run after deploy reads buckets that have no liq_* keys."""
        with tempfile.TemporaryDirectory() as td:
            raw, state = Path(td) / "raw", StateStore(Path(td) / "state")
            m = self.MINUTE
            # exactly what the old code wrote: buy/sell/count only
            state.set(TRADE_BUCKETS_STATE_KEY, {f"BTCUSDT|{m}": {"buy": 5.0, "sell": 1.0, "count": 6.0}})
            self._write(raw, "BTCUSDT", [
                _liq("BTCUSDT", m + 5_000, "Sell", 1.0, 100.0),
                _kline("BTCUSDT", m),
            ])
            normalize_ws_raw(raw, state)  # must not raise KeyError

            row = self._rows(raw, "BTCUSDT")[0]
            self.assertEqual(row["buy_volume"], 5.0)  # the old bucket's contents survive
            self.assertEqual(row["liq_short_usd"], 100.0)

    def test_liquidations_and_trades_share_one_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            raw, state = Path(td) / "raw", StateStore(Path(td) / "state")
            m = self.MINUTE
            self._write(raw, "BTCUSDT", [
                _liq("BTCUSDT", m + 1_000, "Buy", 1.0, 100.0),
                _trade("BTCUSDT", m + 2_000, "Buy", 3.0),
                _kline("BTCUSDT", m),
            ])
            normalize_ws_raw(raw, state)

            row = self._rows(raw, "BTCUSDT")[0]
            self.assertEqual((row["buy_volume"], row["trade_count"]), (3.0, 1))
            self.assertEqual((row["liq_long_usd"], row["liq_count"]), (100.0, 1))


class TestFeatures(unittest.TestCase):
    def _feed(self, rows: list[dict]) -> dict:
        engine = FeatureEngine()
        out = {}
        for i, extra in enumerate(rows):
            out = engine.update({
                "ts": i * 60_000, "symbol": "BTCUSDT",
                "open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 10.0,
                **extra,
            })
        return out

    def test_a_quiet_market_is_all_zeros(self):
        feats = self._feed([{}] * 30)
        self.assertEqual((feats["liq_share_5"], feats["liq_count_5"], feats["liq_imbalance_15"]),
                         (0.0, 0.0, 0.0))

    def test_rows_from_before_the_stream_existed_do_not_crash(self):
        feats = self._feed([{"liq_long_usd": None, "liq_count": None}] * 10)
        self.assertEqual(feats["liq_share_5"], 0.0)

    def test_share_is_liquidated_usd_over_traded_usd(self):
        """5 bars x (close 100 x volume 10) = 5,000 USD traded."""
        rows = [{}] * 20 + [{}, {}, {}, {"liq_long_usd": 300.0, "liq_count": 1},
                            {"liq_short_usd": 200.0, "liq_count": 2}]
        feats = self._feed(rows)
        self.assertAlmostEqual(feats["liq_share_5"], 500.0 / 5000.0)
        self.assertEqual(feats["liq_count_5"], 3.0)

    def test_liquidations_leave_the_window_after_five_bars(self):
        rows = [{"liq_long_usd": 999.0, "liq_count": 1}] + [{}] * 5
        self.assertEqual(self._feed(rows)["liq_share_5"], 0.0)

    def test_imbalance_sign_follows_which_side_was_forced_out(self):
        shorts = self._feed([{}] * 20 + [{"liq_short_usd": 100.0}])
        longs = self._feed([{}] * 20 + [{"liq_long_usd": 100.0}])
        both = self._feed([{}] * 20 + [{"liq_short_usd": 100.0, "liq_long_usd": 100.0}])

        self.assertEqual(shorts["liq_imbalance_15"], 1.0)   # squeeze up
        self.assertEqual(longs["liq_imbalance_15"], -1.0)   # flush down
        self.assertEqual(both["liq_imbalance_15"], 0.0)

    def test_every_model_feature_is_produced(self):
        feats = self._feed([{}] * 5)
        missing = [f for f in MODEL_FEATURES if f not in feats]
        self.assertEqual(missing, [], "the engine does not emit every model feature")


class TestCollectorTopics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("run_ws_collector", REPO / "scripts" / "run_ws_collector.py")
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def test_liquidations_are_subscribed_by_default(self):
        topics = self.mod.build_topics(["BTCUSDT"], {})
        self.assertIn("allLiquidation.BTCUSDT", topics)

    def test_the_flag_turns_them_off(self):
        topics = self.mod.build_topics(["BTCUSDT"], {"collect_liquidations": False})
        self.assertNotIn("allLiquidation.BTCUSDT", topics)
        self.assertIn("kline.1.BTCUSDT", topics)  # nothing else disappears

    def test_existing_streams_are_untouched(self):
        topics = self.mod.build_topics(["BTCUSDT"], {})
        for t in ("kline.1.BTCUSDT", "tickers.BTCUSDT", "publicTrade.BTCUSDT"):
            self.assertIn(t, topics)

    def test_the_live_universe_fits_bybits_args_limit(self):
        """Exceeding it would take the whole collector down, not just this feed."""
        from amber.common.config import ConfigLoader

        cfg = ConfigLoader(REPO).load_yaml("config/amber.yaml")["exchange"]["bybit"]
        topics = self.mod.build_topics(cfg["symbols"], cfg)
        self.assertLess(len(json.dumps(topics)), self.mod.WS_ARGS_CHAR_LIMIT)


class TestSpecConsistency(unittest.TestCase):
    def test_code_and_config_agree_on_the_spec_version(self):
        """A bump in one place only would skip the recompute the new columns need."""
        import yaml

        cfg = yaml.safe_load((REPO / "config" / "features.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["version"], FEATURE_SPEC_VERSION)

    def test_config_lists_exactly_the_model_features(self):
        import yaml

        cfg = yaml.safe_load((REPO / "config" / "features.yaml").read_text(encoding="utf-8"))
        listed = [f["name"] for f in cfg["features"]]
        self.assertEqual(sorted(listed), sorted(MODEL_FEATURES))


class TestModelTrainedBeforeTheChange(unittest.TestCase):
    def test_it_keeps_scoring_on_its_own_feature_list(self):
        from amber.models.infer import infer_row_prob

        old_features = [f for f in MODEL_FEATURES if not f.startswith("liq_")]
        model = {
            "model_type": "logreg_dual_v1",
            "features": old_features,
            "heads": {"move": {"type": "logreg", "weights": {"range_atr_14": 5.0},
                               "bias": -1.0, "label_rate": 0.1}},
        }
        row = {f: 0.1 for f in MODEL_FEATURES}  # a new-style row with liq_* keys
        p = infer_row_prob(model, row, target="move")
        self.assertTrue(0.0 < p < 1.0)


if __name__ == "__main__":
    unittest.main()
