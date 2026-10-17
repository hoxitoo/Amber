"""Score the breakout bracket on shown ignition warnings resolved before it existed.

The bracket (amber/monitoring/ledger.py `bracket_trade`) was fixed on
2026-10-17. Warnings the ignition ledger resolved before then carry no bracket
result; this re-reads their candles from disk and scores the same rule from
the same entry bar, once. The ledger itself is never rewritten: results go to
data/logs/ignition_bracket_backfill.jsonl, which the summary merges in.

Read-only on everything else; safe to run while the services are up and to
run again (it skips what is already scored).

    python scripts/backfill_ignition_bracket.py
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402


def backfill(logs: Path, raw_root: Path, *, now_ms: int | None = None) -> dict[str, int]:
    from amber.monitoring.ledger import DEFAULT_COST, DEFAULT_LAG_BARS, STEP_MS, resolve_alert
    from amber.monitoring.quality_report import _CandleIndex
    from amber.signals.ignition_live import BRACKET_BACKFILL_FILE, BRACKET_TRIGGER, LEDGER_FILE, _read_backfill

    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    done = _read_backfill(logs)
    todo: dict[str, list[dict]] = defaultdict(list)
    ledger = logs / LEDGER_FILE
    if ledger.exists():
        with ledger.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "ok" or not int(r.get("notify", 0)) or "bracket_net" in r:
                    continue
                if (str(r["symbol"]), int(r["event_ts"])) in done:
                    continue
                todo[str(r["symbol"])].append(r)

    counts = {"scored": 0, "filled": 0, "no_candles": 0, "entry_mismatch": 0, "already": len(done)}
    fresh: list[dict] = []
    # One symbol at a time, so only one coin's candles are in memory.
    for symbol, rows in sorted(todo.items()):
        oldest = min(int(r["event_ts"]) for r in rows)
        index = _CandleIndex(raw_root, max_rows=int((now_ms - oldest) // STEP_MS + 128))
        candles, ts = index.candles(symbol), index.timestamps(symbol)
        for r in rows:
            res = resolve_alert(
                candles, ts, int(r["event_ts"]), horizon=int(r["horizon"]), barrier=float(r["barrier"]),
                lag_bars=int(r.get("lag_bars", DEFAULT_LAG_BARS)), cost=float(r.get("cost", DEFAULT_COST)),
                emitted_ms=r.get("emitted_ms"), bracket=BRACKET_TRIGGER,
            )
            if res is None or res.get("status") != "ok":
                counts["no_candles"] += 1
                continue
            # Same entry bar as the ledger booked, or this is not the same trade.
            booked = r.get("entry")
            if booked is None or abs(float(res["entry"]) - float(booked)) > 1e-9 * max(1.0, abs(float(booked))):
                counts["entry_mismatch"] += 1
                continue
            fresh.append({"symbol": symbol, "event_ts": int(r["event_ts"]), "entry": res["entry"],
                          "bracket_side": res["bracket_side"], "bracket_gross": res["bracket_gross"],
                          "bracket_net": res["bracket_net"],
                          "backfilled_ms": now_ms})
            counts["scored"] += 1
            counts["filled"] += int(res["bracket_net"] is not None)
        del index, candles, ts

    if fresh:
        path = logs / BRACKET_BACKFILL_FILE
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in [*done.values(), *fresh]:
                fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        tmp.replace(path)
    return counts


def main() -> int:
    root = enter_project_root(__file__)
    config = ConfigLoader(root).load_yaml("config/amber.yaml")
    storage = config["storage"]
    from amber.signals.ignition_live import format_bracket, summarize_ignition

    c = backfill(Path(storage["logs_dir"]), Path(storage["raw_dir"]))
    print(f"backfill: scored {c['scored']} shown warnings ({c['filled']} filled), "
          f"no candles {c['no_candles']}, entry mismatch {c['entry_mismatch']}, already done {c['already']}")
    summary = summarize_ignition(Path(storage["logs_dir"]))
    print("\n".join(format_bracket(summary["bracket"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
