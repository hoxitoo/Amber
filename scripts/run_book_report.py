"""Is the order book collector recording what D2 will need? Read-only.

    python scripts/run_book_report.py            # last 24 hours
    python scripts/run_book_report.py --hours 2

Per symbol: rows against the expected count, share of rows with a usable
book, how fresh the book was, spread, how deep the received book reaches,
taker trades per hour; and the disk the collector uses.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import gzip
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amber.common.config import ConfigLoader, enter_project_root  # noqa: E402


def read_rows(root: Path, symbol: str, since_ms: int) -> list[dict]:
    from amber.exchange.orderbook import parse_row

    rows = []
    today = datetime.now(timezone.utc)
    for d in sorted({(today - timedelta(days=k)).strftime("%Y-%m-%d") for k in range(0, 3)}):
        for path in (root / symbol / f"{d}.csv", root / symbol / f"{d}.csv.gz"):
            if not path.exists():
                continue
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    r = parse_row(line)
                    if r is not None and r["ts"] >= since_ms:
                        rows.append(r)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    args = ap.parse_args()
    root_dir = enter_project_root(__file__)
    cfg = ConfigLoader(root_dir).load_yaml("config/amber.yaml")
    book = cfg.get("book", {}) if isinstance(cfg.get("book"), dict) else {}
    interval = int(book.get("interval_sec", 5))
    root = Path(cfg["storage"]["raw_dir"]) / "book"
    since = int((time.time() - args.hours * 3600) * 1000)
    expected = int(args.hours * 3600 / interval)

    def med(v):
        v = [x for x in v if x is not None]
        return statistics.median(v) if v else None

    def f(v, fmt):
        return "-" if v is None else format(v, fmt)

    print(f"Стакан за последние {args.hours:g} ч (ожидается ~{expected} строк на монету, каждые {interval} с)")
    print(f"{'монета':<14}{'строк':>7}{'со стаканом':>12}{'свежесть':>10}{'спред bps':>10}"
          f"{'глубина bps':>13}{'сделок/ч':>10}")
    bad = []
    by_symbol = {s: read_rows(root, s, since) for s in cfg["exchange"]["bybit"]["symbols"]}
    first = min((r["ts"] for rows in by_symbol.values() for r in rows), default=None)
    # Rows sit on a 5 s grid, so the first one in a full window is always a few
    # seconds after `since`; only a start later than that is a short collection.
    if first is not None and first > since + 2 * interval * 1000:
        # Collected for less than the asked window: judge against what it could have written.
        expected = int((time.time() * 1000 - first) / 1000 / interval)
        print(f"(сбор идёт с {datetime.fromtimestamp(first / 1000, tz=timezone.utc):%m-%d %H:%M} UTC, "
              f"ожидается ~{expected} строк)")
    for symbol, rows in by_symbol.items():
        with_book = [r for r in rows if r["mid"] is not None]
        share = len(with_book) / len(rows) if rows else 0.0
        age = med([r["book_age_ms"] for r in with_book])
        cov = med([min(r["bid_cov_bps"], r["ask_cov_bps"]) for r in with_book])
        span_h = (len(rows) * interval / 3600) or 1e-9
        trades = sum((r["n_buy"] or 0) + (r["n_sell"] or 0) for r in rows) / span_h
        print(f"{symbol:<14}{len(rows):>7}{share * 100:>11.0f}%{f(age, '.0f') + ' мс':>10}"
              f"{f(med([r['spread_bps'] for r in with_book]), '.2f'):>10}{f(cov, '.0f'):>13}{trades:>10.0f}")
        if len(rows) < 0.9 * expected or share < 0.95:
            bad.append(symbol)
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    print(f"\nдиск: {size / 1e6:.0f} MB в {root}")
    if not root.exists():
        print("ИТОГ: данных нет — служба amber-book-collector не запущена или ещё не записала первую строку")
    elif bad:
        print("ИТОГ: пробелы (меньше 90% строк или меньше 95% со стаканом): " + ", ".join(bad))
    else:
        print("ИТОГ: ok — все монеты пишутся полностью")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
