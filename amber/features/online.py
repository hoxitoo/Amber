from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import sqrt
from statistics import mean
from typing import Any

_MAXLEN = 240  # ~4h of 1m bars: enough for 60-bar returns + rolling stats


@dataclass(slots=True)
class SymbolWindow:
    close: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    high: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    low: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    volume: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    oi: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    funding: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    bid: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    ask: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    buy_vol: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    sell_vol: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))
    trade_count: deque[float] = field(default_factory=lambda: deque(maxlen=_MAXLEN))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sqrt(mean([(x - m) ** 2 for x in values]))


class FeatureEngine:
    """Unified feature logic for offline/live usage.

    - `update(row)` for streaming state updates
    - `latest(symbol)` to retrieve current feature vector
    - `transform_rows(rows)` for offline recomputation using same formulas

    Beyond basic returns/z-scores this computes breakout-precursor features
    (volume surge/acceleration, volatility compression, distance to N-bar
    highs/lows and breakout flags, OI rate-of-change) that target the intraday
    breakout/momentum-ignition patterns the scanner is meant to catch.
    """

    def __init__(self) -> None:
        self.state: dict[str, SymbolWindow] = defaultdict(SymbolWindow)

    def update(self, row: dict[str, Any]) -> dict[str, Any]:
        symbol = str(row["symbol"])
        w = self.state[symbol]

        close = float(row["close"])
        high = float(row.get("high", close))
        low = float(row.get("low", close))
        volume = float(row.get("volume", 0.0))
        oi = float(row.get("oi", 0.0))
        funding = float(row.get("funding", 0.0))
        bid = float(row.get("bid", close))
        ask = float(row.get("ask", close))

        w.close.append(close)
        w.high.append(high)
        w.low.append(low)
        w.volume.append(volume)
        w.oi.append(oi)
        w.funding.append(funding)
        w.bid.append(bid)
        w.ask.append(ask)
        w.buy_vol.append(float(row.get("buy_volume", 0.0)))
        w.sell_vol.append(float(row.get("sell_volume", 0.0)))
        w.trade_count.append(float(row.get("trade_count", 0.0)))

        return self._build_features(
            symbol=symbol,
            ts=row["ts"],
            is_synthetic=bool(row.get("is_synthetic", False)),
        )

    def latest(self, symbol: str) -> dict[str, Any] | None:
        if symbol not in self.state or not self.state[symbol].close:
            return None
        return self._build_features(symbol=symbol, ts=0)

    def transform_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(self.update(row))
        return out

    def _build_features(self, symbol: str, ts: int, is_synthetic: bool = False) -> dict[str, Any]:
        w = self.state[symbol]
        closes = list(w.close)
        highs = list(w.high)
        lows = list(w.low)
        vols = list(w.volume)
        ois = list(w.oi)
        n = len(closes)
        cur = closes[-1] if closes else 0.0

        def _ret(k: int) -> float:
            if n <= k or closes[-(k + 1)] == 0:
                return 0.0
            return (cur / closes[-(k + 1)]) - 1.0

        def _z(values: deque[float], lookback: int) -> float:
            vals = list(values)
            if len(vals) < max(3, lookback):
                return 0.0
            sample = vals[-lookback:]
            m = mean(sample)
            std = _std(sample)
            return 0.0 if std == 0 else (sample[-1] - m) / std

        # --- returns (short to long context) --------------------------------
        ret_1, ret_5, ret_20, ret_60 = _ret(1), _ret(5), _ret(20), _ret(60)

        # --- volume surge & acceleration ------------------------------------
        vol_mean_20 = mean(vols[-20:]) if vols else 0.0
        vol_cur = vols[-1] if vols else 0.0
        vol_short = mean(vols[-3:]) if vols else 0.0
        vol_ratio_20 = (vol_cur / vol_mean_20) if vol_mean_20 > 0 else 0.0
        vol_accel = (vol_short / vol_mean_20) if vol_mean_20 > 0 else 0.0

        # --- volatility compression (squeeze) then expansion ----------------
        rets = [(closes[i] / closes[i - 1] - 1.0) if closes[i - 1] else 0.0 for i in range(1, n)]
        std_short = _std(rets[-5:]) if len(rets) >= 5 else 0.0
        std_long = _std(rets[-20:]) if len(rets) >= 20 else 0.0
        squeeze_ratio = (std_short / std_long) if std_long > 0 else 0.0
        close_20 = closes[-20:]
        bb_width_20 = (_std(close_20) / mean(close_20)) if len(close_20) >= 2 and mean(close_20) > 0 else 0.0
        trs = [highs[i] - lows[i] for i in range(max(0, n - 14), n)]
        range_atr_14 = (mean(trs) / cur) if trs and cur > 0 else 0.0

        # --- distance to N-bar high/low + breakout flags --------------------
        hi_window = highs[-20:]
        lo_window = lows[-20:]
        max_hi = max(hi_window) if hi_window else cur
        min_lo = min(lo_window) if lo_window else cur
        dist_to_high_20 = (cur / max_hi - 1.0) if max_hi > 0 else 0.0
        dist_to_low_20 = (cur / min_lo - 1.0) if min_lo > 0 else 0.0
        prior_hi = highs[-21:-1]
        prior_lo = lows[-21:-1]
        breakout_up_20 = 1.0 if (prior_hi and cur > max(prior_hi)) else 0.0
        breakout_dn_20 = 1.0 if (prior_lo and cur < min(prior_lo)) else 0.0

        # --- open interest positioning --------------------------------------
        oi_roc_5 = (ois[-1] / ois[-6] - 1.0) if len(ois) >= 6 and ois[-6] != 0 else 0.0

        # --- taker order flow (aggressor imbalance / CVD) -------------------
        buys = list(w.buy_vol)
        sells = list(w.sell_vol)
        buy_cur = buys[-1] if buys else 0.0
        sell_cur = sells[-1] if sells else 0.0
        flow = buy_cur + sell_cur
        taker_imbalance = ((buy_cur - sell_cur) / flow) if flow > 0 else 0.0
        # net taker flow over the last 20 bars, normalized by total flow
        net20 = sum(buys[-20:]) - sum(sells[-20:])
        tot20 = sum(buys[-20:]) + sum(sells[-20:])
        cvd_norm_20 = (net20 / tot20) if tot20 > 0 else 0.0
        trade_count_z_20 = _z(w.trade_count, 20)

        # --- microstructure --------------------------------------------------
        mid_price = (w.bid[-1] + w.ask[-1]) / 2.0 if w.bid and w.ask else cur
        spread_bps = 0.0 if mid_price == 0 else ((w.ask[-1] - w.bid[-1]) / mid_price) * 10_000

        return {
            "symbol": symbol,
            "ts": ts,
            # returns
            "ret_1": ret_1,
            "ret_5": ret_5,
            "ret_20": ret_20,
            "ret_60": ret_60,
            # volume
            "vol_z_20": _z(w.volume, 20),
            "vol_ratio_20": vol_ratio_20,
            "vol_accel": vol_accel,
            # open interest / funding
            "oi_z_20": _z(w.oi, 20),
            "oi_roc_5": oi_roc_5,
            "funding_z_20": _z(w.funding, 20),
            # volatility compression / expansion
            "squeeze_ratio": squeeze_ratio,
            "bb_width_20": bb_width_20,
            "range_atr_14": range_atr_14,
            # breakout geometry
            "dist_to_high_20": dist_to_high_20,
            "dist_to_low_20": dist_to_low_20,
            "breakout_up_20": breakout_up_20,
            "breakout_dn_20": breakout_dn_20,
            # taker order flow
            "taker_imbalance": taker_imbalance,
            "cvd_norm_20": cvd_norm_20,
            "trade_count_z_20": trade_count_z_20,
            # microstructure / meta
            "mid_price": mid_price,
            "bid": w.bid[-1] if w.bid else mid_price,
            "ask": w.ask[-1] if w.ask else mid_price,
            "spread_bps": spread_bps,
            "notional_volume_1m": (cur * vol_cur) if closes and vols else 0.0,
            "is_synthetic": is_synthetic,
            "obs": n,
        }
