from __future__ import annotations

import logging
from time import time

from amber.common.types import SignalV1
from amber.storage.state_store import StateStore

logger = logging.getLogger(__name__)


def base_rate_for(model: dict, target: str) -> float:
    """Unconditional event rate P(event) the head was trained on.

    Stored per head as `label_rate`. Used to set a base-rate-relative operating
    threshold: a fixed absolute cut (e.g. 0.65) is structurally unreachable for a
    well-calibrated rare-event head, so nothing would ever fire.
    """
    heads = model.get("heads", {})
    head = heads.get(target, {}) if isinstance(heads, dict) else {}
    try:
        return float(head.get("label_rate", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def effective_prob_min(
    thresholds: dict,
    base_rate: float,
    *,
    absolute_key: str,
) -> float:
    """Operating threshold for a calibrated head.

    When `prob_lift_min` is configured, the signal must clear both an absolute
    floor and a lift over the head's unconditional base rate. The lift is applied
    to the **odds**, not the probability: multiplying the probability breaks down
    once the base rate is no longer small, because probability is capped at 1.
    With a 0.32 base rate a 2x probability lift demands 0.64 — near-certainty
    that a calibrated model almost never reaches, so nothing fires. In odds space
    the same 2x asks for 0.485, which stays selective at any base rate and keeps
    its usual reading ("twice the odds"). Falls back to the legacy absolute cut
    under `absolute_key` when no lift is configured. See audit 2026-07 B3/B7.
    """
    lift_min = thresholds.get("prob_lift_min")
    if lift_min is not None:
        abs_floor = float(thresholds.get("prob_abs_floor", 0.0) or 0.0)
        p = min(max(float(base_rate), 0.0), 1.0)
        if p <= 0.0:
            return max(abs_floor, 0.0)
        if p >= 1.0:
            return 1.0
        odds = float(lift_min) * (p / (1.0 - p))
        return max(abs_floor, odds / (1.0 + odds))
    return float(thresholds.get(absolute_key, 0.65))


def directional_score(signal: SignalV1) -> float:
    return signal.prob_up_calibrated - signal.prob_down_calibrated


def spread_bps(signal: SignalV1) -> float:
    bid = float(signal.market_context.get("bid", 0.0))
    ask = float(signal.market_context.get("ask", 0.0))
    mid = float(signal.market_context.get("mid_price", 0.0))
    if bid <= 0 or ask <= 0 or mid <= 0:
        ctx_spread = signal.market_context.get("spread_bps")
        return float(ctx_spread) if ctx_spread is not None else 0.0
    return ((ask - bid) / mid) * 10_000


class SignalGate:
    """One signal per candle, plus a per-symbol cooldown and a concurrency cap.

    A symbol's slot expires after `slot_ttl_sec` (roughly the signal horizon),
    so the concurrency cap limits *active* signals instead of permanently
    blocking every new symbol after the first N. With a `store`, state survives
    across scanner invocations.

    The candle check matters as much as the cooldown. The scanner rescores the
    newest feature row on every pass, so whenever that row has not advanced —
    the scan interval is shorter than the cooldown, or the pipeline has not
    written a new candle yet — the same candle would emit again once the
    cooldown lapsed. That duplicated live signals (the same symbol, the same
    event_ts, the same probabilities, several times over) and, worse, fed the
    same outcome into the confirmed-outcome statistics repeatedly, inflating the
    signal count and distorting rolling AUC.
    """

    def __init__(
        self,
        cooldown_sec: int,
        concurrent_limit: int,
        *,
        slot_ttl_sec: int | None = None,
        store: StateStore | None = None,
        state_key: str = "signal_gate",
    ) -> None:
        self.cooldown_sec = cooldown_sec
        self.concurrent_limit = concurrent_limit
        self.slot_ttl_sec = slot_ttl_sec if slot_ttl_sec is not None else max(cooldown_sec, 60) * 5
        self.store = store
        self.state_key = state_key
        self.last_emit_ts: dict[str, float] = {}
        self.last_event_ts: dict[str, float] = {}
        if store is not None:
            try:
                stored = store.get(state_key)
                # Older state was a flat {symbol: emit_time} mapping.
                if isinstance(stored.get("emit"), dict) or isinstance(stored.get("event"), dict):
                    self.last_emit_ts = {k: float(v) for k, v in stored.get("emit", {}).items()}
                    self.last_event_ts = {k: float(v) for k, v in stored.get("event", {}).items()}
                else:
                    self.last_emit_ts = {k: float(v) for k, v in stored.items()}
            except Exception:
                logger.warning("could not load signal gate state key=%s; starting fresh", state_key)

    @staticmethod
    def _event_seconds(signal: SignalV1) -> float | None:
        raw = getattr(signal, "event_ts", None)
        if raw is None:
            return None
        try:
            return float(raw.timestamp()) if hasattr(raw, "timestamp") else float(raw)
        except (TypeError, ValueError, OSError):
            return None

    def _persist(self) -> None:
        if self.store is not None:
            self.store.set(self.state_key, {"emit": self.last_emit_ts, "event": self.last_event_ts})

    def allow(self, signal: SignalV1) -> bool:
        now = time()

        # One signal per candle: a rescored but unchanged feature row must not
        # emit again just because the cooldown lapsed.
        event_sec = self._event_seconds(signal)
        if event_sec is not None:
            seen = self.last_event_ts.get(signal.symbol)
            if seen is not None and event_sec <= seen:
                return False

        active = {s: t for s, t in self.last_emit_ts.items() if (now - t) < self.slot_ttl_sec}
        if len(active) >= self.concurrent_limit and signal.symbol not in active:
            return False
        last = self.last_emit_ts.get(signal.symbol)
        if last is not None and (now - last) < self.cooldown_sec:
            return False

        active[signal.symbol] = now
        self.last_emit_ts = active
        if event_sec is not None:
            # Keep only symbols still tracked, so this cannot grow without bound.
            self.last_event_ts = {s: t for s, t in self.last_event_ts.items() if s in active}
            self.last_event_ts[signal.symbol] = event_sec
        self._persist()
        return True


def passes_thresholds(
    signal: SignalV1,
    up_min: float,
    down_min: float,
    directional_min: float,
    spread_max_bps: float,
    move_min: float | None = None,
) -> bool:
    """Gate a signal.

    With `move_min` the decision is "will price travel the barrier at all",
    which is the question the model was shown to answer (roadmap D10). The
    directional filter is deliberately NOT applied in that mode: it demands
    |p_up - p_down| be large, which suppresses exactly the two-sided, volatile
    setups a volatility scanner exists to surface — and it was screening on a
    difference between two heads measured to carry no directional information.

    Without `move_min` the legacy directional gate is kept intact, so models
    trained before the change keep behaving as they did.
    """
    spread_ok = spread_bps(signal) <= spread_max_bps
    if move_min is not None:
        move = signal.prob_move_calibrated
        if move is None:
            move = min(1.0, signal.prob_up_calibrated + signal.prob_down_calibrated)
        return move >= move_min and spread_ok

    prob_ok = signal.prob_up_calibrated >= up_min or signal.prob_down_calibrated >= down_min
    dir_ok = abs(directional_score(signal)) >= directional_min
    return prob_ok and dir_ok and spread_ok
