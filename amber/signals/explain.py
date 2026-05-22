from __future__ import annotations

from typing import Any

from amber.common.types import SignalV1


def _impact_items(signal: SignalV1) -> list[tuple[str, float]]:
    items: list[tuple[str, float]] = []
    for entry in signal.explanation.top_feature_impacts:
        if not entry:
            continue
        name, val = next(iter(entry.items()))
        try:
            items.append((str(name), float(val)))
        except (TypeError, ValueError):
            continue
    return items


def _directional_score(rule_trace: list[dict[str, Any]]) -> float | None:
    for row in rule_trace:
        if row.get("rule") == "directional_score":
            try:
                return float(row.get("value"))
            except (TypeError, ValueError):
                return None
    return None


def to_human_explanation(signal: SignalV1, top_n: int = 3) -> str:
    impacts = _impact_items(signal)
    impacts = sorted(impacts, key=lambda x: abs(x[1]), reverse=True)[: max(1, top_n)]
    impacts_str = ", ".join([f"{k}={v:.4f}" for k, v in impacts]) if impacts else "n/a"

    directional = _directional_score(signal.explanation.rule_trace)
    directional_str = "n/a" if directional is None else f"{directional:+.3f}"

    return (
        f"{signal.symbol} | up_cal={signal.prob_up_calibrated:.3f} "
        f"down_cal={signal.prob_down_calibrated:.3f} "
        f"dir={directional_str} | impacts: {impacts_str}"
    )
