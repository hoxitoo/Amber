from __future__ import annotations

from amber.common.types import SignalV1
from amber.signals.explain import to_human_explanation


def print_alert(signal: SignalV1) -> None:
    print(f"[ALERT] {to_human_explanation(signal)}")
