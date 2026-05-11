from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class RunContext(BaseModel):
    run_id: str
    started_at: datetime
    config_version: str
    feature_spec_version: str


class SignalExplanation(BaseModel):
    top_feature_impacts: list[dict[str, float]] = Field(default_factory=list)
    rule_trace: list[dict[str, Any]] = Field(default_factory=list)


class SignalV1(BaseModel):
    schema_version: Literal["v1"] = "v1"
    signal_id: str
    event_ts: datetime
    symbol: str
    horizon_min: int
    target_up_pct: float
    target_down_pct: float
    prob_up_raw: float
    prob_down_raw: float
    prob_up_calibrated: float
    prob_down_calibrated: float
    regime: str
    market_context: dict[str, Any] = Field(default_factory=dict)
    explanation: SignalExplanation
    model_version: str
    config_version: str
