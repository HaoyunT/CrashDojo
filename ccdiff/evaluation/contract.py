"""Parser for the archived development metric contract (not the main table)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class MetricContract:
    contract_id: str
    frozen: bool
    value: Dict[str, Any]

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "MetricContract":
        if value.get("schema") != "crashdojo_metric_contract_v1":
            raise ValueError("unsupported metric contract schema")
        contract_id = value.get("contract_id")
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError("contract_id must be a non-empty string")
        if value.get("frozen") is not True:
            raise ValueError("the evaluation metric contract must be frozen")
        supports = value.get("time_supports")
        if not isinstance(supports, dict):
            raise ValueError("time_supports must be an object")
        if supports.get("executed_raw", {}).get("dt_seconds") != 0.1:
            raise ValueError("executed_raw dt must be 0.1 seconds")
        if supports.get("executed_logged_matched", {}).get("dt_seconds") != 0.5:
            raise ValueError("executed_logged_matched dt must be 0.5 seconds")
        gates = value.get("absolute_vehicle_gates")
        required_gates = {
            "max_speed_mps",
            "max_abs_acc_mps2",
            "max_abs_jerk_mps3",
            "max_abs_yaw_rate_rps",
        }
        if not isinstance(gates, dict) or set(gates) != required_gates:
            raise ValueError("absolute_vehicle_gates has the wrong keys")
        if any(not isinstance(bound, (int, float)) or bound <= 0 for bound in gates.values()):
            raise ValueError("absolute vehicle gate bounds must be positive")
        denominators = value.get("required_denominators")
        if not isinstance(denominators, list) or len(denominators) != len(set(denominators)):
            raise ValueError("required_denominators must be a unique list")
        if value.get("raw_metrics_only") is not True:
            raise ValueError("raw_metrics_only must be true")
        return cls(contract_id=contract_id, frozen=True, value=value)


def load_metric_contract(path: Path | str) -> MetricContract:
    with Path(path).open("r", encoding="utf-8") as handle:
        return MetricContract.from_dict(json.load(handle))
