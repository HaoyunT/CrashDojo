"""Strict benchmark-manifest validation.

A benchmark may contain unavailable scenes, but it may never silently drop
them.  Every requested scene must be represented by one record and every
excluded record must carry a machine-readable reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


ALLOWED_STATUSES = frozenset({"included", "excluded", "candidate"})


@dataclass(frozen=True)
class SceneRecord:
    """One requested scene or scenario in a benchmark population."""

    identity: str
    status: str
    index: Optional[int] = None
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SceneRecord":
        if not isinstance(value, dict):
            raise ValueError("each manifest record must be an object")
        identity = value.get("identity")
        status = value.get("status")
        if not isinstance(identity, str) or not identity:
            raise ValueError("record.identity must be a non-empty string")
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                f"record {identity!r} has invalid status {status!r}; "
                f"expected one of {sorted(ALLOWED_STATUSES)}"
            )
        reason = value.get("reason")
        if status == "excluded" and (not isinstance(reason, str) or not reason):
            raise ValueError(f"excluded record {identity!r} needs a reason")
        index = value.get("index")
        if index is not None and (not isinstance(index, int) or isinstance(index, bool)):
            raise ValueError(f"record {identity!r} index must be an integer")
        metadata = value.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"record {identity!r} metadata must be an object")
        return cls(identity, status, index, reason, metadata)


@dataclass(frozen=True)
class BenchmarkManifest:
    """Frozen population and inclusion decisions for one benchmark."""

    benchmark_id: str
    domain: str
    dataset: str
    target_count: int
    frozen: bool
    records: Tuple[SceneRecord, ...]
    source: Optional[Dict[str, Any]] = None

    @property
    def included(self) -> Tuple[SceneRecord, ...]:
        return tuple(record for record in self.records if record.status == "included")

    @property
    def excluded(self) -> Tuple[SceneRecord, ...]:
        return tuple(record for record in self.records if record.status == "excluded")

    @property
    def candidates(self) -> Tuple[SceneRecord, ...]:
        return tuple(record for record in self.records if record.status == "candidate")

    @property
    def scene_set_sha256(self) -> str:
        payload = "\n".join(sorted(record.identity for record in self.included))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def summary(self) -> Dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "domain": self.domain,
            "dataset": self.dataset,
            "target_count": self.target_count,
            "requested_count": len(self.records),
            "included_count": len(self.included),
            "excluded_count": len(self.excluded),
            "candidate_count": len(self.candidates),
            "frozen": self.frozen,
            "ready": self.frozen
            and len(self.included) == self.target_count
            and not self.candidates,
            "included_scene_set_sha256": self.scene_set_sha256,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "BenchmarkManifest":
        if value.get("schema") != "crashdojo_benchmark_manifest_v1":
            raise ValueError("unsupported benchmark manifest schema")
        benchmark_id = value.get("benchmark_id")
        domain = value.get("domain")
        dataset = value.get("dataset")
        target_count = value.get("target_count")
        frozen = value.get("frozen")
        if not all(isinstance(item, str) and item for item in (benchmark_id, domain, dataset)):
            raise ValueError("benchmark_id, domain, and dataset must be non-empty strings")
        if not isinstance(target_count, int) or isinstance(target_count, bool) or target_count <= 0:
            raise ValueError("target_count must be a positive integer")
        if not isinstance(frozen, bool):
            raise ValueError("frozen must be boolean")
        raw_records = value.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError("records must be a non-empty list")
        records = tuple(SceneRecord.from_dict(record) for record in raw_records)
        identities = [record.identity for record in records]
        if len(identities) != len(set(identities)):
            raise ValueError("manifest contains duplicate scene identities")
        indices = [record.index for record in records if record.index is not None]
        if len(indices) != len(set(indices)):
            raise ValueError("manifest contains duplicate scene indices")
        included_count = sum(record.status == "included" for record in records)
        candidate_count = sum(record.status == "candidate" for record in records)
        if included_count > target_count:
            raise ValueError("included scene count exceeds target_count")
        if frozen and (included_count != target_count or candidate_count):
            raise ValueError(
                "a frozen manifest must contain exactly target_count included "
                "records and no candidates"
            )
        source = value.get("source")
        if source is not None and not isinstance(source, dict):
            raise ValueError("source must be an object")
        return cls(
            benchmark_id=benchmark_id,
            domain=domain,
            dataset=dataset,
            target_count=target_count,
            frozen=frozen,
            records=records,
            source=source,
        )


def load_manifest(path: Path | str) -> BenchmarkManifest:
    """Load and validate a benchmark manifest from JSON."""

    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return BenchmarkManifest.from_dict(value)


def validate_observed_identities(
    manifest: BenchmarkManifest,
    observed: Iterable[str],
) -> Dict[str, Any]:
    """Compare output identities against the manifest without dropping extras."""

    observed_list = list(observed)
    observed_set = set(observed_list)
    expected_set = {record.identity for record in manifest.included}
    duplicates = sorted({item for item in observed_list if observed_list.count(item) > 1})
    return {
        "expected_count": len(expected_set),
        "observed_count": len(observed_list),
        "observed_unique_count": len(observed_set),
        "missing": sorted(expected_set - observed_set),
        "extra": sorted(observed_set - expected_set),
        "duplicates": duplicates,
        "exact": (
            len(observed_list) == len(expected_set)
            and observed_set == expected_set
            and not duplicates
        ),
    }
