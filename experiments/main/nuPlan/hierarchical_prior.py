"""Fit-log hierarchical context prior used by causal-guidance V26.

The prior describes observed motion intensity. It is not a collision
probability. Sparse cells shrink toward their family/role parent and retain
their support metadata so unsupported scene-road combinations stay auditable.
"""
import math
from collections import defaultdict


MIN_EVENTS = 10
MIN_SCENES = 5
MIN_LOGS = 3
STRICT_EVENTS = 20
STRICT_SCENES = 10
STRICT_LOGS = 3
SHRINK_KAPPA = 30.0
FACTOR_MIN = 0.75
FACTOR_MAX = 1.35

LEVELS = ("joint", "semantic_road", "semantic", "road", "family_role", "family")


def _number(row, key):
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def supported(row, strict=False):
    events = STRICT_EVENTS if strict else MIN_EVENTS
    scenes = STRICT_SCENES if strict else MIN_SCENES
    logs = STRICT_LOGS if strict else MIN_LOGS
    return (_number(row, "events") >= events and
            _number(row, "scenes") >= scenes and
            _number(row, "logs") >= logs)


def request_role(request):
    evidence = request.get("geometry_evidence", {})
    if "actor_role" in evidence:
        return evidence["actor_role"]
    role = evidence.get("longitudinal_role")
    return {
        "adversary_ahead": "lead",
        "adversary_behind": "behind",
    }.get(role, evidence.get("actor_role", "UNKNOWN"))


def _family_row(rows, dataset, family):
    """Return a pooled family baseline when no role-specific parent exists."""
    pooled = [r for r in rows if r.get('split') == 'fit'
              and r.get('dataset') == dataset and r.get('family') == family
              and r.get('level') == 'family_role' and r.get('actor_role') == 'ALL'
              and supported(r)]
    if pooled:
        return dict(max(pooled, key=lambda r: _number(r, 'events')),
                    level='family', support_level='family_pooled')
    candidates = [r for r in rows if r.get("split") == "fit" and
                  r.get("dataset") == dataset and r.get("family") == family and
                  r.get("level") == "family_role" and r.get('actor_role') != 'ALL' and supported(r)]
    if not candidates:
        return None
    weights = [_number(r, "logs") or 1.0 for r in candidates]
    total = sum(weights)
    value = sum(w * _number(r, "log_equal_mean_log_intensity")
                for w, r in zip(weights, candidates)) / total
    return dict(level="family", dataset=dataset, family=family,
                actor_role="ALL", semantic_label="ALL", road_type="ALL",
                traffic_state="ALL", events=sum(_number(r, "events") for r in candidates),
                scenes=sum(_number(r, "scenes") for r in candidates),
                logs=sum(weights), log_equal_mean_log_intensity=value,
                support_level="family_pooled")


def _parent(rows, dataset, family, role):
    for row in rows:
        if (row.get("split") == "fit" and row.get("dataset") == dataset and
                row.get("family") == family and row.get("level") == "family_role" and
                row.get("actor_role") == role and supported(row, strict=True)):
            return row
    return _family_row(rows, dataset, family)


def _matches(row, dataset, family, role, semantic, road, traffic):
    if row.get("split") != "fit" or row.get("dataset") != dataset:
        return False
    if row.get("family") != family or row.get("actor_role") != role:
        return False
    return ((row.get("semantic_label") in ("ALL", semantic)) and
            (row.get("road_type") in ("ALL", road)) and
            (row.get("traffic_state") in ("ALL", traffic)))


def resolve(rows, dataset, request, context):
    """Resolve the most specific supported prior and return an auditable record."""
    family = request.get("family", "NONE")
    if family == "NONE":
        return dict(factor=1.0, reason="no_action", support_level="none",
                    events=0, scenes=0, logs=0, fallback_reason="no_action")
    role = request_role(request)
    semantic = context.get("semantic_label", "UNKNOWN")
    road = context.get("road", {}).get("road_type", "UNKNOWN")
    traffic = context.get("traffic_state", "UNKNOWN")
    parent = _parent(rows, dataset, family, role)
    if parent is None:
        return dict(factor=1.0, reason="unsupported_family", support_level="none",
                    events=0, scenes=0, logs=0, fallback_reason="no_family_fit_support",
                    dataset=dataset, family=family, actor_role=role,
                    semantic_label=semantic, road_type=road, traffic_state=traffic)
    parent_log = _number(parent, "log_equal_mean_log_intensity")
    selected = None
    selected_level = "family"
    # Fine cells use strict support; semantic and road cells can use the
    # relaxed gate because their parent shrinkage keeps them conservative.
    scope = 'exact_role'
    for level in LEVELS[:-2]:
        matches = [r for r in rows if r.get("level") == level and
                   _matches(r, dataset, family, role, semantic, road, traffic)]
        strict = level in ("joint", "semantic_road")
        matches = [r for r in matches if supported(r, strict=strict)]
        if matches:
            selected = max(matches, key=lambda r: (_number(r, "events"), _number(r, "logs")))
            selected_level = level
            break
    # Context can survive role sparsity, using an explicit pooled-role parent.
    # These cells are built from raw fit events, not averaged role summaries.
    pooled_parent = _family_row(rows, dataset, family)
    if selected is None and pooled_parent is not None:
        for level in ('semantic_road', 'semantic', 'road'):
            matches = [r for r in rows if r.get('level') == level
                       and _matches(r, dataset, family, 'ALL', semantic, road, traffic)
                       and supported(r, strict=level == 'semantic_road')]
            if matches:
                selected = max(matches, key=lambda r: (_number(r, 'events'), _number(r, 'logs')))
                selected_level = 'role_pooled_' + level
                scope = 'pooled_roles'
                parent = pooled_parent
                parent_log = _number(parent, 'log_equal_mean_log_intensity')
                break
    if selected is None:
        matches = [r for r in rows if r.get('level') == 'family_role'
                   and _matches(r, dataset, family, role, semantic, road, traffic)
                   and supported(r)]
        if matches:
            selected = max(matches, key=lambda r: (_number(r, 'events'), _number(r, 'logs')))
            selected_level = 'family_role'
    if selected is None:
        selected = parent
        selected_level = parent.get("level", "family")
        reason = "fallback_" + selected_level
    else:
        reason = "supported_" + selected_level
    raw_delta = _number(selected, "log_equal_mean_log_intensity") - parent_log
    n = max(_number(selected, "events"), 1.0)
    shrink = n / (n + SHRINK_KAPPA)
    factor = math.exp(shrink * raw_delta)
    factor = min(FACTOR_MAX, max(FACTOR_MIN, factor))
    return dict(factor=float(factor), reason=reason, support_level=selected_level,
                events=int(_number(selected, "events")), scenes=int(_number(selected, "scenes")),
                logs=int(_number(selected, "logs")), raw_log_delta=float(raw_delta),
                shrink=float(shrink), parent_level=parent.get("level", "family"),
                dataset=dataset, family=family, actor_role=role,
                semantic_label=semantic, road_type=road, traffic_state=traffic,
                support_scope=scope if selected_level != 'family' else 'pooled_family_baseline',
                matched_cell_actor_role=selected.get('actor_role'),
                matched_source_roles=selected.get('source_roles', [selected.get('actor_role')]),
                fallback_reason=None if reason.startswith("supported_") else reason)


def build_table(rows):
    """Return fit-only rows plus a compact support summary for freezing."""
    fit = [r for r in rows if r.get("split") == "fit"]
    summary = defaultdict(lambda: {"cells": 0, "supported": 0})
    for row in fit:
        key = (row.get("dataset"), row.get("family"), row.get("level"))
        summary[key]["cells"] += 1
        summary[key]["supported"] += int(supported(row, strict=row.get("level") in ("joint", "semantic_road")))
    return dict(rows=fit, support_summary={"|".join(k): v for k, v in sorted(summary.items())},
                policy=dict(min_events=MIN_EVENTS, min_scenes=MIN_SCENES, min_logs=MIN_LOGS,
                            strict_events=STRICT_EVENTS, strict_scenes=STRICT_SCENES,
                            strict_logs=STRICT_LOGS, shrink_kappa=SHRINK_KAPPA,
                            factor_min=FACTOR_MIN, factor_max=FACTOR_MAX))
