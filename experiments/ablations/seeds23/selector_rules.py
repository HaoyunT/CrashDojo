"""Evidence-independent request selection for the paper's actor ablations."""
import json

SELECTOR_REVISION = 'paper_actor_selection_v2'
GEOMETRY_KEYS = (
    'family', 'parameters', 'route_world_xy', 'corridor_world_polygons',
    'interaction_window_s', 'route_offset_bounds_m', 'geometry_evidence',
    'route_progress_guidance', 'center_receiving_route',
    'boundary_relative_template', 'phase_specific_intrusion',
)


def candidate_identity_key(candidate):
    """Use template identity only; never read Eq. (2) judgements or scores.

    The revised supplement specifies this within-actor tie rule. Canonical family
    and parameter identity gives an input-order-independent rule without reusing
    failure counts, crash support, conflict quality, strength, gap, or effort.
    """
    request = candidate['request']
    geometry = {k: request.get(k) for k in GEOMETRY_KEYS
                if k not in ('family', 'parameters', 'geometry_evidence')}
    return (candidate['family'],
            json.dumps(request['parameters'], sort_keys=True, separators=(',', ':'), allow_nan=False),
            json.dumps(geometry, sort_keys=True, separators=(',', ':'), allow_nan=False))


def canonical_candidates(pool):
    if not pool:
        raise ValueError('Actor selection requires an eligible candidate pool')
    actors = sorted({candidate['actor'] for candidate in pool})
    return {actor: min((c for c in pool if c['actor'] == actor),
                       key=candidate_identity_key) for actor in actors}


def matches_full_request(candidate, request):
    # Same actor alone must not silently restore Full's ranked request.
    return (candidate['actor'] == request['adversary_vehicle_id']
            and all(candidate['request'].get(k) == request.get(k)
                    for k in GEOMETRY_KEYS))


def selection_seeds(protocol):
    seeds = {job['stream'] for job in protocol.get('jobs', [])
             if job['variant'] in ('A1', 'A2', 'A3')}
    if not seeds:
        seeds = set(protocol.get('seeds', []))
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError('The run manifest must declare integer selection seeds')
    return sorted(seeds)


def validate_selection_revision(selection, arm):
    if arm in ('A1', 'A2') and selection.get('selector_revision') != SELECTOR_REVISION:
        raise ValueError('Regenerate A1/A2 selections with prepare_selectors.py; '
                         'the cached selection uses an older ordering rule')
