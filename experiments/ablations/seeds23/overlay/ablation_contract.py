"""Paper A3: generic contact with common controls retained."""


def generic_contact_only(request):
    return (request.get('ablation_v26', {}).get('arm') == 'A3'
            or request.get('ablation_generic_collision', False))


def generic_contact_request(request):
    """Return a normalized copy without changing the caller's frozen input.

    These settings were already applied by the formal run.py entrypoint.
    Keeping the contract here also covers direct guide/loss entrypoints.
    """
    if not generic_contact_only(request):
        return request
    return dict(request, weights=dict(request['weights'], action=0.),
                ablation_generic_collision=True, contact_phase_window=False,
                qualified_contact_v2=None, causal_parameter_search_v9=False,
                interaction_timing_v11=False, execution_completion_v10=False)
