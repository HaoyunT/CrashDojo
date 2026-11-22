"""Make the original ego-plan broadcast explicit for legacy CUDA indexing."""
import hashlib
import inspect
import textwrap


def install():
    import tbsim.utils.guidance_utils as guidance
    method = guidance.CauseCollisionLossCalculator.calculate_loss
    source = textwrap.dedent(inspect.getsource(method))
    original = 'world_xy_fut_detached[ego_indices] = ego_plan_world[:, None]'
    replacement = ('world_xy_fut_detached[ego_indices] = ego_plan_world[:, None]'
                   '.expand_as(world_xy_fut_detached[ego_indices]).contiguous()')
    if source.count(original) != 1:
        raise ValueError('Unrecognized original SafeSim ego-plan assignment')
    adapted = source.replace(original, replacement)
    namespace = {}
    exec(compile(adapted, '<safesim_explicit_ego_plan_broadcast>', 'exec'), vars(guidance), namespace)
    guidance.CauseCollisionLossCalculator.calculate_loss = namespace['calculate_loss']
    return dict(adapter='explicit_ego_plan_sample_axis_broadcast',
                original_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                patched_source_sha256=hashlib.sha256(adapted.encode()).hexdigest(),
                loss_formula_weights_sampler_and_ranking_unchanged=True,
                same_supplied_ego_plan_repeated_for_each_candidate=True,
                repository_files_modified=False)
