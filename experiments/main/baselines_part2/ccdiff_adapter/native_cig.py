"""Run the released TTC/CIG ranker on the admitted initial actor population."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_initial_history(observation, names, scene, repo, directory):
    from tbsim.utils.agent_centric_transform import get_neighbor_history_relative_states
    import tbsim.utils.agent_centric_transform as transform

    keys = ('history_positions', 'history_yaws', 'history_speeds', 'extent',
            'history_availabilities', 'agent_from_world', 'world_from_agent', 'yaw')
    batch = {key: observation[key].detach().unsqueeze(0) for key in keys}
    with torch.no_grad():
        (_, info), _ = get_neighbor_history_relative_states(batch)
    ttc = np.asarray(info['matrix_ttc'])
    if ttc.ndim != 3 or ttc.shape[:2] != (len(names), len(names)):
        raise ValueError('Released TTC feature shape does not match admitted actor order')
    path = Path(repo) / 'ccdiff/examples/causal_ranker.py'
    spec = importlib.util.spec_from_file_location('admitted_native_cig_ranker', path)
    ranker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ranker)
    ranker.idx, ranker.final_idx = scene, {}
    ranker.causal_ranking(ttc.transpose(2, 0, 1), topK=10)
    ranking = list(map(int, ranker.final_idx[scene]))
    if len(set(ranking)) != len(ranking) or any(i < 0 or i >= len(names) for i in ranking):
        raise ValueError('Released ranker produced invalid actor indices')
    directory = Path(directory)
    inputs = {key: value.detach().cpu().numpy()[0] for key, value in batch.items()}
    np.savez_compressed(directory / 'native_cig_initial_inputs.npz', matrix_ttc=ttc, **inputs)
    report = dict(scene=scene, names=list(names), ranking=ranking,
                  ranked_actor_ids=[names[i] for i in ranking],
                  selection_source='released_cig_recomputed_on_admitted_initial_history',
                  ranker_sha256=sha(path), ttc_transform_sha256=sha(transform.__file__),
                  input_sha256=sha(directory / 'native_cig_initial_inputs.npz'),
                  history_frames=int(ttc.shape[-1]), recorded_future_used=False,
                  reused_ours_target=False)
    (directory / 'native_cig_initial_ranking.json').write_text(json.dumps(report, indent=2))
    return ranking, report
