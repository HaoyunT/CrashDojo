"""Website selection identity contract. No torch/model loading or sampling."""
import hashlib
import json
import math
from pathlib import Path


def load_selection(path, row):
    raw = Path(path).read_bytes()
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if row is None or row < 0 or row >= len(records):
        raise ValueError('selection_row must be an explicit zero-based JSONL record index')
    record = records[row].copy()
    dataset = record.get('dataset')
    namespaces = {'nuscenes': 'nuScenes instance_token', 'nuplan': 'nuPlan track_token (lowercase hex)'}
    if dataset not in namespaces or record.get('vehicle_id_namespace') != namespaces[dataset]:
        raise ValueError('Unsupported dataset or vehicle ID namespace')
    for key in ('scene_id', 'log_id', 'scene_token', 'adversary_vehicle_id'):
        if not isinstance(record.get(key), str) or not record[key]:
            raise ValueError('Missing identity: ' + key)
    if record['adversary_vehicle_id'] == 'ego':
        raise ValueError('Selected adversary cannot be ego')
    t = record.get('start_time_s')
    frame = record.get('start_step_10hz')
    if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
        raise ValueError('Invalid start_time_s')
    if type(frame) is not int or not math.isclose(t * 10, frame, abs_tol=1e-6):
        raise ValueError('start_time_s and start_step_10hz disagree')
    record['_manifest_sha256'] = hashlib.sha256(raw).hexdigest()
    record['_manifest_row'] = row
    return record


def resolve_scene(dataset, record):
    """Resolve only exact native scene names; never reuse export row as dataset index."""
    matches = []
    for index in range(dataset.num_scenes()):
        scene = dataset.get_scene(index)
        if scene.name != record['scene_id']:
            continue
        family = getattr(getattr(scene, 'env_metadata', None), 'name', '')
        expected = 'nusc' if record['dataset'] == 'nuscenes' else 'nuplan'
        if not family.startswith(expected):
            continue
        dt = float(scene.dt)
        if not math.isclose(dt, 0.1, abs_tol=1e-8):
            raise ValueError('Website selection requires the same 10 Hz scene timeline')
        if record['dataset'] == 'nuplan':
            if scene.name != record['log_id'] + '=' + record['scene_token']:
                raise ValueError('nuPlan log/token identity mismatch')
        if record['start_step_10hz'] >= scene.length_timesteps:
            raise ValueError('Selected start is outside the scene')
        matches.append(index)
    if len(matches) != 1:
        raise ValueError('Expected exactly one matching dataset scene, found %d: %s' % (len(matches), record['scene_id']))
    return matches[0]


def resolve_agents(names, record):
    target, ego = record['adversary_vehicle_id'], record.get('ego_vehicle_id', 'ego')
    if len(names) != len(set(names)) or names.count(target) != 1 or names.count(ego) != 1:
        raise ValueError('Selected vehicle/ego missing or ambiguous after scene reset')
    return names.index(target), names.index(ego)


class SelectionBinding:
    """Fail closed on every reset and observation; bind after filtering/reordering."""
    def __init__(self, env, record, scene_index, control_policy, model, action_steps, audit_path, history_num_frames=0):
        self.env, self.record, self.scene_index = env, record, scene_index
        self.control_policy, self.model, self.action_steps = control_policy, model, action_steps
        self.audit_path = Path(audit_path)
        self.history_num_frames = history_num_frames
        self.target_index = self.ego_index = None
        self._last_names = None
        original_reset = env.reset
        original_observation = env.get_observation

        def reset(scene_indices=None, start_frame_index=None):
            if list(scene_indices or []) != [scene_index] or start_frame_index != [record['start_step_10hz']]:
                raise ValueError('Selection reset attempted to change the selected scene/start')
            valid = original_reset(scene_indices=scene_indices, start_frame_index=start_frame_index)
            if not all(valid) or len(valid) != 1:
                raise ValueError('Selected scene is invalid; no silent skip or substitute allowed')
            self.bind()
            return valid

        def observation(*args, **kwargs):
            # Original reset may internally call observation before it returns.
            result = original_observation(*args, **kwargs)
            if self._last_names is not None and list(env.current_agent_names) != self._last_names:
                raise ValueError('Agent order changed before policy observation')
            if self._last_names is not None and self.history_num_frames and not kwargs.get('return_raw', False):
                history = result['agents']['history_availabilities']
                row = history[self.target_index]
                if len(row) < self.history_num_frames + 1 or not bool(row[-(self.history_num_frames + 1):].all()):
                    raise ValueError('Selected vehicle lacks full FM history; preserve selection and stop')
            return result

        env.reset, env.get_observation = reset, observation

    def bind(self):
        names = list(self.env.current_agent_names)
        target, ego = resolve_agents(names, self.record)
        # Changes in active-agent ordering after initial binding cannot safely
        # update all downstream guidance configs. Stop instead of controlling another car.
        if self._last_names is not None and names != self._last_names:
            raise ValueError('Agent order changed after selection binding')
        self.control_policy.set_controllable_set([target])
        self.model.set_guidance_dim([target], self.action_steps)
        self.target_index, self.ego_index = target, ego
        if self._last_names is None:
            audit = dict(self.record, runtime_scene_index=self.scene_index,
                         runtime_target_index=target, runtime_ego_index=ego,
                         runtime_agent_names=names, dt_seconds=0.1,
                         selection_bound=True, closed_loop_validated=False)
            self.audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
            print('WEBSITE_SELECTION_BOUND ' + json.dumps(audit, ensure_ascii=False))
        self._last_names = names
