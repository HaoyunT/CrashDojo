"""Observed-history view for external planners; generator inputs stay untouched.

The first 31 frames come from the frozen common environment. Thereafter only
the last five *observed* frames from the live legacy simulator are appended.
No target_positions, agent_fut, or recorded future is read here.
"""
import copy
import numpy as np


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def transform_history(history, transforms):
    """Transform position, vectors, and heading without inventing missing states."""
    raw = np.asarray(history, dtype=np.float64)
    tf = np.asarray(transforms, dtype=np.float64)
    if raw.ndim != 3 or raw.shape[-1] != 8 or tf.shape != (len(raw), 3, 3):
        raise ValueError('Expected [N,T,8] and [N,3,3]')
    out = raw.copy()
    valid = np.isfinite(raw).all(-1)
    rotation = tf[:, :2, :2]
    out[..., :2] = np.einsum('nij,ntj->nti', rotation, raw[..., :2]) + tf[:, None, :2, 2]
    for start in (2, 4):
        out[..., start:start+2] = np.einsum('nij,ntj->nti', rotation, raw[..., start:start+2])
    yaw = np.arctan2(raw[..., 6], raw[..., 7]) + np.arctan2(tf[:, 1, 0], tf[:, 0, 0])[:, None]
    out[..., 6], out[..., 7] = np.sin(yaw), np.cos(yaw)
    out[~valid] = np.nan
    return out


def parsed_history(raw):
    """Match the current parser: invalid raw xy=0, other raw fields stay NaN."""
    raw = np.asarray(raw, dtype=np.float32).copy()
    valid = np.isfinite(raw).all(-1)
    positions = raw[..., :2]
    yaws = np.zeros(raw.shape[:-1]+(1,),dtype=np.float32)
    observed_yaw = np.arctan2(raw[..., 6][valid], raw[..., 7][valid])
    yaws[..., 0][valid] = (observed_yaw + np.pi) % (2*np.pi) - np.pi
    speeds = np.linalg.norm(raw[..., 2:4], axis=-1)
    positions[~valid] = 0
    yaws[~valid] = 0
    speeds[~valid] = 0
    return dict(agent_hist=raw, history_positions=positions,
                history_yaws=yaws, history_speeds=speeds,
                history_availabilities=valid, curr_speed=speeds[..., -1])


def observed_source(elements, names, timestamp, frames, env):
    """Read uncollated observed history before float32 matrix inversion.

    Millimetre errors otherwise arise from inverting float32 global transforms
    and then projecting through another actor's frame. Never read future fields.
    """
    if [x.agent_name for x in elements] != list(names):raise ValueError('Raw identity order mismatch')
    histories=[];inverse=[]
    if len(env._current_scenes) != 1:
        raise ValueError('One scene is required for observed cache histories')
    cache = env._current_scenes[0].cache
    for element in elements:
        if element.scene_ts != timestamp:raise ValueError('Raw observation timestamp mismatch')
        h = []
        # A missing current frame cannot express earlier local coordinates.
        # Read only already-observed world states, never recorded futures.
        for t in range(timestamp-frames+1, timestamp+1):
            try:
                state = cache.get_raw_state(element.agent_name, t)
                h.append(np.asarray(state.as_format('x,y,xd,yd,xdd,ydd,s,c'), dtype=np.float64))
            except KeyError:
                h.append(np.full(8, np.nan))
        histories.append(h)
        inverse.append(np.asarray(element.agent_from_world_tf,dtype=np.float64))
    inverse=np.stack(inverse);forward=np.linalg.inv(inverse)
    return dict(world_history=np.stack(histories),
                agent_from_world=inverse,world_from_agent=forward,timestamp=timestamp)


class CausalPlannerObservation:
    def __init__(self, row, lane_provider, neighbor_distance):
        self.names = list(row['names'])
        if len(set(self.names)) != len(self.names) or self.names.count('ego') != 1:
            raise ValueError('Common actor identities must be unique and contain ego')
        self.ego = self.names.index('ego')
        self.scene = row['scene_id']
        self.start_frame = int(row['start_frame'])
        self.initial = copy.deepcopy(row['observed_reference'])
        self.n = len(self.names)
        history = array(self.initial['agent_hist'])
        if history.shape != (self.n, 31, 8):
            raise ValueError('A real 31-frame initial observation is required')
        self.world_history = np.asarray(row['observed_source']['world_history'],dtype=np.float64).copy()
        if self.world_history.shape!=history.shape:raise ValueError('Raw source history shape mismatch')
        extents = array(self.initial['agent_hist_extent']).astype(np.float64)
        if extents.shape != (self.n, 31, 3):
            raise ValueError('Need actual initial extent history, not a repeated max')
        self.extent_history = extents.copy()
        self.lane_provider = lane_provider
        self.neighbor_distance = float(neighbor_distance)
        if not self.neighbor_distance > 0:
            raise ValueError('Neighbor distance must come from the common dataset')
        self.last_step = None
        self.audits = []

    def build(self, live, names, step_index, source):
        if list(names) != self.names:
            raise ValueError('Changed actor identity/order')
        expected = 0 if self.last_step is None else self.last_step + 5
        if type(step_index) is not int or step_index != expected:
            raise ValueError('History must advance by exactly five observed frames')
        raw = array(live['agent_hist'])
        if raw.shape != (self.n, 11, 9):
            raise ValueError('Legacy generator input must retain 11x9 layout')
        if 'scene_ts' in live and not np.all(array(live['scene_ts']) == self.start_frame+step_index):
            raise ValueError('Observed history timestamp differs from the common clock')
        mask = array(live['history_availabilities']).astype(bool)
        short = np.asarray(raw[..., [0, 1, 3, 4, 5, 6, 7, 8]], dtype=np.float64).copy()
        short[~mask] = np.nan
        world = np.asarray(source['world_history'],dtype=np.float64)
        if world.shape!=(self.n,11,8) or source['timestamp']!=self.start_frame+step_index:
            raise ValueError('Uncollated live history shape or timestamp mismatch')
        present = array(live['current_pose_available']).astype(bool)
        if not np.array_equal(np.isfinite(world[present]).all(-1),mask[present]):
            raise ValueError('Raw and parsed live history masks differ')
        if mask[~present].any():
            raise ValueError('An absent coordinate frame has an unmasked local history')
        # The overlapping observations must be the same executed trajectory.
        overlap = 11 if step_index == 0 else 6
        old = self.world_history[:, -overlap:]
        new = world[:, :overlap]
        old_valid = np.isfinite(old).all(-1)
        new_valid = np.isfinite(new).all(-1)
        if not np.array_equal(old_valid, new_valid):
            raise ValueError('Live/retained observed history availability differs')
        error = float(np.max(np.abs(old[old_valid]-new[new_valid]))) if old_valid.any() else 0.
        if error > .01:
            raise ValueError('Observed history changed across replans: '+str(error))
        if step_index:
            self.world_history = np.concatenate((self.world_history[:, 5:], world[:, -5:]), axis=1)
            extent = array(live['agent_hist_extent'])
            if extent.shape != (self.n, 11, 3):raise ValueError('Missing actual live extent history')
            self.extent_history = np.concatenate((self.extent_history[:, 5:], extent[:, -5:]), axis=1)
        transforms = np.asarray(source['agent_from_world'],dtype=np.float64)
        local = transform_history(self.world_history, transforms)
        # Retain the initial common arrays exactly, including parser NaN masks.
        agents = parsed_history(local if step_index else self.initial['agent_hist'])
        for key in ('world_from_agent', 'agent_from_world', 'centroid', 'yaw'):
            agents[key] = array(live[key]).copy()
        agents['centroid'][~present] = np.nan
        agents['yaw'][~present] = np.nan
        for key in ('scene_index', 'track_id', 'type'):
            agents[key] = array(self.initial[key]).copy()
        agents['map_names'] = list(self.initial['map_names'])
        agents['extent'] = np.nan_to_num(self.extent_history, nan=0.).max(axis=1).astype(np.float32)
        agents['agent_hist_len'] = agents['history_availabilities'].sum(-1)
        ego = {key:([value[self.ego]] if isinstance(value, list) else value[self.ego:self.ego+1].copy())
               for key,value in agents.items()}
        distances = np.linalg.norm(agents['centroid'] - agents['centroid'][self.ego], axis=-1)
        neighbors = [int(i) for i in np.argsort(distances)
                     if i != self.ego and distances[i] <= self.neighbor_distance]
        if neighbors:
            in_ego = np.repeat(transforms[self.ego:self.ego+1], len(neighbors), axis=0)
            neighbor_raw = parsed_history(transform_history(self.world_history[neighbors], in_ego))['agent_hist']
            extent = agents['extent'][neighbors]
            types = agents['type'][neighbors]
        else:
            # Match maybe_pad_neighbor's one unavailable placeholder; count=0.
            neighbor_raw = parsed_history(np.full((1,31,8),np.nan))['agent_hist']
            extent = np.zeros((1,3),dtype=np.float32)
            types = np.zeros(1,dtype=np.int64)
        ego.update(neigh_hist=neighbor_raw[None], num_neigh=np.array([len(neighbors)]),
                   all_other_agents_indices=np.array([neighbors or [-1]]),
                   all_other_agents_extents=extent[None], all_other_agents_types=types[None],
                   neigh_types=types[None])
        map_view=dict(ego,agent_from_world=transforms[self.ego:self.ego+1],
            world_from_agent=np.asarray(source['world_from_agent'])[self.ego:self.ego+1])
        lanes = self.lane_provider(map_view)
        if np.asarray(lanes).shape != (1,15,80,3):
            raise ValueError('Canonical lane feature shape changed')
        ego['extras'] = {'closest_lane_point': np.asarray(lanes,dtype=np.float32)}
        self.last_step = step_index
        self.audits.append(dict(step_index=step_index,history_frames=31,
            appended_executed_frames=0 if step_index==0 else 5,
            overlap_max_abs_error=error,neighbor_indices=neighbors,
            observed_past_only=True,generator_observation_modified=False))
        return ego, agents
