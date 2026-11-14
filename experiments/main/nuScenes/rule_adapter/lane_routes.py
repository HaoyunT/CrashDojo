"""Observed-pose route shared by the nuScenes IDM and PDM adapters."""
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from nuscenes.map_expansion.map_api import NuScenesMap


def angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class LaneRoutes:
    def __init__(self, name):
        self.name = name
        if name in ['boston','las_vegas','singapore','pittsburgh']:
            from pathlib import Path
            from trajdata.maps.map_api import MapAPI
            from trajdata.maps.vec_map_elements import MapElementType
            root=Path(__file__).resolve().parent.parent/'nuplan_centered_runtime_cache'
            vector=MapAPI(root).get_map('nuplan_mini:'+name)
            native=vector.elements[MapElementType.ROAD_LANE]
            self.lanes={}
            for token,lane in native.items():
                points=lane.center.points
                if len(points)<2:continue
                poses=np.c_[points[:,:2],points[:,3]]
                self.lanes[token]=SimpleNamespace(id=token,poses=poses,speed_limit_mps=13.9)
            self.source=SimpleNamespace(get_outgoing_lane_ids=lambda token:sorted(native[token].next_lanes))
            self.lane_list=list(self.lanes.values())
            self.points=np.concatenate([x.poses for x in self.lane_list])
            self.point_tokens=np.concatenate([[x.id]*len(x.poses) for x in self.lane_list])
            self.point_tree=cKDTree(self.points[:,:2])
            return
        self.source = NuScenesMap(dataroot='/path/to/storage/external/data/nuscenes', map_name=name)
        self.lanes = {}
        for row in self.source.lane + self.source.lane_connector:
            token = row['token']
            poses = np.asarray(self.source.discretize_lanes([token], 1.0)[token])
            if len(poses) >= 2:
                self.lanes[token] = SimpleNamespace(id=token, poses=poses, speed_limit_mps=13.9,
                    polygon=self.source.extract_polygon(row['polygon_token']))
        self.lane_list = list(self.lanes.values())
        self.points = np.concatenate([x.poses for x in self.lane_list])
        self.point_tokens = np.concatenate([[x.id] * len(x.poses) for x in self.lane_list])
        self.point_tree = cKDTree(self.points[:, :2])

    def route_array(self, center):
        dist, idx = self.point_tree.query(center[:2], k=min(150, len(self.points)))
        score = dist + 5 * np.abs(angle(self.points[idx, 2] - center[2]))
        chosen = int(idx[np.argmin(score)])
        if np.min(dist) > 30:
            raise RuntimeError('Initial ego is more than 30m from a mapped lane')
        token = self.point_tokens[chosen]
        route, length = [], 0.
        while token in self.lanes and token not in route and length < 300:
            route.append(token)
            poses = self.lanes[token].poses
            length += np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).sum()
            outgoing = [x for x in self.source.get_outgoing_lane_ids(token) if x in self.lanes and x not in route]
            if not outgoing:
                break
            token = min(outgoing, key=lambda t: (abs(angle(self.lanes[t].poses[0, 2] - poses[-1, 2])), t))
        points = np.concatenate([self.lanes[t].poses for t in route])
        keep = np.r_[True, np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1) > 1e-3]
        return route, points[keep]


class IDMRouteAdapter:
    def __init__(self, planner):
        self.planner, self.maps = planner, {}
        self.key, self.last_step, self.route = None, None, None

    def plan(self, obs, step_index):
        import torch
        assert len(obs['agent_hist']) == 1
        world = np.asarray(obs['world_from_agent'][0])
        inverse = np.asarray(obs['agent_from_world'][0])
        heading = np.arctan2(world[1, 0], world[0, 0])
        name = str(obs['map_names'][0]).split(':')[-1]
        key = (name, int(obs['scene_index'][0]))
        if key != self.key or self.last_step is None or step_index <= self.last_step:
            if name not in self.maps:
                self.maps[name] = LaneRoutes(name)
            self.ids, self.route = self.maps[name].route_array(np.r_[world[:2, 2], heading])
            self.key = key
        self.last_step = step_index
        # Keep graph order through turns; do not sort a curved route by local x.
        nearest = int(np.argmin(np.linalg.norm(self.route[:, :2] - world[:2, 2], axis=-1)))
        points = self.route[min(nearest + 1, len(self.route) - 1):, :2]
        local = points @ inverse[:2, :2].T + inverse[:2, 2]
        path = np.vstack([np.zeros((1, 2)), local])
        keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=-1) > 1e-3]
        path = torch.as_tensor(path[keep], dtype=torch.float32)
        if len(path) < 2:
            raise RuntimeError('IDM route has no forward segment')
        self.planner._reference_paths = lambda observation, batch_size: [path]
        action, info = self.planner.plan(obs, step_index=step_index)
        return action, {'route_lane_ids': self.ids, 'route_uses_future': False,
                        'implementation': 'IDMPlannerPolicy with observed-pose lane route'}
