"""nuScenes inputs for the unmodified official PDM-Closed planning pipeline.

nuScenes has no navigation route or lane speed limits in this interface. Build
a deterministic forward lane-graph route from the INITIAL observed pose only,
keep it fixed for the rollout, use 13.9 m/s and unknown traffic lights. Vehicle
dimensions are observed; the rear-axle geometry scales the nuPlan Pacifica.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
from lane_routes import LaneRoutes
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from tuplan_garage.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy
from tuplan_garage.planning.simulation.planner.pdm_planner.proposal.pdm_proposal import PDMProposalManager
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import parallel_discrete_path
from tuplan_garage.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


def angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class NuScenesMapAdapter(LaneRoutes):
    def __init__(self, name):
        super().__init__(name)
        if name in ['boston', 'las_vegas', 'singapore', 'pittsburgh']:
            # trajdata lane elements expose centerline poses, not NuScenes polygons.
            # PDM only needs a conservative roadblock query for proposal sampling.
            self.lane_polygons = [LineString(x.poses[:, :2]).buffer(2.0) for x in self.lane_list]
            for lane, polygon in zip(self.lane_list, self.lane_polygons):
                lane.polygon = polygon
            self.lane_tree = STRtree(self.lane_polygons)
            self.parks = []
            self.intersections = []
            self.intersection_tree = None
            return
        self.lane_tree = STRtree([x.polygon for x in self.lane_list])
        self.parks = [SimpleNamespace(id='park:' + x['token'], polygon=self.source.extract_polygon(x['polygon_token']))
                      for x in self.source.carpark_area]
        self.intersections = [self.source.extract_polygon(x['polygon_token']) for x in self.source.road_segment if x['is_intersection']]
        self.intersection_tree = STRtree(self.intersections)

    def get_proximal_map_objects(self, position, radius, layers):
        zone = Point(position.x, position.y).buffer(radius)
        lane_ids = self.lane_tree.query(zone, predicate='intersects')
        return {layer: ([SimpleNamespace(interior_edges=[self.lane_list[i]]) for i in lane_ids]
                        if layer == SemanticMapLayer.ROADBLOCK else
                        [x for x in self.parks if x.polygon.intersects(zone)]
                        if layer == SemanticMapLayer.CARPARK_AREA else []) for layer in layers}

    def is_in_layer(self, point, layer):
        if layer != SemanticMapLayer.INTERSECTION:
            raise ValueError('Unsupported map query: ' + str(layer))
        if self.intersection_tree is None:
            return False
        return bool(len(self.intersection_tree.query(Point(point.x, point.y), predicate='intersects')))

    def initial_route(self, center):
        route, points = self.route_array(center)
        return route, [StateSE2(*p) for p in points]



class AdaptedPDM(PDMClosedPlanner):
    """Override dataset route initialization only; retain all official planning stages."""
    def __init__(self, api, center):
        super().__init__(TrajectorySampling(num_poses=80, interval_length=.1),
                         TrajectorySampling(num_poses=40, interval_length=.1),
                         BatchIDMPolicy(speed_limit_fraction=[.2, .4, .6, .8, 1.],
                                        fallback_target_velocity=15., min_gap_to_lead_agent=1.,
                                        headway_time=1.5, accel_max=1.5, decel_max=3.),
                         [-1., 1.], 50.)
        self._map_api = api
        self.route_ids, self.discrete_route = api.initial_route(center)
        self._route_lane_dict = {t: api.lanes[t] for t in self.route_ids}

    def _route_roadblock_correction(self, ego_state):
        # nuScenes has lane connectivity, but no nuPlan roadblock navigation IDs.
        # The observed-pose route was constructed above; never read future labels.
        pass

    def _update_proposal_manager(self, ego_state):
        if self._proposal_manager is None:
            self._centerline = PDMPath(self.discrete_route)
            paths = [self._centerline] + [PDMPath(parallel_discrete_path(self.discrete_route, x)) for x in self._lateral_offsets]
            self._proposal_manager = PDMProposalManager(paths, self._idm_policies)
        self._proposal_manager.update(13.9)


class NuScenesPDMRuntime:
    def __init__(self):
        self.maps = {}
        self.planner = None
        self.scene_key = None
        self.last_step = None
        self.last_objects = []
        self.next_id = 0

    def infer(self, request):
        obs = request['observation']
        hist = np.asarray(obs['agent_hist'], dtype=float)
        assert hist.shape[0] == 1, 'Evaluation uses one scene per batch'
        transform = np.asarray(obs['world_from_agent'][0], dtype=float)
        inverse = np.asarray(obs['agent_from_world'][0], dtype=float)
        heading = np.arctan2(transform[1, 0], transform[0, 0])
        center = np.r_[transform[:2, 2], heading]
        name = str(obs['map_names'][0]).split(':')[-1]
        key = (name, str(obs.get('scene_index', ['unknown'])[0]))
        step = request['step_index']
        if key != self.scene_key or self.last_step is None or step <= self.last_step:
            self.maps.setdefault(name, None)
            if self.maps[name] is None:
                self.maps[name] = NuScenesMapAdapter(name)
            self.planner = AdaptedPDM(self.maps[name], center)
            self.scene_key, self.last_objects, self.next_id = key, [], 0
        delta_t = .5 if self.last_step is None else max(.1, (step - self.last_step) * .1)
        self.last_step = step
        extent = np.asarray(obs['extent'][0], dtype=float)
        pacifica = get_pacifica_parameters()
        scale = extent[0] / pacifica.length
        vehicle = VehicleParameters(width=extent[1], front_length=pacifica.front_length * scale,
                                    rear_length=pacifica.rear_length * scale,
                                    cog_position_from_rear_axle=pacifica.cog_position_from_rear_axle * scale,
                                    wheel_base=pacifica.wheel_base * scale, vehicle_name='nuScenes-observed',
                                    vehicle_type='car', height=extent[2] if len(extent) > 2 else 1.8)
        current = hist[0, -1]
        assert np.isfinite(current).all()
        timestamp = TimePoint(1000000 + step * 100000)
        ego = EgoState.build_from_center(StateSE2(*center), StateVector2D(*current[2:4]),
                                         StateVector2D(*current[4:6]), 0., timestamp, vehicle)
        assert np.allclose([ego.center.x, ego.center.y, ego.center.heading], center)
        neighbors = np.asarray(obs['neigh_hist'][0], dtype=float)
        types = np.asarray(obs['neigh_types'][0])
        extents = np.asarray(obs['all_other_agents_extents'][0], dtype=float)
        valid = np.isfinite(neighbors[:, -1]).all(-1) & (types > 0)
        objects = []
        for j in np.flatnonzero(valid):
            h = neighbors[j, -1]
            xy = transform[:2, :2] @ h[:2] + transform[:2, 2]
            vel = transform[:2, :2] @ h[2:4]
            yaw = angle(np.arctan2(h[6], h[7]) + heading)
            e = extents[j]
            if not np.isfinite(e[:2]).all() or min(e[:2]) <= 0:
                raise ValueError('Invalid observed neighbor extent')
            objects.append({'xy': xy, 'velocity': vel, 'yaw': yaw, 'extent': e, 'type': int(types[j])})
        # Some legacy observations omit persistent neighbor tokens. Associate
        # using current observations and the previous constant-velocity state.
        assigned = {}
        if self.last_objects and objects:
            predicted = np.array([x['xy'] + x['velocity'] * delta_t for x in self.last_objects])
            current_xy = np.array([x['xy'] for x in objects])
            cost = np.linalg.norm(predicted[:, None] - current_xy[None], axis=-1)
            old_idx, new_idx = linear_sum_assignment(cost)
            assigned = {j: self.last_objects[i]['id'] for i, j in zip(old_idx, new_idx) if cost[i, j] < 5}
        agents = []
        for j, obj in enumerate(objects):
            if j in assigned:
                obj['id'] = assigned[j]
            else:
                obj['id'] = self.next_id
                self.next_id += 1
            token = 'neighbor-' + str(obj['id'])
            e = obj['extent']
            # All benchmark environments select VEHICLE; tbsim converts its
            # trajdata type 1 to L5Kit type 3, which must not mean bicycle here.
            if obj['type'] not in (1, 3):
                raise ValueError('Nonvehicle in the vehicle-only rule benchmark')
            kind = TrackedObjectType.VEHICLE
            box = OrientedBox(StateSE2(*obj['xy'], obj['yaw']), e[0], e[1], e[2] if len(e) > 2 else 1.8)
            meta = SceneObjectMetadata(timestamp.time_us, token, obj['id'], token)
            agents.append(Agent(kind, box, StateVector2D(*obj['velocity']), meta))
        self.last_objects = objects
        detections = DetectionsTracks(TrackedObjects(agents))
        history = SimulationHistoryBuffer.initialize_from_list(1, [ego], [detections], .1)
        inp = PlannerInput(SimulationIteration(timestamp, step), history, [])
        trajectory = self.planner.compute_planner_trajectory(inp)
        states = [trajectory.get_state_at_time(TimePoint(timestamp.time_us + int(i * 100000))) for i in range(1, 53)]
        world = np.asarray([[s.center.x, s.center.y, s.center.heading] for s in states])
        positions = world[:, :2] @ inverse[:2, :2].T + inverse[:2, 2]
        yaws = angle(world[:, 2] - heading)
        assert np.isfinite(positions).all() and np.isfinite(yaws).all()
        return {'positions': positions[None].tolist(), 'yaws': yaws[None, :, None].tolist(),
                'info': {'implementation': 'official PDMClosedPlanner core with nuScenes route/map adapter',
                         'proposals': 15, 'route_lane_ids': self.planner.route_ids, 'route_uses_future': False,
                         'speed_limit_mps': 13.9, 'traffic_lights': 'unknown', 'rear_axle_model': 'scaled Pacifica',
                         'neighbor_tracking': 'observed-position constant-velocity association', 'neighbors': len(agents)}}
