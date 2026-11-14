"""Official nuPlan LQR + bicycle execution, isolated from the FM environment.

This is an explicitly separate execution protocol, not a replacement for raw
planner results. Vehicle boxes retain observed extents; axle geometry is the
documented nuPlan Pacifica proxy, not a calibrated nuScenes vehicle model.
"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import yaml
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory

CONFIG = Path("/path/to/workspace/home/planner_sources_20260810/nuplan_source/nuplan-devkit-master/nuplan/planning/script/config/simulation/ego_controller/tracker/lqr_tracker.yaml")


class TrackerRuntime:
    def __init__(self):
        self.config = {k: v for k, v in yaml.safe_load(CONFIG.read_text()).items() if not k.startswith("_")}
        self.config = {k: float(v) if isinstance(v, str) else v for k, v in self.config.items()}
        self.saved = None
        self.next_step = None

    def track(self, request):
        positions = np.asarray(request["positions"], dtype=float)
        yaws = np.asarray(request["yaws"], dtype=float).reshape(-1)
        history = np.asarray(request["history"], dtype=float)
        if positions.shape != (len(yaws), 2) or not np.isfinite(positions).all() or not np.isfinite(yaws).all():
            raise ValueError("Invalid planner reference")
        if len(positions) < 5 or history.ndim != 2 or history.shape[1] != 8:
            raise ValueError("Expected at least five reference poses and [T,8] history")
        length, width = map(float, request["extent"][:2])
        nominal = get_pacifica_parameters()
        offset = nominal.rear_axle_to_center
        if length / 2 <= offset:
            raise ValueError("Observed box too short for the declared axle proxy")
        vehicle = VehicleParameters(
            width=width, front_length=length / 2 + offset, rear_length=length / 2 - offset,
            cog_position_from_rear_axle=nominal.cog_position_from_rear_axle,
            wheel_base=nominal.wheel_base, vehicle_name="observed_box_with_nuplan_axle_proxy",
            vehicle_type="development_proxy", height=float(request["extent"][2]))
        tracker = LQRTracker(vehicle=vehicle, **self.config)
        motion = KinematicBicycleModel(vehicle)
        step = int(request["step_index"])
        if self.saved is not None and step == self.next_step:
            speed, accel, steer, omega, omega_dot, steer_rate = self.saved
            initial_source = "persisted_native_state_at_last_executed_prefix"
        else:
            speed, accel = float(history[-1, 2]), float(history[-1, 4])
            angle = np.arctan2(history[-2:, 6], history[-2:, 7])
            omega = float((angle[-1] - angle[-2] + np.pi) % (2 * np.pi) - np.pi) / .1
            steer = float(np.arctan(omega * vehicle.wheel_base / abs(speed))) if abs(speed) > .2 else 0.
            omega_dot = steer_rate = 0.
            initial_source = "observed_history_initialization"
        initial = EgoState.build_from_rear_axle(
            rear_axle_pose=StateSE2(-offset, 0., 0.),
            rear_axle_velocity_2d=StateVector2D(speed, 0.),
            rear_axle_acceleration_2d=StateVector2D(accel, 0.),
            tire_steering_angle=steer, time_point=TimePoint(0), vehicle_parameters=vehicle,
            angular_vel=omega, angular_accel=omega_dot, tire_steering_rate=steer_rate)
        reference = [initial]
        for i, (xy, yaw) in enumerate(zip(positions, yaws), start=1):
            rear = xy - offset * np.array([np.cos(yaw), np.sin(yaw)])
            reference.append(EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(float(rear[0]), float(rear[1]), float(yaw)),
                rear_axle_velocity_2d=StateVector2D(0., 0.),
                rear_axle_acceleration_2d=StateVector2D(0., 0.), tire_steering_angle=0.,
                time_point=TimePoint(i * 100000), vehicle_parameters=vehicle))
        trajectory = InterpolatedTrajectory(reference)
        current = initial
        tracked, commands = [], []
        for i in range(len(positions)):
            now = SimulationIteration(TimePoint(i * 100000), i)
            after = SimulationIteration(TimePoint((i + 1) * 100000), i + 1)
            command = tracker.track_trajectory(now, after, current, trajectory)
            current = motion.propagate_state(current, command, TimePoint(100000))
            tracked.append([current.center.x, current.center.y, current.center.heading])
            commands.append([command.rear_axle_acceleration_2d.x, command.tire_steering_rate])
            if i == 4:
                state = current.dynamic_car_state
                saved = (state.rear_axle_velocity_2d.x, state.rear_axle_acceleration_2d.x,
                         current.tire_steering_angle, state.angular_velocity,
                         state.angular_acceleration, state.tire_steering_rate)
        if not np.isfinite(tracked).all() or not np.isfinite(commands).all():
            raise FloatingPointError("Nonfinite native controller trajectory; no fallback/clipping")
        self.saved, self.next_step = saved, step + 5
        return dict(positions=np.asarray(tracked)[:, :2].tolist(),
                    yaws=np.asarray(tracked)[:, 2:3].tolist(), commands=commands,
                    provenance=dict(protocol="native_nuplan_lqr_bicycle_v1",
                                    initialization=initial_source, executed_prefix_frames=5,
                                    tracker_config=str(CONFIG), tracker_parameters=self.config,
                                    wheelbase_m=vehicle.wheel_base, rear_axle_to_center_m=offset,
                                    vehicle_geometry_scope="observed_box_with_uncalibrated_nuplan_axle_proxy",
                                    posthoc_trajectory_smoothing=False, acceptance_thresholds_changed=False))


def main():
    runtime = TrackerRuntime()
    print(json.dumps(dict(ready=True, protocol="native_nuplan_lqr_bicycle_v1")), flush=True)
    for line in sys.stdin:
        try:
            response = dict(ok=True, **runtime.track(json.loads(line)))
        except Exception:
            response = dict(ok=False, error=traceback.format_exc())
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
