import numpy as np
import pytest

from tbsim.envs.env_metrics import CollisionRate, DiskCollisionRate


def _state(history_availabilities):
    return {
        "scene_index": np.asarray([10, 10, 20, 20]),
        "track_id": np.asarray([0, 1, 0, 1]),
        "centroid": np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [100.0, 0.0],
                [100.1, 0.0],
            ],
            dtype=np.float64,
        ),
        "yaw": np.zeros((4, 1), dtype=np.float64),
        "extent": np.asarray([[4.0, 2.0, 1.5]] * 4, dtype=np.float64),
        "history_availabilities": np.asarray(history_availabilities, dtype=np.float64),
    }


def test_collision_metrics_share_per_scene_active_pair_support():
    # Scene 10 contains an overlapping unavailable ghost. Scene 20 contains
    # the same geometry with two active agents. This catches both the old
    # global-index bug and the old failure to mask unavailable neighbours.
    state = _state(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ]
    )
    scenes = np.asarray([10, 20])

    obb, _ = CollisionRate.compute_per_step(state, scenes)
    disk, _ = DiskCollisionRate.compute_per_step(state, scenes)

    np.testing.assert_array_equal(obb["coll_any"], [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(disk["coll_any"], [0.0, 0.0, 1.0, 1.0])
    assert np.all(disk["coll_any"] <= obb["coll_any"])


def test_all_unavailable_agents_are_excluded_from_both_metrics():
    state = _state(np.zeros((4, 2), dtype=np.float64))
    scenes = np.asarray([10, 20])

    obb, _ = CollisionRate.compute_per_step(state, scenes)
    disk, _ = DiskCollisionRate.compute_per_step(state, scenes)

    assert not np.any(obb["coll_any"])
    assert not np.any(disk["coll_any"])


@pytest.mark.parametrize("metric", [CollisionRate, DiskCollisionRate])
def test_availability_rows_must_match_agent_rows(metric):
    state = _state(np.ones((4, 2), dtype=np.float64))
    state["history_availabilities"] = np.ones((3, 2), dtype=np.float64)

    with pytest.raises(ValueError, match="one row per agent"):
        metric.compute_per_step(state, np.asarray([10, 20]))
