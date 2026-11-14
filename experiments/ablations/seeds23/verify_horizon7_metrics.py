"""Check horizon adaptation against retained simulator outputs, outside formal runs."""
import importlib.util
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

from common import ROOT, HERE, QUEUE, load, save, sha
import final_metrics as current


def main():
    old_source = ROOT / 'launch/crashdojo_A0_A8_nusc100_s123_20261107/final_metrics.py'
    spec = importlib.util.spec_from_file_location('previous_v3_metrics', old_source)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    source = ROOT / 'eval/crashdojo_A0_A8_nusc100_s123_20261107/runs/scene-1062_s1_idm_A0'
    before, after = previous.executed(source), current.executed(source, 100)
    for a, b in zip(before[2:6], after[2:6]):
        np.testing.assert_array_equal(a, b)
    assert before[6] == after[6]

    fixture = QUEUE / 'qa/h70_cropped_recorded_simulator_fixture'
    fixture.mkdir(parents=True, exist_ok=True)
    source_hdf5 = after[0]
    shutil.copy2(source_hdf5.with_name('selection_binding.json'), fixture / 'selection_binding.json')
    with h5py.File(source_hdf5) as original, h5py.File(fixture / 'data.hdf5', 'w') as cropped:
        name = next(iter(original))
        src, dst = original[name], cropped.create_group(name)
        for key in ['centroid', 'yaw', 'extent', 'availability', 'action_positions', 'action_yaws']:
            dst.create_dataset(key, data=np.asarray(src[key])[:, :70])
    short = current.executed(fixture, 70)
    np.testing.assert_array_equal(short[2][:, :70], after[2][:, :70])
    # The 7-second reconstruction can be checked against a real recorded state
    # from the retained 10-second run, not against another reconstruction.
    endpoint_error = float(np.linalg.norm(short[2][:, 70] - after[2][:, 70], axis=-1).max())
    assert endpoint_error < .01, endpoint_error
    assert short[2].shape == (2, 71, 2)

    protocol = current.freeze_reference()
    assert protocol['trajectory_times_s'] == dict(initial=0., first_future=.1,
        last_future=7., dt=.1, frames_including_initial=71)
    cohort = load(QUEUE / 'cohort.json')
    assert len(cohort) == len({x['scene'] for x in cohort}) == 100
    reference_coverage = {}
    for job in cohort:
        start = load(job['manifest'])['start_step_10hz']
        xy, yaw, valid = current.reference(job, job['selected_actor'], np.arange(start, start + 71))
        assert valid[0], job['scene']
        f = current.features(xy, yaw, valid)
        if valid.all():
            assert {k: len(v) for k, v in f.items()} == dict(lon_accel=13, lat_accel=13, jerk=12)
        reference_coverage[job['scene']] = dict(valid_frames=int(valid.sum()),
            endpoint_available=bool(valid[-1]), feature_counts={k: len(v) for k, v in f.items()})
    save(QUEUE / 'qa/h7_metrics_validation.json', dict(passed=True, checked_unix=time.time(),
        old_100_step_pose_regression_exact=True, cropped_70_step_frames_including_initial=71,
        reconstructed_7s_endpoint_vs_recorded_pose_max_error_m=endpoint_error,
        reference_coverage=reference_coverage,
        reference_endpoint_available=sum(x['endpoint_available'] for x in reference_coverage.values()),
        missing_endpoint_policy='FDE remains missing; do not substitute an earlier endpoint.',
        metric_source_sha256=sha(HERE / 'final_metrics.py'), old_metric_source_sha256=sha(old_source),
        fixture_path=str(fixture), fixture_is_not_an_experiment_result=True,
        protocol_sha256=sha(QUEUE / 'metric_protocol.json')))
    print(dict(passed=True, endpoint_error_m=endpoint_error, reference_scenes=100), flush=True)


if __name__ == '__main__':
    main()
