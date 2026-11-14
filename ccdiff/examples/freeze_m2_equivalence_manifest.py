#!/usr/bin/env python3
"""Freeze a deterministic 100-scene M1/M2 equivalence manifest."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from trajdata import UnifiedDataset

from tbsim.utils.config_utils import (
    get_experiment_config_from_file,
    translate_pass_trajdata_cfg,
)
from tbsim.utils.trajdata_utils import TRAJDATA_AGENT_TYPE_MAP


SALT = "CCDiff-M2-equivalence-v2-20260804"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_frozen_json(path, payload):
    path = Path(path)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != serialized:
            raise RuntimeError(
                "Refusing to overwrite different frozen manifest: {}".format(
                    path
                )
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as stream:
        stream.write(serialized)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    experiment_config = Path(args.experiment_config).resolve()
    selection_manifest = Path(args.selection_manifest).resolve()
    output_dir = Path(args.output_dir).resolve()

    experiment = get_experiment_config_from_file(str(experiment_config))
    data = translate_pass_trajdata_cfg(experiment)
    only_types = [
        TRAJDATA_AGENT_TYPE_MAP[value]
        for value in data.trajdata_only_types
    ]
    only_predict = None
    if data.trajdata_predict_types is not None:
        only_predict = [
            TRAJDATA_AGENT_TYPE_MAP[value]
            for value in data.trajdata_predict_types
        ]

    dataset = UnifiedDataset(
        cache_location=Path(args.cache_path).resolve(),
        desired_data=["nusc_trainval-val"],
        desired_dt=data.step_time,
        future_sec=(
            data.future_num_frames * data.step_time,
            data.future_num_frames * data.step_time,
        ),
        history_sec=(
            data.history_num_frames * data.step_time,
            data.history_num_frames * data.step_time,
        ),
        data_dirs={"nusc_trainval": str(Path(args.dataset_path).resolve())},
        only_types=only_types,
        only_predict=only_predict,
        agent_interaction_distances=defaultdict(
            lambda: data.max_agents_distance
        ),
        incl_raster_map=False,
        incl_vector_map=False,
        centric="agent",
        scene_description_contains=data.trajdata_scene_desc_contains,
        standardize_data=data.trajdata_standardize_data,
        verbose=False,
        num_workers=0,
        rebuild_cache=False,
        rebuild_maps=False,
    )
    scene_names = [scene.name for scene in dataset.scenes()]
    if len(scene_names) != 150 or len(set(scene_names)) != 150:
        raise RuntimeError(
            "Expected 150 unique nusc_trainval-val scenes, got {}".format(
                len(scene_names)
            )
        )

    with open(selection_manifest, "r", encoding="utf-8") as stream:
        selection_payload = json.load(stream)
    selection_indices = selection_payload["scene_indices"]
    if len(selection_indices) != 20 or len(set(selection_indices)) != 20:
        raise RuntimeError("selection20 is not exactly 20 unique indices")
    if min(selection_indices) < 0 or max(selection_indices) >= 150:
        raise RuntimeError("selection20 index is outside the val population")

    remaining = [
        index for index in range(150) if index not in set(selection_indices)
    ]
    ranked = sorted(
        remaining,
        key=lambda index: hashlib.sha256(
            "{}|{}|{}".format(SALT, index, scene_names[index]).encode(
                "utf-8"
            )
        ).hexdigest(),
    )
    equivalence_indices = ranked[:100]
    excluded_indices = ranked[100:]
    if set(equivalence_indices) & set(selection_indices):
        raise RuntimeError("equivalence100 overlaps selection20")

    base_payload = {
        "assertions": {
            "equivalence100_selection20_disjoint": True,
            "equivalence100_unique": True,
            "population_unique": True,
            "population_size_is_150": True,
        },
        "dataset": "nusc_trainval-val",
        "equivalence_scene_indices": equivalence_indices,
        "equivalence_scene_names": [
            scene_names[index] for index in equivalence_indices
        ],
        "excluded_remaining_indices": excluded_indices,
        "experiment_config_path": str(experiment_config),
        "experiment_config_sha256": sha256_file(experiment_config),
        "freeze_time": "2026-08-04T11:22:00+08:00",
        "generation": {
            "algorithm": "ascending SHA-256 rank",
            "key": "salt|trajdata_index|scene_name",
            "salt": SALT,
            "take": 100,
        },
        "population_index_to_scene_name": {
            str(index): name for index, name in enumerate(scene_names)
        },
        "population_size": len(scene_names),
        "protocol": "M2 Gate 3 paired transport equivalence",
        "selection_manifest_path": str(selection_manifest),
        "selection_manifest_sha256": sha256_file(selection_manifest),
        "selection_scene_indices": selection_indices,
    }

    main_path = output_dir / "val_equivalence_100_v2_20260804.json"
    write_frozen_json(main_path, base_payload)
    shard_paths = []
    for shard_index in range(4):
        shard_indices = equivalence_indices[
            shard_index * 25:(shard_index + 1) * 25
        ]
        shard_payload = {
            "dataset": "nusc_trainval-val",
            "parent_manifest": str(main_path),
            "scene_indices": shard_indices,
            "scene_names": [scene_names[index] for index in shard_indices],
            "shard_count": 4,
            "shard_index": shard_index,
        }
        shard_path = output_dir / (
            "val_equivalence_100_v2_20260804_shard{}.json".format(
                shard_index
            )
        )
        write_frozen_json(shard_path, shard_payload)
        shard_paths.append(str(shard_path))

    report = {
        "equivalence_count": len(equivalence_indices),
        "equivalence_selection_overlap": len(
            set(equivalence_indices) & set(selection_indices)
        ),
        "main_manifest": str(main_path),
        "main_manifest_sha256": sha256_file(main_path),
        "population_count": len(scene_names),
        "selection_count": len(selection_indices),
        "shards": shard_paths,
        "status": "pass",
    }
    print("M2_EQUIVALENCE_MANIFEST {}".format(json.dumps(report, sort_keys=True)))


if __name__ == "__main__":
    main()
