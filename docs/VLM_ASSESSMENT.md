# VLM-assisted interaction inspection

This document records the protocol described in the manuscript supplementary
material. It is a protocol summary, not a claim that this source snapshot includes
the original VLM runner, prompts, video processor, or inference outputs.

## Purpose and decision rule

The fixed VLM describes visible continuity and identity with numerical evidence
supplied by code. The original labels are `Reasonable`, `Clearly anomalous`, and
`Insufficient evidence`, followed by a reason specifying the time and vehicle.
`Reasonable` means no identified visible spatiotemporal inconsistency under the
supplied evidence. It is not a physical-validity, maneuver-completion, or accident-
responsibility label. Collision alone is not an anomaly criterion.

The instruction requires a locatable contradiction for `Clearly anomalous` and
`Insufficient evidence` for a target with no visible frame. Physical threshold
judgments are prohibited when no threshold is provided. Output format compliance
must be distinguished from factual correctness and instruction compliance.

## Reported settings

| Setting | Manuscript value |
|---|---|
| Model | Qwen3-VL-32B-Instruct |
| Model revision | `0cfaf48183f594c31` |
| Inference | BF16, greedy decoding |
| Maximum new tokens | 512 |
| Input | One MP4 video per case |
| Video-pixel budget | 40,960,000 |
| Encoding | 10 Hz; 100 recorded frames for nuScenes, 70 for nuPlan |
| System instruction | Chinese; reported SHA-256 prefix `42bcf582` |

These settings are reported by the manuscript and were not reconstructed from
an inference run in this archive. Encoding all recorded frames does not determine
the model processor's effective frame sampling.

## Evidence supplied to the model

- Fixed-world rendering with timestamps, ego E, selected target T1, surrounding
  vehicles, and native-map drivable mask.
- Recorded-box ego-target overlap intervals.
- Per-vehicle visible-frame counts.
- E/T1 maximum adjacent-frame displacement and surrounding 0.1 s displacement
  vectors.

Method names and prior judgments are withheld. Signals, right of way, lane
direction, off-road checks, other-vehicle contact, and post-contact dynamics are
listed as unavailable. Repeating supplied numbers is not independent detection
or verification of those numbers.

## Case and output accounting

The manuscript reports 12 illustrative cases and a 36-case extension, plus three
technical controls reused with the extension. The extension was selected before
inference using hash ordering within request family and round-robin road classes;
contact status was not a selection criterion. Some families permit two planners
per scene. Counts are rollouts rather than necessarily independent scenes.

All 48 case labels are `Reasonable`. The reported extension contains 18 of 36
explanations with unsupported physical/normal-range statements. This is an
explanation issue count, not a physical-invalidity rate; the remaining explanations
are not thereby certified correct. The manuscript identifies a numerical error
and unsupported-range statements among illustrative explanations as well.

The three controls demonstrate category reachability. They do not establish
natural-case accuracy, and the available numeric metadata can itself reveal the
injected jump or absent target. The reported stationary-target description needs
trajectory verification before inferring maneuver non-completion.

## Information still needed for exact reproduction

The source snapshot does not supply the complete original system/user prompts,
the original input assembly, processor/decoder versions and options, or effective
frame indices and timestamps. A hash prefix is not a replacement for prompt text.
Do not reconstruct a guessed prompt and label its output as the original study.

A future evaluation should save those settings and keep new results separate.
Comparing video-only, metadata-only, and joint input on the same fixed cases can
measure the added visual contribution. Such comparisons and human annotations
are proposed checks, not completed experiments in this release.
