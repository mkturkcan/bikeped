# Dataset

The 24 conformance scenarios are published on Hugging Face as [`mehmetkeremturkcan/bikeped`](https://huggingface.co/datasets/mehmetkeremturkcan/bikeped), with per-frame ground-truth labels and three videos per scenario. This page describes what the dataset contains and how to read it.

## Contents

The dataset card names it "bikeped Conformance Scenarios". Agents follow hand-authored waypoint paths that are interpolated at 30 fps, and each frame carries a danger label from the [kinematic safety model](system/safety-model.md). The licence is CC BY 4.0.

| Path | Files | Contents |
|---|---|---|
| `scenarios.jsonl` | 1 | One row per scenario: metadata, waypoint paths and video paths. |
| `frames.jsonl` | 1 | One row per scenario, agent and frame, with position and labels. |
| `videos/sim/` | 24 | Schematic simulated render of each scenario. |
| `videos/sim_2cam/` | 24 | A second schematic set. The folder name matches the `_2cam` suffix that `sim_visualizer.py --two-cam` gives its output. |
| `videos/carla/` | 24 | CARLA 0.9.16 renders from the pole camera on `Town10HD_Opt`. |

Videos are named by index and slug, for example `videos/carla/07_fast_approach.mp4`. The dataset page reports a total file size of 389 MB.

## Configurations

The two JSONL files are exposed as two configurations with a single `train` split each.

### scenarios

24 rows.

| Column | Type | Meaning |
|---|---|---|
| `scenario_id` | int | Index of the scenario, 0 to 23, in the order of `decision_testbench.SCENARIOS`. |
| `name` | string | Display name, for example `Fast Approach`. |
| `expected_dominant` | string | `idle`, `safe`, `warning` or `alert`. |
| `vehicle` | string | Braking profile of the cyclist: `bicycle` or `ebike`. |
| `duration_s` | float | Scenario length in seconds. |
| `n_person`, `n_bike` | int | Number of pedestrians and cyclists. |
| `agents` | list | One entry per agent with `type`, `agent_id` and `path`, a list of `{x, y, t}` waypoints. |
| `sim_video`, `carla_video` | string | Paths of the schematic and the CARLA video inside the dataset. |

### frames

12,901 rows.

| Column | Type | Meaning |
|---|---|---|
| `scenario_id`, `scenario_name` | int, string | Scenario the row belongs to. |
| `frame_idx`, `time_s` | int, float | Frame index and time at 30 fps. |
| `agent_id`, `agent_type` | int, string | Agent the position belongs to. |
| `x_m`, `y_m` | float | Interpolated position in the scenario frame, in metres. |
| `is_danger` | bool | Ground-truth danger label of the frame. |
| `actionable` | bool | Whether a warning at this frame can still be acted on. |
| `severity` | float | Severity, scaled by speed relative to 12 m/s. |
| `ttc_s` | float | Time to collision in seconds. |
| `closing_speed_ms`, `bike_speed_ms` | float | Closing speed of the pair and cyclist speed, in m/s. |
| `stopping_dist_m` | float | Cyclist stopping distance in metres. |
| `warning_budget_s` | float | Warning budget in seconds. |
| `distance_m` | float | Distance to the nearest counterpart in metres. |

## Coordinate frame

Positions use the scenario frame of the testbench. The crosswalk runs along the y axis at x = 5 m and is about 3 m wide, cyclists travel along x, and the card places the deployed camera at (x = 5, y = -13, z = 3.66) in metres.

## Ground-truth labels

The card describes the label as a conservative kinematic model: a frame is dangerous when the pair is closing, the closest approach is within 5 m, and either the stopping distance or the time to collision says the cyclist cannot avoid the conflict. Conventional bicycles use a reaction time of 0.84 s and a deceleration of 1.96 m/s², and e-bikes a deceleration of 6.0 m/s². 21 of the 24 scenarios contain a danger interval. The same model is implemented by `compute_ground_truth_danger` in `decision_testbench.py`, documented under [safety model](system/safety-model.md).

## Loading

=== "datasets"

    ```python
    from datasets import load_dataset

    scenarios = load_dataset("mehmetkeremturkcan/bikeped", "scenarios", split="train")
    frames = load_dataset("mehmetkeremturkcan/bikeped", "frames", split="train")
    ```

=== "Plain files"

    ```console
    $ curl -sL -O https://huggingface.co/datasets/mehmetkeremturkcan/bikeped/resolve/main/scenarios.jsonl
    $ wc -l < scenarios.jsonl
    24
    ```

    Each line is one JSON object with the columns listed above.

=== "Everything, with videos"

    ```bash
    huggingface-cli download mehmetkeremturkcan/bikeped --repo-type dataset --local-dir bikeped-dataset
    ```

## Relation to the repository

The waypoint paths in `scenarios.jsonl` are those of `decision_testbench.SCENARIOS`, so a result computed from the dataset can be checked against `python decision_testbench.py`. The repository also renders the videos: `sim_visualizer.py` writes the schematic ones and the [CARLA renderer](simulation/carla/index.md) the photorealistic ones.

## Related

- [Conformance scenarios](scenarios/conformance.md)
- [File formats](reference/formats.md)
- [Citing](citing.md)
