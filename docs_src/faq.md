# FAQ

Short answers to questions that come up when reading the paper, running the testbench or setting the system up at a new site. Each answer points to the page with the details.

## What do the four states mean?

<span class="state state-idle">idle</span> means no pedestrian is tracked. <span class="state state-safe">safe</span> means a pedestrian is present and no cyclist has been seen within the cyclist memory. <span class="state state-warning">warning</span> means a pedestrian is present and a cyclist is or recently was in the scene, but no pair passes the closing check. <span class="state state-alert">alert</span> means a cyclist and a pedestrian are within the distance bounds, the distance between them is shrinking, and the cyclist is moving. The light shows blue, green, orange and red for the four states. See [The decision pipeline](system/decision.md).

## Why is there a warning state between safe and alert?

Stage 3 returns `alert` only for a pair that is closing. When a cyclist is in the scene with a pedestrian and no pair meets that test, the pipeline falls through to `warning` and does not go back to `safe`. The light therefore shows that a cyclist is around before any conflict is detected.

## Why does the pipeline remember cyclists?

Stage 2 keeps a memory of the times at which a cyclist was in the frame. It stays open for `bike_memory_frames` frames after the last one, 58 frames or about 1.9 s at 30 FPS with the shipped parameters. A cyclist that the detector drops for a few frames does not send the state back to `safe`. The testbench models such dropouts, see [Detection and localisation uncertainty](system/uncertainty.md).

## Why pairwise closing and not time-to-collision or distance alone?

`python decision_testbench.py --compare` runs the default rule and three baseline rules on the 24 conformance scenarios. The baselines keep the default memory and lower bound. `baseline_distance_only` uses a 10 m upper bound and `baseline_ttc` a 3 s threshold. Totals over all frames, with fisheye localisation error enabled:

| Rule | Config name | TP | FP | FN | TN | Sensitivity | Specificity |
|---|---|---|---|---|---|---|---|
| Pairwise closing | `default` | 1228 | 370 | 88 | 4418 | 93.3% | 92.3% |
| Distance only | `baseline_distance_only` | 761 | 1429 | 555 | 3359 | 57.8% | 70.2% |
| Naive closing | `baseline_naive_closing` | 1229 | 460 | 87 | 4328 | 93.4% | 90.4% |
| Time-to-collision | `baseline_ttc` | 1109 | 173 | 207 | 4615 | 84.3% | 96.4% |

The time-to-collision rule misses 207 danger frames where pairwise closing misses 88. Naive closing matches the sensitivity and raises 90 more false-positive frames. [The decision pipeline](system/decision.md) defines each rule.

## Do I need a GPU?

Not for the testbench, the figures or the paper tables. They need numpy, scipy, matplotlib, pyyaml and opencv-python. The live loop runs a YOLO detector at a 1280 px input size, and for that the README recommends a CUDA-capable GPU. The starter kit accepts `YOLO_DEVICE = "cpu"`. The repository gives no throughput figure for that case.

## Can I use a different detector?

Yes. In the starter kit, point `YOLO_HF_REPO` and `YOLO_HF_FILE` at other ultralytics weights. The kit assumes COCO class IDs: 0 is a pedestrian, and 1 and 3 are cyclists. With a detector outside ultralytics, replace the `model.predict` call. The decision pipeline itself only needs a track ID and a ground position in metres for each pedestrian and cyclist, passed to `DecisionPipeline.update_track` and `evaluate`.

## Why does a cyclist not trigger an alert against their own body?

Detectors usually output a separate `person` box for the rider. The kit relabels a `person` box that overlaps a bicycle or motorcycle box as a cyclist, and the pipeline ignores pairs closer than `prox_min`, 1.9 m by default. Both thresholds depend on the camera geometry. See [Tuning for a new site](deployment/tuning.md#rider-on-bike-suppression).

## Why is the fisheye not calibrated with OpenCV's fisheye module?

OpenCV's fisheye calibration fits the Kannala-Brandt model with polynomial distortion coefficients. According to the docstring of `calibrate_fisheye.py`, that model does not fit the 200° spherical lens on a 16:9 sensor used here. The script fits the equidistant model `r = f · θ` directly with a bundle adjustment, with optional `k1` and `k2` correction terms. See [Fisheye calibration](calibration/index.md).

## Where is the live MQTT bridge of the deployed system?

`mqtt_bridge.py` is not in this repository. [`bridge_starter/`](deployment/index.md) is a self-contained version of it with the same decision pipeline, a simulated light and optional sound, meant to be copied and adapted.

## Does editing config.yaml change the testbench results?

Only `camera.height` and `camera.pitch_deg` are read by the testbench. The lens intrinsics come from `camera_calibration.json`, and the decision parameters come from `PIPELINE_CONFIGS` in `decision_testbench.py`. The other keys of `config.yaml` belong to the live bridge. See [Configuration reference](reference/config.md#who-reads-what).

## Can I try the decision logic without a camera or any install?

Yes. The [browser simulator](simulation/browser.md) runs the pipeline on the conformance scenarios in a web page. For numbers, run [the testbench](simulation/testbench.md), which needs Python and the packages in `requirements.txt`.

## How do I cite the project?

See [Citing](citing.md) for the BibTeX entry of the paper, arXiv 2604.17046.
