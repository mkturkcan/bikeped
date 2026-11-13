# Browser simulator

The browser simulator runs the decision pipeline on the scenario suites in a bird's eye view, one frame at a time, with no install. This page describes its controls and how its numbers relate to the offline testbench.

<a href="../../simulator/">Open the simulator</a>. From a checkout, open `simulator/index.html` in any browser; it is a single self-contained HTML file, and `simulator/scenarios.extended.js` next to it adds the suites beyond conformance.

## Panels

| Panel | Contents |
|---|---|
| Bird's eye view | Road, crosswalk, cameras with their field of view, agents and their tracks, in the [scenario frame](../scenarios/index.md) |
| Decision pipeline | A flowchart of the stages (detection, classification and ground projection, tracking, then the three decision checks) with the active path lit and the current output: <span class="state state-idle">idle</span> blue, <span class="state state-safe">safe</span> green, <span class="state state-warning">warning</span> orange, <span class="state state-alert">alert</span> red blink |
| Timeline | Scenario time, duration, play, reset, loop and a speed slider |
| Telemetry | Number of objects, number of tracks and the minimum bike-pedestrian distance |
| Systematic error budget | The localisation error that the chosen camera setup and uncertainty parameters introduce for the scenario being played |

## Suites and scenarios

The suite bar along the top switches between suites. Conformance holds the paper's 24 scenarios and is always present. Further suites (extended, stress) appear when `scenarios.extended.js` is present next to the page; without it the page runs with the conformance suite alone.

Selecting a scenario writes its slug into the URL fragment, so a link such as `simulator/#fast-approach` opens that scenario directly. Other pages of these docs use this to link to individual scenarios.

## Pipeline variants

The pipeline selector offers seven configurations. The first is the deployed configuration; the last three are the decision-rule baselines from the paper, run with the same front end.

| Variant | Memory (frames) | d_min (m) | d_max (m) | Min displacement (m) | Lookback (frames) | Speed-adaptive | Rule |
|---|---|---|---|---|---|---|---|
| Selected | 58 | 1.9 | 24.8 | 0.147 | 2 | yes | pairwise |
| Conservative | 60 | 1.5 | 15.0 | 0.05 | 3 | no | pairwise |
| Speed-Adaptive | 60 | 1.0 | 20.0 | 0.05 | 2 | yes | pairwise |
| Proximity Only | 0 | 2.0 | 10.0 | 0.1 | 3 | no | pairwise |
| Distance Only | 58 | 1.9 | 10.0 | 0.0 | 2 | no | distance_only |
| Naive Closing | 58 | 1.9 | 24.8 | 0.147 | 2 | yes | naive_closing |
| TTC | 58 | 1.9 | 24.8 | 0.0 | 2 | yes | ttc, threshold 3 s |

The values are written into the page and mirror `decision_testbench.PIPELINE_CONFIGS`. [Decision pipeline](../system/decision.md) explains each parameter and rule.

## Cameras

The setup buttons place cameras in four presets: single pole, cross-cam (a second pole at (5, 13) facing the first across the crosswalk), centre and overhead. The `+` and `−` buttons add and remove cameras, and the sliders set the selected camera's position (X, Y), height, field of view and heading. The default camera stands at (5, −13) m, 5.5 m high, with a 220° field of view facing the crosswalk.

## Uncertainty parameters

The uncertainty panel sets the detector and tracker model that turns ground truth into noisy observations:

| Group | Parameters |
|---|---|
| YOLO detection | Average recall and precision for small, medium and large objects (COCO size buckets), latency, high and low confidence thresholds |
| BEV projection | Ground tilt uncertainty (the street is not perfectly level) |
| ByteTrack Kalman filter | Position and velocity noise, frames before a lost track is dropped, IoU threshold, frame rate |

Changing them changes how often agents drop out and how far their projected positions wander, which is the quickest way to see why the decision stages need memory and a lookback.

## Relation to the testbench

The simulator is a JavaScript port written for inspection. The numbers in the paper and on these pages come from the Python [offline testbench](testbench.md), which shares its code with the deployed system. Use the simulator to understand a scenario and the testbench to measure one.

## Related

- [Simulators compared](index.md)
- [Conformance suite](../scenarios/conformance.md)
- [Uncertainty models](../system/uncertainty.md)
