# API reference

These pages are built from the docstrings of each module. Most modules are scripts at the repository root and are imported by their file name, so run Python from the repository root (or put it on `PYTHONPATH`) before importing them.

```python
from decision_pipeline import DecisionPipeline, stopping_distance
from decision_testbench import SCENARIOS, PIPELINE_CONFIGS, make_pipeline, simulate_scenario
```

<div class="grid-cards" markdown>

- [decision_pipeline](decision_pipeline.md)

    The three-stage decision module shared with the live system.

- [decision_testbench](decision_testbench.md)

    Scenarios, fisheye camera, detection and tracking models, metrics, optimizer.

- [sim_visualizer](sim_visualizer.md)

    Schematic scenario videos through the fisheye model.

- [generate_figures](generate_figures.md)

    Publication figures.

- [run_experiments](run_experiments.md)

    One-command reproduction of the paper.

- [run_height_pitch_sweep](run_height_pitch_sweep.md)

    Camera height and pitch sweep.

- [crosswalk_analysis](crosswalk_analysis.md)

    Road width and deployment geometry.

- [calibrate_fisheye](calibrate_fisheye.md)

    Fisheye calibration by bundle adjustment.

- [capture_calibration](capture_calibration.md)

    Checkerboard frame capture.

- [compare_fisheye_models](compare_fisheye_models.md)

    Projection model comparison.

- [eval_models](eval_models.md)

    Detector evaluation on fisheye-augmented COCO.

- [latency_report](latency_report.md)

    Per-step latency statistics.

- [bikeped.scenarios](bikeped_scenarios.md)

    Scenario registry, suites and generators.

- [bikeped.sim](bikeped_sim.md)

    Data types and imaging code of the CARLA renderer.

</div>
