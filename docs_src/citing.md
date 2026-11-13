# Citing

How to cite the paper, the dataset and the starter kit. All three use the same reference.

## Paper

Mehmet Kerem Turkcan. *A Real-Time Bike-Pedestrian Safety System with Wide-Angle Perception and Evaluation Testbed for Urban Intersections.* arXiv:2604.17046 [cs.CV], submitted 18 April 2026. <https://arxiv.org/abs/2604.17046>

```bibtex
@misc{turkcan2026realtimebikepedestriansafetywideangle,
  title         = {A Real-Time Bike-Pedestrian Safety System with Wide-Angle Perception and Evaluation Testbed for Urban Intersections},
  author        = {Mehmet Kerem Turkcan},
  year          = {2026},
  eprint        = {2604.17046},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2604.17046}
}
```

## Abstract

> Collisions between cyclists and pedestrians at urban intersections remain a persistent source of injuries, yet few systems attempt real-time warnings to unequipped road users using commodity hardware. We present a prototype collision warning system that runs on a single edge device with a wide-angle fisheye camera, producing audible and visual alerts at 30 fps. The system makes four contributions. First, we develop a calibration pipeline for ultra-wide fisheye lenses that overcomes corner-detection failure and optimizer divergence through perspective remapping and direct bundle adjustment. Second, we combine fisheye-aware object detection with a closed-form ground-plane projection via a precomputed lookup table. Third, we introduce a design-time conformance simulation with 24 scripted hazard scenarios, stochastic size-aware detection failures, and a latency sweep showing that a first-order kinematic predictor maintains the mean warning budget above the distracted-pedestrian reaction time across realistic camera latencies. Fourth, we formalize the decision layer as a separable, auditable testbench with explicit deployment gates, contestability mechanisms, and a residual risk register. Under conformance testing with fisheye localization error, the selected pipeline configuration achieves 93.3% sensitivity and 92.3% specificity, with a mean warning budget of 3.3 s. The system design was informed by community-aided design workshops. Code and replication scripts are available at https://github.com/mkturkcan/bikeped.

Where each contribution lives in this repository:

| Contribution | Code | Documentation |
|---|---|---|
| Fisheye calibration by perspective remapping and bundle adjustment | `calibration/` | [Calibration](calibration/index.md) |
| Ground-plane projection | `FisheyeCamera` in `decision_testbench.py` | [Perception](system/perception.md) |
| Conformance simulation, detection failures, latency sweep | `decision_testbench.py` | [Testbench](simulation/testbench.md), [uncertainty](system/uncertainty.md), [latency](experiments/latency.md) |
| Decision layer | `decision_pipeline.py` | [Decision pipeline](system/decision.md) |

The sensitivity and specificity in the abstract are the `default` row of `python decision_testbench.py --compare`. The [quickstart](start/quickstart.md#compare-decision-rules) shows that run.

## Dataset

The scenario dataset is at <https://huggingface.co/datasets/mehmetkeremturkcan/bikeped> under CC BY 4.0. Its card asks for the paper citation above. See [dataset](dataset.md) for the contents.

## Starter kit

If the [bridge starter kit](deployment/index.md) informs published work, cite the paper with the same entry.
