# bikeped bridge — adaptation starter kit

A slim, self-contained version of the deployed bikeped collision-warning
bridge, designed for teams adapting the system to a new intersection or
research environment.

What you get:

- **One file** ([bridge.py](bridge.py)) with a clearly-marked **USER CONFIG** block at the top.
  Everything you'll typically need to change (camera intrinsics, MQTT
  broker, sounds, model name) lives in the first ~80 lines.
- **No hardware dependencies.** The original system drove a USB busylight;
  this kit ships a `SimulatedLight` class that prints colour transitions
  to stdout. Subclass it to wire up real hardware later.
- **No mandatory audio.** Sound playback is optional and skipped silently
  if `play-sounds` isn't installed or the WAV files are missing.
- **Auto-managed model.** First run downloads
  [`yolo11xxl.pt`](https://huggingface.co/mehmetkeremturkcan/YOLO11xxl)
  from Hugging Face, then best-effort exports it to a TensorRT engine.
  Falls back to PyTorch with a clear warning if export fails.
- **Same latency profiler** as the production system — output is
  compatible with the workspace's `latency_report.py`.

The kit is one folder; copy it anywhere and run.

## Install

Python 3.10+ recommended. CUDA-capable GPU strongly recommended for
real-time throughput.

```bash
pip install -r requirements.txt
```

Optional extras:

```bash
pip install play-sounds          # enable audio cues
pip install tensorrt              # for engine export (platform-specific)
```

## Configure

Open [bridge.py](bridge.py) and edit the **USER CONFIG** block. The fields
you'll almost certainly want to set:

| Field                 | What                                            |
|-----------------------|-------------------------------------------------|
| `USE_WEBCAM` / `VIDEO_FILE` | Live webcam vs. an MP4 file               |
| `CAMERA_HEIGHT_M`, `CAMERA_FOV_DEG`, `CAMERA_RADIUS_PX`, `CAMERA_PITCH_DEG`, `CAMERA_CX`, `CAMERA_CY` | Your fisheye intrinsics (run `calibrate_fisheye.py` from the main repo to obtain these) |
| `MQTT_BROKER`, `MQTT_PORT`, `MQTT_TOPIC` | Where to publish track messages |
| `ALERT_SOUND`, `SAFE_SOUND` | Optional WAV paths (silent if missing)     |
| `YOLO_HF_REPO` / `YOLO_HF_FILE` | Override the default detector model       |

The decision-pipeline parameters (`DEC_*`) ship with the optimizer-derived
values from the paper. Tune for your road geometry / cyclist speed mix as
needed.

## Run

```bash
python bridge.py                  # honour USER CONFIG defaults
python bridge.py --video clip.mp4 # quick offline test on a video file
python bridge.py --no-mqtt        # local-only, no broker
python bridge.py --no-window      # headless (e.g. for systemd units)
```

On a fresh checkout the first invocation prints something like:

```
[model] yolo11xxl.pt not found locally — downloading from huggingface.co/mehmetkeremturkcan/YOLO11xxl
[model] saved to models/yolo11xxl.pt
[model] no TensorRT engine found at models/yolo11xxl.engine
[model] attempting one-time export yolo11xxl.pt -> yolo11xxl.engine (this can take a few minutes) ...
[model] export OK: models/yolo11xxl.engine
[bev] building ground LUT (3500x3500) ...
[bev] valid pixels: 9621123/12250000
[mqtt] connecting to mqtt.example.com:1883 ...
======================================================================
 Source: WEBCAM
 Model:  models/yolo11xxl.engine
 MQTT:   enabled (mqtt.example.com)
 Window: on  (q to quit)
======================================================================
[light] BLUE
[light] GREEN
[light] RED
...
```

Each colour transition fires once, mirroring what a real indicator light
would show.

## Pipeline

```
capture  ->  YOLO detect  ->  fisheye -> ground LUT  ->  GroundPlaneTracker
                                                              |
              SimulatedLight + (optional) sound cue  <----  DecisionPipeline
                                                              |
                                                          MQTT publish
```

`DecisionPipeline` lives in [decision_pipeline.py](decision_pipeline.py)
and is shared verbatim with the main repo / paper. Modifying decision
behaviour upstream automatically flows here.

## Tuning the proximity floor (rider-on-bike confusion)

YOLO almost always emits a separate `person` box for the cyclist's torso
on top of the `bicycle` / `motorcycle` box. Without filtering, the
cyclist's body is ~0.3–0.7 m from their own bike, which would otherwise
trip the proximity check on every cyclist that enters the scene.

The kit handles this with two layers of defence — both worth tuning for
your detector and camera geometry:

1. **Rider-overlap suppression.** [`find_rider_indices()`](bridge.py)
   tags `person` boxes whose 2D box overlaps a bike's 2D box and the
   main loop relabels them as `bike` before they hit the tracker. The
   thresholds live in `RIDER_X_OVERLAP_FRAC` and `RIDER_Y_OVERLAP_FRAC`
   in the USER CONFIG block.
2. **Hard distance floor.** `DEC_PROX_MIN_M` is a lower bound on the
   bike↔ped distance the decision pipeline will ever alert on. Anything
   closer is treated as rider self-overlap that slipped past layer 1.

**When to raise `DEC_PROX_MIN_M`** — if you see spurious alerts from
solo cyclists (no nearby pedestrians) or your alert rate is unexpectedly
high in cyclist-heavy scenes:

| Symptom | Likely cause | Fix |
|---|---|---|
| Cyclist self-trips alert at low height | Person bbox bottom ≠ bike bbox bottom under barrel distortion | Raise `DEC_PROX_MIN_M` to 2.5–3.0 m |
| A walking adult next to a parked bike trips alert | Real pedestrian at <1.9 m from a bike | Lower `DEC_PROX_MIN_M`, then check if rider suppression mis-fired (look for a person being relabelled to bike when they shouldn't) |
| Group cycling never triggers | Rider suppression eats real pedestrians next to bikes | Reduce `RIDER_X_OVERLAP_FRAC` (e.g. 0.4) or `RIDER_Y_OVERLAP_FRAC` (e.g. 0.15) |

The paper's value of 1.9 m worked for the wide-angle ceiling-mounted
deployment. With a low pole-mounted camera or a narrower lens you'll
likely need 2.5–3.0 m because the bbox bottom-centre projection error
grows.

## Wiring up real hardware

Replace `SimulatedLight` in [bridge.py](bridge.py) with your own class.
The interface is one method:

```python
class MyHardwareLight:
    def set_color(self, rgb):  # rgb is a 3-tuple in 0..255
        ...
    on = set_color             # busylight-compatible alias
```

Then `light = MyHardwareLight()`. The rest of the pipeline doesn't care.

## Latency profiling

Set `LATENCY_ENABLED = True` (default) and the script writes
`latency_data.npy` periodically. Generate a stats report with
`latency_report.py` from the main repo:

```bash
python ../latency_report.py latency_data.npy
```

You'll get per-step mean / median / p95 / p99 in milliseconds, suitable
for budget tracking.

## Citation

If this kit informs published work, please cite the bikeped paper:

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
