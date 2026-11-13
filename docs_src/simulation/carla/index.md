# CARLA renderer

`bikeped.sim` plays a scenario from the [scenario library](../../scenarios/index.md) in a CARLA world and records what a camera at the intersection would see, together with the decision state and ground-truth boxes for every frame. One command line drives both engine generations: Unreal Engine 4 (CARLA 0.9.14 to 0.9.16) and Unreal Engine 5 (CARLA 0.10.0).

## How a render runs

The runner connects to a CARLA server and puts the world in synchronous mode with a fixed step of `1 / fps` seconds, so simulated time advances only when the runner ticks. It maps the scenario frame onto a crosswalk from `crosswalks.json`, spawns one actor per scripted agent and one CARLA sensor per camera view, then steps through the scenario. On every tick it moves the agents along their scripted paths, reads back the simulated poses and images, evaluates the decision pipeline and the ground-truth danger label, and passes a `FrameRecord` to the `Recorder`. When the scenario ends, or the run fails, the session destroys the actors it spawned and restores the world settings.

<figure class="diagram">
--8<-- "assets/figures/carla-dataflow.svg"
<figcaption>One render: the runner drives the server tick by tick and the recorder writes the run directory.</figcaption>
</figure>

The decision pipeline and the danger model read the scripted positions (`sx`, `sy` in the scenario frame), so the state shown in the overlay does not depend on how closely a simulated actor followed its path. The gap between the scripted and the simulated position is stored for every agent as `track_error_m`.

## UE4 and UE5

| | UE4 | UE5 |
|---|---|---|
| CARLA releases | 0.9.14 to 0.9.16 | 0.10.0 |
| Launcher script | `CarlaUE4.sh` | `CarlaUnreal.sh` |
| Python client wheels | Python 3.10 to 3.12 | `cp310` only (Python 3.10) |
| Rendering | Unreal Engine 4 | Unreal Engine 5 with Lumen and Nanite |
| Maps | Town01 to Town15, `Town10HD_Opt` by default | `Town10HD_Opt` and `Mine_01` |
| Actor catalogue | Full set of two-wheelers and walkers, wheelchair walkers on 0.9.16 | Four-wheeled vehicles and walkers; no rideable two-wheeler, cyclists need `--cyclist-proxy walker` |
| Validation | Developed and run end to end against CARLA 0.9.16 | Tested against an in-memory stand-in for the 0.10.0 API; not yet run against a real 0.10.0 server ([current limits](ue5.md#current-limits)) |
| Install root variable | `CARLA_UE4_ROOT` | `CARLA_UE5_ROOT` |
| `--engine` value | `ue4` | `ue5` |

The two lines differ in what they can spawn, so the renderer asks the server instead of assuming. `python -m bikeped.sim probe` reports the engine, the versions, the available maps and the capabilities the actor code depends on. Setup for each line is on its own page: [UE4](ue4.md) and [UE5](ue5.md).

## A minimal session

```bash
export CARLA_UE4_ROOT=/path/to/CARLA_0.9.16   # directory that holds CarlaUE4.sh

python -m bikeped.sim server --engine ue4     # terminal 1, keeps running
python -m bikeped.sim probe                   # terminal 2, check the connection
python -m bikeped.sim render --scenario fast-approach --rig pole,fisheye
```

The render writes `renders/ue4_fast-approach_clear_noon/`: a manifest, one video per view, per-frame ground truth and a few key stills. The layout is described in [Run output](output.md).

## Commands

| Command | What it does |
|---|---|
| `python -m bikeped.sim probe` | Connects to a server and prints engine, version, maps and capabilities |
| `python -m bikeped.sim server` | Starts a CARLA server from `CARLA_UE4_ROOT` or `CARLA_UE5_ROOT` |
| `python -m bikeped.sim presets` | Lists the weather, rig and quality presets |
| `python -m bikeped.sim render` | Renders one scenario, a suite, or every scenario |
| `python -m bikeped.sim export` | Converts the ground truth of a finished run to COCO or YOLO |

Every flag is listed in the [CLI reference](../../reference/cli.md).

## In this section

<div class="grid-cards" markdown>

- [UE4 setup](ue4.md)

    CARLA 0.9.14 to 0.9.16 with the full two-wheeler and walker catalogue.

- [UE5 setup](ue5.md)

    CARLA 0.10.0 with Lumen and Nanite, and what to check before rendering.

- [Cameras](cameras.md)

    The pole, fisheye, chase, orbit, street and top-down views, and how the fisheye is built.

- [Environment](environment.md)

    Map, crosswalk, weather, ambient traffic, seed and frame rate.

- [Actors](actors.md)

    How scripted agents become CARLA walkers and vehicles, and how they are driven.

- [Ground truth](ground-truth.md)

    Boxes, visibility and danger labels per frame, and export to COCO or YOLO.

- [Run output](output.md)

    The files of one run and the structure of `manifest.json`.

- [Troubleshooting](troubleshooting.md)

    Connection, version and blueprint problems and what to do about them.

</div>
