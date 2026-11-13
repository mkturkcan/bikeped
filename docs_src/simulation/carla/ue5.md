# CARLA on Unreal Engine 5

This page sets up a CARLA 0.10.0 server and its Python client for `bikeped.sim`, and lists what to check before a first render. The UE5 line renders with Lumen and Nanite and ships a smaller set of maps and vehicles than UE4.

!!! note "Validation status"
    The renderer is developed and run end to end against CARLA 0.9.16 on Unreal Engine 4. The Unreal Engine 5 path follows the CARLA 0.10.0 Python API and is covered by tests against an in-memory stand-in for that API; it has not yet been run against a real CARLA 0.10.0 server. See [Current limits](#current-limits).

## Requirements

| Item | Value |
|---|---|
| CARLA release | 0.10.0 |
| Launcher script | `CarlaUnreal.sh`, in the `Linux/` folder of the extracted release |
| Python client | `carla-0.10.0-cp310-cp310-linux_x86_64.whl`, shipped in the release under `PythonAPI/carla/dist/` |
| Python version | 3.10, the only version the wheel supports |
| Install root variable | `CARLA_UE5_ROOT` |
| Maps | `Town10HD_Opt` and `Mine_01` |

## Install the client

The 0.10.0 wheel is built for CPython 3.10 only. The rest of the repository runs on Python 3.10 or newer, so one 3.10 environment covers the testbench and both renderers.

```bash
export CARLA_UE5_ROOT=/path/to/Carla-0.10.0-Linux-Shipping

python3.10 -m venv .venv-ue5
source .venv-ue5/bin/activate
pip install -r requirements.txt
pip install "$CARLA_UE5_ROOT/PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl"
```

## Start the server

```bash
python -m bikeped.sim server --engine ue5
```

| Flag | Default | Meaning |
|---|---|---|
| `--root PATH` | `$CARLA_UE5_ROOT` | Folder of the extracted CARLA release |
| `--port` | `2000` | RPC port. The server also uses the two ports after it for streaming |
| `--quality` | `Epic` | Accepted and ignored: the UE5 launcher has no quality flag |
| `--windowed` | off | Open a window. Without it the server renders off screen |

The launcher is looked up at the top of the root folder and under `Linux/`, where the 0.10.0 archive keeps it, so `CARLA_UE5_ROOT` can point at either. As on UE4, the command refuses to start when the port is taken, waits until a client can read the server version, and writes the server's output to `carla_ue5_<port>.log` in the system temporary folder.

The launcher call it builds is equivalent to:

```bash
cd "$CARLA_UE5_ROOT/Linux"
./CarlaUnreal.sh -RenderOffScreen -carla-rpc-port=2000
```

## Probe before rendering

```bash
python -m bikeped.sim probe
```

The probe reads the server's blueprint library at connect time and prints the engine, the versions, the maps and the capabilities the actor code depends on, including whether a rideable two-wheeler exists. On a CARLA 0.10.0 server that check is what decides whether scenarios with cyclists can render at all (see below). [Actors](actors.md) describes how scenario models map to blueprints.

## Current limits

CARLA 0.10.0 ships no rideable bicycle or motorcycle blueprint: its catalogue is four-wheeled vehicles plus walkers. On UE5, `render` therefore refuses a scenario that contains cyclists unless you pass `--cyclist-proxy walker`. With that flag a walker stands in for each cyclist, moving along the cyclist's scripted path, and the manifest records the substitution, so a dataset built from the run can tell proxies from real two-wheelers. Pedestrian-only scenarios render as on UE4.

The decision state and the danger label do not change under the proxy, because both read the scripted positions rather than the simulated actor.

## First render

```bash
python -m bikeped.sim render --scenario fast-approach --engine ue5 --rig pole,fisheye --cyclist-proxy walker
```

The run directory is `renders/ue5_fast-approach_clear_noon/`. The engine is part of the run id, so UE4 and UE5 renders of the same scenario sit next to each other and can be compared view by view.

## Notes for this line

The default map and crosswalk carry over: `Town10HD_Opt` exists on both lines and `crosswalks.json` was extracted from it. `Mine_01` has no entry in `crosswalks.json`.

The fisheye view renders one camera per cube face where a pinhole view renders one camera in total. Use `--quality draft` while checking a setup and switch to `standard` or `cinematic` for the final pass. See [Cameras](cameras.md#quality-presets).

## Related

- [UE4 setup](ue4.md)
- [Troubleshooting](troubleshooting.md)
- [CLI reference](../../reference/cli.md)
