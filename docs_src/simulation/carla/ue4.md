# CARLA on Unreal Engine 4

This page sets up a CARLA 0.9.14, 0.9.15 or 0.9.16 server and its Python client for `bikeped.sim` and runs a first render. The UE4 line carries the full catalogue of two-wheelers and walkers, so it is the one to use when the agent models matter.

## What you need

| Item | Value |
|---|---|
| CARLA release | 0.9.14, 0.9.15 or 0.9.16 |
| Launcher script | `CarlaUE4.sh`, at the top of the extracted release |
| Python client | `carla` wheel of the same release as the server, available for Python 3.10 to 3.12 |
| Install root variable | `CARLA_UE4_ROOT` |
| Maps | Town01 to Town15, with `Town10HD_Opt` as the default here |

## Install the client

The client wheel has to come from the same CARLA release as the server.

```bash
pip install carla==0.9.16
```

`bikeped.sim` imports `carla` only when a command talks to a server, so the package imports, and its tests run, on a machine without the wheel.

## Start the server

```bash
export CARLA_UE4_ROOT=/path/to/CARLA_0.9.16
python -m bikeped.sim server --engine ue4
```

| Flag | Default | Meaning |
|---|---|---|
| `--root PATH` | `$CARLA_UE4_ROOT` | Folder of the extracted CARLA release |
| `--port` | `2000` | RPC port. The server also uses the two ports after it for streaming |
| `--quality` | `Epic` | `Epic` or `Low`, passed to the launcher as `-quality-level=` |
| `--windowed` | off | Open a window. Without it the server renders off screen |

With `--root` the command looks only there. Without it, it tries `CARLA_UE4_ROOT` and then a short list of common install paths. It refuses to start when something already listens on the port, runs the launcher in its own process group, and waits until a client can read the server version. The server's output is written to `carla_ue4_<port>.log` in the system temporary folder.

The launcher call it builds is equivalent to:

```bash
cd "$CARLA_UE4_ROOT"
./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000 -quality-level=Epic
```

## Probe the server

```bash
python -m bikeped.sim probe --host 127.0.0.1 --port 2000
```

The probe connects, reads the client and server versions, and reports the engine generation, the maps the server can load and the capabilities the renderer depends on. Run it once after installing: a version mismatch or a missing map shows up here, before a render starts.

## First render

```bash
python -m bikeped.sim render --scenario fast-approach --engine ue4 \
    --rig pole,fisheye --weather clear_noon
```

The run directory is `renders/ue4_fast-approach_clear_noon/`. Its contents are described in [Run output](output.md).

## Notes for this line

Wheelchair walkers exist on 0.9.16, so that is the release to use for the accessibility scenarios. The probe reports whether the connected server has them. See [Actors](actors.md) for how scenario models map to blueprints.

`crosswalks.json` holds the 16 crosswalks of `Town10HD_Opt`. To render on another town, extract its crosswalks first with `tools/carla_find_crosswalks.py`, as described in [Environment](environment.md#extracting-crosswalks-from-another-map).

`carla_scenario.py` in the repository root still accepts the options of the single-file runner it replaced and forwards them to `python -m bikeped.sim render`.

## Related

- [UE5 setup](ue5.md)
- [Troubleshooting](troubleshooting.md)
- [CLI reference](../../reference/cli.md)
