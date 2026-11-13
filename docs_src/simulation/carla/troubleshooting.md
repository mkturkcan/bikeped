# Troubleshooting

Problems that come up when starting a CARLA server, connecting to it or rendering a scenario, with the message the tools print and what to do about it. For anything on the client side, start with `python -m bikeped.sim probe`.

## Starting the server

### Launcher not found

```text
CarlaUnreal.sh not found in /data/carla or /data/carla/Linux. Point --root or CARLA_UE5_ROOT at the folder of the extracted CARLA release.
```

`--root` was given and the launcher is in neither place. Without `--root` the message reads `No CARLA ue5 install found` and lists every path that was tried: the environment variable first, then the common install paths. The launcher is `CarlaUE4.sh` for `--engine ue4` and `CarlaUnreal.sh` for `--engine ue5`, so a UE5 folder passed with `--engine ue4` fails the same way.

### Port already in use

```text
port 2000 on 127.0.0.1 is already in use; stop the server that listens there or choose another --port
```

Another server, often one left from an earlier session, is listening. Stop it, or start the new one elsewhere and pass the same port to `probe` and `render`. A server uses the RPC port and the two ports after it, so keep two servers at least three ports apart.

```bash
python -m bikeped.sim server --engine ue5 --port 2010
python -m bikeped.sim probe --port 2010
```

### Server exits during start-up

```text
the CARLA server exited with code 1 before it accepted connections; its output is in /tmp/carla_ue4_2000.log
```

The message is followed by the last 20 lines of the server's output. The full log is in the file it names, `carla_<engine>_<port>.log` in the system temporary folder.

### No answer from the server

```text
no answer from the CARLA server on 127.0.0.1:2000 after 180 s
```

The process is alive and no client could read its version within the timeout. The message includes the last client error and the path of the log file.

## Connecting

### Missing `carla` module

The Python client is not installed in the environment that runs the command. Install the wheel that matches the server release:

=== "UE4"

    ```bash
    pip install carla==0.9.16
    ```

=== "UE5"

    ```bash
    pip install "$CARLA_UE5_ROOT/PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl"
    ```

The 0.10.0 wheel is `cp310` only. In an environment with a newer Python, pip rejects it as not supported on the platform. Create a Python 3.10 environment for UE5 work.

### Client and server version mismatch

The client wheel and the server have to come from the same CARLA release. The probe prints both versions, and each manifest stores them as `client_version` and `server_version`. Install the wheel of the server's release.

## Rendering

### Server unreachable

```text
no CARLA server at 127.0.0.1:2000: ...
```

`render` and `probe` connect to `--host` and `--port` (default `127.0.0.1:2000`). Start a server first, or pass `--launch` to `render` so it starts a managed server for the run and stops it afterwards.

### Engine mismatch

```text
--engine ue5 requested but the server is ue4
```

`--engine` on `render` states which engine you expect; the default `auto` accepts whatever the server runs. Connect to the other server or drop the flag.

### No rideable two-wheeler

```text
ue5 server 0.10.0 has no rideable two-wheeler blueprint, so cyclists cannot be rendered. Pass --cyclist-proxy walker to render them as walkers.
```

The server has no rideable two-wheeler, which is the case for CARLA 0.10.0. Pass `--cyclist-proxy walker`, or render the scenario on a UE4 server. See [UE5 current limits](ue5.md#current-limits).

### Sensor timeout

```text
sensor 'fisheye/front' delivered nothing for frame ... within ... s; the server may be overloaded (try --quality draft or fewer views)
```

Each fisheye view runs one camera per cube face, so a run with several views can exceed what the GPU renders within the timeout. Lower `--quality`, render fewer views per run, or close other GPU work.

### Crosswalks on other maps

`crosswalks.json` describes `Town10HD_Opt`. On another map the renderer reads crosswalk polygons from the live map, and `--crosswalk` indexes those instead; use `--crosswalks-file` for a file extracted with `tools/carla_find_crosswalks.py` ([Environment](environment.md#extracting-crosswalks-from-another-map)). The probe lists the maps the server can load.

### Unexpected blueprint

The server lacks the preferred blueprint for that model and the renderer used the next one on its list, as described in [Actors](actors.md#two-wheelers). Wheelchair walkers exist from CARLA 0.9.16; before that a wheelchair agent is an adult walker. The `blueprint` field of each agent in `manifest.json` names what was spawned.

## Related

- [UE4 setup](ue4.md)
- [UE5 setup](ue5.md)
- [CLI reference](../../reference/cli.md)
