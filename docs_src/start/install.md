# Install

The testbench, the figure scripts and the calibration tools need Python 3.10 or newer and five packages. This page covers that install, the optional pieces (CARLA client, bridge kit, docs toolchain) and a check that it worked.

## Core install

```bash
git clone https://github.com/mkturkcan/bikeped.git
cd bikeped
pip install -r requirements.txt
```

`requirements.txt` pins lower bounds only:

| Package | Minimum | Used for |
|---|---|---|
| `numpy` | 1.24 | Camera model, scenario interpolation, Monte Carlo. |
| `scipy` | 1.11 | Differential evolution in the optimizer, least-squares bundle adjustment in the calibration tools. |
| `matplotlib` | 3.7 | Publication figures and the latency sweep plot. |
| `pyyaml` | 6.0 | Reading `config.yaml`. |
| `opencv-python` | 4.8 | Scenario videos from `sim_visualizer.py`, checkerboard detection and frame capture. |

None of this needs a GPU. The testbench simulates the detector with a statistical model, so the scenarios run on a laptop CPU.

Run every script from the repository root. The scripts open `config.yaml` and `camera_calibration.json` by relative path, and the testbench falls back to built-in camera defaults with the message `No camera_calibration.json found, using defaults` when it is started somewhere else.

## Optional pieces

=== "CARLA on UE4"

    The renderer in `bikeped/sim` talks to a running CARLA server through the `carla` Python client. For the Unreal Engine 4 line, CARLA 0.9.14 to 0.9.16, client wheels exist for Python 3.10 to 3.12:

    ```bash
    pip install carla==0.9.16
    ```

    The client version has to match the server. `python -m bikeped.sim probe` prints both.

=== "CARLA on UE5"

    CARLA 0.10.0 ships a single client wheel, built for CPython 3.10, inside the server package:

    ```bash
    pip install Carla-0.10.0-Linux-Shipping/PythonAPI/carla/dist/carla-0.10.0-cp310-cp310-linux_x86_64.whl
    ```

    If you want to render on both engines from one environment, create it with Python 3.10. The two client versions cannot be installed side by side, so keep one virtual environment per engine.

=== "Bridge starter kit"

    The kit in `bridge_starter/` has its own requirements file, because it runs a real detector:

    ```bash
    cd bridge_starter
    pip install -r requirements.txt
    ```

    This adds `ultralytics` (the YOLO runtime), `huggingface_hub` (downloads the detector weights on first run) and `paho-mqtt` (track publishing). `play-sounds` for audio cues and TensorRT for engine export are installed separately and are both optional. See [the starter kit](../deployment/index.md).

=== "Docs site"

    ```bash
    pip install mkdocs-material "mkdocstrings[python]" pymdown-extensions
    python tools/build_docs.py --serve
    ```

    The second command builds this site and serves a live preview.

`carla` is imported lazily. Every module in the repository imports without it, and only the commands that connect to a server need it.

## Check the install

```console
$ python decision_testbench.py
Loaded calibration: FOV=197.90 cx=1752.7 cy=1804.5 f=1013.3 R=1750 frame=3500x3500 pad=3840 pitch=0 k1=0 k2=0
Running all scenarios with Default pipeline...
```

The first line confirms that `camera_calibration.json` was found and read. A table with one row per scenario follows, and the run ends with the totals over all 24 scenarios:

```text
  TOTAL                        1209  361  107 4427 91.9% 92.5%  8.4%

  Alert fatigue (mean fraction of time in alert): 31.9%
```

The columns are true positives, false positives, false negatives, true negatives (in frames), sensitivity, specificity and severity-weighted false-negative rate. If your totals match, the install is complete. The [quickstart](quickstart.md) explains the table and why these totals differ from the headline numbers in the paper.

## Related

- [Quickstart](quickstart.md)
- [Command line reference](../reference/cli.md)
- [CARLA overview](../simulation/carla/index.md)
