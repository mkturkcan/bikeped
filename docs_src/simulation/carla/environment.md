# Environment

This page covers where a scenario is placed in the CARLA world and under which conditions it is rendered: map, crosswalk, weather, ambient traffic, seed and frame rate.

## Map

`--map` loads a map before the render if the server has a different one open. Without it the renderer keeps whatever map is loaded, which on a freshly started server is `Town10HD_Opt` on both engine lines.

## Crosswalks

Scenarios are written in the [scenario frame](../../scenarios/index.md): metres, `x` along the road, `y` across it, crosswalk centre at `(5, 0)`. To place a scenario in a town, the renderer anchors that frame on a real crosswalk taken from `crosswalks.json` in the repository root. The file lists the 16 crosswalks of `Town10HD_Opt`, and `--crosswalk` picks one by index. The default is 4. On any other map the renderer reads the crosswalk polygons from the live map instead (`world.get_map().get_crosswalks()`), and `--crosswalks-file` points at another file in the same format.

Each entry has these fields:

| Field | Unit | Meaning |
|---|---|---|
| `id` | | Index used by `--crosswalk` |
| `center` | m | `[x, y, z]` of the crosswalk centre in the CARLA world frame |
| `corners` | m | Four `[x, y, z]` corners of the crosswalk polygon |
| `width_m` | m | Length of the short side, measured along the road |
| `length_m` | m | Length of the long side, the distance a pedestrian crosses |
| `road_heading_deg` | degrees | Heading of the road at the crosswalk |

The crosswalk centre becomes scenario `(5, 0)`. Scenario `+y` runs along the crosswalk, the direction pedestrians walk, and scenario `+x` runs along the road, the direction cyclists ride. The pole camera stands where the testbench puts it, at scenario `(5, -13)` and 3.6576 m (12 ft) high, read from `decision_testbench.FisheyeCamera`.

??? info "The 16 crosswalks of Town10HD_Opt"

    | id | Centre (x, y) | `width_m` | `length_m` | `road_heading_deg` |
    |---:|---|---:|---:|---:|
    | 0 | (-46.9, 118.5) | 2.5 | 18.9 | -179.7 |
    | 1 | (42.2, 58.5) | 3.1 | 17.8 | 179.5 |
    | 2 | (-46.8, -47.9) | 2.6 | 19.4 | 0.2 |
    | 3 | (-31.2, -62.9) | 2.6 | 18.7 | -90.1 |
    | 4 | (-46.9, 36.5) | 2.9 | 22.2 | 0.1 |
    | 5 | (-47.3, 3.0) | 2.6 | 20.6 | -179.7 |
    | 6 | (-62.7, -63.3) | 2.5 | 18.3 | 91.0 |
    | 7 | (104.4, 2.0) | 2.5 | 18.7 | -179.1 |
    | 8 | (87.5, 20.8) | 2.6 | 24.3 | 90.6 |
    | 9 | (103.8, 37.4) | 2.8 | 19.8 | 0.0 |
    | 10 | (91.3, 68.1) | 3.0 | 17.0 | 90.2 |
    | 11 | (-93.1, 20.3) | 2.6 | 25.7 | -89.5 |
    | 12 | (-108.7, 39.0) | 2.5 | 18.9 | 0.4 |
    | 13 | (-107.7, 3.5) | 2.9 | 20.3 | -179.3 |
    | 14 | (42.2, 35.1) | 3.5 | 20.3 | -0.3 |
    | 15 | (-30.6, 67.9) | 2.5 | 13.0 | -90.5 |

    Values are rounded from `crosswalks.json`.

### Extracting crosswalks from another map

`tools/carla_find_crosswalks.py` reads the crosswalk polygons of the map a running server has loaded and writes them in the same format:

```bash
python tools/carla_find_crosswalks.py --map Town10HD_Opt --export crosswalks.json
python tools/carla_find_crosswalks.py --goto 4        # move the spectator to crosswalk 4
```

Use `--goto` to look at a candidate in the simulator window before rendering on it.

## Weather

`--weather` takes one of nine presets. A scenario can also carry a hint in its `render` block, for example `'render': {'weather': 'wet_night'}`, which is used when the flag is not given.

```console
$ python -m bikeped.sim presets
weather
  clear_noon   sun    60 deg, clouds 10, rain 0, fog 0
  golden_hour  sun     9 deg, clouds 20, rain 0, fog 3
  overcast     sun    45 deg, clouds 90, rain 0, fog 5
  light_rain   sun    35 deg, clouds 80, rain 30, fog 8
  heavy_rain   sun    30 deg, clouds 100, rain 90, fog 20
  dusk         sun    -3 deg, clouds 40, rain 0, fog 5, street lights on
  clear_night  sun   -40 deg, clouds 10, rain 0, fog 0, street lights on
  wet_night    sun   -40 deg, clouds 70, rain 10, fog 6, street lights on
  fog          sun    25 deg, clouds 60, rain 0, fog 45
```

The sun angle is the altitude. The sun's azimuth is set relative to the pole camera's viewing direction rather than as an absolute angle, so the pole view is lit from behind or from the side whatever the orientation of the crosswalk. Fields that a server does not know are skipped, so the same presets serve CARLA 0.9.x and 0.10.0. Without `--weather` and without a hint the preset is `clear_noon`.

The weather name is the last part of the run id, so renders of one scenario under different conditions land in separate directories. `python -m bikeped.sim presets` prints the presets with their parameters.

## Ambient traffic

`--ambient` adds background vehicles and walkers that are not part of the scenario. The table gives the number of actors requested per level.

| Level | Vehicles | Walkers |
|---|---:|---:|
| `none` | 0 | 0 |
| `light` | 6 | 8 |
| `busy` | 20 | 24 |

Spawning is best effort: a map without spawn points or a full navigation mesh leaves fewer ambient actors and a log message, and never stops the render. Ambient actors are kept out of the scripted corridor, which is the bounding box of all scripted paths plus a margin. Vehicles spawn outside it, drive under CARLA's Traffic Manager, and are stopped when they approach the corridor. Walkers spawn on the navigation mesh outside the corridor, do not cross the road, and are sent back to where they started if they wander in. The street looks inhabited and the scripted encounter plays out undisturbed.

## Seed and frame rate

`--seed` (default `0`) fixes every random choice of a run, such as which walker blueprint an agent gets and where ambient traffic spawns. `--fps` (default `30`) sets both the video frame rate and the simulation step: the world runs in synchronous mode with a fixed step of `1 / fps` seconds, so a render takes as long as it takes and the result does not depend on how fast the machine is.

## Related

- [Actors](actors.md)
- [Scenario format](../../reference/formats.md)
- [CLI reference](../../reference/cli.md)
