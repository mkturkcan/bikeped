# Actors

This page describes how the agents of a scenario become CARLA actors, how they follow their scripted paths, and what happens when the server lacks a blueprint. The code is in `bikeped/sim/actors.py`.

## From agent to actor

A scenario agent has a `type`, which is what the decision pipeline uses, and an optional `model`, which only changes how it looks. A `person` becomes a walker. A `bike` becomes a two-wheeled vehicle with a rider, or a walker standing in for the cyclist on engines without one (see [below](#engines-without-a-rideable-two-wheeler)).

```python
{'type': 'bike', 'id': 2, 'model': 'ebike', 'label': 'e-bike',
 'path': [{'x': 34, 'y': 2, 't': 0}, {'x': 5, 'y': 0, 't': 4}]}
```

### Walkers

Walker blueprints are the server's `walker.pedestrian.*` blueprints, sorted by id, and one is drawn with the run's seed.

| `model` | Blueprint pool |
|---|---|
| `adult` (default), `runner` | Walkers whose `age` attribute is not `child` |
| `child` | Walkers with `age` = `child`, or the adult pool if there are none |
| `wheelchair` | Walkers with `can_use_wheelchair` = `true`, spawned in a wheelchair; the adult pool if the server has none (CARLA 0.9.16 has them) |

### Two-wheelers

For a `bike` agent the renderer keeps the vehicle blueprints the server reports as two-wheelers and takes the first entry of the model's preference list that exists:

| `model` | Preferred blueprints, in order |
|---|---|
| `bicycle` (default) | `vehicle.bh.crossbike`, `vehicle.gazelle.omafiets`, `vehicle.diamondback.century` |
| `road_bike` | `vehicle.diamondback.century`, `vehicle.bh.crossbike` |
| `ebike` | `vehicle.bh.crossbike`, `vehicle.gazelle.omafiets` |
| `cargo_bike` | `vehicle.gazelle.omafiets`, `vehicle.bh.crossbike` |
| `escooter` | `vehicle.vespa.zx125`, `vehicle.bh.crossbike` |
| `moped` | `vehicle.vespa.zx125`, `vehicle.yamaha.yzf`, `vehicle.kawasaki.ninja` |

If none of them exists, it picks a random two-wheeler whose `base_type` is `bicycle`, and failing that any two-wheeler. The blueprint that was spawned is stored per agent in the manifest as `blueprint`, next to the requested `model`, so a dataset built from renders can be filtered by what is actually in the image.

The model changes pixels only. The decision pipeline treats every `bike` alike, and the braking profile of the danger label comes from the scenario's `vehicle` key ([Safety model](../../system/safety-model.md)).

## Drive modes

`--drive` selects how actors follow their paths. The default is `kinematic`.

| Mode | How the actor moves |
|---|---|
| `kinematic` | The actor is placed on its scripted pose every tick. Height comes from a ground profile sampled before any actor exists, yaw from the smoothed heading and, for two-wheelers, roll from the lean angle. Walkers also get a `WalkerControl` so the walk cycle plays. |
| `velocity` | Walkers are steered with `WalkerControl` (the scripted velocity plus a proportional correction on position error). Riders get target linear and angular velocities plus a light throttle and steer, so wheels, pedals and handlebars move. An actor is put back on its scripted pose only when its position error exceeds a tolerance. |

Under `velocity` the simulated actor can drift from the script, for example where a path turns sharply. The runner stores that drift per agent and frame as `track_error_m`, the ground-plane distance in metres between the scripted and the simulated position.

Positions reported to the decision pipeline and to the danger model always come from the script, in both modes. Drift moves the actor in the image, and therefore the ground-truth boxes, which describe the simulated actor.

## Engines without a rideable two-wheeler

CARLA 0.10.0 ships no rideable bicycle or motorcycle blueprint. On such a server `render` refuses a scenario with `bike` agents unless `--cyclist-proxy walker` is passed. With the flag, a walker follows each cyclist's scripted path and the manifest records the substitution. Pedestrian-only scenarios are unaffected. `python -m bikeped.sim probe` shows in advance whether the server has a rideable two-wheeler; [UE5 setup](ue5.md#current-limits) has the details.

## Ambient actors

Actors added by `--ambient` carry `role: "ambient"` and have no scenario path; the simulator drives them. See [Environment](environment.md#ambient-traffic).

## Related

- [Scenario format](../../reference/formats.md)
- [Ground truth](ground-truth.md)
- [Troubleshooting](troubleshooting.md)
