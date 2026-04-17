#!/usr/bin/env python3
"""
Run the full height x pitch Monte Carlo sweep for the paper heatmap.

Heights: 1.5 to 7.5 m in 0.5 m steps (13 values)
Pitches: 0 to -90 deg in 10 deg steps (10 values)
Road width: 40 ft
Trials per cell: 20
Total: 130 cells x 20 trials = 2600 Monte Carlo runs

Output: height_pitch_sweep.json (loaded by generate_figures.py)

Runtime: ~30-60 minutes depending on hardware.
"""

import json
import numpy as np
from crosswalk_analysis import (
    run_mc, make_scenario, FEET_TO_METERS,
    load_uncertainty_from_cache, UncertaintyParams,
)

ROAD_FT = 40
HEIGHTS = [round(h, 1) for h in np.arange(1.5, 8.0, 0.5)]
PITCHES = list(range(0, -91, -10))
N_TRIALS = 20


def main():
    width_m = ROAD_FT * FEET_TO_METERS
    scenario = make_scenario(width_m)

    unc = load_uncertainty_from_cache('eval_cache.json', 'best')
    if unc is None:
        print('Warning: eval_cache.json not found, using flat AR defaults')
        unc = UncertaintyParams()

    print(f"Height-Pitch Sweep: {len(HEIGHTS)} heights x {len(PITCHES)} pitches "
          f"= {len(HEIGHTS) * len(PITCHES)} cells, {N_TRIALS} trials each")
    print(f"Road: {ROAD_FT} ft ({width_m:.1f} m)")
    print(f"Heights: {HEIGHTS}")
    print(f"Pitches: {PITCHES}")
    print()

    data = np.zeros((len(HEIGHTS), len(PITCHES)))

    header = f"  {'h\\p':>5}"
    for p in PITCHES:
        header += f"  {p:>5}"
    print(header)
    print(f"  {'-' * (7 + 7 * len(PITCHES))}")

    for i, h in enumerate(HEIGHTS):
        row = f"  {h:>4.1f}m"
        for j, p in enumerate(PITCHES):
            r = run_mc(scenario, 1, width_m, unc=unc,
                       cam_height=h, cam_pitch=float(p), n_trials=N_TRIALS)
            data[i, j] = r['mean_sens'] * 100
            row += f"  {data[i, j]:>5.1f}"
        print(row)

    result = {
        'heights': HEIGHTS,
        'pitches': PITCHES,
        'road_ft': ROAD_FT,
        'n_trials': N_TRIALS,
        'sensitivity_pct': data.tolist(),
    }

    out_path = 'height_pitch_sweep.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
