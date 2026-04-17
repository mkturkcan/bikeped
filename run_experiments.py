#!/usr/bin/env python3
"""
Reproduce all paper experiments and generate all figures.

This script runs every experiment reported in the paper in a single
invocation.  All results are printed to stdout and all figures are
saved to paper/.

Runtime: ~45 min with optimizer, ~10 min with --skip-optimizer.

Usage:
    python run_experiments.py                  # full pipeline
    python run_experiments.py --skip-optimizer  # skip optimizer
"""

import argparse
import subprocess
import sys


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  $ {cmd}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n*** FAILED: {desc} (exit code {result.returncode}) ***")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Reproduce all paper experiments')
    parser.add_argument('--skip-optimizer', action='store_true',
                        help='Skip the optimizer (saves ~30 min)')
    args = parser.parse_args()

    # ── Primary results (with fisheye localization error) ──

    # Table 2: Pipeline configuration comparison
    run('python decision_testbench.py --compare',
        'Table 2: Pipeline comparison (with fisheye localization error)')

    # Figure 4: Camera latency sweep
    run('python decision_testbench.py --latency-sweep',
        'Figure 4: Camera latency sweep (0-500 ms, with fisheye error)')

    # Monte Carlo with size-aware stochastic detection (1-camera)
    run('python decision_testbench.py --monte-carlo --mc-trials 50',
        'Monte Carlo: 1-camera, size-aware detection drops, 50 trials')

    # ── Ablation: without fisheye localization error ──

    run('python decision_testbench.py --no-bbox-noise --compare',
        'Ablation: Pipeline comparison WITHOUT fisheye localization error')

    run('python decision_testbench.py --no-bbox-noise --latency-sweep',
        'Ablation: Latency sweep WITHOUT fisheye error')

    # ── Optimizer ──

    if not args.skip_optimizer:
        run('python decision_testbench.py --optimize',
            'Optimizer: Differential evolution with fisheye error')
    else:
        print("\n  [Skipping optimizer as requested]\n")

    # ── Supporting analyses ──

    # BBox error analysis
    run('python decision_testbench.py --error-map',
        'BBox localization error summary')

    # ── Figures and videos ──

    run('python generate_figures.py',
        'Generate all publication figures (EPS/PDF)')

    run('python sim_visualizer.py --no-display',
        'Render scenario simulation videos')

    print(f"\n{'='*70}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Figures saved to: paper/")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
