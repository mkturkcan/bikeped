"""
Latency report generator for the bikeped collision-warning system.

Loads the .npy file saved by mqtt_bridge.py and prints per-step statistics.

Usage:
    python latency_report.py                         # default path
    python latency_report.py latency_data.npy        # explicit path
    python latency_report.py --csv                   # also write CSV
"""

import sys
import numpy as np

COLUMNS = [
    'frame_acquire',
    'preprocess',
    'yolo_preprocess',
    'yolo_inference',
    'yolo_postprocess',
    'bev_tracking',
    'decision',
    'viz_submit',
    'total',
    'pipeline_only',
]

# Derived groups for the summary
GROUPS = {
    'Frame acquisition':     [0],
    'Preprocessing':         [1],
    'YOLO preprocess':       [2],
    'YOLO inference':        [3],
    'YOLO postprocess':      [4],
    'Detection total':       [2, 3, 4],
    'BEV tracking':          [5],
    'Decision + feedback':   [6],
    'Viz submit (threaded)': [7],
    'Total (all steps)':     [8],
    'Pipeline only':         [9],
}


def load_data(path):
    data = np.load(path)
    if data.ndim != 2 or data.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, got shape {data.shape}. "
            f"The .npy file may have been recorded with an older format.")
    return data


def compute_stats(values):
    return {
        'mean': np.mean(values),
        'median': np.median(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'p95': np.percentile(values, 95),
        'p99': np.percentile(values, 99),
    }


def print_report(data):
    n = data.shape[0]
    total_time_s = np.sum(data[:, COLUMNS.index('total')]) / 1000.0
    avg_fps = n / total_time_s if total_time_s > 0 else 0

    print()
    print('=' * 90)
    print('  LATENCY REPORT')
    print('=' * 90)
    print(f'  Frames analyzed:  {n}')
    print(f'  Total wall time:  {total_time_s:.1f}s')
    print(f'  Mean FPS:         {avg_fps:.1f}')
    print()
    print(f'  {"Step":<25} {"Mean":>8} {"Median":>8} {"Std":>8} '
          f'{"Min":>8} {"Max":>8} {"P95":>8} {"P99":>8}   (ms)')
    print(f'  {"-" * 87}')

    for name, col_indices in GROUPS.items():
        values = np.sum(data[:, col_indices], axis=1)
        s = compute_stats(values)
        print(f'  {name:<25} {s["mean"]:>8.2f} {s["median"]:>8.2f} {s["std"]:>8.2f} '
              f'{s["min"]:>8.2f} {s["max"]:>8.2f} {s["p95"]:>8.2f} {s["p99"]:>8.2f}')

    print(f'  {"-" * 87}')

    # Breakdown: what fraction of total time is each step
    print()
    print(f'  {"Step":<25} {"% of total":>10}')
    print(f'  {"-" * 37}')
    total_mean = np.mean(data[:, COLUMNS.index('total')])
    for name, col_indices in GROUPS.items():
        if name in ('Total (all steps)', 'Pipeline only', 'Detection total'):
            continue
        step_mean = np.mean(np.sum(data[:, col_indices], axis=1))
        pct = step_mean / total_mean * 100 if total_mean > 0 else 0
        print(f'  {name:<25} {pct:>9.1f}%')

    # Pipeline-only stats (excluding frame acquisition, preprocessing, and visualization)
    print()
    pipeline = data[:, COLUMNS.index('pipeline_only')]
    ps = compute_stats(pipeline)
    pipeline_fps = 1000.0 / ps['mean'] if ps['mean'] > 0 else 0
    print(f'  Pipeline-only (excl. acquire + preprocess + visualization):')
    print(f'    Mean: {ps["mean"]:.2f} ms  ->  {pipeline_fps:.1f} FPS capacity')
    print(f'    P95:  {ps["p95"]:.2f} ms  P99: {ps["p99"]:.2f} ms')

    # Viz submit stats
    viz = data[:, COLUMNS.index('viz_submit')]
    vs = compute_stats(viz)
    print()
    print(f'  Viz submit (frame copy + handoff to draw thread):')
    print(f'    Mean: {vs["mean"]:.2f} ms  P95: {vs["p95"]:.2f} ms  P99: {vs["p99"]:.2f} ms')
    print(f'    Drawing runs concurrently and does not block the pipeline.')

    print()
    print('=' * 90)


def write_csv(data, path):
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in data:
            writer.writerow([f'{v:.3f}' for v in row])
    print(f'CSV written to {path}')


def main():
    path = 'latency_data.npy'
    do_csv = False

    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    flags = [a for a in sys.argv[1:] if a.startswith('-')]

    if args:
        path = args[0]
    if '--csv' in flags:
        do_csv = True

    try:
        data = load_data(path)
    except FileNotFoundError:
        print(f'Error: {path} not found. Run mqtt_bridge.py with latency.enabled: true first.')
        sys.exit(1)

    print_report(data)

    if do_csv:
        csv_path = path.replace('.npy', '.csv')
        write_csv(data, csv_path)


if __name__ == '__main__':
    main()
