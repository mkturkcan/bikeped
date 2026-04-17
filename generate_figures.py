"""
Generate publication-quality EPS figures for the CVPR paper.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'text.usetex': False,
})

OUTDIR = 'paper'


def _load_height_pitch_data():
    """Load height-pitch sweep data from JSON (produced by run_height_pitch_sweep.py)."""
    try:
        with open('height_pitch_sweep.json', 'r') as f:
            d = json.load(f)
        return d['heights'], d['pitches'], np.array(d['sensitivity_pct'])
    except FileNotFoundError:
        print('height_pitch_sweep.json not found — run: python run_height_pitch_sweep.py')
        return None, None, None


def fig_height_pitch_heatmap():
    """Figure: Joint height x pitch sensitivity heatmap for 1-camera deployment."""
    heights, pitches, data_1cam = _load_height_pitch_data()
    if data_1cam is None:
        return

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    cmap = LinearSegmentedColormap.from_list('sens',
        ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'])
    vmin = max(np.nanmin(data_1cam) - 2, 0)
    vmax = np.nanmax(data_1cam) + 2
    im = ax.imshow(data_1cam, cmap=cmap, aspect='auto',
                   vmin=vmin, vmax=vmax, interpolation='nearest')

    ax.set_xticks(range(len(pitches)))
    ax.set_xticklabels([f'{p}\u00b0' for p in pitches], fontsize=6)
    ax.set_yticks(range(len(heights)))
    ax.set_yticklabels([f'{h}' for h in heights], fontsize=6)
    ax.set_xlabel('Pitch (degrees)')
    ax.set_ylabel('Height (m)')

    for i in range(len(heights)):
        for j in range(len(pitches)):
            val = data_1cam[i, j]
            if np.isnan(val):
                continue
            color = 'white' if val < (vmin + (vmax - vmin) * 0.3) else 'black'
            ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                    fontsize=4.5, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Sensitivity (%)', fontsize=8)

    fig.savefig(f'{OUTDIR}/fig_height_pitch.eps', format='eps')
    fig.savefig(f'{OUTDIR}/fig_height_pitch.pdf', format='pdf')
    print('Saved fig_height_pitch.eps/pdf')
    plt.close()


def fig_ar_curve():
    """Figure: AR vs bbox area curve from eval cache, with distance annotations."""
    try:
        with open('eval_cache.json', 'r') as f:
            cache = json.load(f)
    except FileNotFoundError:
        print('eval_cache.json not found, skipping AR curve figure')
        return

    model = cache.get('best', {})
    mg = model.get('merged', {})

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    colors = {'pedestrian': '#d62728', 'cyclist': '#2ca02c', 'vehicle': '#1f77b4'}
    labels = {'pedestrian': 'Pedestrian', 'cyclist': 'Cyclist', 'vehicle': 'Vehicle'}

    for group in ['pedestrian', 'cyclist', 'vehicle']:
        curve = mg.get(group, {}).get('ar_curve', {})
        if not curve:
            continue
        centers = np.array(curve['bin_centers'])
        ar = np.array(curve['ar'])
        n_gt = np.array(curve['n_gt'])
        # Only plot bins with data
        mask = n_gt > 2
        ax.plot(centers[mask], ar[mask] * 100, '-o', color=colors[group],
                label=labels[group], markersize=3, linewidth=1.2)

    # Add distance annotations for cyclist (from crosswalk, default pole)
    # Placed at top of plot with staggered y to avoid overlap
    dist_areas = [(25, 673), (15, 1336), (5, 1694)]
    for i, (dist, area) in enumerate(dist_areas):
        ax.axvline(area, color='gray', linestyle=':', linewidth=0.5, alpha=0.6)
        y_pos = 8 + i * 7
        ax.annotate(f'{dist} m', xy=(area, y_pos), fontsize=6.5,
                    ha='center', color='#555555',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white',
                              ec='none', alpha=0.8))

    ax.set_xscale('log')
    ax.set_xlabel('Bounding box area (px$^2$ at YOLO input)')
    ax.set_ylabel('Recall (%)')
    ax.set_xlim(50, 300000)
    ax.set_ylim(-2, 105)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.savefig(f'{OUTDIR}/fig_ar_curve.eps', format='eps')
    fig.savefig(f'{OUTDIR}/fig_ar_curve.pdf', format='pdf')
    print('Saved fig_ar_curve.eps/pdf')
    plt.close()


def fig_deployment_comparison():
    """Figure: Bar chart comparing 1-cam vs 2-cam across road widths,
    default vs optimized height."""
    widths = [30, 40, 50, 60]

    # Default h=5.5m
    sens_1_default = [81.3, 82.2, 80.1, 76.2]
    sens_2_default = [95.0, 95.3, 93.8, 89.3]

    # Optimal h=2.5m, pitch=-40
    sens_1_optimal = [88.0, 89.2, 88.8, 84.1]
    sens_2_optimal = [96.4, 97.3, 96.7, 93.6]

    x = np.arange(len(widths))
    w = 0.2

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    bars1 = ax.bar(x - 1.5*w, sens_1_default, w, label='1-cam, h=5.5m',
                   color='#bdd7e7', edgecolor='#6baed6', linewidth=0.5)
    bars2 = ax.bar(x - 0.5*w, sens_1_optimal, w, label='1-cam, h=2.5m',
                   color='#6baed6', edgecolor='#2171b5', linewidth=0.5)
    bars3 = ax.bar(x + 0.5*w, sens_2_default, w, label='2-cam, h=5.5m',
                   color='#bae4b3', edgecolor='#74c476', linewidth=0.5)
    bars4 = ax.bar(x + 1.5*w, sens_2_optimal, w, label='2-cam, h=2.5m',
                   color='#74c476', edgecolor='#238b45', linewidth=0.5)

    ax.set_xlabel('Road width (ft)')
    ax.set_ylabel('Sensitivity (%)')
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_ylim(70, 100)
    ax.legend(loc='lower left', fontsize=6.5, ncol=2, framealpha=0.9)
    ax.grid(True, axis='y', alpha=0.2)

    fig.savefig(f'{OUTDIR}/fig_deployment.eps', format='eps')
    fig.savefig(f'{OUTDIR}/fig_deployment.pdf', format='pdf')
    print('Saved fig_deployment.eps/pdf')
    plt.close()


def fig_pipeline_comparison():
    """Figure: Sensitivity vs specificity scatter for pipeline configs."""
    configs = {
        'Baseline':       (56.1, 97.4, 61.5),
        'Conservative':   (77.1, 92.6, 31.0),
        'Speed-adaptive': (93.1, 91.8, 9.4),
        'Selected':       (93.3, 92.3, 7.6),
    }

    fig, ax = plt.subplots(figsize=(3.4, 2.4))

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    markers = ['s', '^', 'D', 'o']

    for i, (name, (sens, spec, sevfn)) in enumerate(configs.items()):
        size = max(140 - sevfn * 5, 30)
        ax.scatter(spec, sens, s=size, c=colors[i], marker=markers[i],
                   edgecolors='black', linewidth=0.5, zorder=3, label=name)

    ax.set_xlabel('Specificity (%)')
    ax.set_ylabel('Sensitivity (%)')
    ax.set_xlim(84, 98)
    ax.set_ylim(55, 100)
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.savefig(f'{OUTDIR}/fig_pipeline.eps', format='eps')
    fig.savefig(f'{OUTDIR}/fig_pipeline.pdf', format='pdf')
    print('Saved fig_pipeline.eps/pdf')
    plt.close()


def fig_combined_pipeline_heatmap():
    """Two-column figure: pipeline scatter (left) + height-pitch heatmap (right)."""
    heights, pitches, data = _load_height_pitch_data()
    if data is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.4))

    # Left: pipeline scatter (symbols reduced ~30%)
    configs = {
        'Baseline':       (56.1, 97.4, 61.5),
        'Conservative':   (77.1, 92.6, 31.0),
        'Speed-adaptive': (93.1, 91.8, 9.4),
        'Selected':       (93.3, 92.3, 7.6),
    }
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    markers = ['s', '^', 'D', 'o']
    for i, (name, (sens, spec, sevfn)) in enumerate(configs.items()):
        size = max(140 - sevfn * 5, 30)
        ax1.scatter(spec, sens, s=size, c=colors[i], marker=markers[i],
                    edgecolors='black', linewidth=0.5, zorder=3, label=name)
    ax1.set_xlabel('Specificity (%)')
    ax1.set_ylabel('Sensitivity (%)')
    ax1.set_xlim(84, 98)
    ax1.set_ylim(55, 100)
    ax1.legend(loc='lower left', fontsize=6.5, framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.set_title('(a) Pipeline configurations', fontsize=9)

    # Right: height-pitch heatmap from sweep data
    cmap = LinearSegmentedColormap.from_list('sens',
        ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60', '#1a9850'])
    vmin = max(np.nanmin(data) - 2, 0)
    vmax = np.nanmax(data) + 2
    im = ax2.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
                    interpolation='nearest')
    ax2.set_xticks(range(len(pitches)))
    ax2.set_xticklabels([f'{p}' for p in pitches], fontsize=5)
    ax2.set_yticks(range(len(heights)))
    ax2.set_yticklabels([f'{h}' for h in heights], fontsize=5)
    ax2.set_xlabel('Pitch (deg)')
    ax2.set_ylabel('Height (m)')
    for i in range(len(heights)):
        for j in range(len(pitches)):
            val = data[i, j]
            if np.isnan(val):
                continue
            color = 'white' if val < (vmin + (vmax - vmin) * 0.3) else 'black'
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=3.5, color=color)
    cbar = fig.colorbar(im, ax=ax2, shrink=0.85, pad=0.02)
    cbar.set_label('Sens (%)', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    ax2.set_title('(b) Camera height vs. pitch', fontsize=9)

    fig.tight_layout()
    fig.savefig(f'{OUTDIR}/fig_combined.eps', format='eps')
    fig.savefig(f'{OUTDIR}/fig_combined.pdf', format='pdf')
    print('Saved fig_combined.eps/pdf')
    plt.close()


def fig_warning_budget():
    """Figure: Warning budget per danger scenario with PRT thresholds.

    Runs the testbench to get live data rather than using stale hardcoded values.
    """
    from decision_testbench import (
        SCENARIOS, PIPELINE_CONFIGS, FisheyeCamera,
        load_camera_calibration, run_all_scenarios,
    )

    calib = load_camera_calibration()
    if calib:
        cam = FisheyeCamera(
            fov_deg=calib.get('fov_deg', 197.90),
            radius_px=int(calib.get('radius_px', 1750)),
            frame_width=int(calib.get('frame_width', 3500)),
            frame_height=int(calib.get('frame_height', 3500)),
            pad_size=int(calib.get('pad_size', 3840)),
            cx_override=calib.get('cx'),
            cy_override=calib.get('cy'),
        )
    else:
        cam = FisheyeCamera()

    results = run_all_scenarios(PIPELINE_CONFIGS['default'], cam=cam,
                                verbose=False)

    # Collect danger scenarios with valid warning budgets
    names = []
    budgets = []
    for r in results:
        wb = r['min_warning_budget_s']
        if not np.isnan(wb):
            # Only danger scenarios (those that have alert onset)
            if r['true_pos'] + r['false_neg'] > 0:
                names.append(r['scenario'])
                budgets.append(wb)

    if not names:
        print('No danger scenarios with warning budgets, skipping')
        return

    # Sort by budget descending
    order = np.argsort(budgets)[::-1]
    names = [names[i] for i in order]
    budgets = [budgets[i] for i in order]

    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    y_pos = np.arange(len(names))

    # Color by PRT threshold (Fugger et al. 2000, combined averages)
    colors = []
    for b in budgets:
        if b >= 1.87:
            colors.append('#4CAF50')   # exceeds distracted PRT
        elif b >= 0.84:
            colors.append('#FF9800')   # exceeds attentive PRT
        else:
            colors.append('#F44336')   # below attentive PRT

    ax.barh(y_pos, budgets, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel('Warning budget (s)')

    # PRT thresholds (Fugger et al. 2000)
    ax.axvline(0.77, color='#9C27B0', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axvline(0.84, color='#1565C0', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.axvline(1.87, color='#E65100', linewidth=0.8, linestyle='--', alpha=0.7)

    ax.annotate('Anticipating\n0.77 s', xy=(0.77, len(names) - 0.5),
                fontsize=5.5, color='#9C27B0', ha='right', va='top')
    ax.annotate('Attentive\n0.84 s', xy=(0.84, len(names) - 0.5),
                fontsize=5.5, color='#1565C0', ha='left', va='top')
    ax.annotate('Distracted\n1.87 s', xy=(1.87, len(names) - 0.5),
                fontsize=5.5, color='#E65100', ha='left', va='top')

    ax.set_xlim(0, max(budgets) * 1.15)
    ax.grid(True, axis='x', alpha=0.2)

    fig.savefig(f'{OUTDIR}/fig_warning_budget.eps', format='eps',
                bbox_inches='tight')
    fig.savefig(f'{OUTDIR}/fig_warning_budget.pdf', format='pdf',
                bbox_inches='tight')
    print('Saved fig_warning_budget.eps/pdf')
    plt.close()


if __name__ == '__main__':
    fig_height_pitch_heatmap()
    fig_ar_curve()
    fig_deployment_comparison()
    fig_pipeline_comparison()
    fig_combined_pipeline_heatmap()
    fig_warning_budget()
    print('\nAll figures generated.')
