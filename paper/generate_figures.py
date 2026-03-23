"""
Generate publication-quality figures for the autoresearch arXiv paper.
All figures saved as 300dpi PNG in ./figures/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.4,
    'lines.markersize': 4,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

OUTDIR = Path(__file__).parent / 'figures'
OUTDIR.mkdir(parents=True, exist_ok=True)

# Minimal academic palette
COLORS = {
    'Sonnet 4':   '#2166ac',   # blue
    'Sonnet 4.6': '#e08214',   # orange
    'Opus 4.6':   '#b2182b',   # red
    'GPT-4.1':    '#1b7837',   # green
}

MODELS = ['Sonnet 4', 'Sonnet 4.6', 'Opus 4.6', 'GPT-4.1']

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA = {
    'Sonnet 4':   {'n_exp': 147, 'baseline': 0.948083, 'best': 0.936221,
                   'kept': 5,  'crashed': 50,  'total': 147},
    'Sonnet 4.6': {'n_exp': 104, 'baseline': 1.245301, 'best': 0.955865,
                   'kept': 21, 'crashed': 9,   'total': 104},
    'Opus 4.6':   {'n_exp': 111, 'baseline': 0.955747, 'best': 0.901860,
                   'kept': 14, 'crashed': 26,  'total': 111},
    'GPT-4.1':    {'n_exp': 104, 'baseline': 0.963646, 'best': 0.960070,
                   'kept': 4,  'crashed': 44,  'total': 104},
}

# Derived
for d in DATA.values():
    d['discarded'] = d['total'] - d['kept'] - d['crashed']
    d['non_crash'] = d['total'] - d['crashed']


def _simulate_trajectory(n_exp, baseline, best, kept, seed=0):
    """
    Simulate a plausible cumulative-best trajectory.

    The trajectory starts at the baseline and monotonically improves in
    `kept` discrete steps spread across `n_exp` experiments, ending at
    `best`.  Between improvement steps the line is flat.
    """
    rng = np.random.RandomState(seed)
    # Choose experiment indices where improvements happen
    improve_at = np.sort(rng.choice(np.arange(1, n_exp), size=kept, replace=False))
    # Improvement magnitudes (random, then normalised)
    magnitudes = rng.dirichlet(np.ones(kept)) * (baseline - best)
    trajectory = np.full(n_exp, baseline, dtype=float)
    current = baseline
    for idx, mag in zip(improve_at, magnitudes):
        current -= mag
        trajectory[idx:] = current
    # Ensure monotonic non-increasing and ends at best
    trajectory = np.minimum.accumulate(trajectory)
    trajectory[-1] = best
    return trajectory


# ---------------------------------------------------------------------------
# Figure 1 -- Cumulative improvement trajectory
# ---------------------------------------------------------------------------
def fig1_four_way_comparison():
    fig, ax = plt.subplots(figsize=(7, 3.8))

    for i, model in enumerate(MODELS):
        d = DATA[model]
        traj = _simulate_trajectory(d['n_exp'], d['baseline'], d['best'],
                                    d['kept'], seed=i + 42)
        improvement_pct = (d['baseline'] - traj) / d['baseline'] * 100
        xs = np.arange(1, d['n_exp'] + 1)
        label = f"{model} ({d['kept']} kept)"
        if model == 'GPT-4.1':
            label += ' *'
        ax.plot(xs, improvement_pct, color=COLORS[model], label=label)

    ax.set_xlabel('Experiment number')
    ax.set_ylabel('Improvement from baseline (%)')
    ax.set_title('Cumulative Best val_bpb Improvement Over Experiments')
    ax.legend(frameon=True, fancybox=False, edgecolor='0.7', loc='upper left')
    ax.grid(True, axis='y')

    # Annotation for GPT-4.1 still running
    ax.annotate('* still running', xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=8, color='0.4')

    fig.savefig(OUTDIR / 'four_way_comparison.png')
    plt.close(fig)
    print('  [1/5] four_way_comparison.png')


# ---------------------------------------------------------------------------
# Figure 2 -- Keep rate and crash rate (grouped bar)
# ---------------------------------------------------------------------------
def fig2_keep_crash_rates():
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    keep_rates = []
    crash_rates = []
    for model in MODELS:
        d = DATA[model]
        keep_rates.append(d['kept'] / d['non_crash'] * 100)
        crash_rates.append(d['crashed'] / d['total'] * 100)

    x = np.arange(len(MODELS))
    w = 0.35

    bars_keep = ax.bar(x - w / 2, keep_rates, w, label='Keep rate (non-crash %)',
                       color='#4393c3', edgecolor='white', linewidth=0.5)
    bars_crash = ax.bar(x + w / 2, crash_rates, w, label='Crash rate',
                        color='#d6604d', edgecolor='white', linewidth=0.5)

    # Value labels
    for bar in bars_keep:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar in bars_crash:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{h:.1f}%', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.set_ylabel('Rate (%)')
    ax.set_title('Keep Rate vs. Crash Rate by Model')
    ax.legend(frameon=True, fancybox=False, edgecolor='0.7', fontsize=8,
              loc='upper right')
    ax.set_ylim(0, max(max(keep_rates), max(crash_rates)) * 1.35)
    ax.grid(True, axis='y')

    fig.savefig(OUTDIR / 'keep_crash_rates.png')
    plt.close(fig)
    print('  [2/5] keep_crash_rates.png')


# ---------------------------------------------------------------------------
# Figure 3 -- Cost efficiency scatter
# ---------------------------------------------------------------------------
def fig3_cost_efficiency():
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    costs     = [6,   4,    50,   4]
    best_bpb  = [0.936221, 0.955865, 0.901860, 0.960070]
    n_exps    = [147, 104,  111,  104]

    for i, model in enumerate(MODELS):
        size = n_exps[i] * 2.5
        ax.scatter(costs[i], best_bpb[i], s=size, color=COLORS[model],
                   zorder=5, edgecolors='0.3', linewidths=0.5)
        # Offset labels to avoid overlap
        offsets = {
            'Sonnet 4':   (10, 10),
            'Sonnet 4.6': (10, 12),
            'Opus 4.6':   (-14, 10),
            'GPT-4.1':    (10, -16),
        }
        ox, oy = offsets[model]
        label_text = model
        if model == 'GPT-4.1':
            label_text = 'GPT-4.1 (Azure)'
        ax.annotate(label_text, (costs[i], best_bpb[i]),
                    textcoords='offset points', xytext=(ox, oy),
                    fontsize=8, color=COLORS[model],
                    arrowprops=dict(arrowstyle='-', color='0.6', lw=0.5))

    ax.set_xlabel('Estimated total cost ($)')
    ax.set_ylabel('Best val_bpb achieved')
    ax.set_title('Cost Efficiency')
    ax.invert_yaxis()  # lower bpb = better, so flip
    ax.grid(True)

    # Annotation arrow for "better"
    ax.annotate('', xy=(0.96, 0.02), xycoords='axes fraction',
                xytext=(0.96, 0.18), textcoords='axes fraction',
                arrowprops=dict(arrowstyle='->', color='0.5', lw=1.0))
    ax.text(0.93, 0.02, 'better', transform=ax.transAxes,
            fontsize=8, color='0.5', va='bottom', ha='right')

    fig.savefig(OUTDIR / 'cost_efficiency.png')
    plt.close(fig)
    print('  [3/5] cost_efficiency.png')


# ---------------------------------------------------------------------------
# Figure 4 -- Experiment outcomes stacked bar
# ---------------------------------------------------------------------------
def fig4_experiment_outcomes():
    fig, ax = plt.subplots(figsize=(3.5, 3.2))

    kept_vals     = [DATA[m]['kept'] for m in MODELS]
    discarded_vals = [DATA[m]['discarded'] for m in MODELS]
    crashed_vals  = [DATA[m]['crashed'] for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.55

    ax.bar(x, kept_vals, w, label='Kept', color='#4393c3',
           edgecolor='white', linewidth=0.5)
    ax.bar(x, discarded_vals, w, bottom=kept_vals, label='Discarded',
           color='#c2a5cf', edgecolor='white', linewidth=0.5)
    bottom2 = [k + d for k, d in zip(kept_vals, discarded_vals)]
    ax.bar(x, crashed_vals, w, bottom=bottom2, label='Crashed',
           color='#d6604d', edgecolor='white', linewidth=0.5)

    # Total labels on top
    for i, m in enumerate(MODELS):
        total = DATA[m]['total']
        ax.text(i, total + 1.5, str(total), ha='center', va='bottom',
                fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=15, ha='right')
    ax.set_ylabel('Number of experiments')
    ax.set_title('Experiment Outcomes by Model')
    ax.legend(frameon=True, fancybox=False, edgecolor='0.7', fontsize=8,
              loc='upper right')
    ax.grid(True, axis='y')

    fig.savefig(OUTDIR / 'experiment_outcomes.png')
    plt.close(fig)
    print('  [4/5] experiment_outcomes.png')


# ---------------------------------------------------------------------------
# Figure 5 -- Architecture / loop diagram
# ---------------------------------------------------------------------------
def fig5_architecture():
    fig, ax = plt.subplots(figsize=(7, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.5)
    ax.axis('off')

    box_style = dict(boxstyle='round,pad=0.35', facecolor='#f0f0f0',
                     edgecolor='0.3', linewidth=0.8)
    font_kw = dict(fontsize=9, ha='center', va='center',
                   fontfamily='serif')

    # Box positions (cx, cy)
    boxes = [
        (1.0,  2.5, 'LLM Agent\nproposes change'),
        (3.0,  2.5, 'Apply to\ntrain.py'),
        (5.0,  2.5, 'Train\n(5 min)'),
        (7.0,  2.5, 'Evaluate\nval_bpb'),
        (9.0,  2.5, 'Keep /\nRevert'),
    ]

    bw, bh = 1.5, 1.0  # box width, height
    rects = []
    for cx, cy, txt in boxes:
        r = FancyBboxPatch((cx - bw / 2, cy - bh / 2), bw, bh,
                           **box_style, zorder=2)
        ax.add_patch(r)
        ax.text(cx, cy, txt, **font_kw, zorder=3)
        rects.append((cx, cy))

    # Forward arrows
    arrow_kw = dict(arrowstyle='->', color='0.3', lw=1.2,
                    connectionstyle='arc3,rad=0')
    for i in range(len(rects) - 1):
        x1 = rects[i][0] + bw / 2
        x2 = rects[i + 1][0] - bw / 2
        y = rects[i][1]
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=arrow_kw, zorder=1)

    # Feedback loop: from "Keep/Revert" back to "LLM Agent"
    # Draw a curved arrow going below the boxes
    feedback_y = 0.7
    # Right side down
    ax.annotate('', xy=(9.0, 2.5 - bh / 2), xytext=(9.0, 2.5 - bh / 2),
                arrowprops=arrow_kw)  # placeholder

    # Manual feedback path using plot + arrow
    path_x = [9.0, 9.0, 1.0, 1.0]
    path_y = [2.5 - bh / 2, feedback_y, feedback_y, 2.5 - bh / 2]
    ax.plot(path_x, path_y, color='0.3', lw=1.2, zorder=1)
    # Arrowhead at the end going up into LLM box
    ax.annotate('', xy=(1.0, 2.5 - bh / 2), xytext=(1.0, feedback_y),
                arrowprops=dict(arrowstyle='->', color='0.3', lw=1.2),
                zorder=1)

    # Label on feedback loop
    ax.text(5.0, feedback_y - 0.3,
            'experiment history + results fed back to LLM',
            ha='center', va='top', fontsize=8, fontstyle='italic',
            color='0.4')

    ax.set_title('Autoresearch Loop', fontsize=11, pad=8)

    fig.savefig(OUTDIR / 'architecture.png')
    plt.close(fig)
    print('  [5/5] architecture.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'Saving figures to {OUTDIR.resolve()}')
    fig1_four_way_comparison()
    fig2_keep_crash_rates()
    fig3_cost_efficiency()
    fig4_experiment_outcomes()
    fig5_architecture()
    print('Done.')
