"""generate_4dut_figures.py — IEEE-styled figures for the 4-DUT modeling results.

reads the committed result JSONs and emits publication figures:
  1. lobo_perboard_balacc      : RF balanced accuracy per held-out board, grouped
                                 by task (heavy_vs_rest / fresh_vs_aged / 3class)
  2. lobo_task_summary         : mean +/- std balanced accuracy per task (RF), the
                                 headline board-independent numbers
  3. ml_vs_baseline_lobo       : RF vs best non-ML baseline, mean LOBO balanced
                                 accuracy, per binary task (added-value-of-ML)
  4. per_dut_random_split      : per-DUT accuracy under the random 80/20 split

all figures use plot_style.py (IEEE, Sixers palette, no titles).
"""
import os, sys, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import plot_style
from plot_style import SIXERS_COLORS, save, format_ax
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')
MODELS = os.path.join(BASE, 'models')

DUTS = ['SN004', 'SN005', 'SN006', 'SN007']
TASK_ORDER = ['fresh_vs_aged', 'heavy_vs_rest', '3class']
TASK_LABEL = {
    'fresh_vs_aged': 'Fresh vs Aged',
    'heavy_vs_rest': 'Heavy vs Rest',
    '3class': '3-Class',
}


def held(fold_key):
    return fold_key.split('test_')[-1]


def load():
    lo = json.load(open(os.path.join(RES, 'lobo_baseline_relative.json')))
    bl = json.load(open(os.path.join(RES, 'baselines.json')))
    pd = json.load(open(os.path.join(RES, 'per_dut_accuracy.json')))
    return lo, bl, pd


def fig_lobo_perboard(lo):
    """Grouped bars: RF balanced accuracy per held-out board, grouped by task."""
    fig, ax = plt.subplots(figsize=(plot_style.STYLE['fig_width'] * 2,
                                    plot_style.STYLE['fig_height']))
    x = np.arange(len(DUTS))
    width = 0.8 / len(TASK_ORDER)
    colors = [SIXERS_COLORS[3], SIXERS_COLORS[0], SIXERS_COLORS[1]]  # green, blue, red
    for i, task in enumerate(TASK_ORDER):
        vals = {held(k): v['rf']['balanced_accuracy'] for k, v in lo[task].items()}
        y = [100 * vals[d] for d in DUTS]
        off = (i - len(TASK_ORDER) / 2 + 0.5) * width
        ax.bar(x + off, y, width, label=TASK_LABEL[task], color=colors[i],
               edgecolor='white', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(DUTS)
    ax.set_ylim(0, 105)
    format_ax(ax, xlabel='Held-Out Board', ylabel='Balanced Accuracy (%)',
              legend=True, legend_loc='lower right')
    fig.tight_layout()
    save(fig, 'lobo_perboard_balacc', output_dir=MODELS)
    plt.close()


def fig_lobo_task_summary(lo):
    """Mean +/- std balanced accuracy per task (RF) — the headline numbers."""
    fig, ax = plt.subplots(figsize=(plot_style.STYLE['fig_width'] * 1.4,
                                    plot_style.STYLE['fig_height']),
                           layout='constrained')
    means, stds, labels = [], [], []
    two_line = {'fresh_vs_aged': 'Fresh\nvs Aged',
                'heavy_vs_rest': 'Heavy\nvs Rest',
                '3class': '3-Class'}
    for task in TASK_ORDER:
        vals = np.array([v['rf']['balanced_accuracy'] for v in lo[task].values()]) * 100
        means.append(vals.mean()); stds.append(vals.std(ddof=1)); labels.append(two_line[task])
    x = np.arange(len(TASK_ORDER))
    colors = [SIXERS_COLORS[3], SIXERS_COLORS[0], SIXERS_COLORS[1]]
    ax.bar(x, means, 0.6, yerr=stds, capsize=4, color=colors,
           edgecolor='white', linewidth=0.5)
    # place the value label to the RIGHT of each bar's error whisker so the
    # errorbar line does not run through the text; put the +/- spread below it.
    for xi, m, s in zip(x, means, stds):
        y_top = m + s
        ax.text(xi + 0.32, y_top, f'{m:.1f}', ha='left', va='center',
                fontsize=plot_style.STYLE['tick_size'], fontweight='bold')
        ax.text(xi + 0.32, y_top - 4.5, f'(\u00b1{s:.1f})', ha='left', va='center',
                fontsize=plot_style.STYLE['tick_size'] - 2, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 108)
    format_ax(ax, xlabel=None, ylabel='Balanced Accuracy (%)', legend=False)
    save(fig, 'lobo_task_summary', output_dir=MODELS)
    plt.close()


def fig_ml_vs_baseline(lo, bl):
    """RF vs best non-ML baseline: mean LOBO balanced accuracy per binary task."""
    tasks = ['fresh_vs_aged', 'heavy_vs_rest']
    rf_means, base_means = [], []
    for task in tasks:
        rf = np.mean([v['rf']['balanced_accuracy'] for v in lo[task].values()]) * 100
        rf_means.append(rf)
        blk = bl[f'lobo_{task}']
        # best baseline = max over the two baseline methods of their mean bal acc
        methods = ['nearest_centroid', 'frequency_threshold']
        best = 0.0
        for mth in methods:
            mvals = [blk[f][mth]['balanced_accuracy'] for f in blk]
            best = max(best, np.mean(mvals))
        base_means.append(best * 100)
    fig, ax = plt.subplots(figsize=(plot_style.STYLE['fig_width'] * 1.4,
                                    plot_style.STYLE['fig_height']))
    x = np.arange(len(tasks)); w = 0.36
    ax.bar(x - w / 2, rf_means, w, label='Random Forest', color=SIXERS_COLORS[0],
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, base_means, w, label='Best Non-ML Baseline',
           color=SIXERS_COLORS[7], edgecolor='white', linewidth=0.5)
    for xi, m in zip(x - w / 2, rf_means):
        ax.text(xi, m + 1.5, f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for xi, m in zip(x + w / 2, base_means):
        ax.text(xi, m + 1.5, f'{m:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([TASK_LABEL[t] for t in tasks])
    ax.set_ylim(0, 118)
    format_ax(ax, xlabel=None, ylabel='Mean LOBO Balanced Acc. (%)',
              legend=True, legend_loc='lower center')
    fig.tight_layout()
    save(fig, 'ml_vs_baseline_lobo', output_dir=MODELS)
    plt.close()


def fig_per_dut(pd):
    """Per-DUT accuracy under the random 80/20 split (RF)."""
    fig, ax = plt.subplots(figsize=(plot_style.STYLE['fig_width'] * 1.4,
                                    plot_style.STYLE['fig_height']))
    accs = [100 * pd['per_dut'][d]['accuracy'] for d in DUTS]
    bals = [100 * pd['per_dut'][d]['balanced_accuracy'] for d in DUTS]
    x = np.arange(len(DUTS)); w = 0.36
    ax.bar(x - w / 2, accs, w, label='Accuracy', color=SIXERS_COLORS[0],
           edgecolor='white', linewidth=0.5)
    ax.bar(x + w / 2, bals, w, label='Balanced Acc.', color=SIXERS_COLORS[4],
           edgecolor='white', linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(DUTS)
    ax.set_ylim(80, 102)
    format_ax(ax, xlabel='DUT', ylabel='Accuracy (%)', legend=True, legend_loc='lower left')
    fig.tight_layout()
    save(fig, 'per_dut_random_split', output_dir=MODELS)
    plt.close()


def main():
    lo, bl, pd = load()
    fig_lobo_perboard(lo)
    fig_lobo_task_summary(lo)
    fig_ml_vs_baseline(lo, bl)
    fig_per_dut(pd)
    print("Done. Generated 4 figures in models/:")
    for b in ['lobo_perboard_balacc', 'lobo_task_summary',
              'ml_vs_baseline_lobo', 'per_dut_random_split']:
        print(f"  {b}.{{png,pdf,svg}}")


if __name__ == '__main__':
    main()
