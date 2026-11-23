"""
plot_feature_importance_grouped.py — Aggregated RF feature importance by category.

Loads the trained Random Forest model and produces a horizontal bar chart showing
the *total* (summed) MDI feature importance for each feature group:
    - S11 Magnitude, S11 Phase, S11 Group Delay, S11 Mag Slope
    - S21 Magnitude, S21 Phase, S21 Group Delay, S21 Mag Slope
    - Statistical

Uses IEEE styling from plot_style.py (no title, bold axes, Sixers palette).

Usage:
    python plot_feature_importance_grouped.py
"""

import sys
import os
import numpy as np
import joblib

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import plot_style
from plot_style import SIXERS_COLORS, save, format_ax

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# =============================================================================
# CONSTANTS (must match train_classifier.py)
# =============================================================================
N_FREQ_POINTS = 201
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'rf_model.joblib')

# feature group definitions: (label, start_index, length)
# order in the feature vector:
#   S11_mag_dB:      0 .. 200        (201)
#   S11_phase_deg:   201 .. 401      (201)
#   S21_mag_dB:      402 .. 602      (201)
#   S21_phase_deg:   603 .. 803      (201)
#   S11_grpdelay_ns: 804 .. 1003     (200)
#   S21_grpdelay_ns: 1004 .. 1203    (200)
#   S11_mag_slope:   1204 .. 1403    (200)
#   S21_mag_slope:   1404 .. 1603    (200)
#   Statistical:     1604 .. 1611    (8)

n = N_FREQ_POINTS  # 201
dn = n - 1         # 200

GROUPS = [
    ('S11 Magnitude',   0,              n),
    ('S11 Phase',       n,              n),
    ('S21 Magnitude',   2*n,            n),
    ('S21 Phase',       3*n,            n),
    ('S11 Group Delay', 4*n,            dn),
    ('S21 Group Delay', 4*n + dn,       dn),
    ('S11 Mag Slope',   4*n + 2*dn,     dn),
    ('S21 Mag Slope',   4*n + 3*dn,     dn),
    ('Statistical',     4*n + 4*dn,     8),
]

# colors for each group (Sixers palette)
GROUP_COLORS = [
    SIXERS_COLORS[0],   # S11 Magnitude — Blue
    SIXERS_COLORS[2],   # S11 Phase — Navy
    SIXERS_COLORS[4],   # S11 Group Delay — Light Blue
    SIXERS_COLORS[13],  # S11 Mag Slope — Medium Teal
    SIXERS_COLORS[1],   # S21 Magnitude — Red
    SIXERS_COLORS[5],   # S21 Phase — Brown
    SIXERS_COLORS[8],   # S21 Group Delay — Orange-Yellow
    SIXERS_COLORS[6],   # S21 Mag Slope — Gold
    SIXERS_COLORS[11],  # Statistical — Dark Plum
]


def main():
    # load the trained RF model
    if not os.path.isfile(RF_MODEL_PATH):
        print(f"ERROR: RF model not found at {RF_MODEL_PATH}")
        print("       Run train_classifier.py first.")
        sys.exit(1)

    print(f"Loading RF model from {RF_MODEL_PATH}...")
    rf_model = joblib.load(RF_MODEL_PATH)
    importances = rf_model.feature_importances_
    print(f"  Total features: {len(importances)}")

    # aggregate importances by group (sum within each category)
    group_labels = []
    group_importances = []
    for label, start, length in GROUPS:
        group_labels.append(label)
        group_importances.append(importances[start:start + length].sum())

    group_importances = np.array(group_importances)

    # sort by importance (descending)
    sort_idx = np.argsort(group_importances)[::-1]
    group_labels = [group_labels[i] for i in sort_idx]
    group_importances = group_importances[sort_idx]
    sorted_colors = [GROUP_COLORS[i] for i in sort_idx]

    # print summary
    print("\n  Aggregated Feature Importance by Category:")
    print(f"  {'Category':<20s}  {'Sum MDI':>10s}  {'% Total':>8s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*8}")
    for label, imp in zip(group_labels, group_importances):
        print(f"  {label:<20s}  {imp:10.4f}  {100*imp:7.2f}%")

    # =========================================================================
    # PLOT
    # =========================================================================
    fig, ax = plt.subplots(figsize=(plot_style.STYLE['fig_width'] * 2, plot_style.STYLE['fig_height'] * 1.4))

    y_pos = np.arange(len(group_labels))
    ax.barh(y_pos, group_importances, color=sorted_colors, edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels)
    ax.invert_yaxis()

    # format with IEEE style (no title, bold axes)
    format_ax(ax, xlabel='Cumulative Feature Importance (MDI)', ylabel=None, legend=False)

    # build legend
    legend_elements = []
    for i in sort_idx:
        label, _, _ = GROUPS[i]
        legend_elements.append(Patch(facecolor=GROUP_COLORS[i], edgecolor='none', label=label))

    leg = ax.legend(handles=legend_elements, loc='lower right',
                    fontsize=plot_style.STYLE['legend_size'])
    leg.get_frame().set_linewidth(plot_style.STYLE['legend_frame_width'])
    leg.get_frame().set_edgecolor('black')
    for text in leg.get_texts():
        text.set_fontweight('bold')

    fig.tight_layout()

    # save
    basename = 'random_forest_feature_importance_grouped'
    save(fig, basename, output_dir=MODEL_DIR)
    plt.close()
    print(f"\nDone. Output in {MODEL_DIR}/{basename}.{{png,pdf,svg}}")


if __name__ == '__main__':
    main()
