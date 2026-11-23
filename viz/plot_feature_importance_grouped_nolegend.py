"""
plot_feature_importance_grouped_nolegend.py

regenerates the aggregated RF feature-importance bar chart identical to
plot_feature_importance_grouped.py, but WITHOUT the legend (the y-axis tick
labels already name each category, making the legend redundant).

output: models/random_forest_feature_importance_grouped_regen.{png,pdf,svg}
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

# reuse the exact group/color definitions from the original script.
from plot_feature_importance_grouped import (
    GROUPS, GROUP_COLORS, MODEL_DIR, RF_MODEL_PATH,
)


def main():
    if not os.path.isfile(RF_MODEL_PATH):
        print(f"ERROR: RF model not found at {RF_MODEL_PATH}")
        sys.exit(1)

    print(f"Loading RF model from {RF_MODEL_PATH}...")
    rf_model = joblib.load(RF_MODEL_PATH)
    importances = rf_model.feature_importances_

    # aggregate importances by group (sum within each category)
    group_labels = []
    group_importances = []
    for label, start, length in GROUPS:
        group_labels.append(label)
        group_importances.append(importances[start:start + length].sum())
    group_importances = np.array(group_importances)

    # sort descending
    sort_idx = np.argsort(group_importances)[::-1]
    group_labels = [group_labels[i] for i in sort_idx]
    group_importances = group_importances[sort_idx]
    sorted_colors = [GROUP_COLORS[i] for i in sort_idx]

    # plot (identical sizing to the original)
    fig, ax = plt.subplots(
        figsize=(plot_style.STYLE['fig_width'] * 2,
                 plot_style.STYLE['fig_height'] * 1.4)
    )

    y_pos = np.arange(len(group_labels))
    ax.barh(y_pos, group_importances, color=sorted_colors, edgecolor='none', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(group_labels)
    ax.invert_yaxis()

    # IEEE style, no title, no legend
    format_ax(ax, xlabel='Cumulative Feature Importance (MDI)', ylabel=None, legend=False)

    fig.tight_layout()

    basename = 'random_forest_feature_importance_grouped_regen'
    save(fig, basename, output_dir=MODEL_DIR)
    plt.close()
    print(f"\nDone. Output in {MODEL_DIR}/{basename}.{{png,pdf,svg}}")


if __name__ == '__main__':
    main()
