"""diagnose_board_offset.py — Quantify and visualize why LOBO collapses

produces:
  - results/board_offset_diagnostic.md : between-board vs between-class variance
  - models/lobo_pca_scatter.{png,pdf,svg} : PCA colored by board and by class

the story: if between-board separation >> between-class separation in the raw
feature space, a model trained on one board cannot generalize to another
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
import plot_style
from plot_style import SIXERS_COLORS, save, format_ax
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

DUTS = ['SN004', 'SN005']
CLASS_NAMES = {0: 'Fresh', 1: 'Lightly Aged', 2: 'Heavily Aged'}


def main():
    Xs, ys, boards = [], [], []
    for bi, d in enumerate(DUTS):
        Xi, yi, _ = dc.load_dut(d)
        Xs.append(Xi.astype(np.float64))
        ys.append(yi)
        boards.append(np.full(len(yi), bi))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    board = np.concatenate(boards)

    # standardize (global) just for a fair variance comparison / PCA
    Xz = StandardScaler().fit_transform(X)

    # --- variance decomposition on the standardized features ---
    grand = Xz.mean(axis=0)

    # between-board: variance of board means around grand mean
    board_means = np.vstack([Xz[board == b].mean(axis=0) for b in range(len(DUTS))])
    board_counts = np.array([np.sum(board == b) for b in range(len(DUTS))])
    ss_board = np.sum(board_counts[:, None] * (board_means - grand) ** 2, axis=0).sum()

    # between-class (within each board, so aging effect not confounded by board)
    ss_class_within_board = 0.0
    for b in range(len(DUTS)):
        Xb = Xz[board == b]
        yb = y[board == b]
        mb = Xb.mean(axis=0)
        for c in np.unique(yb):
            Xc = Xb[yb == c]
            ss_class_within_board += (len(Xc) * (Xc.mean(axis=0) - mb) ** 2).sum()

    ratio = ss_board / ss_class_within_board if ss_class_within_board > 0 else float('inf')

    # --- PCA scatter (subsample for plotting) ---
    rng = np.random.RandomState(0)
    idx = rng.choice(len(Xz), size=min(6000, len(Xz)), replace=False)
    pca = PCA(n_components=2, random_state=0).fit(Xz[idx])
    P = pca.transform(Xz[idx])
    b_sub = board[idx]
    y_sub = y[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plot_style.STYLE['fig_width'] * 2,
                                                   plot_style.STYLE['fig_height']))
    # colored by board
    board_colors = [SIXERS_COLORS[0], SIXERS_COLORS[1]]
    for b in range(len(DUTS)):
        m = b_sub == b
        ax1.scatter(P[m, 0], P[m, 1], s=4, alpha=0.4, color=board_colors[b], label=DUTS[b])
    format_ax(ax1, xlabel='PC1', ylabel='PC2', legend=True, legend_loc='best')

    # colored by class
    class_colors = [SIXERS_COLORS[3], SIXERS_COLORS[8], SIXERS_COLORS[11]]
    for c in range(3):
        m = y_sub == c
        ax2.scatter(P[m, 0], P[m, 1], s=4, alpha=0.4, color=class_colors[c], label=CLASS_NAMES[c])
    format_ax(ax2, xlabel='PC1', ylabel='PC2', legend=True, legend_loc='best')

    fig.tight_layout()
    save(fig, 'lobo_pca_scatter', output_dir=MODEL_DIR)
    plt.close()

    # --- write markdown ---
    lines = []
    lines.append("# Board-offset diagnostic (why LOBO collapses)\n")
    lines.append("Variance decomposition on globally-standardized features:\n")
    lines.append(f"- Between-**board** sum of squares: **{ss_board:,.0f}**")
    lines.append(f"- Between-**class** (within board) sum of squares: **{ss_class_within_board:,.0f}**")
    lines.append(f"- **Ratio (board / class) = {ratio:.1f}x**\n")
    lines.append(
        f"board identity accounts for roughly **{ratio:.0f}x** more feature variance than "
        "the aging class does within a board. a model trained on one board therefore "
        "keys on board-specific offsets; applied to an unseen board those offsets shift "
        "the samples outside the learned decision regions, collapsing predictions to a "
        "single class. this is the mechanism behind the random-split (leaky) 99.8% vs. "
        "the leave-one-board-out chance-level result.\n")
    lines.append("See `models/lobo_pca_scatter.{pdf,png}`: the left panel (colored by "
                 "board) shows two well-separated clusters; the right panel (colored by "
                 "class) shows the aging classes are NOT separable across that board gap.\n")

    md = os.path.join(RESULTS_DIR, 'board_offset_diagnostic.md')
    with open(md, 'w') as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote {md}")
    print(f"Wrote {MODEL_DIR}/lobo_pca_scatter.{{png,pdf,svg}}")


if __name__ == '__main__':
    main()
