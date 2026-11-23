"""make_trajectory_figure.py -- IEEE-styled predicted-vs-true aging trajectory.

trains the Ridge trajectory regressor (raw hours) under LOBO with SN007 held
out (longest aging span), then plots predicted aging hours vs. the measurement
timeline for the held-out board. saves to both the classifier models/ dir and
the paper figure folder. uses plot_style for IEEE formatting.
"""
import os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
import task9_trajectory as t9
import plot_style as ps
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

RNG = 42
PAPER_FIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')


def main():
    held = 'SN007'
    train = [d for d in t9.ALL_DUTS if d != held]
    feats, hrs = {}, {}
    for d in t9.ALL_DUTS:
        X, y, _ = dc.load_dut(d); X = X.astype(np.float64)
        X = X - X[y == 0].mean(axis=0)
        feats[d], hrs[d] = X, t9.aging_hours_for_dut(d)

    Xtr = np.vstack([feats[d] for d in train])
    htr = np.concatenate([hrs[d] for d in train])
    Xte, hte = feats[held], hrs[held]

    sc = StandardScaler().fit(Xtr)
    pca = PCA(50, whiten=True, svd_solver='randomized', random_state=RNG).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr)); Zte = pca.transform(sc.transform(Xte))
    rg = Ridge(alpha=1.0).fit(Ztr, htr)
    pred = rg.predict(Zte)

    # sort test samples by true aging hours for a clean trajectory view.
    order = np.argsort(hte)
    t_true = hte[order]; t_pred = pred[order]
    # bin to reduce overplotting of ~24k points: mean predicted per true-hour bin.
    bins = np.linspace(t_true.min(), t_true.max(), 60)
    idx = np.digitize(t_true, bins)
    bx, bmean, blo, bhi = [], [], [], []
    for b in range(1, len(bins)):
        m = idx == b
        if m.sum() == 0:
            continue
        bx.append(t_true[m].mean()); bmean.append(t_pred[m].mean())
        blo.append(np.percentile(t_pred[m], 10)); bhi.append(np.percentile(t_pred[m], 90))
    bx, bmean, blo, bhi = map(np.array, (bx, bmean, blo, bhi))

    ps.apply()
    fig, ax = ps.figure()
    lim = [0, max(t_true.max(), t_pred.max()) * 1.02]
    ax.plot(lim, lim, color=ps.colors[7], linestyle='--', linewidth=1.2,
            label='Ideal (y = x)')
    ax.fill_between(bx, blo, bhi, color=ps.colors[0], alpha=0.25,
                    label='10--90th pct.')
    ax.plot(bx, bmean, color=ps.colors[0], marker='o', markersize=3,
            label='Predicted (mean)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ps.format_ax(ax, xlabel='True aging (hours)',
                 ylabel='Predicted aging (hours)', legend_loc='upper left')
    os.makedirs(PAPER_FIG, exist_ok=True)
    ps.save(fig, 'trajectory_SN007', output_dir='models')
    ps.save(fig, 'trajectory_SN007', output_dir=PAPER_FIG)
    plt.close(fig)
    mae = np.mean(np.abs(pred - hte))
    print(f"SN007 held out: Ridge trajectory MAE={mae:.2f} h over span {hte.max():.0f} h")


if __name__ == '__main__':
    main()
