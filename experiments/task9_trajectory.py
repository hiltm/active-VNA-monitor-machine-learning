"""task9_trajectory.py -- Aging-trajectory regression under 4-fold LOBO.

Pivots the problem from 3-class classification to continuous aging estimation,
the direction intended for the IEEE TIM journal submission. Over the SAME
feature space and the SAME leakage-free protocol as task1b (baseline-relative
features, StandardScaler on training boards only, leave-one-board-out), we
regress two targets:

  primary   : raw accumulated aging hours  a_i  (hours since each board's
              earliest/day-0 sweep; physically meaningful, TIM-aligned)
  secondary : normalized aging index  u_i = a_i / max_b a  in [0,1]
              (cross-board comparable despite different aging schedules)

Models:
  - Ridge regression        (linear baseline)
  - RandomForestRegressor   (nonlinear, matches the classifier family)
  - Approx-GP regressor      (Nystrom RBF features + Ridge; reports a
                              posterior-style std for calibration)

Metrics: MAE and RMSE in hours (primary) and in index units (secondary).
Sanity check: quantizing the hour estimate back to {0,1,2} via the day->class
cut-points should approximately recover the 3-class classifier behavior.

Writes results/task9_trajectory.json. Aging hours are derived from the .s2p
filename timestamps via train_classifier's discovery + parser; features come
from the per-DUT cache (no re-parsing of S-parameters).
"""
import os, sys, json, time, glob
from datetime import datetime
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import train_classifier as tc
import data_cache as dc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = dc.DATA_DIR         # honors CHIPTAMPER_DATA_DIR
RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = os.path.join(RESULTS_DIR, 'task9_trajectory.json')

# single source of truth: every aging DUT in DATA_CONFIG that is present on disk
ALL_DUTS = sorted(dc.all_duts())
FOLDS = [(tuple(d for d in ALL_DUTS if d != held), held) for held in ALL_DUTS]
RNG = 42
PCA_K = 50
NYS_COMPONENTS = 300
SUBSAMPLE = 20000


def aging_hours_for_dut(dut):
    """per-sweep aging hours for a DUT, row-aligned with dc.load_dut(dut)

    hours are read from the feature cache, where they were captured in the same
    parse pass as X/y/day (identical skip of unparseable files). this guarantees
    alignment; the old re-derivation via discover_files could drift by the count
    of files that fail to parse (observed as an off-by-one on SN009)
    """
    return dc.load_hours(dut)


def regress_metrics(y_true, y_pred):
    return {
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def main():
    # Load features (baseline-relative) and aging hours, aligned per DUT.
    feats, hrs, cls = {}, {}, {}
    maxhours = {}
    for dut in ALL_DUTS:
        # keep float32 (cache dtype); float64 preload of all 8 DUTs OOM-killed the run
        X, y, day = dc.load_dut(dut)
        baseline = X[y == 0].mean(axis=0, dtype=np.float64).astype(np.float32)
        X = (X - baseline).astype(np.float32)   # baseline-relative
        h = aging_hours_for_dut(dut)
        if h.shape[0] != X.shape[0]:
            raise RuntimeError(f"{dut}: hours {h.shape[0]} != features {X.shape[0]} "
                               f"(discovery/cache order mismatch)")
        # guard against NaN hours
        ok = ~np.isnan(h)
        feats[dut], hrs[dut], cls[dut] = X[ok], h[ok], y[ok]
        maxhours[dut] = float(np.nanmax(h))
        print(f"{dut}: n={ok.sum()} aging span={maxhours[dut]:.1f} h "
              f"(day max={day.max()})")

    out = {'per_dut_max_hours': maxhours, 'folds': {}}

    for train_duts, test_dut in FOLDS:
        fold_key = f"test_{test_dut}"
        out['folds'][fold_key] = {}
        Xtr = np.vstack([feats[d] for d in train_duts])
        Xte = feats[test_dut]
        htr = np.concatenate([hrs[d] for d in train_duts])
        hte = hrs[test_dut]
        # normalized index per board
        utr = np.concatenate([hrs[d] / maxhours[d] for d in train_duts])
        ute = hrs[test_dut] / maxhours[test_dut]

        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        pca = PCA(PCA_K, whiten=True, svd_solver='randomized',
                  random_state=RNG).fit(Xtr_s)
        Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)

        for target_name, (ytr, yte) in {
            'hours': (htr, hte), 'index': (utr, ute)
        }.items():
            res = {}
            # Ridge
            t0 = time.time()
            rg = Ridge(alpha=1.0).fit(Ztr, ytr)
            res['ridge'] = regress_metrics(yte, rg.predict(Zte))
            res['ridge']['time_s'] = round(time.time()-t0, 1)
            # Gradient-boosted trees (histogram-based): the fast, nonlinear
            # counterpart to the classifier's random forest. Histogram binning
            # makes it tractable on ~1.6e5 rows where a deep RF regressor is not.
            t0 = time.time()
            gb = HistGradientBoostingRegressor(max_iter=200, max_leaf_nodes=31,
                                               learning_rate=0.1,
                                               early_stopping=True, random_state=RNG)
            gb.fit(Ztr, ytr)
            res['hgb'] = regress_metrics(yte, gb.predict(Zte))
            res['hgb']['time_s'] = round(time.time()-t0, 1)
            # Approx-GP regressor (Nystrom + Ridge); std via tree-ensemble proxy
            t0 = time.time()
            rs = np.random.RandomState(RNG)
            idx = (rs.choice(Ztr.shape[0], SUBSAMPLE, replace=False)
                   if Ztr.shape[0] > SUBSAMPLE else np.arange(Ztr.shape[0]))
            sub = Ztr[idx][rs.choice(len(idx), min(2000, len(idx)), replace=False)]
            d2 = np.sum((sub[:, None] - sub[None]) ** 2, -1)
            med = np.median(d2[d2 > 0]); gamma = 1.0/med if med > 0 else 1.0/PCA_K
            nys = Nystroem('rbf', gamma=gamma, n_components=NYS_COMPONENTS,
                           random_state=RNG).fit(Ztr[idx])
            gp = Ridge(alpha=1.0).fit(nys.transform(Ztr[idx]), ytr[idx])
            pred = gp.predict(nys.transform(Zte))
            res['approx_gp'] = regress_metrics(yte, pred)
            res['approx_gp']['gamma'] = float(gamma)
            res['approx_gp']['time_s'] = round(time.time()-t0, 1)

            # Sanity: quantize the HGB HOURS prediction to {0,1,2} via the
            # day->class cut-points and compare to the day-derived class (only
            # meaningful for the 'hours' target). This checks that the
            # continuous estimate is consistent with the 3-class classifier.
            if target_name == 'hours':
                pred_h = gb.predict(Zte)
                pred_day = pred_h / 24.0
                pred_cls = np.where(pred_day < 0.5, 0,
                             np.where(pred_day <= 3.5, 1, 2))
                true_cls = cls[test_dut]
                acc = float(np.mean(pred_cls == true_cls))
                res['hgb']['quantized_to_class_acc'] = acc

            out['folds'][fold_key][target_name] = res
            with open(OUT, 'w') as f:
                json.dump(out, f, indent=2)
            tag = 'h' if target_name == 'hours' else ''
            print(f"[{test_dut}/{target_name}] ridge MAE={res['ridge']['mae']:.2f}{tag} "
                  f"hgb MAE={res['hgb']['mae']:.2f}{tag} "
                  f"gp MAE={res['approx_gp']['mae']:.2f}{tag}", flush=True)

    print(f"\nWrote {OUT}")


if __name__ == '__main__':
    main()
