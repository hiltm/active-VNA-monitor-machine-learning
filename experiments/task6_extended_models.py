"""task6_extended_models.py -- Batch-B extended models under 4-fold LOBO.

Runs, under the SAME leakage-free protocol as task1b (baseline-relative
features, StandardScaler fit on training boards only, leave-one-board-out):

  - PCA(k) -> RandomForest / LinearSVC         (dimensionality reduction)
  - PLS(k) regression -> thresholded classifier (supervised DR)
  - MLP on engineered features                  (learned-feature NN; stands in
                                                 for the 1-D CNN, which needs a
                                                 GPU/torch not available here)
  - Approx. Gaussian Process via RBF Nystrom features + logistic head, with
    predicted probabilities for calibration (uncertainty-aware model)

Tasks evaluated: 3class and heavy_vs_rest (the two most informative).
Writes results/task6_extended.json. Reuses the per-DUT feature cache; no
.s2p re-parsing.
"""
import os, sys, json, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix, brier_score_loss)

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = os.path.join(RESULTS_DIR, 'task6_extended.json')

# single source of truth: every aging DUT in DATA_CONFIG that is present on disk
ALL_DUTS = sorted(dc.all_duts())
FOLDS = [(tuple(d for d in ALL_DUTS if d != held), held) for held in ALL_DUTS]
TASKS = {
    'heavy_vs_rest': (lambda y: (y == 2).astype(int), [0, 1]),
    '3class':        (lambda y: y.copy(),             [0, 1, 2]),
}
RNG = 42
PCA_K = 50          # retained components
PLS_K = 10          # latent components
NYS_COMPONENTS = 300  # Nystrom landmarks for the approx-GP
GP_SUBSAMPLE = 15000  # cap training rows for the approx-GP (tractability)
NN_SUBSAMPLE = 15000  # cap training rows for the MLP (tractability)


def load_baseline_relative(dut):
    # keep float32 (cache dtype); float64 preload of all 8 DUTs OOM-killed the run
    X, y, _ = dc.load_dut(dut)
    baseline = X[y == 0].mean(axis=0, dtype=np.float64).astype(np.float32)
    return (X - baseline).astype(np.float32), y


def metrics(y_true, y_pred, labels):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main():
    out = {}
    cache = {}
    for dut in ALL_DUTS:
        Xr, y = load_baseline_relative(dut)
        cache[dut] = (Xr, y)
        print(f"Loaded {dut}: {Xr.shape}")

    for task_name, (relabel, labels) in TASKS.items():
        out[task_name] = {}
        for train_duts, test_dut in FOLDS:
            fold_key = f"test_{test_dut}"
            out[task_name][fold_key] = {}
            Xtr = np.vstack([cache[d][0] for d in train_duts])
            ytr = relabel(np.concatenate([cache[d][1] for d in train_duts]))
            Xte, yte = cache[test_dut][0], relabel(cache[test_dut][1])

            scaler = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            # --- PCA -> RF and PCA -> LinearSVC ---
            # Whiten so downstream kernel/NN models see unit-scale components.
            pca = PCA(n_components=PCA_K, whiten=True, random_state=RNG).fit(Xtr_s)
            Ztr, Zte = pca.transform(Xtr_s), pca.transform(Xte_s)
            evr = float(pca.explained_variance_ratio_.sum())
            t0 = time.time()
            rf = RandomForestClassifier(n_estimators=200, min_samples_split=5,
                                        min_samples_leaf=2, random_state=RNG, n_jobs=4)
            rf.fit(Ztr, ytr)
            m = metrics(yte, rf.predict(Zte), labels)
            m['explained_var'] = evr; m['time_s'] = round(time.time()-t0, 1)
            out[task_name][fold_key]['pca_rf'] = m

            t0 = time.time()
            svc = LinearSVC(C=1.0, dual=False, max_iter=2000, tol=1e-3, random_state=RNG)
            svc.fit(Ztr, ytr)
            m = metrics(yte, svc.predict(Zte), labels)
            m['explained_var'] = evr; m['time_s'] = round(time.time()-t0, 1)
            out[task_name][fold_key]['pca_svc'] = m

            # --- PLS regression -> nearest-integer / threshold classifier ---
            t0 = time.time()
            pls = PLSRegression(n_components=PLS_K)
            pls.fit(Xtr_s, ytr.astype(float))
            pred_cont = pls.predict(Xte_s).ravel()
            pred_pls = np.clip(np.rint(pred_cont), labels[0], labels[-1]).astype(int)
            m = metrics(yte, pred_pls, labels)
            m['time_s'] = round(time.time()-t0, 1)
            out[task_name][fold_key]['pls_thresh'] = m

            # --- MLP (learned-feature NN; CNN stand-in) ---
            # On whitened PCA features; training subsampled for tractability
            # (flagged preliminary in the report; true CNN deferred to GPU/8-DUT).
            t0 = time.time()
            rs0 = np.random.RandomState(RNG)
            if Ztr.shape[0] > NN_SUBSAMPLE:
                sidx = rs0.choice(Ztr.shape[0], NN_SUBSAMPLE, replace=False)
                Zm, ym = Ztr[sidx], ytr[sidx]
            else:
                Zm, ym = Ztr, ytr
            mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                                alpha=1e-4, max_iter=40, early_stopping=True,
                                n_iter_no_change=5, tol=1e-3, random_state=RNG)
            import threadpoolctl
            with threadpoolctl.threadpool_limits(limits=8):
                mlp.fit(Zm, ym)
            m = metrics(yte, mlp.predict(Zte), labels)
            m['time_s'] = round(time.time()-t0, 1)
            m['input'] = f'whitenedPCA{PCA_K}, n_train={Zm.shape[0]}'
            out[task_name][fold_key]['mlp'] = m

            # --- Approx-GP (Nystrom RBF features + logistic), probs for calibration ---
            # gamma via median-distance heuristic on a landmark subsample.
            t0 = time.time()
            rs = np.random.RandomState(RNG)
            if Ztr.shape[0] > GP_SUBSAMPLE:
                idx = rs.choice(Ztr.shape[0], GP_SUBSAMPLE, replace=False)
                Xg, yg = Ztr[idx], ytr[idx]
            else:
                Xg, yg = Ztr, ytr
            sub = Xg[rs.choice(Xg.shape[0], min(2000, Xg.shape[0]), replace=False)]
            d2 = np.sum((sub[:, None, :] - sub[None, :, :]) ** 2, axis=-1)
            med = np.median(d2[d2 > 0])
            gamma = 1.0 / med if med > 0 else 1.0 / Ztr.shape[1]
            nys = Nystroem(kernel='rbf', gamma=gamma,
                           n_components=NYS_COMPONENTS, random_state=RNG).fit(Xg)
            Ptr, Pte = nys.transform(Xg), nys.transform(Zte)
            clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
            clf.fit(Ptr, yg)
            pred_gp = clf.predict(Pte)
            m = metrics(yte, pred_gp, labels)
            m['time_s'] = round(time.time()-t0, 1); m['gamma'] = float(gamma)
            if len(labels) == 2:  # Brier score on positive class for calibration
                proba = clf.predict_proba(Pte)[:, 1]
                m['brier'] = float(brier_score_loss((yte == 1).astype(int), proba))
            out[task_name][fold_key]['approx_gp'] = m

            with open(OUT, 'w') as f:
                json.dump(out, f, indent=2)
            print(f"[{task_name}/{test_dut}] done: "
                  + ", ".join(f"{k}={v['balanced_accuracy']:.3f}"
                              for k, v in out[task_name][fold_key].items()))

    print(f"\nWrote {OUT}")


if __name__ == '__main__':
    main()
