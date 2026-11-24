"""task1b_lobo_baseline_relative.py — Leakage-free LOBO with baseline-relative features.

Motivation: absolute S-parameter features encode board identity (~2.7x more
variance than the aging class), so a model trained on one board fails on an
unseen board (chance-level). Referencing each board to its OWN day-0 (Fresh)
mean spectrum cancels the board-specific offset and exposes the transferable
aging signal.

Feature transform (per board, no cross-board leakage):
    X_rel = X - mean(X[y==0])      # subtract this board's Fresh baseline

Evaluated tasks (2-fold LOBO: SN004<->SN005):
    - 3class          : Fresh / Lightly Aged / Heavily Aged
    - heavy_vs_rest   : Heavily Aged (4+ days) vs everything else   <- transfers best
    - fresh_vs_aged   : Fresh vs any aging

Models: LinearSVC (primary for recentered features) and Random Forest (reported
for completeness; note RF does not adapt well to recentering).

Checkpoint: results/lobo_baseline_relative.json, written after each
(task, fold, model). Re-running resumes.
"""
import os, sys, json, time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
CKPT = os.path.join(RESULTS_DIR, 'lobo_baseline_relative.json')

# Proper leave-one-board-out over all cached aging DUTs: each fold trains on
# all-but-one DUT and tests on the held-out DUT.
# single source of truth: every aging DUT in DATA_CONFIG that is present on disk
# (SN003, the rework board, is not in DATA_CONFIG so it is naturally excluded)
ALL_DUTS = sorted(dc.all_duts())
FOLDS = [(tuple(d for d in ALL_DUTS if d != held), held) for held in ALL_DUTS]

# Order: fast/important binary tasks first, 3-class (slowest) last.
TASKS = {
    'heavy_vs_rest': (lambda y: (y == 2).astype(int), [0, 1]),
    'fresh_vs_aged': (lambda y: (y > 0).astype(int),  [0, 1]),
    '3class':        (lambda y: y.copy(),            [0, 1, 2]),
}
MODELS = ['linearsvc', 'rf']
MODEL_NAMES = {'linearsvc': 'LinearSVC', 'rf': 'Random Forest'}


# cap RF worker count: at 8 DUTs (~1.1M x 1612) n_jobs=-1 forked 20 workers and
# OOM-killed the process (26 GB RSS). 4 workers keeps peak memory in budget;
# results are identical, only wall-time changes
RF_N_JOBS = 4


def make_model(mt):
    if mt == 'linearsvc':
        # n_samples >> n_features -> dual=False (primal) is much faster.
        return LinearSVC(C=1.0, dual=False, max_iter=2000, tol=1e-3, random_state=42)
    if mt == 'rf':
        return RandomForestClassifier(n_estimators=200, min_samples_split=5,
                                      min_samples_leaf=2, random_state=42,
                                      n_jobs=RF_N_JOBS)
    raise ValueError(mt)


def load_baseline_relative(dut):
    """Load features referenced to this board's own Fresh (day-0) mean.

    NOTE: features are kept float32 (the cache dtype). at 8 DUTs an upcast to
    float64 doubled the preload to ~14 GB and contributed to an OOM kill;
    float32 is numerically fine for the baseline subtraction, StandardScaler,
    and the downstream models
    """
    X, y, _ = dc.load_dut(dut)
    baseline = X[y == 0].mean(axis=0, dtype=np.float64).astype(np.float32)
    return (X - baseline).astype(np.float32), y


def evaluate(y_true, y_pred, labels):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def load_ckpt():
    if os.path.isfile(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {}


def save_ckpt(c):
    tmp = CKPT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(c, f, indent=2)
    os.replace(tmp, CKPT)


def main():
    ckpt = load_ckpt()

    # Preload baseline-relative features for all boards once.
    cache = {}
    for dut in ALL_DUTS:
        Xr, y = load_baseline_relative(dut)
        cache[dut] = (Xr, y)
        print(f"Loaded {dut}: {Xr.shape} (baseline-relative)")

    for task_name, (relabel, labels) in TASKS.items():
        ckpt.setdefault(task_name, {})
        for train_duts, test_dut in FOLDS:
            fold_key = f"train_{'+'.join(train_duts)}__test_{test_dut}"
            ckpt[task_name].setdefault(fold_key, {})

            # Concatenate baseline-relative features across the training DUTs.
            Xtr = np.vstack([cache[d][0] for d in train_duts])
            ytr_raw = np.concatenate([cache[d][1] for d in train_duts])
            Xte, yte_raw = cache[test_dut]
            ytr, yte = relabel(ytr_raw), relabel(yte_raw)

            scaler = StandardScaler().fit(Xtr)   # fit on TRAIN boards only
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            for mt in MODELS:
                if mt in ckpt[task_name][fold_key]:
                    print(f"[skip] {task_name}/{fold_key}/{mt}")
                    continue
                print(f"\n[run] {task_name} / {fold_key} / {MODEL_NAMES[mt]}")
                t0 = time.time()
                model = make_model(mt)
                model.fit(Xtr_s, ytr)
                pred = model.predict(Xte_s)
                metrics = evaluate(yte, pred, labels)
                metrics['train_time_s'] = round(time.time() - t0, 2)
                ckpt[task_name][fold_key][mt] = metrics
                save_ckpt(ckpt)
                print(f"    acc={metrics['accuracy']:.4f} "
                      f"bal_acc={metrics['balanced_accuracy']:.4f} "
                      f"macroF1={metrics['macro_f1']:.4f} ({metrics['train_time_s']}s)")

    print(f"\nDone. Checkpoint: {CKPT}")


if __name__ == '__main__':
    main()
