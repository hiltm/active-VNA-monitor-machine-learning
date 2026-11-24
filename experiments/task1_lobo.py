"""task1_lobo.py — Leave-One-Board-Out (LOBO) evaluation

two aged DUTs exist (SN004, SN005), so LOBO is 2-fold:
    Fold A: train on SN004, test on SN005 (fully unseen board)
    Fold B: train on SN005, test on SN004 (fully unseen board)

for each fold:
  - StandardScaler is fit ONLY on the training board (no test-board leakage).
  - Three models are evaluated: KNN, LinearSVC (no probability), Random Forest.
  - Metrics: accuracy, balanced accuracy, macro-F1, weighted-F1 + confusion matrix.

Checkpointing: results are written to results/lobo_checkpoint.json after every
(fold, model) completes. Re-running skips already-completed (fold, model) pairs,
so an interruption resumes rather than restarts.

Usage:
    python task1_lobo.py
    python task1_lobo.py --force   # ignore checkpoint, recompute everything
"""
import os
import sys
import json
import time
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
)

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
CKPT_PATH = os.path.join(RESULTS_DIR, 'lobo_checkpoint.json')

# LOBO folds: (train_dut, test_dut)
FOLDS = [('SN004', 'SN005'), ('SN005', 'SN004')]
MODEL_ORDER = ['knn', 'linearsvc', 'rf']
MODEL_NAMES = {'knn': 'KNN', 'linearsvc': 'LinearSVC', 'rf': 'Random Forest'}


def make_model(model_type):
    if model_type == 'knn':
        return KNeighborsClassifier(n_neighbors=5, weights='distance',
                                    metric='euclidean', n_jobs=-1)
    if model_type == 'linearsvc':
        # No probability (LinearSVC has none by design) — fast, deterministic.
        return LinearSVC(C=1.0, random_state=42, dual='auto', max_iter=5000)
    if model_type == 'rf':
        return RandomForestClassifier(n_estimators=200, max_depth=None,
                                      min_samples_split=5, min_samples_leaf=2,
                                      random_state=42, n_jobs=-1)
    raise ValueError(model_type)


def load_checkpoint():
    if os.path.isfile(CKPT_PATH):
        with open(CKPT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(ckpt):
    tmp = CKPT_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp, CKPT_PATH)


def evaluate(y_true, y_pred):
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true', help='Ignore checkpoint')
    args = ap.parse_args()

    ckpt = {} if args.force else load_checkpoint()

    # Preload both DUT caches once.
    cache = {}
    for dut in set([d for fold in FOLDS for d in fold]):
        X, y, _ = dc.load_dut(dut)
        cache[dut] = (X.astype(np.float64), y)
        print(f"Loaded {dut}: {X.shape}")

    for train_dut, test_dut in FOLDS:
        fold_key = f"train_{train_dut}__test_{test_dut}"
        ckpt.setdefault(fold_key, {})

        Xtr, ytr = cache[train_dut]
        Xte, yte = cache[test_dut]

        # Scaler fit on TRAIN board only.
        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        for model_type in MODEL_ORDER:
            if model_type in ckpt[fold_key]:
                print(f"[skip] {fold_key} / {model_type} (already in checkpoint)")
                continue

            print(f"\n[run] {fold_key} / {MODEL_NAMES[model_type]}")
            t0 = time.time()
            model = make_model(model_type)
            model.fit(Xtr_s, ytr)
            y_pred = model.predict(Xte_s)
            metrics = evaluate(yte, y_pred)
            metrics['train_time_s'] = round(time.time() - t0, 2)
            metrics['n_train'] = int(Xtr_s.shape[0])
            metrics['n_test'] = int(Xte_s.shape[0])

            ckpt[fold_key][model_type] = metrics
            save_checkpoint(ckpt)  # checkpoint after every (fold, model)

            print(f"    acc={metrics['accuracy']:.4f} "
                  f"bal_acc={metrics['balanced_accuracy']:.4f} "
                  f"macroF1={metrics['macro_f1']:.4f} "
                  f"wF1={metrics['weighted_f1']:.4f} "
                  f"({metrics['train_time_s']}s)")

    print(f"\nAll folds complete. Checkpoint: {CKPT_PATH}")


if __name__ == '__main__':
    main()
