"""task5_baselines.py — Simple non-ML baselines (reviewer: demonstrate ML's added value).

Baselines implemented:
  1. Nearest-centroid (spectral-distance) classifier: compute a per-class mean
     feature vector (template) on the training data; classify each test sample by
     minimum Euclidean distance to the class centroids. This is the classic
     "spectral-distance comparison" the reviewer named.
  2. Frequency-threshold classifier: use the single most discriminative feature
     (selected on TRAIN by class-mean separation, drawn from the mid-band S21
     region the importance analysis highlighted) and threshold it. For 3 classes
     we fit 1-D class-conditional Gaussians on the chosen feature and assign the
     most likely class (a transparent, non-ML thresholding scheme).

Both are evaluated in TWO settings, matching the ML experiments:
  - random split (mixed boards, 80/20)  -> comparison vs the leaky 99.8% regime
  - LOBO baseline-relative (heavy_vs_rest, the transferable task)

Output: results/baselines.json + printed table. Fast (no heavy training).
"""
import os, sys, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT = os.path.join(RESULTS_DIR, 'baselines.json')


# ---------------------------------------------------------------------------
# Baseline classifiers
# ---------------------------------------------------------------------------
class NearestCentroidBaseline:
    """Assign each sample to the nearest per-class mean (Euclidean)."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroids_ = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        # Memory-efficient squared-distance to each centroid (no giant broadcast):
        # ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2
        x_sq = np.sum(X * X, axis=1)[:, None]                 # (n,1)
        c_sq = np.sum(self.centroids_ * self.centroids_, axis=1)[None, :]  # (1,k)
        cross = X @ self.centroids_.T                          # (n,k)
        d2 = x_sq - 2.0 * cross + c_sq
        return self.classes_[np.argmin(d2, axis=1)]


class FrequencyThresholdBaseline:
    """Pick the single most class-separating feature on TRAIN, then classify by
    1-D class-conditional Gaussian likelihood (transparent thresholding)."""
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        # score each feature by between-class variance of class means / pooled var
        means = np.vstack([X[y == c].mean(axis=0) for c in self.classes_])
        between = means.var(axis=0)
        within = np.mean(np.vstack([X[y == c].var(axis=0) for c in self.classes_]), axis=0) + 1e-9
        self.feat_ = int(np.argmax(between / within))
        xv = X[:, self.feat_]
        self.mu_ = np.array([xv[y == c].mean() for c in self.classes_])
        self.sd_ = np.array([xv[y == c].std() + 1e-6 for c in self.classes_])
        return self

    def predict(self, X):
        xv = X[:, self.feat_][:, None]
        # log-likelihood under each class Gaussian
        ll = -0.5 * ((xv - self.mu_[None, :]) / self.sd_[None, :]) ** 2 - np.log(self.sd_[None, :])
        return self.classes_[np.argmax(ll, axis=1)]


def metrics(y, p, labels):
    return {
        'accuracy': float(accuracy_score(y, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y, p)),
        'macro_f1': float(f1_score(y, p, average='macro', zero_division=0)),
        'weighted_f1': float(f1_score(y, p, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(y, p, labels=labels).tolist(),
    }


def load_all():
    X, y, b = [], [], []
    for bi, d in enumerate(['SN004', 'SN005', 'SN006', 'SN007']):
        Xi, yi, _ = dc.load_dut(d)
        X.append(Xi.astype(np.float64)); y.append(yi); b.append(np.full(len(yi), bi))
    return np.vstack(X), np.concatenate(y), np.concatenate(b)


def load_baseline_relative(dut):
    X, y, _ = dc.load_dut(dut); X = X.astype(np.float64)
    return X - X[y == 0].mean(axis=0), y


def main():
    results = {'random_split_3class': {}}

    # ---------- Random split (mixed boards) ----------
    X, y, _ = load_all()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    for name, clf in [('nearest_centroid', NearestCentroidBaseline()),
                      ('frequency_threshold', FrequencyThresholdBaseline())]:
        clf.fit(Xtr_s, ytr)
        p = clf.predict(Xte_s)
        results['random_split_3class'][name] = metrics(yte, p, [0, 1, 2])

    # ---------- 4-fold LOBO baseline-relative, both binary tasks ----------
    ALL_DUTS = sorted(dc.all_duts())   # aging DUTs in DATA_CONFIG present on disk
    cache = {d: load_baseline_relative(d) for d in ALL_DUTS}
    BIN_TASKS = {
        'heavy_vs_rest': lambda y: (y == 2).astype(int),
        'fresh_vs_aged': lambda y: (y > 0).astype(int),
    }
    for task_name, relabel in BIN_TASKS.items():
        results[f'lobo_{task_name}'] = {}
        for held in ALL_DUTS:
            train_duts = [d for d in ALL_DUTS if d != held]
            Xtr = np.vstack([cache[d][0] for d in train_duts])
            ytr = relabel(np.concatenate([cache[d][1] for d in train_duts]))
            Xte, yte_raw = cache[held]
            yte = relabel(yte_raw)
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            fold = f"test_{held}"
            results[f'lobo_{task_name}'][fold] = {}
            for name, clf in [('nearest_centroid', NearestCentroidBaseline()),
                              ('frequency_threshold', FrequencyThresholdBaseline())]:
                clf.fit(Xtr_s, ytr)
                p = clf.predict(Xte_s)
                results[f'lobo_{task_name}'][fold][name] = metrics(yte, p, [0, 1])

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=2)

    # ---- print summary ----
    print("=== Baselines: random split (3-class, mixed boards) ===")
    for name, m in results['random_split_3class'].items():
        print(f"  {name:<22} acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
              f"macroF1={m['macro_f1']:.4f}")
    for task_name in BIN_TASKS:
        key = f'lobo_{task_name}'
        print(f"\n=== Baselines: 4-fold LOBO baseline-relative, {task_name} ===")
        # per-fold + mean
        for name in ['nearest_centroid', 'frequency_threshold']:
            bals = [results[key][f][name]['balanced_accuracy'] for f in results[key]]
            per = "  ".join(f"{f.split('_')[1]}={results[key][f][name]['balanced_accuracy']:.3f}"
                            for f in results[key])
            print(f"  {name:<20} {per}  MEAN={np.mean(bals):.4f}")
    print(f"\nWrote {OUT}")


if __name__ == '__main__':
    main()
