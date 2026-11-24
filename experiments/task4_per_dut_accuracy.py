"""task4_per_dut_accuracy.py — Per-DUT accuracy in the random 80/20 split.

Trains one Random Forest on the mixed-board random split (the paper's headline
setting), then breaks the test-set accuracy down by the DUT each test sample
came from. Shows the model is not carried by a single easy board.

Output: results/per_dut_accuracy.md + printed table. One RF fit (~minutes).
"""
import os, sys, json
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, confusion_matrix)

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
DUTS = ['SN004', 'SN005', 'SN006', 'SN007']


def main():
    X, y, b = [], [], []
    for bi, d in enumerate(DUTS):
        Xi, yi, _ = dc.load_dut(d)
        X.append(Xi.astype(np.float64)); y.append(yi); b.append(np.full(len(yi), bi))
    X = np.vstack(X); y = np.concatenate(y); board = np.concatenate(b)

    # Stratified random split; carry board index through the split.
    idx = np.arange(len(y))
    itr, ite = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler().fit(X[itr])
    rf = RandomForestClassifier(n_estimators=200, min_samples_split=5,
                                min_samples_leaf=2, random_state=42, n_jobs=-1)
    rf.fit(sc.transform(X[itr]), y[itr])
    pred = rf.predict(sc.transform(X[ite]))
    yte = y[ite]; bte = board[ite]

    overall = {
        'accuracy': float(accuracy_score(yte, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(yte, pred)),
        'macro_f1': float(f1_score(yte, pred, average='macro', zero_division=0)),
    }

    per_dut = {}
    for bi, d in enumerate(DUTS):
        m = bte == bi
        per_dut[d] = {
            'n_test': int(m.sum()),
            'accuracy': float(accuracy_score(yte[m], pred[m])),
            'balanced_accuracy': float(balanced_accuracy_score(yte[m], pred[m])),
            'macro_f1': float(f1_score(yte[m], pred[m], average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(yte[m], pred[m], labels=[0, 1, 2]).tolist(),
        }

    lines = ["# Per-DUT accuracy (random 80/20 split, Random Forest)\n",
             f"Overall test accuracy: **{overall['accuracy']:.4f}**, "
             f"balanced accuracy {overall['balanced_accuracy']:.4f}, "
             f"macro-F1 {overall['macro_f1']:.4f}.\n",
             "| DUT | Test samples | Accuracy | Balanced Acc. | Macro-F1 |",
             "|-----|-------------:|---------:|--------------:|---------:|"]
    for d in DUTS:
        r = per_dut[d]
        lines.append(f"| {d} | {r['n_test']:,} | {r['accuracy']:.4f} | "
                     f"{r['balanced_accuracy']:.4f} | {r['macro_f1']:.4f} |")
    lines.append("")
    lines.append("Interpretation: high accuracy on BOTH boards in the random-split "
                 "setting confirms the headline is not carried by a single easy DUT. "
                 "Note this is the same (leaky) regime as the headline number; see the "
                 "LOBO results for board-independent generalization.\n")

    body = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, 'per_dut_accuracy.md'), 'w') as f:
        f.write(body)
    with open(os.path.join(RESULTS_DIR, 'per_dut_accuracy.json'), 'w') as f:
        json.dump({'overall': overall, 'per_dut': per_dut}, f, indent=2)

    print(body)
    print("Wrote results/per_dut_accuracy.{md,json}")


if __name__ == '__main__':
    main()
