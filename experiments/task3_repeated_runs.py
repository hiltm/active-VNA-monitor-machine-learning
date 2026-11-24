"""task3_repeated_runs.py — Variability across repeated training runs (random split).

Runs the random 80/20 mixed-board split N times with different seeds (the seed
controls both the split and the RF bootstrap). Reports mean +/- std and a 95%
confidence interval for accuracy and macro-F1, per model.

Models: RandomForest and LinearSVC. KNN is intentionally EXCLUDED — its full
120k-test prediction costs ~6 min/run and adds little to a variability study.

Checkpoint: results/repeated_runs.json, written after every (seed, model), so
an interruption resumes rather than restarts.

Usage: python task3_repeated_runs.py [--n 10]
"""
import os, sys, json, time, argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)
CKPT = os.path.join(RESULTS_DIR, 'repeated_runs.json')

MODELS = ['linearsvc', 'rf']
MODEL_NAMES = {'linearsvc': 'LinearSVC', 'rf': 'Random Forest'}


def make_model(mt, seed):
    if mt == 'linearsvc':
        return LinearSVC(C=1.0, dual='auto', max_iter=5000, random_state=seed)
    if mt == 'rf':
        return RandomForestClassifier(n_estimators=200, min_samples_split=5,
                                      min_samples_leaf=2, random_state=seed, n_jobs=-1)
    raise ValueError(mt)


def load_all():
    X, y = [], []
    for d in ['SN004', 'SN005', 'SN006', 'SN007']:
        Xi, yi, _ = dc.load_dut(d)
        X.append(Xi.astype(np.float64)); y.append(yi)
    return np.vstack(X), np.concatenate(y)


def load_ckpt():
    if os.path.isfile(CKPT):
        with open(CKPT) as f:
            return json.load(f)
    return {'runs': {}}


def save_ckpt(c):
    tmp = CKPT + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(c, f, indent=2)
    os.replace(tmp, CKPT)


def summarize(vals):
    a = np.array(vals, dtype=float)
    n = len(a)
    mean, std = float(a.mean()), float(a.std(ddof=1)) if n > 1 else 0.0
    # 95% CI via normal approx (n=10 -> report t would be ~2.26; use 1.96 for simplicity, note in writeup)
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {'mean': mean, 'std': std, 'ci95': float(ci), 'n': n,
            'values': [round(v, 5) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=10)
    args = ap.parse_args()

    X, y = load_all()
    print(f"Loaded mixed dataset: {X.shape}")

    ckpt = load_ckpt()

    for seed in range(args.n):
        skey = str(seed)
        ckpt['runs'].setdefault(skey, {})
        # Split depends on seed; only compute once per seed.
        need = [m for m in MODELS if m not in ckpt['runs'][skey]]
        if not need:
            print(f"[skip] seed {seed} (complete)")
            continue

        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

        for mt in need:
            t0 = time.time()
            model = make_model(mt, seed)
            model.fit(Xtr_s, ytr)
            p = model.predict(Xte_s)
            rec = {
                'accuracy': float(accuracy_score(yte, p)),
                'balanced_accuracy': float(balanced_accuracy_score(yte, p)),
                'macro_f1': float(f1_score(yte, p, average='macro', zero_division=0)),
                'weighted_f1': float(f1_score(yte, p, average='weighted', zero_division=0)),
                'time_s': round(time.time() - t0, 1),
            }
            ckpt['runs'][skey][mt] = rec
            save_ckpt(ckpt)
            print(f"  seed {seed} {MODEL_NAMES[mt]:<14} acc={rec['accuracy']:.4f} "
                  f"macroF1={rec['macro_f1']:.4f} ({rec['time_s']}s)")

    # Aggregate
    summary = {}
    for mt in MODELS:
        accs = [ckpt['runs'][s][mt]['accuracy'] for s in ckpt['runs'] if mt in ckpt['runs'][s]]
        f1s = [ckpt['runs'][s][mt]['macro_f1'] for s in ckpt['runs'] if mt in ckpt['runs'][s]]
        bals = [ckpt['runs'][s][mt]['balanced_accuracy'] for s in ckpt['runs'] if mt in ckpt['runs'][s]]
        summary[mt] = {
            'accuracy': summarize(accs),
            'macro_f1': summarize(f1s),
            'balanced_accuracy': summarize(bals),
        }
    ckpt['summary'] = summary
    save_ckpt(ckpt)

    print("\n=== Repeated-run summary (random split, N runs) ===")
    for mt in MODELS:
        s = summary[mt]
        print(f"\n{MODEL_NAMES[mt]} (n={s['accuracy']['n']}):")
        print(f"  Accuracy : {s['accuracy']['mean']:.4f} +/- {s['accuracy']['std']:.4f} "
              f"(95% CI +/-{s['accuracy']['ci95']:.4f})")
        print(f"  Macro-F1 : {s['macro_f1']['mean']:.4f} +/- {s['macro_f1']['std']:.4f} "
              f"(95% CI +/-{s['macro_f1']['ci95']:.4f})")
    print(f"\nCheckpoint: {CKPT}")


if __name__ == '__main__':
    main()
