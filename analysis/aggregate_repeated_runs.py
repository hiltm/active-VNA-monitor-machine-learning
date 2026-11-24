"""aggregate_repeated_runs.py — Summarize the repeated-run checkpoint over the
seeds that completed BOTH models. emits mean +/- std and a 95% CI (Student-t)
for accuracy, balanced accuracy, and macro-F1. Writes results/repeated_runs_summary.md
"""
import os, sys, json
import numpy as np

# outputs live under CHIPTAMPER_OUTPUT_DIR (defaults to repo root)
BASE = os.environ.get('CHIPTAMPER_OUTPUT_DIR',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(BASE, 'results', 'repeated_runs.json')
OUT = os.path.join(BASE, 'results', 'repeated_runs_summary.md')

MODELS = ['linearsvc', 'rf']
NAMES = {'linearsvc': 'LinearSVC', 'rf': 'Random Forest'}
# student-t 97.5% quantiles for small n (df = n-1)
T = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
     7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def stat(vals):
    a = np.array(vals, float); n = len(a)
    mean = float(a.mean()); std = float(a.std(ddof=1)) if n > 1 else 0.0
    df = n - 1
    ci = (T.get(df, 2.0) * std / np.sqrt(n)) if n > 1 else 0.0
    return mean, std, ci, n


def main():
    d = json.load(open(CKPT))
    runs = d['runs']
    # keep only seeds with BOTH models complete
    complete = [s for s, v in runs.items() if all(m in v for m in MODELS)]
    complete = sorted(complete, key=int)
    n = len(complete)

    lines = [f"# Repeated-run variability (random 80/20 split, mixed boards)\n",
             f"Seeds with both models complete: **N = {n}** (seeds {', '.join(complete)}).\n",
             "95% CI uses the Student-t quantile for df = N-1.\n"]

    metric_keys = [('accuracy', 'Accuracy'),
                   ('balanced_accuracy', 'Balanced Acc.'),
                   ('macro_f1', 'Macro-F1')]

    lines.append("| Model | Metric | Mean | Std | 95% CI |")
    lines.append("|-------|--------|-----:|----:|:------:|")
    summary = {}
    for m in MODELS:
        summary[m] = {}
        for mk, mlabel in metric_keys:
            vals = [runs[s][m][mk] for s in complete]
            mean, std, ci, _ = stat(vals)
            summary[m][mk] = {'mean': mean, 'std': std, 'ci95': ci, 'n': n,
                              'values': [round(v, 5) for v in vals]}
            lines.append(f"| {NAMES[m]} | {mlabel} | {mean:.4f} | {std:.4f} | ±{ci:.4f} |")

    body = "\n".join(lines) + "\n"
    with open(OUT, 'w') as f:
        f.write(body)
    # also stash a clean summary back into the checkpoint json
    d['summary_final'] = {'n': n, 'seeds': complete, 'metrics': summary}
    tmp = CKPT + '.tmp'; json.dump(d, open(tmp, 'w'), indent=2); os.replace(tmp, CKPT)

    print(body)
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
