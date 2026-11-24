"""aggregate_lobo4.py — Summarize 4-fold LOBO (baseline-relative) results.

Reads results/lobo_baseline_relative.json and produces per-fold tables plus
mean +/- std across the 4 held-out-board folds, for each task and model.
Writes results/lobo4_summary.md.
"""
import os, json
import numpy as np

# outputs live under CHIPTAMPER_OUTPUT_DIR (defaults to repo root)
BASE = os.environ.get('CHIPTAMPER_OUTPUT_DIR',
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CKPT = os.path.join(BASE, 'results', 'lobo_baseline_relative.json')
OUT = os.path.join(BASE, 'results', 'lobo4_summary.md')

TASK_LABEL = {
    'heavy_vs_rest': 'Heavily-Aged (4+ d) vs rest',
    'fresh_vs_aged': 'Fresh vs any-Aged',
    '3class': '3-class (Fresh/Light/Heavy)',
}
MODEL_LABEL = {'linearsvc': 'LinearSVC', 'rf': 'Random Forest'}
HELD = ['SN004', 'SN005', 'SN006', 'SN007']


def held_from_key(k):
    return k.split('__test_')[1]


def main():
    d = json.load(open(CKPT))
    lines = ["# 4-fold Leave-One-Board-Out (baseline-relative features)\n",
             "Each fold trains on 3 aging DUTs and tests on the fully held-out 4th DUT. "
             "Features are referenced to each board's own day-0 (Fresh) mean before "
             "training; the scaler is fit on the training boards only.\n"]

    for task in ['heavy_vs_rest', 'fresh_vs_aged', '3class']:
        if task not in d:
            continue
        lines.append(f"## {TASK_LABEL[task]}\n")
        lines.append("Balanced accuracy per held-out board:\n")
        lines.append("| Model | " + " | ".join(HELD) + " | Mean | Std |")
        lines.append("|-------|" + "|".join(["------:"] * (len(HELD) + 2)) + "|")
        for m in ['linearsvc', 'rf']:
            row_vals = {}
            for fold_key, md in d[task].items():
                if m in md:
                    row_vals[held_from_key(fold_key)] = md[m]['balanced_accuracy']
            vals = [row_vals.get(h) for h in HELD]
            arr = np.array([v for v in vals if v is not None], float)
            mean = arr.mean() if arr.size else float('nan')
            std = arr.std(ddof=1) if arr.size > 1 else 0.0
            cells = " | ".join(f"{v:.4f}" if v is not None else "-" for v in vals)
            lines.append(f"| {MODEL_LABEL[m]} | {cells} | {mean:.4f} | {std:.4f} |")
        lines.append("")

    # Also emit accuracy means for the master table
    lines.append("## Accuracy (mean over 4 folds)\n")
    lines.append("| Task | Model | Mean Acc | Mean Bal.Acc | Mean Macro-F1 |")
    lines.append("|------|-------|---------:|-------------:|--------------:|")
    for task in ['heavy_vs_rest', 'fresh_vs_aged', '3class']:
        if task not in d:
            continue
        for m in ['linearsvc', 'rf']:
            accs, bals, f1s = [], [], []
            for md in d[task].values():
                if m in md:
                    accs.append(md[m]['accuracy'])
                    bals.append(md[m]['balanced_accuracy'])
                    f1s.append(md[m]['macro_f1'])
            if accs:
                lines.append(f"| {TASK_LABEL[task]} | {MODEL_LABEL[m]} | "
                             f"{np.mean(accs):.4f} | {np.mean(bals):.4f} | {np.mean(f1s):.4f} |")
    lines.append("")

    body = "\n".join(lines)
    with open(OUT, 'w') as f:
        f.write(body)
    print(body)
    print(f"Wrote {OUT}")


if __name__ == '__main__':
    main()
