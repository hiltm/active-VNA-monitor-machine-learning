"""task2_sample_breakdown.py — DUT x class x aging-day sample-count breakdown.

Reads the per-DUT feature cache (day + class metadata) and emits:
  - results/sample_breakdown_by_day.csv   (DUT, day, class, n_samples)
  - results/sample_breakdown.md            (human-readable tables)

Also documents precisely what one ML sample is. No training.
"""
import os
import sys
import numpy as np
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import data_cache as dc
import train_classifier as tc

RESULTS_DIR = dc.RESULTS_DIR   # honors CHIPTAMPER_OUTPUT_DIR
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASS_LABELS = tc.CLASS_LABELS


def main():
    duts = dc.all_duts()

    # (dut, day, class) -> count
    by_day = defaultdict(int)
    # (dut, class) -> count
    by_class = defaultdict(int)
    dut_totals = defaultdict(int)

    for dut in duts:
        X, y, day = dc.load_dut(dut)
        for d, c in zip(day.tolist(), y.tolist()):
            by_day[(dut, d, c)] += 1
            by_class[(dut, c)] += 1
            dut_totals[dut] += 1

    # --- CSV ---
    csv_path = os.path.join(RESULTS_DIR, 'sample_breakdown_by_day.csv')
    with open(csv_path, 'w') as f:
        f.write("dut,aging_day,class,class_label,n_samples\n")
        for (dut, d, c) in sorted(by_day.keys()):
            f.write(f"{dut},{d},{c},{CLASS_LABELS[c].replace(',', ';')},{by_day[(dut, d, c)]}\n")

    # --- Markdown ---
    lines = []
    lines.append("# Task 2 — Sample Breakdown\n")
    lines.append("## Definition of one ML sample\n")
    lines.append(
        "One ML **sample** is a single VNA frequency sweep of one DUT: 201 frequency "
        "points from 1 MHz to 4 GHz, capturing complex S11 and S21. Each sweep is "
        "converted into a single **1,612-dimensional feature vector** (see methodology). "
        "One `.s2p` file = one sweep = one sample. Class is assigned from the aging day "
        "of the measurement session: day 0 = Fresh (Class 0), days 1-3 = Lightly Aged "
        "(Class 1), days 4+ = Heavily Aged (Class 2). The first 5 minutes of each "
        "session are discarded as a thermal-settling guard window.\n")

    # Table 1: DUT x class
    lines.append("## DUT x Class totals\n")
    lines.append("| DUT | Class 0 (Fresh) | Class 1 (Lightly Aged) | Class 2 (Heavily Aged) | Total |")
    lines.append("|-----|----------------:|-----------------------:|-----------------------:|------:|")
    grand = [0, 0, 0]
    for dut in duts:
        row = [by_class[(dut, c)] for c in range(3)]
        for c in range(3):
            grand[c] += row[c]
        lines.append(f"| {dut} | {row[0]:,} | {row[1]:,} | {row[2]:,} | {sum(row):,} |")
    lines.append(f"| **Total** | **{grand[0]:,}** | **{grand[1]:,}** | **{grand[2]:,}** | **{sum(grand):,}** |")
    lines.append("")

    # Table 2: DUT x day x class
    lines.append("## DUT x Aging-Day x Class\n")
    lines.append("| DUT | Aging Day | Class | n_samples |")
    lines.append("|-----|----------:|:-----:|----------:|")
    for (dut, d, c) in sorted(by_day.keys()):
        lines.append(f"| {dut} | {d} | {c} | {by_day[(dut, d, c)]:,} |")
    lines.append("")

    # Class balance percentages
    lines.append("## Overall class balance\n")
    total = sum(grand)
    lines.append("| Class | Label | Count | Percent |")
    lines.append("|:-----:|-------|------:|--------:|")
    for c in range(3):
        lines.append(f"| {c} | {CLASS_LABELS[c]} | {grand[c]:,} | {100*grand[c]/total:.1f}% |")
    lines.append(f"| — | **Total** | **{total:,}** | 100.0% |")
    lines.append("")

    md_path = os.path.join(RESULTS_DIR, 'sample_breakdown.md')
    with open(md_path, 'w') as f:
        f.write("\n".join(lines))

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}\n")
    print("\n".join(lines))


if __name__ == '__main__':
    main()
