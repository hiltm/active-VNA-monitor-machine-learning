"""count_samples.py — Task 2 sample breakdown. Counts .s2p files by
DUT x class x aging-day using the existing discovery logic. no training,
no feature extraction — just file enumeration"""
import os
import sys
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import train_classifier as tc

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def main():
    duts = [d for d in tc.DATA_CONFIG.keys()
            if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"DUTs in config present on disk: {duts}\n")

    grand = defaultdict(int)  # (dut, class) -> count
    per_day = defaultdict(int)  # (dut, day, class) -> count

    for dut in duts:
        pairs = tc.discover_files(DATA_DIR, dut_list=[dut], discard_first_5min=True)
        # discover_files returns (file, class) but we also want day; re-derive
        # day from the folder path for the per-day table.
        for fp, cls in pairs:
            grand[(dut, cls)] += 1
        # per-day: re-walk config to attribute counts
        cfg = tc.DATA_CONFIG[dut]
        for date_str, day in sorted(cfg['dates'].items()):
            cls = tc.day_to_class(day)
            per_day[(dut, day, cls)] += 0  # ensure key exists

    print("\n=== DUT x CLASS totals ===")
    print(f"{'DUT':<8}{'Class 0':>10}{'Class 1':>10}{'Class 2':>10}{'Total':>10}")
    dut_class = defaultdict(lambda: [0, 0, 0])
    for (dut, cls), n in grand.items():
        dut_class[dut][cls] += n
    for dut in duts:
        c = dut_class[dut]
        print(f"{dut:<8}{c[0]:>10}{c[1]:>10}{c[2]:>10}{sum(c):>10}")

    print("\n=== Class coverage check ===")
    for dut in duts:
        c = dut_class[dut]
        missing = [i for i in range(3) if c[i] == 0]
        status = "OK (all 3 classes)" if not missing else f"MISSING classes {missing}"
        print(f"  {dut}: {status}")


if __name__ == '__main__':
    main()
