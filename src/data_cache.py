"""data_cache.py — Reusable, cached per-DUT feature loading.

extracts the 1612-dim feature vector from every .s2p file ONCE per DUT and
caches the result to models/cache/<DUT>_features.npz

cache contents per DUT:
    X    : (n_samples, 1612) float32 feature matrix
    y    : (n_samples,)      int class labels (0/1/2)
    day  : (n_samples,)      int aging-day number
    dut  : str               DUT name

feature extraction and discovery reuse train_classifier.py verbatim, so the
cached features are identical to what the deployed model was trained on

usage (as a module):
    from data_cache import load_dut, all_duts
    X, y, day = load_dut('SN005')

usage (CLI, to pre-build the cache):
    python data_cache.py            # build cache for all configured DUTs
    python data_cache.py --rebuild  # force re-extraction
"""
import os
import sys
import argparse
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
import train_classifier as tc

# resolve data/output through train_classifier so the env-var overrides
# (CHIPTAMPER_DATA_DIR / CHIPTAMPER_OUTPUT_DIR) apply everywhere, single source of truth
DATA_DIR = tc.DATA_ROOT
OUTPUT_ROOT = tc.OUTPUT_ROOT
CACHE_DIR = os.path.join(OUTPUT_ROOT, 'models', 'cache')
RESULTS_DIR = os.path.join(OUTPUT_ROOT, 'results')

os.makedirs(CACHE_DIR, exist_ok=True)


def all_duts():
    """DUTs present both in DATA_CONFIG and on disk"""
    return [d for d in tc.DATA_CONFIG.keys()
            if os.path.isdir(os.path.join(DATA_DIR, d))]


def _cache_path(dut):
    return os.path.join(CACHE_DIR, f'{dut}_features.npz')


def _day_for_file(filepath, dut):
    """recover the aging-day number for a file from its path, using DATA_CONFIG

    keys on the '<run>_<YYYYMMDD>' capture-session folder, NOT the filename
    timestamp. a capture session can span midnight, so day-N files sometimes
    carry a day-(N-1) timestamp in their name (observed on SN010); matching the
    folder avoids that collision
    """
    cfg = tc.DATA_CONFIG[dut]
    run_prefix = f"{cfg['run']}_"
    # walk the path components looking for the '<run>_<YYYYMMDD>' session folder
    for part in filepath.split(os.sep):
        if part.startswith(run_prefix):
            date_str = part[len(run_prefix):]
            if date_str in cfg['dates']:
                return cfg['dates'][date_str]
    return -1


def build_dut_cache(dut, rebuild=False, atomic=True):
    """extract features for one DUT and cache to disk. returns (X, y, day)"""
    path = _cache_path(dut)
    if os.path.isfile(path) and not rebuild:
        data = np.load(path, allow_pickle=True)
        return data['X'], data['y'], data['day']

    print(f"[cache] Building feature cache for {dut} ...")
    pairs = tc.discover_files(DATA_DIR, dut_list=[dut], discard_first_5min=True)
    if not pairs:
        raise RuntimeError(f"No files discovered for {dut}")

    X_list, y_list, day_list = [], [], []
    total = len(pairs)
    for i, (fp, label) in enumerate(pairs):
        if (i + 1) % 2000 == 0 or (i + 1) == total:
            print(f"    {dut}: {i+1}/{total} ({100*(i+1)/total:.1f}%)", end='\r')
        freqs, s11, s21 = tc.parse_s2p_file(fp)
        if freqs is None:
            continue
        feats = tc.extract_features(s11, s21, freqs)
        X_list.append(feats.astype(np.float32))
        y_list.append(label)
        day_list.append(_day_for_file(fp, dut))
    print()

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    day = np.asarray(day_list, dtype=np.int64)

    # atomic write: temp file + rename, so an interrupted write can't corrupt cache
    # NOTE: np.savez_compressed appends '.npz' to the given path if not present,
    # so we build the temp base without a trailing extension and account for it
    if atomic:
        tmp_base = path + '.tmp'          # np will write tmp_base + '.npz'
        tmp_written = tmp_base + '.npz'
        np.savez_compressed(tmp_base, X=X, y=y, day=day, dut=dut)
        os.replace(tmp_written, path)
    else:
        np.savez_compressed(path, X=X, y=y, day=day, dut=dut)

    print(f"[cache] {dut}: {X.shape[0]} samples x {X.shape[1]} features -> {path}")
    return X, y, day


def load_dut(dut, rebuild=False):
    """load cached features for a DUT (building the cache if needed)"""
    return build_dut_cache(dut, rebuild=rebuild)


def main():
    ap = argparse.ArgumentParser(description="Build per-DUT feature cache.")
    ap.add_argument('--rebuild', action='store_true', help='Force re-extraction')
    ap.add_argument('--duts', nargs='*', default=None, help='Subset of DUTs')
    args = ap.parse_args()

    duts = args.duts or all_duts()
    print(f"Building cache for DUTs: {duts}")
    for dut in duts:
        X, y, day = build_dut_cache(dut, rebuild=args.rebuild)
        unique, counts = np.unique(y, return_counts=True)
        dist = ", ".join(f"C{u}={c}" for u, c in zip(unique, counts))
        print(f"  {dut}: X{X.shape}  classes: {dist}")


if __name__ == '__main__':
    main()
