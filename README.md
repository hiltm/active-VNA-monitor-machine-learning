# RF-Reflectometry Board-Aging Classifier

Detects recycled/counterfeit PCBs from active RF reflectometry. Each sample is
one VNA sweep (201 points, 1 MHz–4 GHz, complex `S11` + `S21`) of a powered
DUT; a 1,612-dimensional feature vector is classified into an aging state
(Fresh / Lightly Aged / Heavily Aged).

## Repository layout

```
src/          train_classifier.py, data_cache.py       # shared core: features + caching
experiments/  task1_lobo, task1b, task2 … task9        # training / evaluation runs
analysis/     aggregate_lobo4, aggregate_repeated_runs,  # summaries & diagnostics
              count_samples, diagnose_board_offset
viz/          plot_style, plot_feature_importance_*,     # figures
              generate_4dut_figures, make_trajectory_figure
export/       export_to_mat.py, generate_figures.m       # MATLAB export
deploy/       classify_measurement.py, package_for_pi.sh,# Raspberry Pi inference
              requirements-inference.txt
data/         (not tracked) raw .s2p measurements
models/       (not tracked) feature cache, trained models, figures
results/      (not tracked) generated JSON / markdown summaries
```

All scripts resolve `data/`, `models/`, and `results/` relative to the
repository root, so they must be run from the repo root (e.g.
`python experiments/task1b_lobo_baseline_relative.py`), not from inside a
subfolder.

## Data layout

```
data/<DUT>/<run>_<YYYYMMDD>/<YYYYMMDD>/*.s2p   # per-day aging measurements
data/<DUT>/<run>_<YYYYMMDD>/baseline/*.s2p     # day-0 Fresh baseline
```

DUTs, runs, and day→class mapping are declared in `DATA_CONFIG` at the top of
`src/train_classifier.py`. Day 0 = Fresh (class 0), days 1–3 = Lightly Aged
(class 1), days 4+ = Heavily Aged (class 2). The first 5 minutes of each
session are discarded as a warm-up guard.

## One sample → one feature vector (d = 1612)

| Group | Count | Definition |
|-------|------:|------------|
| S11/S21 magnitude (dB) | 2×201 | `20·log10(|S|)` |
| S11/S21 phase (deg)    | 2×201 | `angle(S)` |
| S11/S21 group delay (ns) | 2×200 | `−dφ/dω`, unwrapped phase, finite diff. |
| S11/S21 mag slope (dB/GHz) | 2×200 | `d(|S|dB)/df`, finite diff. |
| Statistical summaries | 8 | mean/std of magnitude & phase |

Implemented in `src/train_classifier.py::extract_features`.

## Reproduce the headline (board-independent) results

Run from the repository root:

```bash
source .venv/bin/activate
python src/data_cache.py                              # build per-DUT feature caches (once)
python experiments/task1b_lobo_baseline_relative.py   # 4-fold LOBO, baseline-relative
python analysis/aggregate_lobo4.py                    # mean ± std summary tables
```

`task1b` writes `results/lobo_baseline_relative.json`; `aggregate_lobo4.py`
summarizes it into `results/lobo4_summary.md`. Headline (random forest):
3-class 93.5% ± 8.5% balanced accuracy, Heavy-vs-rest 89.4% ± 15.7%,
Fresh-vs-aged 99.7% ± 0.4%.

> Do **not** use the 80/20 random split in `src/train_classifier.py` as a
> generalization measure — it reaches ~99.8% by memorizing board identity
> (board variance ≈ 2.7× class variance). LOBO is the honest protocol.

## Extended models (dimensionality reduction + learned/uncertainty)

```bash
python experiments/task6_extended_models.py           # PCA/PLS/MLP/approx-GP under LOBO
```

Writes `results/task6_extended.json`. PLS(10) matches full-dim RF; the
Nyström approx-GP returns calibrated probabilities (Brier score reported).

## Script index

| Script | Role |
|--------|------|
| `src/train_classifier.py` | feature extraction Φ; random-split training; plots |
| `src/data_cache.py` | extract Φ once per DUT → `models/cache/<DUT>_features.npz` |
| `experiments/task1b_lobo_baseline_relative.py` | 4-fold LOBO on baseline-relative features (**headline**) |
| `experiments/task6_extended_models.py` | PCA/PLS/MLP/approx-GP under the same LOBO |
| `experiments/task9_trajectory.py` | continuous aging-hours regression under LOBO |
| `analysis/aggregate_lobo4.py` | per-fold and mean±std summary tables |
| `analysis/diagnose_board_offset.py` | between-board vs between-class variance (the 2.7× result) |
| `analysis/count_samples.py` | sample/class/day breakdown |
| `deploy/classify_measurement.py` | run a saved model on new `.s2p` files |
| `viz/plot_style.py` | IEEE figure style (png/pdf/svg/pkl) |
| `export/export_to_mat.py`, `export/generate_figures.m` | export results + MATLAB `.fig` figures |

## Adding the new DUTs (path to the 8-DUT run)

1. Drop the new DUTs under `data/<DUT>/...` in the same layout.
2. Add each to `DATA_CONFIG` in `src/train_classifier.py` (run number + day→date).
3. Add the DUT names to `ALL_DUTS` in
   `experiments/task1b_lobo_baseline_relative.py` and
   `experiments/task6_extended_models.py` (the LOBO fold list is generated from
   this set).
4. Re-run the three headline commands above. Everything else is unchanged.
