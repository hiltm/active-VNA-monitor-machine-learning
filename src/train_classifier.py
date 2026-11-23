#!/usr/bin/env python3
"""
train_classifier.py - RF Reflectometry Board Aging Classifier (Training)

trains a 3-class classifier to determine board aging state from .s2p measurements:
  Class 0: Fresh / Unaged (baseline measurements)
  Class 1: Lightly Aged (Days 1-3 of accelerated aging)
  Class 2: Heavily Aged (Days 4+ of accelerated aging)

primary model: K-Nearest Neighbors (KNN)
optional models: Support Vector Machine (SVM), Random Forest (RF)

features extracted per measurement (1,608 total):
  - S11 magnitude (dB) at each of 201 frequency points
  - S11 phase (degrees) at each of 201 frequency points
  - S21 magnitude (dB) at each of 201 frequency points
  - S21 phase (degrees) at each of 201 frequency points
  - S11 group delay (ns) at each of 200 frequency intervals
  - S21 group delay (ns) at each of 200 frequency intervals
  - S11 magnitude slope (dB/GHz) at each of 200 frequency intervals
  - S21 magnitude slope (dB/GHz) at each of 200 frequency intervals
  - 8 statistical summary features (mean/std of magnitudes and phases)

data format: 2-port S-parameter files (.s2p) with 201 frequency points (1 MHz - 4 GHz)
note: the VNA captures S11 and S21, but the logging script writes S21 data into the S12 columns
columns labeled "S21" in the file are zeros; actual S21 is in S12 columns

usage:
    python train_classifier.py [--data-dir DATA_DIR] [--sample-rate RATE] [--model MODEL]
                               [--no-discard] [--output-dir OUTPUT_DIR]

author: Michael Hilt
date: 2026-07-18
"""

import os
import sys
import glob
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, auc
)
from sklearn.inspection import permutation_importance
import joblib

# try importing matplotlib (optional, for plots)
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_CONFIG = {
    'SN005': {
        'run': 32,
        'dates': {
            '20260708': 0,   # Day 0 - baseline
            '20260709': 1,   # Day 1
            '20260710': 2,   # Day 2
            '20260711': 3,   # Day 3
            '20260712': 4,   # Day 4
            '20260713': 5,   # Day 5
            '20260714': 6,   # Day 6
            '20260715': 7,   # Day 7
        },
        'baseline_folder': 'baseline',
    },
    'SN004': {
        'run': 29,
        'dates': {
            '20260621': 0,   # Day 0 - baseline
            '20260622': 1,   # Day 1
            '20260623': 2,   # Day 2
            '20260624': 3,   # Day 3
            '20260625': 4,   # Day 4
            '20260626': 5,   # Day 5
        },
        'baseline_folder': 'baseline',
    },
    'SN006': {
        'run': 33,
        'dates': {
            '20260717': 0,   # Day 0 - baseline
            '20260718': 1,   # Day 1
            '20260719': 2,   # Day 2
            '20260720': 3,   # Day 3
            '20260721': 4,   # Day 4
            '20260722': 5,   # Day 5
        },
        'baseline_folder': 'baseline',
    },
    'SN007': {
        'run': 34,
        'dates': {
            '20260724': 0,   # Day 0 - baseline
            '20260725': 1,   # Day 1
            '20260726': 2,   # Day 2
            '20260727': 3,   # Day 3
            '20260728': 4,   # Day 4
            '20260729': 5,   # Day 5
            '20260730': 6,   # Day 6
            '20260731': 7,   # Day 7
        },
        'baseline_folder': 'baseline',
    },
}

CLASS_LABELS = {
    0: 'Fresh (Baseline)',
    1: 'Lightly Aged (Days 1-3)',
    2: 'Heavily Aged (Days 4+)',
}

# Sixers colors palette (used in listed order throughout all plots)
SIXERS_COLORS = [
    '#006bb6',   # blue
    '#ed174c',   # red
    '#002b5c',   # navy
    '#007a33',   # green
    '#00a9e0',   # light blue
    '#963821',   # brown
    '#bb9754',   # gold
    '#2d2d2d',   # charcoal
    '#f5b400',   # orange-yellow
    '#324b1e',   # dark olive
    '#fabece',   # light pink
    '#781478',   # dark plum
    '#fff5b4',   # light yellow
    '#009696',   # medium teal
    '#ffa07a',   # light coral
]

CLASS_COLORS = {
    0: SIXERS_COLORS[0],   # blue
    1: SIXERS_COLORS[1],   # red
    2: SIXERS_COLORS[2],   # navy
}

N_FREQ_POINTS = 201
DISCARD_MINUTES = 5


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def day_to_class(day_number):
    """map a day number to a class label."""
    if day_number == 0:
        return 0
    elif 1 <= day_number <= 3:
        return 1
    else:
        return 2


def parse_timestamp_from_filename(filename):
    """extract datetime from filename like 'measurement_YYYYMMDD_HHMMSS_testdata.s2p'."""
    basename = os.path.basename(filename)
    parts = basename.split('_')
    if len(parts) >= 3:
        try:
            date_str = parts[1]
            time_str = parts[2]
            return datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            return None
    return None


def get_discard_boundary(file_list):
    """determine the timestamp boundary after which data should be kept"""
    earliest = None
    for f in file_list[:100]:
        ts = parse_timestamp_from_filename(f)
        if ts is not None:
            if earliest is None or ts < earliest:
                earliest = ts
    if earliest is None:
        return None
    return earliest + timedelta(minutes=DISCARD_MINUTES)


def get_frequency_axis():
    """return the frequency axis in GHz for plotting"""
    return np.linspace(1e6, 4e9, N_FREQ_POINTS) / 1e9


def parse_s2p_file(filepath):
    """parse a .s2p file and extract S11 and S21 (from S12 columns) data
    
    returns:
        freqs: numpy array of frequency points (Hz)
        s11_complex: numpy array of complex S11 values
        s21_complex: numpy array of complex S21 values (read from S12 columns)
    returns (None, None, None) if parsing fails
    """
    try:
        freqs = []
        s11_re, s11_im = [], []
        s21_re, s21_im = [], []
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('!') or line.startswith('#') or not line:
                    continue
                parts = line.split()
                if len(parts) >= 7:
                    freqs.append(float(parts[0]))
                    s11_re.append(float(parts[1]))
                    s11_im.append(float(parts[2]))
                    s21_re.append(float(parts[5]))
                    s21_im.append(float(parts[6]))
        
        if len(freqs) != N_FREQ_POINTS:
            return None, None, None
        
        freqs = np.array(freqs)
        s11_complex = np.array(s11_re) + 1j * np.array(s11_im)
        s21_complex = np.array(s21_re) + 1j * np.array(s21_im)
        return freqs, s11_complex, s21_complex
    except (IOError, ValueError):
        return None, None, None


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_features(s11_complex, s21_complex, freqs=None):
    """extract expanded feature vector from S-parameter data
    
    features extracted:
      1. S11 magnitude (dB) - 201 features
      2. S11 phase (degrees) - 201 features
      3. S21 magnitude (dB) - 201 features
      4. S21 phase (degrees) - 201 features
      5. S11 group delay (ns) - 200 features (derivative of unwrapped phase)
      6. S21 group delay (ns) - 200 features
      7. S11 magnitude slope (dB/GHz) - 200 features (captures resonance shifts)
      8. S21 magnitude slope (dB/GHz) - 200 features
      9. statistical summaries - 8 features
    
    total: 201*4 + 200*4 + 8 = 1,612 features
    
    rationale:
      - magnitude/phase: direct RF fingerprint of the board
      - group delay: sensitive to subtle path-length changes caused by aging
      - magnitude slope: captures shifts in resonant peaks/nulls
      - statistics: provide global summary that aids distance-based classifiers
    """
    if freqs is None:
        freqs = np.linspace(1e6, 4e9, N_FREQ_POINTS)
    
    # --- core features: magnitude and phase ---
    s11_mag_db = 20 * np.log10(np.maximum(np.abs(s11_complex), 1e-10))
    s21_mag_db = 20 * np.log10(np.maximum(np.abs(s21_complex), 1e-10))
    s11_phase = np.angle(s11_complex, deg=True)
    s21_phase = np.angle(s21_complex, deg=True)
    
    # --- group delay: -d(phase)/d(freq) in nanoseconds ---
    # unwrap phase to avoid discontinuities at +/-180 degrees
    s11_phase_unwrap = np.unwrap(np.angle(s11_complex))  # radians, unwrapped
    s21_phase_unwrap = np.unwrap(np.angle(s21_complex))
    
    freq_diff = np.diff(freqs)  # Hz between adjacent points
    s11_group_delay_ns = -np.diff(s11_phase_unwrap) / (2 * np.pi * freq_diff) * 1e9
    s21_group_delay_ns = -np.diff(s21_phase_unwrap) / (2 * np.pi * freq_diff) * 1e9
    
    # --- magnitude slope: d(mag_dB)/d(freq_GHz) ---
    freq_diff_ghz = freq_diff / 1e9
    s11_mag_slope = np.diff(s11_mag_db) / freq_diff_ghz
    s21_mag_slope = np.diff(s21_mag_db) / freq_diff_ghz
    
    # --- statistical summaries ---
    stats = np.array([
        np.mean(s11_mag_db),
        np.std(s11_mag_db),
        np.mean(s21_mag_db),
        np.std(s21_mag_db),
        np.mean(s11_phase),
        np.std(s11_phase),
        np.mean(s21_phase),
        np.std(s21_phase),
    ])
    
    # concatenate all features
    features = np.concatenate([
        s11_mag_db,          # 201
        s11_phase,           # 201
        s21_mag_db,          # 201
        s21_phase,           # 201
        s11_group_delay_ns,  # 200
        s21_group_delay_ns,  # 200
        s11_mag_slope,       # 200
        s21_mag_slope,       # 200
        stats,               # 8
    ])
    
    return features


def get_feature_names():
    """generate descriptive names for all features (for importance plotting)"""
    freq_ghz = get_frequency_axis()
    names = []
    
    # mag/phase features (201 each)
    for prefix in ['S11_mag_dB', 'S11_phase_deg', 'S21_mag_dB', 'S21_phase_deg']:
        for i in range(N_FREQ_POINTS):
            names.append(f"{prefix}_{freq_ghz[i]:.3f}GHz")
    
    # derivative features (200 each)
    for prefix in ['S11_grpdelay_ns', 'S21_grpdelay_ns', 'S11_mag_slope', 'S21_mag_slope']:
        for i in range(N_FREQ_POINTS - 1):
            mid_freq = (freq_ghz[i] + freq_ghz[i+1]) / 2
            names.append(f"{prefix}_{mid_freq:.3f}GHz")
    
    # statistical features
    stat_names = [
        'S11_mag_mean', 'S11_mag_std',
        'S21_mag_mean', 'S21_mag_std',
        'S11_phase_mean', 'S11_phase_std',
        'S21_phase_mean', 'S21_phase_std',
    ]
    names.extend(stat_names)
    
    return names


# =============================================================================
# DATA LOADING
# =============================================================================

def discover_files(data_dir, dut_list=None, discard_first_5min=True):
    """discover and organize all .s2p files with their class labels"""
    file_label_pairs = []
    
    if dut_list is None:
        dut_list = [d for d in DATA_CONFIG.keys()
                    if os.path.isdir(os.path.join(data_dir, d))]
    
    for dut in dut_list:
        if dut not in DATA_CONFIG:
            print(f"  [WARN] DUT '{dut}' not in configuration, skipping.")
            continue
        
        config = DATA_CONFIG[dut]
        run_num = config['run']
        dut_dir = os.path.join(data_dir, dut)
        
        if not os.path.isdir(dut_dir):
            print(f"  [WARN] DUT directory not found: {dut_dir}")
            continue
        
        print(f"  Processing {dut} (run {run_num})...")
        
        for date_str, day_num in sorted(config['dates'].items()):
            class_label = day_to_class(day_num)
            folder_name = f"{run_num}_{date_str}"
            
            if day_num == 0:
                baseline_dir = os.path.join(dut_dir, folder_name, config['baseline_folder'])
                data_subdir = os.path.join(dut_dir, folder_name, date_str)
                
                if os.path.isdir(baseline_dir):
                    files = sorted(glob.glob(os.path.join(baseline_dir, '*.s2p')))
                    for f in files:
                        file_label_pairs.append((f, class_label))
                    print(f"    {date_str} baseline: {len(files)} files -> Class {class_label}")
                
                if os.path.isdir(data_subdir):
                    files = sorted(glob.glob(os.path.join(data_subdir, '*.s2p')))
                    if discard_first_5min and files:
                        boundary = get_discard_boundary(files)
                        if boundary:
                            files = [f for f in files
                                     if parse_timestamp_from_filename(f) is not None
                                     and parse_timestamp_from_filename(f) >= boundary]
                    for f in files:
                        file_label_pairs.append((f, class_label))
                    print(f"    {date_str} data: {len(files)} files -> Class {class_label}")
            else:
                data_subdir = os.path.join(dut_dir, folder_name, date_str)
                if not os.path.isdir(data_subdir):
                    print(f"    {date_str}: directory not found, skipping.")
                    continue
                
                files = sorted(glob.glob(os.path.join(data_subdir, '*.s2p')))
                if discard_first_5min and files:
                    boundary = get_discard_boundary(files)
                    if boundary:
                        files = [f for f in files
                                 if parse_timestamp_from_filename(f) is not None
                                 and parse_timestamp_from_filename(f) >= boundary]
                
                for f in files:
                    file_label_pairs.append((f, class_label))
                print(f"    {date_str} (Day {day_num}): {len(files)} files -> Class {class_label}")
    
    if not discard_first_5min:
        print("\n  [NOTE] First 5-minute discard window is DISABLED.")
    
    return file_label_pairs


def subsample_files(file_label_pairs, sample_rate=100, oversample_baseline=True):
    """subsample files. keeps all Class 0 (baseline) data by default"""
    by_class = {}
    for fp, label in file_label_pairs:
        by_class.setdefault(label, []).append((fp, label))
    
    subsampled = []
    for label in sorted(by_class.keys()):
        files = by_class[label]
        if label == 0 and oversample_baseline:
            sampled = files
            print(f"    Class {label} ({CLASS_LABELS[label]}): "
                  f"{len(files)} -> {len(sampled)} samples (ALL KEPT - baseline oversample)")
        else:
            sampled = files[::sample_rate]
            print(f"    Class {label} ({CLASS_LABELS[label]}): "
                  f"{len(files)} -> {len(sampled)} samples (1/{sample_rate})")
        subsampled.extend(sampled)
    
    return subsampled


def load_dataset(file_label_pairs):
    """load and extract features from all files"""
    X_list = []
    y_list = []
    failed = []
    total = len(file_label_pairs)
    print(f"\n  Loading {total} files...")
    
    for i, (filepath, label) in enumerate(file_label_pairs):
        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"    Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%)", end='\r')
        
        freqs, s11, s21 = parse_s2p_file(filepath)
        if freqs is None:
            failed.append(filepath)
            continue
        
        features = extract_features(s11, s21, freqs)
        X_list.append(features)
        y_list.append(label)
    
    print()
    if not X_list:
        return None, None, failed
    
    return np.array(X_list), np.array(y_list), failed


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_knn(X_train, y_train, n_neighbors=5):
    """train a K-Nearest Neighbors classifier
    
    KNN classifies based on proximity in feature space — boards with similar
    RF signatures cluster together. distance-weighted voting reduces sensitivity
    to the choice of k.
    """
    knn = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights='distance',
        metric='euclidean',
        n_jobs=-1,
    )
    knn.fit(X_train, y_train)
    return knn


def train_svm(X_train, y_train, tune_hyperparams=True):
    """train a Support Vector Machine classifier with optional hyperparameter tuning
    
    SVM finds optimal decision boundaries in high-dimensional space. the RBF kernel
    handles non-linear class boundaries that arise from complex RF aging patterns.
    
    when tuning is enabled, performs grid search over C and gamma to find the
    best regularization trade-off for this specific dataset
    """
    if tune_hyperparams and len(X_train) < 5000:
        print("    Running hyperparameter grid search (C, gamma)...")
        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.01, 0.001],
        }
        base_svm = SVC(
            kernel='rbf',
            decision_function_shape='ovr',
            probability=True,
            random_state=42,
        )
        grid_search = GridSearchCV(
            base_svm, param_grid, cv=3, scoring='f1_weighted',
            n_jobs=-1, verbose=0,
        )
        grid_search.fit(X_train, y_train)
        print(f"    Best params: C={grid_search.best_params_['C']}, "
              f"gamma={grid_search.best_params_['gamma']}")
        print(f"    CV F1 score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    else:
        svm = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            decision_function_shape='ovr',
            probability=True,
            random_state=42,
        )
        svm.fit(X_train, y_train)
        return svm


def train_random_forest(X_train, y_train, tune_hyperparams=True):
    """train a Random Forest classifier with optional hyperparameter tuning
    
    when tuning is enabled, searches over tree depth and leaf sizes to balance
    model complexity against overfitting risk
    """
    if tune_hyperparams and len(X_train) < 5000:
        print("    Running hyperparameter grid search (n_estimators, max_depth)...")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 30, 50],
            'min_samples_leaf': [1, 2, 4],
        }
        base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            base_rf, param_grid, cv=3, scoring='f1_weighted',
            n_jobs=-1, verbose=0,
        )
        grid_search.fit(X_train, y_train)
        print(f"    Best params: n_estimators={grid_search.best_params_['n_estimators']}, "
              f"max_depth={grid_search.best_params_['max_depth']}, "
              f"min_samples_leaf={grid_search.best_params_['min_samples_leaf']}")
        print(f"    CV F1 score: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_
    else:
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        return rf


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """evaluate a trained model and print metrics"""
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"  {model_name} Results")
    print(f"{'='*60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f} (weighted)")
    print(f"  Recall:    {rec:.4f} (weighted)")
    print(f"  F1 Score:  {f1:.4f} (weighted)")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=[CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())],
        zero_division=0,
    ))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"  {'':>25} Predicted")
    print(f"  {'':>15} {'Class 0':>10} {'Class 1':>10} {'Class 2':>10}")
    for i, row in enumerate(cm):
        print(f"  {'Actual '+str(i):>15} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
    
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
        'confusion_matrix': cm, 'y_pred': y_pred,
    }


# =============================================================================
# PLOTTING / VISUALIZATION
# =============================================================================

def save_figure(fig, output_dir, basename, dpi=150):
    """save a figure in .png, .pdf, and .svg formats"""
    for ext in ['png', 'pdf', 'svg']:
        path = os.path.join(output_dir, f'{basename}.{ext}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  Plot saved: {os.path.join(output_dir, basename)}.{{png,pdf,svg}}")


def plot_confusion_matrix(cm, model_name, output_dir):
    """save a publication-quality confusion matrix plot"""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # normalize for color intensity (percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, label='Proportion')
    
    classes = [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())]
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=f'{model_name} — Confusion Matrix',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')
    
    # annotate with count and percentage
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_norm[i, j] * 100
            txt = f"{cm[i, j]}\n({pct:.1f}%)"
            ax.text(j, i, txt, ha='center', va='center', fontsize=10,
                    color='white' if cm_norm[i, j] > thresh else 'black')
    
    fig.tight_layout()
    basename = f'{model_name.lower().replace(" ", "_")}_confusion_matrix'
    save_figure(fig, output_dir, basename)
    plt.close()


def plot_feature_importance(importances, feature_names, model_name, output_dir, top_n=30):
    """plot top feature importances as a horizontal bar chart grouped by feature type"""
    if not HAS_MATPLOTLIB:
        return
    
    top_indices = np.argsort(importances)[-top_n:][::-1]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    # color by feature type (Sixers palette)
    type_colors = {
        'S11_mag': SIXERS_COLORS[0],     # blue
        'S11_phase': SIXERS_COLORS[2],   # navy
        'S11_grp': SIXERS_COLORS[4],     # light blue
        'S11_slope': SIXERS_COLORS[13],  # medium teal
        'S21_mag': SIXERS_COLORS[1],     # red
        'S21_phase': SIXERS_COLORS[5],   # brown
        'S21_grp': SIXERS_COLORS[8],     # orange-yellow
        'S21_slope': SIXERS_COLORS[6],   # gold
        'stat': SIXERS_COLORS[11],       # dark plum
    }
    
    colors = []
    for name in top_names:
        if name.startswith('S11_mag_dB'):
            colors.append(type_colors['S11_mag'])
        elif name.startswith('S11_phase'):
            colors.append(type_colors['S11_phase'])
        elif name.startswith('S11_grpdelay'):
            colors.append(type_colors['S11_grp'])
        elif name.startswith('S11_mag_slope'):
            colors.append(type_colors['S11_slope'])
        elif name.startswith('S21_mag_dB'):
            colors.append(type_colors['S21_mag'])
        elif name.startswith('S21_phase'):
            colors.append(type_colors['S21_phase'])
        elif name.startswith('S21_grpdelay'):
            colors.append(type_colors['S21_grp'])
        elif name.startswith('S21_mag_slope'):
            colors.append(type_colors['S21_slope'])
        else:
            colors.append(type_colors['stat'])
    
    fig, ax = plt.subplots(figsize=(10, 10))
    y_pos = np.arange(top_n)
    ax.barh(y_pos, top_importances, color=colors, edgecolor='none')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'{model_name} — Top {top_n} Feature Importances')
    
    # legend
    legend_elements = [
        Patch(facecolor=type_colors['S11_mag'], label='S11 Magnitude'),
        Patch(facecolor=type_colors['S11_phase'], label='S11 Phase'),
        Patch(facecolor=type_colors['S11_grp'], label='S11 Group Delay'),
        Patch(facecolor=type_colors['S11_slope'], label='S11 Mag Slope'),
        Patch(facecolor=type_colors['S21_mag'], label='S21 Magnitude'),
        Patch(facecolor=type_colors['S21_phase'], label='S21 Phase'),
        Patch(facecolor=type_colors['S21_grp'], label='S21 Group Delay'),
        Patch(facecolor=type_colors['S21_slope'], label='S21 Mag Slope'),
        Patch(facecolor=type_colors['stat'], label='Statistical'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    fig.tight_layout()
    basename = f'{model_name.lower().replace(" ", "_")}_feature_importance'
    save_figure(fig, output_dir, basename)
    plt.close()


def plot_feature_importance_by_frequency(importances, feature_names, model_name, output_dir):
    """plot feature importance mapped back onto the frequency axis
    
    shows which frequency bands are most discriminative for classification,
    broken out by feature type (S11/S21, mag/phase/delay/slope)
    """
    if not HAS_MATPLOTLIB:
        return
    
    freq_ghz = get_frequency_axis()
    n = N_FREQ_POINTS
    
    # segment importances by feature group
    groups = {
        'S11 Magnitude (dB)':    importances[0:n],
        'S11 Phase (deg)':       importances[n:2*n],
        'S21 Magnitude (dB)':    importances[2*n:3*n],
        'S21 Phase (deg)':       importances[3*n:4*n],
        'S11 Group Delay (ns)':  importances[4*n:4*n+(n-1)],
        'S21 Group Delay (ns)':  importances[4*n+(n-1):4*n+2*(n-1)],
        'S11 Mag Slope (dB/GHz)': importances[4*n+2*(n-1):4*n+3*(n-1)],
        'S21 Mag Slope (dB/GHz)': importances[4*n+3*(n-1):4*n+4*(n-1)],
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    fig.suptitle(f'{model_name} — Feature Importance by Frequency', fontsize=14, y=0.98)
    
    colors = [SIXERS_COLORS[0], SIXERS_COLORS[2], SIXERS_COLORS[1], SIXERS_COLORS[5],
              SIXERS_COLORS[4], SIXERS_COLORS[8], SIXERS_COLORS[13], SIXERS_COLORS[6]]
    
    for idx, (label, imp) in enumerate(groups.items()):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        # use midpoints for derivative features
        if len(imp) == n - 1:
            x = (freq_ghz[:-1] + freq_ghz[1:]) / 2
        else:
            x = freq_ghz
        
        ax.fill_between(x, imp, alpha=0.3, color=colors[idx])
        ax.plot(x, imp, color=colors[idx], linewidth=0.8)
        ax.set_ylabel('Importance', fontsize=8)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(freq_ghz[0], freq_ghz[-1])
    
    axes[-1, 0].set_xlabel('Frequency (GHz)')
    axes[-1, 1].set_xlabel('Frequency (GHz)')
    
    fig.tight_layout()
    basename = f'{model_name.lower().replace(" ", "_")}_importance_by_freq'
    save_figure(fig, output_dir, basename)
    plt.close()


def plot_model_comparison(results, output_dir):
    """bar chart comparing all trained models on key metrics"""
    if not HAS_MATPLOTLIB or len(results) < 2:
        return
    
    models = list(results.keys())
    model_display = {'knn': 'KNN', 'svm': 'SVM', 'rf': 'Random Forest'}
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    colors = [SIXERS_COLORS[0], SIXERS_COLORS[1], SIXERS_COLORS[3]]
    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model_display.get(model, model),
                      color=colors[i % len(colors)], edgecolor='white', linewidth=0.5)
        # add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison — Classification Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.9, 1.01)  # zoom in since all models are high-performing
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    fig.tight_layout()
    save_figure(fig, output_dir, 'model_comparison')
    plt.close()


def plot_class_distribution(y_train, y_test, output_dir):
    """Pie/bar chart of class distribution in train and test sets."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for ax, data, title in [(ax1, y_train, 'Training Set'), (ax2, y_test, 'Test Set')]:
        unique, counts = np.unique(data, return_counts=True)
        labels = [f"Class {c}\n{CLASS_LABELS[c]}" for c in unique]
        colors = [CLASS_COLORS[c] for c in unique]
        
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 9}
        )
        ax.set_title(f'{title}\n(n={len(data)})', fontsize=11)
    
    fig.suptitle('Class Distribution', fontsize=13, y=1.02)
    fig.tight_layout()
    save_figure(fig, output_dir, 'class_distribution')
    plt.close()


def plot_per_class_metrics(results, output_dir):
    """grouped bar chart showing precision/recall/f1 per class for each model"""
    if not HAS_MATPLOTLIB or not results:
        return
    
    # pick the best model's detailed per-class breakdown
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5), squeeze=False)
    
    model_display = {'knn': 'KNN', 'svm': 'SVM', 'rf': 'Random Forest'}
    
    for idx, (model_key, res) in enumerate(results.items()):
        ax = axes[0, idx]
        cm = res['confusion_matrix']
        n_classes = cm.shape[0]
        
        # calculate per-class precision, recall, f1
        precisions, recalls, f1s = [], [], []
        for c in range(n_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2*p*r / (p+r) if (p+r) > 0 else 0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
        
        x = np.arange(n_classes)
        w = 0.25
        ax.bar(x - w, precisions, w, label='Precision', color=SIXERS_COLORS[0])
        ax.bar(x, recalls, w, label='Recall', color=SIXERS_COLORS[3])
        ax.bar(x + w, f1s, w, label='F1', color=SIXERS_COLORS[1])
        
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{c}" for c in range(n_classes)])
        ax.set_ylim(0.85, 1.02)
        ax.set_ylabel('Score')
        ax.set_title(model_display.get(model_key, model_key))
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)
    
    fig.suptitle('Per-Class Performance Metrics', fontsize=13)
    fig.tight_layout()
    save_figure(fig, output_dir, 'per_class_metrics')
    plt.close()


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train a 3-class aging classifier for RF reflectometry data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Classes:
  0 - Fresh / Unaged (baseline measurements before aging)
  1 - Lightly Aged (Days 1-3 of accelerated thermal aging)
  2 - Heavily Aged (Days 4+ of accelerated thermal aging)

models (pass as comma-separated string or 'all'):
  knn - K-Nearest Neighbors (simple, fast, distance-based)
  svm - Support Vector Machine (finds optimal boundaries, good for high-dim data)
  rf  - Random Forest (non-linear, provides feature importance rankings)

examples:
  # train KNN only (default, fastest)
  python train_classifier.py

  # train SVM and Random Forest
  python train_classifier.py --model svm,rf

  # train all three models
  python train_classifier.py --model all

  # full dataset training (no subsampling)
  python train_classifier.py --sample-rate 1 --model all

  # use both DUTs
  python train_classifier.py --duts SN004 SN005 --model all
        """
    )
    
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                        help='Root directory containing SN004/SN005 data folders')
    parser.add_argument('--sample-rate', type=int, default=100,
                        help='Subsampling rate: keep 1 in N files (default: 100)')
    parser.add_argument('--model', type=str, default='knn',
                        help='Model(s) to train: knn, svm, rf, or comma-separated '
                             '(e.g. "knn,svm"), or "all" (default: knn)')
    parser.add_argument('--duts', nargs='+', default=['SN005'],
                        help='Which DUTs to use for training (default: SN005)')
    parser.add_argument('--no-discard', action='store_true',
                        help='Disable 5-minute initial discard window')
    parser.add_argument('--output-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
                        help='Directory to save trained models and results')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data for testing (default: 0.2)')
    parser.add_argument('--knn-neighbors', type=int, default=5,
                        help='Number of neighbors for KNN (default: 5)')
    parser.add_argument('--no-tune', action='store_true',
                        help='Disable hyperparameter tuning for SVM/RF (faster training)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # parse model selection
    if args.model.lower() == 'all':
        models_to_train = ['knn', 'svm', 'rf']
    else:
        models_to_train = [m.strip().lower() for m in args.model.split(',')]
        valid_models = {'knn', 'svm', 'rf'}
        for m in models_to_train:
            if m not in valid_models:
                print(f"[ERROR] Unknown model '{m}'. Choose from: knn, svm, rf, all")
                sys.exit(1)
    
    # create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("  RF Reflectometry Board Aging Classifier - Training")
    print("=" * 60)
    print(f"\n  Data directory:    {args.data_dir}")
    print(f"  DUTs:              {args.duts}")
    print(f"  Sample rate:       1/{args.sample_rate} (baseline kept in full)")
    print(f"  Model(s):          {', '.join(models_to_train)}")
    print(f"  Hyperparam tuning: {not args.no_tune}")
    print(f"  Discard first 5m:  {not args.no_discard}")
    print(f"  Test split:        {args.test_size*100:.0f}%")
    print(f"  Output directory:  {args.output_dir}")
    print()
    
    # =========================================================================
    # Step 1: discover files
    # =========================================================================
    print("[1/5] Discovering data files...")
    start_time = time.time()
    
    file_label_pairs = discover_files(
        args.data_dir, dut_list=args.duts,
        discard_first_5min=(not args.no_discard),
    )
    
    if not file_label_pairs:
        print("  [ERROR] No data files found. Check --data-dir path.")
        sys.exit(1)
    
    print(f"\n  Total files discovered: {len(file_label_pairs)}")
    print(f"  Time: {time.time() - start_time:.1f}s")
    
    # =========================================================================
    # Step 2: subsample
    # =========================================================================
    print(f"\n[2/5] Subsampling data (rate: 1/{args.sample_rate}, baseline: ALL)...")
    
    if args.sample_rate > 1:
        file_label_pairs = subsample_files(file_label_pairs, args.sample_rate)
    else:
        print("  No subsampling (using all files).")
        dist = Counter(label for _, label in file_label_pairs)
        for label in sorted(dist.keys()):
            print(f"    Class {label} ({CLASS_LABELS[label]}): {dist[label]} samples")
    
    print(f"\n  Dataset size: {len(file_label_pairs)} samples")
    
    # =========================================================================
    # Step 3: load data and extract features
    # =========================================================================
    print(f"\n[3/5] Loading data and extracting features...")
    start_time = time.time()
    
    X, y, failed = load_dataset(file_label_pairs)
    
    if X is None:
        print("  [ERROR] No valid data loaded.")
        sys.exit(1)
    
    if failed:
        print(f"  [WARN] {len(failed)} files failed to parse.")
    
    elapsed = time.time() - start_time
    n_features = X.shape[1]
    print(f"  Loaded: {X.shape[0]} samples, {n_features} features each")
    print(f"  Feature breakdown:")
    print(f"    - S11/S21 magnitude (dB):     201 x 2 = 402")
    print(f"    - S11/S21 phase (deg):        201 x 2 = 402")
    print(f"    - S11/S21 group delay (ns):   200 x 2 = 400")
    print(f"    - S11/S21 mag slope (dB/GHz): 200 x 2 = 400")
    print(f"    - Statistical summaries:      8")
    print(f"    - TOTAL:                      {n_features}")
    print(f"  Time: {elapsed:.1f}s")
    
    # class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Class distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"    Class {cls} ({CLASS_LABELS[cls]}): {cnt} samples ({100*cnt/len(y):.1f}%)")
    
    # =========================================================================
    # Step 4: train/test split and scaling
    # =========================================================================
    print(f"\n[4/5] Splitting data and scaling features...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    scaler_path = os.path.join(args.output_dir, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")
    
    # plot class distribution
    plot_class_distribution(y_train, y_test, args.output_dir)
    
    # =========================================================================
    # Step 5: train and evaluate
    # =========================================================================
    print(f"\n[5/5] Training and evaluating model(s)...")
    
    results = {}
    feature_names = get_feature_names()
    
    for model_type in models_to_train:
        start_time = time.time()
        
        if model_type == 'knn':
            print(f"\n  Training KNN (k={args.knn_neighbors})...")
            model = train_knn(X_train_scaled, y_train, n_neighbors=args.knn_neighbors)
            model_name = "KNN"
            
        elif model_type == 'svm':
            print(f"\n  Training SVM (RBF kernel)...")
            model = train_svm(X_train_scaled, y_train, tune_hyperparams=(not args.no_tune))
            model_name = "SVM"
            
        elif model_type == 'rf':
            print(f"\n  Training Random Forest...")
            model = train_random_forest(X_train_scaled, y_train, tune_hyperparams=(not args.no_tune))
            model_name = "Random Forest"
        
        elapsed = time.time() - start_time
        print(f"  Training time: {elapsed:.1f}s")
        
        # evaluate
        result = evaluate_model(model, X_test_scaled, y_test, model_name)
        results[model_type] = result
        
        # save model
        model_path = os.path.join(args.output_dir, f'{model_type}_model.joblib')
        joblib.dump(model, model_path)
        print(f"  Model saved: {model_path}")
        
        # plots
        plot_confusion_matrix(result['confusion_matrix'], model_name, args.output_dir)
        
        # feature importance for Random Forest
        if model_type == 'rf' and hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            plot_feature_importance(importances, feature_names, model_name, args.output_dir)
            plot_feature_importance_by_frequency(importances, feature_names, model_name, args.output_dir)
            
            top_indices = np.argsort(importances)[-20:][::-1]
            print(f"\n  Top 20 Most Important Features ({model_name}):")
            for idx in top_indices:
                print(f"    {feature_names[idx]:>30s}: {importances[idx]:.4f}")
    
    # multi-model comparison plots
    if len(results) > 1:
        plot_model_comparison(results, args.output_dir)
        plot_per_class_metrics(results, args.output_dir)
    
    # =========================================================================
    # save metadata
    # =========================================================================
    metadata = {
        'n_freq_points': N_FREQ_POINTS,
        'n_features': n_features,
        'feature_names': feature_names,
        'class_labels': CLASS_LABELS,
        'duts_used': args.duts,
        'sample_rate': args.sample_rate,
        'discard_first_5min': not args.no_discard,
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'models_trained': models_to_train,
        'results': {k: {kk: vv for kk, vv in v.items()
                        if kk not in ('y_pred',)}
                    for k, v in results.items()},
        'trained_at': datetime.now().isoformat(),
    }
    metadata_path = os.path.join(args.output_dir, 'training_metadata.joblib')
    joblib.dump(metadata, metadata_path)
    print(f"\n  Metadata saved: {metadata_path}")
    
    # final summary
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"{'='*60}")
    print(f"  Models saved to: {args.output_dir}/")
    print(f"  Files:")
    print(f"    - scaler.joblib (feature scaler)")
    for mt in models_to_train:
        print(f"    - {mt}_model.joblib ({results[mt]['accuracy']:.1%} accuracy)")
    print(f"    - training_metadata.joblib")
    if HAS_MATPLOTLIB:
        print(f"\n  Plots generated (each in .png, .pdf, .svg):")
        print(f"    - class_distribution")
        for mt in models_to_train:
            mn = {'knn': 'knn', 'svm': 'svm', 'rf': 'random_forest'}[mt]
            print(f"    - {mn}_confusion_matrix")
        if 'rf' in models_to_train:
            print(f"    - random_forest_feature_importance")
            print(f"    - random_forest_importance_by_freq")
        if len(results) > 1:
            print(f"    - model_comparison")
            print(f"    - per_class_metrics")
    
    print(f"\n  To classify a new measurement:")
    print(f"    python classify_measurement.py <path_to_file.s2p>")
    print()


if __name__ == '__main__':
    main()
