#!/usr/bin/env python3
"""
export_to_mat.py - Export training results to a MATLAB .mat file.

exports all pre-computed results (metrics, confusion matrices, feature importances,
class distributions, frequency axis) so that MATLAB can regenerate all figures as .fig

if confusion matrices are not in the saved metadata, this script will re-evaluate
the saved models on a test split of the data (prediction only, no retraining)

usage:
    python export_to_mat.py [--output results_for_matlab.mat]
"""

import os
import sys
import argparse
import numpy as np
import joblib
import scipy.io as sio

# import from train_classifier for data loading/feature extraction
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _d in ('src', 'viz', 'experiments'):
    sys.path.insert(0, os.path.join(_ROOT, _d))
from train_classifier import (
    DATA_CONFIG, CLASS_LABELS, N_FREQ_POINTS, DISCARD_MINUTES,
    discover_files, subsample_files, load_dataset, extract_features,
    get_feature_names, get_frequency_axis, day_to_class
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Export classifier training results to a MATLAB .mat file.'
    )
    parser.add_argument('--output', type=str, default='models/classifier_results.mat',
                        help='Output .mat file path (default: models/classifier_results.mat)')
    parser.add_argument('--models-dir', type=str, default='models',
                        help='Directory containing saved models and metadata')
    parser.add_argument('--data-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                        help='Data directory (only needed if confusion matrices must be regenerated)')
    args = parser.parse_args()

    models_dir = args.models_dir
    print("=" * 60)
    print("  Exporting Training Results to MATLAB .mat")
    print("=" * 60)

    # =========================================================================
    # load metadata
    # =========================================================================
    meta_path = os.path.join(models_dir, 'training_metadata.joblib')
    if not os.path.exists(meta_path):
        print(f"[ERROR] Metadata not found at {meta_path}")
        print("  Run train_classifier.py first.")
        sys.exit(1)

    meta = joblib.load(meta_path)
    print(f"\n  Loaded metadata from: {meta_path}")
    print(f"  Training date: {meta['trained_at']}")
    print(f"  Models: {meta['models_trained']}")
    print(f"  Samples: {meta['training_samples']} train / {meta['test_samples']} test")

    # =========================================================================
    # extract metrics
    # =========================================================================
    models_trained = meta['models_trained']
    feature_names = meta['feature_names']
    n_features = meta['n_features']
    freq_ghz = get_frequency_axis()

    # per-model metrics
    model_metrics = {}
    for model_key in models_trained:
        res = meta['results'][model_key]
        model_metrics[model_key] = {
            'accuracy': float(res['accuracy']),
            'precision': float(np.asarray(res['precision'])),
            'recall': float(np.asarray(res['recall'])),
            'f1': float(np.asarray(res['f1'])),
        }

    # =========================================================================
    # check for confusion matrices in metadata
    # =========================================================================
    has_cm = all('confusion_matrix' in meta['results'][m] for m in models_trained)

    if has_cm:
        print("\n  Confusion matrices found in metadata.")
        confusion_matrices = {m: np.array(meta['results'][m]['confusion_matrix'])
                              for m in models_trained}
    else:
        print("\n  Confusion matrices not in metadata — regenerating from models...")
        print("  (This requires loading a subset of data for prediction only.)")
        confusion_matrices = regenerate_confusion_matrices(
            models_dir, args.data_dir, meta
        )

    # =========================================================================
    # feature importances from RF
    # =========================================================================
    rf_importances = None
    rf_path = os.path.join(models_dir, 'rf_model.joblib')
    if os.path.exists(rf_path):
        rf_model = joblib.load(rf_path)
        if hasattr(rf_model, 'feature_importances_'):
            rf_importances = rf_model.feature_importances_
            print(f"\n  RF feature importances loaded: {rf_importances.shape[0]} features")

    # =========================================================================
    # build export dictionary
    # =========================================================================
    print("\n  Assembling .mat export structure...")

    mat_dict = {
        # metadata
        'n_freq_points': np.int32(N_FREQ_POINTS),
        'n_features': np.int32(n_features),
        'freq_ghz': freq_ghz.astype(np.float64),
        'training_samples': np.int32(meta['training_samples']),
        'test_samples': np.int32(meta['test_samples']),
        'sample_rate': np.int32(meta['sample_rate']),
        'duts_used': np.array(meta['duts_used'], dtype=object),
        'trained_at': meta['trained_at'],

        # Class info
        'class_labels': np.array(
            [CLASS_LABELS[i] for i in sorted(CLASS_LABELS.keys())], dtype=object
        ),
        'n_classes': np.int32(len(CLASS_LABELS)),

        # models trained (as cell array of strings)
        'models_trained': np.array(models_trained, dtype=object),

        # feature names (cell array of strings)
        'feature_names': np.array(feature_names, dtype=object),
    }

    # per-model metrics and confusion matrices
    for model_key in models_trained:
        prefix = model_key  # e.g., 'knn', 'svm', 'rf'
        mat_dict[f'{prefix}_accuracy'] = model_metrics[model_key]['accuracy']
        mat_dict[f'{prefix}_precision'] = model_metrics[model_key]['precision']
        mat_dict[f'{prefix}_recall'] = model_metrics[model_key]['recall']
        mat_dict[f'{prefix}_f1'] = model_metrics[model_key]['f1']
        mat_dict[f'{prefix}_confusion_matrix'] = confusion_matrices[model_key].astype(np.float64)

    # RF feature importances
    if rf_importances is not None:
        mat_dict['rf_feature_importances'] = rf_importances.astype(np.float64)

        # also export top-30 for bar chart
        top_n = 30
        top_indices = np.argsort(rf_importances)[-top_n:][::-1]
        mat_dict['rf_top30_indices'] = (top_indices + 1).astype(np.int32)  # 1-indexed for MATLAB
        mat_dict['rf_top30_importances'] = rf_importances[top_indices].astype(np.float64)
        mat_dict['rf_top30_names'] = np.array(
            [feature_names[i] for i in top_indices], dtype=object
        )

        # feature importance segmented by type for frequency plots
        n = N_FREQ_POINTS
        mat_dict['rf_imp_s11_mag'] = rf_importances[0:n].astype(np.float64)
        mat_dict['rf_imp_s11_phase'] = rf_importances[n:2*n].astype(np.float64)
        mat_dict['rf_imp_s21_mag'] = rf_importances[2*n:3*n].astype(np.float64)
        mat_dict['rf_imp_s21_phase'] = rf_importances[3*n:4*n].astype(np.float64)
        mat_dict['rf_imp_s11_grpdelay'] = rf_importances[4*n:4*n+(n-1)].astype(np.float64)
        mat_dict['rf_imp_s21_grpdelay'] = rf_importances[4*n+(n-1):4*n+2*(n-1)].astype(np.float64)
        mat_dict['rf_imp_s11_slope'] = rf_importances[4*n+2*(n-1):4*n+3*(n-1)].astype(np.float64)
        mat_dict['rf_imp_s21_slope'] = rf_importances[4*n+3*(n-1):4*n+4*(n-1)].astype(np.float64)

    # class distribution (from training/test counts in metadata)
    # store counts per class, needed to regenerate from data or approximate
    # use confusion matrix row sums for test set distribution
    if confusion_matrices:
        first_cm = list(confusion_matrices.values())[0]
        test_class_counts = first_cm.sum(axis=1).astype(np.float64)
        mat_dict['test_class_counts'] = test_class_counts

        # approximate training class counts from ratio
        total_test = test_class_counts.sum()
        total_train = meta['training_samples']
        train_class_counts = test_class_counts * (total_train / total_test)
        mat_dict['train_class_counts'] = train_class_counts

    # =========================================================================
    # save .mat file
    # =========================================================================
    output_path = args.output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    sio.savemat(output_path, mat_dict, do_compression=True)
    print(f"\n  Saved: {output_path}")
    print(f"  Variables exported: {len(mat_dict)}")
    print(f"\n  Load in MATLAB with:")
    print(f"    data = load('{os.path.basename(output_path)}');")
    print()


def regenerate_confusion_matrices(models_dir, data_dir, meta):
    """re-evaluate saved models on test data to get confusion matrices
    
    uses a subsample for speed — just needs predictions, no training
    """
    from train_classifier import discover_files, subsample_files, load_dataset

    duts = meta['duts_used']
    print(f"  Loading data from {duts}...")

    # discover files
    file_label_pairs = discover_files(data_dir, dut_list=duts, discard_first_5min=True)
    
    # use a subsample for speed (1/10 is enough for accurate confusion matrices)
    sample_rate = max(meta['sample_rate'], 10)
    if sample_rate > 1:
        file_label_pairs = subsample_files(file_label_pairs, sample_rate)
    
    print(f"  Using {len(file_label_pairs)} samples for evaluation...")

    # load and extract features
    X, y, failed = load_dataset(file_label_pairs)
    if X is None:
        print("  [ERROR] Could not load data. Exporting without confusion matrices.")
        return {m: np.zeros((3, 3), dtype=int) for m in meta['models_trained']}

    # split same as training (same random seed = same split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # scale
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    X_test_scaled = scaler.transform(X_test)

    # predict with each model
    confusion_matrices = {}
    for model_key in meta['models_trained']:
        model_path = os.path.join(models_dir, f'{model_key}_model.joblib')
        if not os.path.exists(model_path):
            print(f"  [WARN] Model file not found: {model_path}")
            confusion_matrices[model_key] = np.zeros((3, 3), dtype=int)
            continue

        model = joblib.load(model_path)
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[model_key] = cm
        print(f"  {model_key}: confusion matrix computed ({cm.sum()} samples)")

    return confusion_matrices


if __name__ == '__main__':
    main()
