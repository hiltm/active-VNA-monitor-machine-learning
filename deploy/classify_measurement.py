#!/usr/bin/env python3
"""
classify_measurement.py - RF Reflectometry Board Aging Classifier (Inference)

classifies a single .s2p measurement file (or directory of files) as:
  Class 0: Fresh / Unaged (baseline)
  Class 1: Lightly Aged (Days 1-3)
  Class 2: Heavily Aged (Days 4+)

requires a trained model from train_classifier.py.

usage:
    python classify_measurement.py <path_to_file.s2p>
    python classify_measurement.py <directory_of_s2p_files>
    python classify_measurement.py <file.s2p> --model svm
    python classify_measurement.py <file.s2p> --model-dir /path/to/models

author: Michael Hilt
date: 2026-07-18
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import joblib


# =============================================================================
# CONFIGURATION (must match train_classifier.py)
# =============================================================================

N_FREQ_POINTS = 201

CLASS_LABELS = {
    0: 'Fresh (Baseline)',
    1: 'Lightly Aged (Days 1-3)',
    2: 'Heavily Aged (Days 4+)',
}

# confidence thresholds for reporting
CONFIDENCE_HIGH = 0.8
CONFIDENCE_MEDIUM = 0.5


# =============================================================================
# DATA PARSING (shared logic with train_classifier.py)
# =============================================================================

def parse_s2p_file(filepath):
    """parse a .s2p file and extract S11 and S21 (from S12 columns) data
    
    the VNA captures S11 and S21, but the logging script writes S21 data
    into the S12 columns. Columns labeled "S21" in the file are zeros
    
    returns:
        freqs: numpy array of frequency points (Hz)
        s11_complex: numpy array of complex S11 values
        s21_complex: numpy array of complex S21 values (read from S12 columns)
        
    returns (None, None, None) if parsing fails
    """
    try:
        freqs = []
        s11_re = []
        s11_im = []
        s21_re = []
        s21_im = []
        
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
                    # S21 data is in S12 columns (indices 5, 6)
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


def extract_features(s11_complex, s21_complex, freqs=None):
    """extract expanded feature vector from S-parameter data
    
    features:
      - S11/S21 magnitude (dB) and phase (deg): 201 x 4 = 804
      - S11/S21 group delay (ns): 200 x 2 = 400
      - S11/S21 magnitude slope (dB/GHz): 200 x 2 = 400
      - Statistical summaries: 8
    total: 1,612 features
    
    must match train_classifier.py extract_features exactly
    """
    if freqs is None:
        freqs = np.linspace(1e6, 4e9, N_FREQ_POINTS)
    
    # magnitude in dB
    s11_mag_db = 20 * np.log10(np.maximum(np.abs(s11_complex), 1e-10))
    s21_mag_db = 20 * np.log10(np.maximum(np.abs(s21_complex), 1e-10))
    
    # phase in degrees
    s11_phase = np.angle(s11_complex, deg=True)
    s21_phase = np.angle(s21_complex, deg=True)
    
    # group delay: -d(unwrapped_phase)/d(freq) in nanoseconds
    s11_phase_unwrap = np.unwrap(np.angle(s11_complex))
    s21_phase_unwrap = np.unwrap(np.angle(s21_complex))
    freq_diff = np.diff(freqs)
    s11_group_delay_ns = -np.diff(s11_phase_unwrap) / (2 * np.pi * freq_diff) * 1e9
    s21_group_delay_ns = -np.diff(s21_phase_unwrap) / (2 * np.pi * freq_diff) * 1e9
    
    # magnitude slope: d(mag_dB)/d(freq_GHz)
    freq_diff_ghz = freq_diff / 1e9
    s11_mag_slope = np.diff(s11_mag_db) / freq_diff_ghz
    s21_mag_slope = np.diff(s21_mag_db) / freq_diff_ghz
    
    # statistical summaries
    stats = np.array([
        np.mean(s11_mag_db), np.std(s11_mag_db),
        np.mean(s21_mag_db), np.std(s21_mag_db),
        np.mean(s11_phase), np.std(s11_phase),
        np.mean(s21_phase), np.std(s21_phase),
    ])
    
    features = np.concatenate([
        s11_mag_db, s11_phase, s21_mag_db, s21_phase,
        s11_group_delay_ns, s21_group_delay_ns,
        s11_mag_slope, s21_mag_slope,
        stats,
    ])
    return features


# =============================================================================
# CLASSIFICATION
# =============================================================================

def load_model_and_scaler(model_dir, model_type='knn'):
    """load the trained model and feature scaler
    
    args:
        model_dir: Directory containing saved model files
        model_type: Which model to load ('knn', 'svm', or 'rf')
    
    returns:
        model: trained sklearn classifier
        scaler: fitted StandardScaler
        metadata: training metadata dict
    """
    model_path = os.path.join(model_dir, f'{model_type}_model.joblib')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    metadata_path = os.path.join(model_dir, 'training_metadata.joblib')
    
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print(f"  Available models in {model_dir}:")
        for f in glob.glob(os.path.join(model_dir, '*_model.joblib')):
            print(f"    - {os.path.basename(f)}")
        sys.exit(1)
    
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scaler file not found: {scaler_path}")
        print("  Run train_classifier.py first to generate model files.")
        sys.exit(1)
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    metadata = None
    if os.path.exists(metadata_path):
        metadata = joblib.load(metadata_path)
    
    return model, scaler, metadata


def classify_file(filepath, model, scaler):
    """classify a single .s2p file
    
    args:
        filepath: path to the .s2p file
        model: trained classifier
        scaler: fitted StandardScaler
    
    returns:
        dict with keys: 'class', 'label', 'confidence', 'probabilities'
        or None if parsing fails.
    """
    freqs, s11, s21 = parse_s2p_file(filepath)
    if freqs is None:
        return None
    
    features = extract_features(s11, s21, freqs)
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # predict class
    predicted_class = model.predict(features_scaled)[0]
    
    # get confidence/probability if available
    probabilities = None
    confidence = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features_scaled)[0]
        confidence = probabilities[predicted_class]
    
    return {
        'class': int(predicted_class),
        'label': CLASS_LABELS[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities,
    }


def format_result(filepath, result, verbose=False):
    """Format classification result for display."""
    filename = os.path.basename(filepath)
    
    if result is None:
        return f"  {filename}: [PARSE ERROR]"
    
    conf_str = ""
    if result['confidence'] is not None:
        conf = result['confidence']
        if conf >= CONFIDENCE_HIGH:
            conf_str = f" (confidence: {conf:.1%} - HIGH)"
        elif conf >= CONFIDENCE_MEDIUM:
            conf_str = f" (confidence: {conf:.1%} - MEDIUM)"
        else:
            conf_str = f" (confidence: {conf:.1%} - LOW)"
    
    line = f"  {filename}: Class {result['class']} - {result['label']}{conf_str}"
    
    if verbose and result['probabilities'] is not None:
        probs = result['probabilities']
        line += f"\n    Probabilities: "
        for i, p in enumerate(probs):
            line += f"C{i}={p:.3f} "
    
    return line


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Classify .s2p measurements as Fresh, Lightly Aged, or Heavily Aged.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python classify_measurement.py measurement_20260715_120000_testdata.s2p
  python classify_measurement.py /path/to/data/folder/
  python classify_measurement.py file.s2p --model svm --verbose
        """
    )
    
    parser.add_argument('input', type=str,
                        help='Path to a .s2p file or directory of .s2p files')
    parser.add_argument('--model', type=str, default='knn',
                        choices=['knn', 'svm', 'rf'],
                        help='Which trained model to use (default: knn)')
    parser.add_argument('--model-dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'),
                        help='Directory containing trained model files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed probability output')
    parser.add_argument('--batch-summary', action='store_true',
                        help='Show summary statistics when classifying a directory')
    
    args = parser.parse_args()
    
    # load model
    print(f"Loading {args.model.upper()} model from {args.model_dir}...")
    model, scaler, metadata = load_model_and_scaler(args.model_dir, args.model)
    
    if metadata:
        acc = metadata.get('results', {}).get(args.model, {}).get('accuracy', None)
        if acc:
            print(f"  Model training accuracy: {acc:.1%}")
    
    # determine input files
    input_path = args.input
    files_to_classify = []
    
    if os.path.isfile(input_path):
        files_to_classify = [input_path]
    elif os.path.isdir(input_path):
        files_to_classify = sorted(glob.glob(os.path.join(input_path, '*.s2p')))
        if not files_to_classify:
            # try recursive search
            files_to_classify = sorted(glob.glob(os.path.join(input_path, '**', '*.s2p'), recursive=True))
        print(f"  Found {len(files_to_classify)} .s2p files in directory.")
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)
    
    if not files_to_classify:
        print("[ERROR] No .s2p files found.")
        sys.exit(1)
    
    # classify
    print(f"\nClassification Results ({args.model.upper()}):")
    print("-" * 60)
    
    results = []
    for filepath in files_to_classify:
        result = classify_file(filepath, model, scaler)
        results.append((filepath, result))
        print(format_result(filepath, result, verbose=args.verbose))
    
    # summary for batch classification
    if (len(files_to_classify) > 1 and args.batch_summary) or len(files_to_classify) > 10:
        print(f"\n{'='*60}")
        print("  Batch Summary")
        print(f"{'='*60}")
        
        valid_results = [(f, r) for f, r in results if r is not None]
        failed_count = len(results) - len(valid_results)
        
        if valid_results:
            classes = [r['class'] for _, r in valid_results]
            unique, counts = np.unique(classes, return_counts=True)
            
            print(f"  Total files:   {len(results)}")
            if failed_count:
                print(f"  Parse errors:  {failed_count}")
            print(f"  Distribution:")
            for cls, cnt in zip(unique, counts):
                pct = 100 * cnt / len(valid_results)
                print(f"    Class {cls} ({CLASS_LABELS[cls]}): {cnt} ({pct:.1f}%)")
            
            # average confidence
            confs = [r['confidence'] for _, r in valid_results if r['confidence'] is not None]
            if confs:
                print(f"\n  Average confidence: {np.mean(confs):.1%}")
                print(f"  Min confidence:     {np.min(confs):.1%}")


if __name__ == '__main__':
    main()
