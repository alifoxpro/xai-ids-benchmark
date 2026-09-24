"""
run_cross_dataset.py
====================
Cross-dataset evaluation: train on CIC IoT-DIAD 2024, test on CICIDS2017.

The two datasets share 39 common CICFlowMeter features. This script:
  1. Loads a pre-trained model from CIC IoT-DIAD 2024
  2. Loads CICIDS2017 and projects it onto the common feature subspace
  3. Maps CICIDS2017 attack labels to CIC IoT-DIAD label space
  4. Reports source vs target accuracy, balanced accuracy, F1, ROC-AUC
  5. Computes domain distance metrics (Maximum Mean Discrepancy)

Usage
-----
python experiments/run_cross_dataset.py \
       --source-mode multiclass \
       --target-dataset CICIDS2017 \
       --models GRU,ResNet1D,FT-Transformer
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import MODEL_REGISTRY
    from src.stage3_dl_models import load_trained_model
    from src.stage1_preprocess import (load_preprocessed_data,
                                       fit_scaler_from_dict)
except ImportError as exc:
    print(f"[WARN] Project modules not importable ({exc}).")
    MODEL_REGISTRY = ['GRU', 'ResNet1D', 'FT-Transformer']
    load_trained_model = None


# --------------------------------------------------------- feature alignment

# Common feature names between CIC IoT-DIAD 2024 and CICIDS2017 / TON_IoT.
# Verified against the CICFlowMeter v4 documentation.
COMMON_FEATURES = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max',
    'Fwd IAT Min', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd Header Length', 'Bwd Header Length', 'FWD Packets/s',
    'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'ACK Flag Count',
]


# CICIDS2017 -> CIC IoT-DIAD 2024 label mapping
# (CICIDS2017 lacks Mirai; the closest IoT-DIAD class is DOS or Benign)
LABEL_MAP_2017_TO_DIAD = {
    'BENIGN': 'Benign',
    'DDoS': 'DDOS',
    'DoS Hulk': 'DOS', 'DoS GoldenEye': 'DOS',
    'DoS slowloris': 'DOS', 'DoS Slowhttptest': 'DOS',
    'PortScan': 'Recon',
    'FTP-Patator': 'BruteForce', 'SSH-Patator': 'BruteForce',
    'Web Attack \x96 Brute Force': 'Web-Based',
    'Web Attack \x96 XSS': 'Web-Based',
    'Web Attack \x96 Sql Injection': 'Web-Based',
    'Infiltration': 'Spoofing',
    'Bot': 'Mirai',                    # closest analogue
    'Heartbleed': 'Web-Based',
}


# --------------------------------------------------------- MMD distance

def maximum_mean_discrepancy(X_s: np.ndarray, X_t: np.ndarray,
                              kernel: str = 'rbf', gamma: float = 1.0,
                              max_n: int = 5000) -> float:
    """Approximate MMD^2 between source and target feature distributions."""
    rng = np.random.default_rng(42)
    if X_s.shape[0] > max_n:
        X_s = X_s[rng.choice(X_s.shape[0], max_n, replace=False)]
    if X_t.shape[0] > max_n:
        X_t = X_t[rng.choice(X_t.shape[0], max_n, replace=False)]

    def rbf(a, b):
        a_norm = np.sum(a * a, axis=1).reshape(-1, 1)
        b_norm = np.sum(b * b, axis=1).reshape(1, -1)
        return np.exp(-gamma * (a_norm + b_norm - 2 * a @ b.T))

    k_ss = rbf(X_s, X_s).mean()
    k_tt = rbf(X_t, X_t).mean()
    k_st = rbf(X_s, X_t).mean()
    return float(k_ss + k_tt - 2 * k_st)


# --------------------------------------------------------- evaluator

@torch.no_grad()
def evaluate(model, X: np.ndarray, y: np.ndarray, batch: int = 4096) -> dict:
    model.eval()
    device = next(model.parameters()).device
    preds, probs = [], []
    for i in range(0, X.shape[0], batch):
        chunk = torch.as_tensor(X[i:i+batch], dtype=torch.float32, device=device)
        logits = model(chunk)
        preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
        probs.extend(torch.softmax(logits, dim=-1).cpu().numpy().tolist())
    preds, probs = np.asarray(preds), np.asarray(probs)
    metrics = {
        'accuracy': float(accuracy_score(y, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(y, preds)),
        'f1_macro': float(f1_score(y, preds, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, preds, average='weighted',
                                       zero_division=0)),
    }
    try:
        # multi-class ROC-AUC requires probability matrix
        metrics['roc_auc'] = float(
            roc_auc_score(y, probs, multi_class='ovr', average='macro'))
    except ValueError:
        metrics['roc_auc'] = None
    return metrics


# --------------------------------------------------------- target loader

def load_cicids2017(dataset_root: str, feature_subset: list[str],
                    label_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """Load CICIDS2017 from CSV files in `dataset_root`."""
    csv_files = list(Path(dataset_root).glob('*.csv'))
    print(f"[INFO] Loading CICIDS2017 from {len(csv_files)} CSV files ...")
    frames = []
    for f in csv_files:
        df = pd.read_csv(f, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        # Normalize label column
        label_col = next((c for c in df.columns if 'label' in c.lower()), None)
        if label_col is None:
            continue
        df = df.rename(columns={label_col: 'Label'})
        df['Label'] = df['Label'].astype(str).str.strip()
        df['MappedLabel'] = df['Label'].map(label_map).fillna('Other')
        df = df[df['MappedLabel'] != 'Other']
        # Restrict to common features
        keep = [f for f in feature_subset if f in df.columns]
        if len(keep) < 10:
            print(f"[WARN] {f.name} only has {len(keep)} common features, skipping")
            continue
        df = df[keep + ['MappedLabel']]
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        frames.append(df)
    if not frames:
        raise RuntimeError("CICIDS2017 produced no usable samples")
    full = pd.concat(frames, ignore_index=True)
    X = full[feature_subset].to_numpy(dtype=np.float32)
    y_str = full['MappedLabel'].to_numpy()
    return X, y_str


# --------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-mode', choices=['binary', 'multiclass'],
                        default='multiclass')
    parser.add_argument('--target-dataset', default='CICIDS2017')
    parser.add_argument('--target-root', type=str,
                        default='datasets/CICIDS2017')
    parser.add_argument('--models', default='GRU,ResNet1D,FT-Transformer')
    parser.add_argument('--output-dir', default='results_2026_03_03')
    args = parser.parse_args()

    if load_trained_model is None:
        print("[FATAL] Project modules unavailable; aborting.")
        sys.exit(1)

    models = args.models.split(',')

    # Load source (CIC IoT-DIAD 2024)
    src = load_preprocessed_data(mode=args.source_mode)
    feature_intersection = [f for f in COMMON_FEATURES
                            if f in src['feature_names']]
    src_idx = [src['feature_names'].index(f) for f in feature_intersection]
    X_src = src['X_test'][:, src_idx]

    # Load target (CICIDS2017)
    X_tgt_raw, y_tgt_str = load_cicids2017(
        args.target_root, feature_intersection, LABEL_MAP_2017_TO_DIAD)

    # Apply the source scaler so feature distributions are comparable
    scaler = fit_scaler_from_dict(src, restrict_to=src_idx)
    X_tgt = scaler.transform(X_tgt_raw)

    # Map labels to source label index
    label_to_idx = src['label_to_idx']
    y_tgt = np.asarray([label_to_idx.get(s, -1) for s in y_tgt_str])
    mask = y_tgt != -1
    X_tgt, y_tgt = X_tgt[mask], y_tgt[mask]
    print(f"[INFO] target rows after label filter = {y_tgt.shape[0]}")

    # Domain distance
    mmd = maximum_mean_discrepancy(X_src, X_tgt)
    print(f"[INFO] MMD source vs target = {mmd:.4f}")

    out_dir = Path(args.output_dir) / args.source_mode / 'cross_dataset'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {'target_dataset': args.target_dataset,
               'common_features': feature_intersection,
               'mmd': mmd, 'models': {}}

    for name in models:
        print(f"\n--- {name} ---")
        model = load_trained_model(name, mode=args.source_mode)
        src_metrics = evaluate(model, X_src, src['y_test'])
        tgt_metrics = evaluate(model, X_tgt, y_tgt)
        gap = {k: src_metrics[k] - tgt_metrics[k]
               for k in src_metrics if src_metrics[k] is not None
               and tgt_metrics[k] is not None}
        ratio = {k: (tgt_metrics[k] / src_metrics[k]
                     if src_metrics[k] not in (None, 0) else None)
                 for k in src_metrics}
        print(f"  source: {src_metrics}")
        print(f"  target: {tgt_metrics}")
        print(f"  gap:    {gap}")
        summary['models'][name] = {
            'source': src_metrics, 'target': tgt_metrics,
            'gap': gap, 'ratio': ratio,
        }

    out_path = out_dir / 'cross_dataset_summary.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Written {out_path}")


if __name__ == '__main__':
    main()
