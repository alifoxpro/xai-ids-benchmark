"""
run_ablation_design.py
======================
Design-choice ablation: train a fixed architecture under several variants and
compare against the reference configuration.

Variants exercised
------------------
  1. SMOTE                : on (reference) / off
  2. Class weights        : on (reference) / off
  3. Batch size           : 128 / 256 / 512 (reference) / 1024
  4. Training data fraction: 25 / 50 / 75 / 100% (reference)
  5. Attention heads (FT-Transformer only): 2 / 4 / 8 (reference) / 16

Produces the per-variant rows of Table 10 (Design Choice Ablation) in the paper.

Usage
-----
python experiments/run_ablation_design.py --mode binary --model GRU
python experiments/run_ablation_design.py --mode multiclass --model FT-Transformer
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.stage1_preprocess import load_preprocessed_data
    from src.stage3_dl_models import train_and_evaluate_model
    from src.config import MODEL_REGISTRY
except ImportError as exc:
    print(f"[WARN] Project modules not importable ({exc}).")
    MODEL_REGISTRY = ['GRU', 'FT-Transformer']
    load_preprocessed_data = None
    train_and_evaluate_model = None


# Reference configuration (matches configs/train_default.yaml)
REFERENCE = {
    'use_smote': True,
    'use_class_weights': True,
    'batch_size': 512,
    'data_fraction': 1.0,
    'attention_heads': 8,   # only used by FT-Transformer
}


def run_variant(model_name: str, mode: str, variant_cfg: dict,
                base_data: dict) -> dict:
    """Run a single training+evaluation under variant_cfg overrides."""
    if train_and_evaluate_model is None:
        return {'accuracy': float('nan'), 'balanced_accuracy': float('nan')}

    # Apply data-fraction subsampling deterministically
    data = copy.copy(base_data)
    frac = variant_cfg.get('data_fraction', 1.0)
    if frac < 1.0:
        rng = np.random.default_rng(42)
        n = int(data['X_train'].shape[0] * frac)
        idx = rng.choice(data['X_train'].shape[0], size=n, replace=False)
        data['X_train'] = data['X_train'][idx]
        data['y_train'] = data['y_train'][idx]

    metrics = train_and_evaluate_model(
        model_name=model_name, data=data, seed=42,
        use_smote=variant_cfg.get('use_smote', True),
        use_class_weights=variant_cfg.get('use_class_weights', True),
        batch_size=variant_cfg.get('batch_size', 512),
        attention_heads=variant_cfg.get('attention_heads', 8),
    )
    return metrics


def build_variants(model_name: str) -> dict[str, dict]:
    variants = {}

    # Reference
    variants['reference'] = REFERENCE.copy()

    # Class balancing
    variants['no_smote'] = {**REFERENCE, 'use_smote': False}
    variants['no_class_weights'] = {**REFERENCE, 'use_class_weights': False}
    variants['no_smote_no_weights'] = {**REFERENCE,
                                       'use_smote': False,
                                       'use_class_weights': False}

    # Batch size sweep
    for bs in [128, 256, 1024]:
        variants[f'batch_{bs}'] = {**REFERENCE, 'batch_size': bs}

    # Training data fraction
    for frac in [0.25, 0.50, 0.75]:
        variants[f'data_{int(frac*100):02d}pct'] = {**REFERENCE,
                                                    'data_fraction': frac}

    # Attention heads (FT-Transformer only)
    if model_name == 'FT-Transformer':
        for h in [2, 4, 16]:
            variants[f'heads_{h:02d}'] = {**REFERENCE, 'attention_heads': h}

    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['binary', 'multiclass'],
                        default='multiclass')
    parser.add_argument('--model', default='FT-Transformer',
                        choices=MODEL_REGISTRY)
    parser.add_argument('--output-dir', default='results_2026_03_03')
    parser.add_argument('--variants', default='all',
                        help='Comma-separated subset, or "all"')
    args = parser.parse_args()

    if load_preprocessed_data is None:
        print("[FATAL] Project modules unavailable; aborting.")
        sys.exit(1)

    base_data = load_preprocessed_data(mode=args.mode)
    variants = build_variants(args.model)
    if args.variants != 'all':
        chosen = set(args.variants.split(','))
        variants = {k: v for k, v in variants.items() if k in chosen}

    out_dir = Path(args.output_dir) / args.mode / 'ablation_design'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for variant_name, cfg in variants.items():
        print(f"\n=== {args.model} @ {variant_name} ===")
        t0 = time.time()
        metrics = run_variant(args.model, args.mode, cfg, base_data)
        metrics['wall_time_sec'] = round(time.time() - t0, 2)
        metrics['config'] = cfg
        summary[variant_name] = metrics
        print(f"  acc = {metrics.get('accuracy', float('nan')):.4f}  "
              f"bal = {metrics.get('balanced_accuracy', float('nan')):.4f}  "
              f"time = {metrics['wall_time_sec']:.1f}s")

    out_path = out_dir / f'{args.model}_ablation.json'
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Written {out_path}")


if __name__ == '__main__':
    main()
