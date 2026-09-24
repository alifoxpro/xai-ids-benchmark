"""
run_multiple_seeds.py
=====================
Train each of the 10 DL architectures using 5 random seeds, collect per-seed
metrics, and aggregate mean ± standard deviation for variance estimation.

Used to produce the "Bootstrap Mean" column and variance estimates referenced
in Tables 3, 4, and 10 of the paper.

Usage
-----
python experiments/run_multiple_seeds.py --mode multiclass
python experiments/run_multiple_seeds.py --mode binary --seeds 17,42,123

Output
------
results_2026_03_03/<mode>/seed_variance/<model>_seeds.json   (per-model)
results_2026_03_03/<mode>/seed_variance/summary.json         (aggregated)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Make project importable when run from repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports — assumes the project's stage modules exist
try:
    from src.stage3_dl_models import train_and_evaluate_model
    from src.stage1_preprocess import load_preprocessed_data
    from src.config import MODEL_REGISTRY, DEFAULT_SEEDS
except ImportError as exc:   # graceful fallback for documentation runs
    print(f"[WARN] Project modules not importable ({exc}). "
          "Script will only validate arguments and config.")
    train_and_evaluate_model = None
    load_preprocessed_data = None
    MODEL_REGISTRY = ['GRU', 'ResNet1D', 'FT-Transformer', 'CNN-LSTM',
                     'BiLSTM-Attention', 'TabNet', 'TCN', 'Transformer',
                     'AE-Classifier', 'VotingEnsemble']
    DEFAULT_SEEDS = [17, 42, 123, 2024, 31337]


def set_all_seeds(seed: int) -> None:
    """Seed every stochastic component for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def aggregate_metrics(per_seed_results: list[dict]) -> dict:
    """Compute mean and std for each metric across seeds."""
    if not per_seed_results:
        return {}
    keys = per_seed_results[0].keys()
    summary = {}
    for key in keys:
        if isinstance(per_seed_results[0][key], (int, float)):
            values = [r[key] for r in per_seed_results]
            summary[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['binary', 'multiclass'],
                        default='multiclass')
    parser.add_argument('--seeds', type=str, default=','.join(str(s) for s in DEFAULT_SEEDS),
                        help='Comma-separated seeds, e.g. "17,42,123,2024,31337"')
    parser.add_argument('--models', type=str, default='all',
                        help='Comma-separated subset of models, or "all"')
    parser.add_argument('--output-dir', type=str, default='results_2026_03_03')
    parser.add_argument('--quick', action='store_true',
                        help='Use quick-profile config (1M rows, 20 epochs)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    models = MODEL_REGISTRY if args.models == 'all' else args.models.split(',')

    output_dir = Path(args.output_dir) / args.mode / 'seed_variance'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Models: {models}")
    print(f"[INFO] Output: {output_dir}")

    if train_and_evaluate_model is None:
        print("[FATAL] Project modules unavailable; cannot run training.")
        sys.exit(1)

    # Load data once (preprocessing is deterministic given fixed seed=42)
    data = load_preprocessed_data(mode=args.mode, quick=args.quick)

    summary = {}
    for model_name in models:
        print(f"\n{'='*60}\n  Model: {model_name}\n{'='*60}")
        per_seed = []
        for seed in seeds:
            set_all_seeds(seed)
            t0 = time.time()
            metrics = train_and_evaluate_model(
                model_name=model_name,
                data=data,
                seed=seed,
                quick=args.quick,
            )
            metrics['seed'] = seed
            metrics['wall_time_sec'] = round(time.time() - t0, 2)
            per_seed.append(metrics)
            print(f"  seed={seed:>5}  acc={metrics.get('accuracy', float('nan')):.4f}  "
                  f"bal={metrics.get('balanced_accuracy', float('nan')):.4f}  "
                  f"time={metrics['wall_time_sec']:.1f}s")

        # Save per-seed and aggregated
        with open(output_dir / f'{model_name}_seeds.json', 'w') as f:
            json.dump(per_seed, f, indent=2)

        agg = aggregate_metrics(per_seed)
        summary[model_name] = agg
        print(f"  -> mean acc = {agg.get('accuracy', {}).get('mean', float('nan')):.4f} "
              f"± {agg.get('accuracy', {}).get('std', float('nan')):.4f}")

    # Final summary
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[DONE] Summary written to {output_dir / 'summary.json'}")


if __name__ == '__main__':
    main()
