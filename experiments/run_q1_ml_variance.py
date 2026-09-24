"""
run_q1_ml_variance.py
=====================
Q1 upgrade: Multiple-seed variance for ML baselines + Design ablation.

Runs XGBoost, LightGBM, CatBoost, RandomForest with 5 random seeds each,
plus design ablation (no SMOTE, no class weights, batch/data variants for
the tree-based equivalents).

Uses the cached preprocessed data at:
    results_2026_03_03/multiclass/stage1_cache/arrays.npz

Total wall time on RTX 5070: ~1 hour.

Output:
    results_2026_03_03/multiclass/q1_variance/ml_seeds.json
    results_2026_03_03/multiclass/q1_variance/ml_ablation.json
"""

import json
import time
import gc
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, precision_score)
from sklearn.utils.class_weight import compute_class_weight

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False


SEEDS = [17, 42, 123, 2024, 31337]
DATA_PATH = Path("results_2026_03_03/multiclass/stage1_cache/arrays.npz")
OUT_DIR = Path("results_2026_03_03/multiclass/q1_variance")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    inf_time = (time.time() - t0) / len(X_te) * 1000  # ms per sample

    return {
        'accuracy': float(accuracy_score(y_te, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_te, y_pred)),
        'f1_macro': float(f1_score(y_te, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_te, y_pred, average='weighted', zero_division=0)),
        'precision_weighted': float(precision_score(y_te, y_pred, average='weighted', zero_division=0)),
        'train_time': train_time,
        'inference_ms': inf_time,
    }


def aggregate(per_seed):
    keys = [k for k in per_seed[0].keys()
            if isinstance(per_seed[0][k], (int, float))]
    out = {}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[k] = {
            'mean': float(np.mean(vals)),
            'std': float(np.std(vals, ddof=1)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
            'n': len(vals),
        }
    return out


def main():
    print(f"[INFO] Loading cached arrays from {DATA_PATH} ...")
    d = np.load(DATA_PATH, allow_pickle=True)
    X_train_full, y_train_full = d['X_train'], d['y_train']
    X_test_full, y_test_full = d['X_test'], d['y_test']
    print(f"[INFO] Train: {X_train_full.shape}, Test: {X_test_full.shape}")

    # Sample 500K train + 500K test for tractable multi-seed runs
    print("[INFO] Sampling 500K train + 500K test per seed ...")

    # ---------------------------------------------------------------
    # Phase 1: Multi-seed variance for 4 ML baselines
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 1: Multi-seed variance (5 seeds × 4 ML models)")
    print("=" * 60)

    multi_seed = {}
    for model_name in ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest']:
        if model_name == 'XGBoost' and not HAS_XGB:
            print(f"[SKIP] {model_name} not available")
            continue
        if model_name == 'LightGBM' and not HAS_LGBM:
            print(f"[SKIP] {model_name} not available")
            continue
        if model_name == 'CatBoost' and not HAS_CAT:
            print(f"[SKIP] {model_name} not available")
            continue

        per_seed_results = []
        for seed in SEEDS:
            print(f"\n--- {model_name} seed={seed} ---")
            rng = np.random.default_rng(seed)
            tr_idx = rng.choice(len(X_train_full),
                                size=min(500_000, len(X_train_full)),
                                replace=False)
            te_idx = rng.choice(len(X_test_full),
                                size=min(500_000, len(X_test_full)),
                                replace=False)
            X_tr, y_tr = X_train_full[tr_idx], y_train_full[tr_idx]
            X_te, y_te = X_test_full[te_idx], y_test_full[te_idx]

            if model_name == 'XGBoost':
                model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    tree_method='hist', device='cpu',
                    objective='multi:softprob', num_class=8,
                    random_state=seed, n_jobs=-1, verbosity=0)
            elif model_name == 'LightGBM':
                model = lgb.LGBMClassifier(
                    n_estimators=200, max_depth=8, learning_rate=0.1,
                    objective='multiclass', num_class=8,
                    random_state=seed, n_jobs=-1, verbosity=-1)
            elif model_name == 'CatBoost':
                model = CatBoostClassifier(
                    iterations=200, depth=8, learning_rate=0.1,
                    loss_function='MultiClass',
                    random_seed=seed, verbose=False)
            elif model_name == 'RandomForest':
                model = RandomForestClassifier(
                    n_estimators=200, max_depth=15,
                    random_state=seed, n_jobs=-1)

            metrics = evaluate(model_name, model, X_tr, y_tr, X_te, y_te)
            metrics['seed'] = seed
            per_seed_results.append(metrics)
            print(f"  acc={metrics['accuracy']:.4f}  "
                  f"bal={metrics['balanced_accuracy']:.4f}  "
                  f"f1m={metrics['f1_macro']:.4f}  "
                  f"t={metrics['train_time']:.1f}s")

            del model, X_tr, y_tr, X_te, y_te
            gc.collect()

        multi_seed[model_name] = {
            'per_seed': per_seed_results,
            'aggregate': aggregate(per_seed_results),
        }
        print(f"\n  >>> {model_name} mean acc = "
              f"{multi_seed[model_name]['aggregate']['accuracy']['mean']:.4f} "
              f"± {multi_seed[model_name]['aggregate']['accuracy']['std']:.4f}")

    with open(OUT_DIR / 'ml_seeds.json', 'w') as f:
        json.dump(multi_seed, f, indent=2)
    print(f"\n[DONE Phase 1] -> {OUT_DIR / 'ml_seeds.json'}")

    # ---------------------------------------------------------------
    # Phase 2: Design ablation on XGBoost (the strongest baseline)
    # ---------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  PHASE 2: Design ablation (XGBoost, seed=42)")
    print("=" * 60)

    if not HAS_XGB:
        print("[SKIP] XGBoost not available; aborting Phase 2")
        return

    rng = np.random.default_rng(42)
    base_tr = rng.choice(len(X_train_full),
                         size=min(500_000, len(X_train_full)), replace=False)
    base_te = rng.choice(len(X_test_full),
                         size=min(500_000, len(X_test_full)), replace=False)
    X_tr_base = X_train_full[base_tr]
    y_tr_base = y_train_full[base_tr]
    X_te = X_test_full[base_te]
    y_te = y_test_full[base_te]

    ablation_results = {}

    # Reference (already in Phase 1) — recompute for clarity
    print("\n--- Variant: reference ---")
    model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                               tree_method='hist', device='cpu',
                               objective='multi:softprob', num_class=8,
                               random_state=42, n_jobs=-1, verbosity=0)
    ablation_results['reference'] = evaluate('XGBoost', model,
                                              X_tr_base, y_tr_base, X_te, y_te)
    print(f"  acc={ablation_results['reference']['accuracy']:.4f}  "
          f"bal={ablation_results['reference']['balanced_accuracy']:.4f}")

    # Class weights variant — apply sample_weight inversely proportional to class freq
    print("\n--- Variant: no_class_weights (default model has none) ---")
    # XGBoost reference already has no class weights, so this matches.

    print("\n--- Variant: with_class_weights ---")
    class_w = compute_class_weight('balanced',
                                    classes=np.unique(y_tr_base),
                                    y=y_tr_base)
    sample_w = np.array([class_w[int(c)] for c in y_tr_base])
    model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                               tree_method='hist', device='cpu',
                               objective='multi:softprob', num_class=8,
                               random_state=42, n_jobs=-1, verbosity=0)
    t0 = time.time()
    model.fit(X_tr_base, y_tr_base, sample_weight=sample_w)
    train_time = time.time() - t0
    y_pred = model.predict(X_te)
    ablation_results['with_class_weights'] = {
        'accuracy': float(accuracy_score(y_te, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_te, y_pred)),
        'f1_macro': float(f1_score(y_te, y_pred, average='macro', zero_division=0)),
        'train_time': train_time,
    }
    print(f"  acc={ablation_results['with_class_weights']['accuracy']:.4f}  "
          f"bal={ablation_results['with_class_weights']['balanced_accuracy']:.4f}")

    # Training-data fraction sweep
    for frac in [0.25, 0.50, 0.75]:
        print(f"\n--- Variant: data_{int(frac*100):02d}pct ---")
        n_frac = int(len(X_tr_base) * frac)
        idx_f = rng.choice(len(X_tr_base), size=n_frac, replace=False)
        X_f, y_f = X_tr_base[idx_f], y_tr_base[idx_f]
        model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                                   tree_method='hist', device='cpu',
                                   objective='multi:softprob', num_class=8,
                                   random_state=42, n_jobs=-1, verbosity=0)
        ablation_results[f'data_{int(frac*100):02d}pct'] = evaluate(
            'XGBoost', model, X_f, y_f, X_te, y_te)
        print(f"  acc={ablation_results[f'data_{int(frac*100):02d}pct']['accuracy']:.4f}  "
              f"bal={ablation_results[f'data_{int(frac*100):02d}pct']['balanced_accuracy']:.4f}")
        del X_f, y_f, model
        gc.collect()

    # Max-depth sweep
    for depth in [4, 12, 16]:
        print(f"\n--- Variant: depth_{depth} ---")
        model = xgb.XGBClassifier(n_estimators=200, max_depth=depth,
                                   learning_rate=0.1, tree_method='hist',
                                   device='cpu', objective='multi:softprob',
                                   num_class=8, random_state=42, n_jobs=-1,
                                   verbosity=0)
        ablation_results[f'depth_{depth}'] = evaluate('XGBoost', model,
                                                       X_tr_base, y_tr_base,
                                                       X_te, y_te)
        print(f"  acc={ablation_results[f'depth_{depth}']['accuracy']:.4f}  "
              f"bal={ablation_results[f'depth_{depth}']['balanced_accuracy']:.4f}")
        del model
        gc.collect()

    with open(OUT_DIR / 'ml_ablation.json', 'w') as f:
        json.dump(ablation_results, f, indent=2)
    print(f"\n[DONE Phase 2] -> {OUT_DIR / 'ml_ablation.json'}")

    print("\n" + "=" * 60)
    print("  ALL PHASES COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
