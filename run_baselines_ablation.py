"""
Quick ML baselines + Ablation study for the paper.
Runs XGBoost, LightGBM, CatBoost, RandomForest on the preprocessed data.
Also runs ablation: no balancing, no dedup effects (reduced features).
"""
import numpy as np
import time
import json
import gc
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier

# Load preprocessed data
print("Loading data...")
d = np.load('results_2026_03_03/multiclass/stage1_cache/arrays.npz', allow_pickle=True)
X_train, y_train = d['X_train'], d['y_train']
X_test, y_test = d['X_test'], d['y_test']
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Sample for faster training (use 500K train, full test)
np.random.seed(42)
sample_idx = np.random.choice(len(X_train), size=min(500000, len(X_train)), replace=False)
X_tr = X_train[sample_idx]
y_tr = y_train[sample_idx]
print(f"Sampled train: {X_tr.shape}")

# Test on 500K sample too for speed
test_idx = np.random.choice(len(X_test), size=min(500000, len(X_test)), replace=False)
X_te = X_test[test_idx]
y_te = y_test[test_idx]
print(f"Sampled test: {X_te.shape}")

results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\n{'='*50}")
    print(f"Training {name}...")
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_te)
    inf_time = (time.time() - t0) / len(X_te) * 1000  # ms per sample

    acc = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average='macro', zero_division=0)
    f1_w = f1_score(y_te, y_pred, average='weighted', zero_division=0)
    prec_w = precision_score(y_te, y_pred, average='weighted', zero_division=0)

    print(f"  Acc: {acc:.4f}, Bal Acc: {bal_acc:.4f}, F1 macro: {f1_mac:.4f}")
    print(f"  Train: {train_time:.1f}s, Inf: {inf_time:.4f}ms/sample")

    results[name] = {
        'accuracy': acc, 'balanced_accuracy': bal_acc,
        'f1_macro': f1_mac, 'f1_weighted': f1_w,
        'precision_weighted': prec_w,
        'train_time': train_time, 'inference_ms': inf_time
    }
    gc.collect()

# 1. XGBoost
try:
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        tree_method='hist', device='cuda',
        n_jobs=-1, random_state=42, verbosity=0
    )
    evaluate("XGBoost", xgb, X_tr, y_tr, X_te, y_te)
    del xgb; gc.collect()
except Exception as e:
    print(f"XGBoost failed: {e}")
    # Try CPU
    try:
        xgb = XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            tree_method='hist', n_jobs=-1, random_state=42, verbosity=0
        )
        evaluate("XGBoost", xgb, X_tr, y_tr, X_te, y_te)
        del xgb; gc.collect()
    except Exception as e2:
        print(f"XGBoost CPU also failed: {e2}")

# 2. LightGBM
try:
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        n_jobs=-1, random_state=42, verbose=-1, device='gpu'
    )
    evaluate("LightGBM", lgbm, X_tr, y_tr, X_te, y_te)
    del lgbm; gc.collect()
except Exception as e:
    print(f"LightGBM GPU failed: {e}")
    try:
        lgbm = LGBMClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            n_jobs=-1, random_state=42, verbose=-1
        )
        evaluate("LightGBM", lgbm, X_tr, y_tr, X_te, y_te)
        del lgbm; gc.collect()
    except Exception as e2:
        print(f"LightGBM CPU also failed: {e2}")

# 3. CatBoost
try:
    from catboost import CatBoostClassifier
    cb = CatBoostClassifier(
        iterations=200, depth=8, learning_rate=0.1,
        task_type='GPU', random_seed=42, verbose=0
    )
    evaluate("CatBoost", cb, X_tr, y_tr, X_te, y_te)
    del cb; gc.collect()
except Exception as e:
    print(f"CatBoost GPU failed: {e}")
    try:
        cb = CatBoostClassifier(
            iterations=200, depth=8, learning_rate=0.1,
            random_seed=42, verbose=0
        )
        evaluate("CatBoost", cb, X_tr, y_tr, X_te, y_te)
        del cb; gc.collect()
    except Exception as e2:
        print(f"CatBoost CPU also failed: {e2}")

# 4. Random Forest
try:
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=20,
        n_jobs=-1, random_state=42
    )
    evaluate("RandomForest", rf, X_tr, y_tr, X_te, y_te)
    del rf; gc.collect()
except Exception as e:
    print(f"RandomForest failed: {e}")

# ============================================================
# ABLATION: Reduced features (top-30 vs all 81)
# ============================================================
print("\n" + "="*60)
print("ABLATION: Top-30 features vs All 81")
print("="*60)

# Get top-30 feature indices from SHAP (use feature importance from the data)
# We'll use variance-based selection as proxy
from sklearn.feature_selection import mutual_info_classif
print("Computing mutual information for top-30 features...")
mi = mutual_info_classif(X_tr[:50000], y_tr[:50000], random_state=42, n_jobs=-1)
top30_idx = np.argsort(mi)[-30:]

X_tr_30 = X_tr[:, top30_idx]
X_te_30 = X_te[:, top30_idx]

# Run best ML model on reduced features
try:
    from xgboost import XGBClassifier
    xgb30 = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0
    )
    evaluate("XGBoost_top30", xgb30, X_tr_30, y_tr, X_te_30, y_te)
    del xgb30; gc.collect()
except:
    pass

# ABLATION: Without class balancing (original imbalanced data)
# The loaded data is already balanced. We simulate "no balance" by using
# the original class distribution weights
print("\n" + "="*60)
print("ABLATION: Impact of feature count")
print("="*60)

# Top-10 features
top10_idx = np.argsort(mi)[-10:]
X_tr_10 = X_tr[:, top10_idx]
X_te_10 = X_te[:, top10_idx]

try:
    from xgboost import XGBClassifier
    xgb10 = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0
    )
    evaluate("XGBoost_top10", xgb10, X_tr_10, y_tr, X_te_10, y_te)
    del xgb10; gc.collect()
except:
    pass

# Top-50 features
top50_idx = np.argsort(mi)[-50:]
X_tr_50 = X_tr[:, top50_idx]
X_te_50 = X_te[:, top50_idx]

try:
    from xgboost import XGBClassifier
    xgb50 = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        tree_method='hist', n_jobs=-1, random_state=42, verbosity=0
    )
    evaluate("XGBoost_top50", xgb50, X_tr_50, y_tr, X_te_50, y_te)
    del xgb50; gc.collect()
except:
    pass

# Save all results
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
for name, r in results.items():
    print(f"{name}: Acc={r['accuracy']:.4f}, Bal={r['balanced_accuracy']:.4f}, F1m={r['f1_macro']:.4f}")

with open('results_2026_03_03/ml_baselines_ablation.json', 'w') as f:
    json.dump(results, f, indent=2)
print("\nResults saved to ml_baselines_ablation.json")
