# -*- coding: utf-8 -*-
"""
compute_params_flops.py
=======================
Load each trained multiclass model and report:
  - parameter count
  - model size (MB, fp32)
  - FLOPs per sample (via fvcore, best-effort)
Writes results_2026_03_03/multiclass/efficiency/params_flops.csv
"""
import io, os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))  # so pickle can resolve stage3_ml_models classes
MODELS = ROOT / 'models' / 'multiclass' / 'stage3'
META = ROOT / 'results_2026_03_03' / 'multiclass' / 'stage1_cache' / 'metadata.json'
OUT = ROOT / 'results_2026_03_03' / 'multiclass' / 'efficiency'
OUT.mkdir(parents=True, exist_ok=True)

n_features = len(json.load(open(META))['feature_columns'])
print(f'[INFO] n_features = {n_features}')

try:
    from fvcore.nn import FlopCountAnalysis
    HAVE_FVCORE = True
except Exception:
    HAVE_FVCORE = False


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def flops_per_sample(m, n_feat):
    if not HAVE_FVCORE:
        return None
    try:
        m.eval()
        dev = next(m.parameters()).device
        x = torch.randn(1, n_feat, device=dev)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fa = FlopCountAnalysis(m, x)
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            return int(fa.total())
    except Exception as e:
        print(f'    [flops failed] {e}')
        return None


rows = []
for mf in sorted(MODELS.glob('*.pkl')):
    name = mf.stem
    if mf.stat().st_size == 0:
        print(f'--- {name}: empty checkpoint, skip')
        continue
    try:
        model = pickle.load(open(mf, 'rb'))
        if hasattr(model, 'cuda'):
            model = model.cuda()
        params = count_params(model)
        size_mb = params * 4 / 1e6
        flops = flops_per_sample(model, n_features)
        rows.append({'Model': name, 'Params': params,
                     'Size_MB': round(size_mb, 3),
                     'FLOPs': flops,
                     'FLOPs_M': round(flops / 1e6, 3) if flops else None})
        print(f'--- {name}: params={params:,}  size={size_mb:.2f}MB  '
              f'flops={flops/1e6 if flops else None}M')
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f'--- {name}: ERROR {e}')

import csv
out_csv = OUT / 'params_flops.csv'
with open(out_csv, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['Model', 'Params', 'Size_MB', 'FLOPs', 'FLOPs_M'])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f'\n[DONE] {out_csv}')
for r in sorted(rows, key=lambda r: r['Params']):
    print(f"  {r['Model']:<18} {r['Params']:>12,}  {r['Size_MB']:>8.2f} MB  "
          f"{str(r['FLOPs_M'])+' M' if r['FLOPs_M'] else 'n/a':>10}")
