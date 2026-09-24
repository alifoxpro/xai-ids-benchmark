# -*- coding: utf-8 -*-
"""
topsis_synthesis.py
===================
Multi-criteria decision synthesis (TOPSIS) over the per-model results, turning the
seven-axis benchmark into a single defensible ranking + a radar comparison. This
operationalises the "Multi-Criteria Benchmark" of the title: a model wins only if
it is jointly strong on effectiveness, generalization, and efficiency.

Criteria (per model, benefit unless noted):
  balanced_accuracy, f1_macro, roc_auc, cross_dataset_bal_acc,
  params (cost), train_time (cost), infer_time (cost)

Output: results_2026_03_03/multiclass/synthesis/topsis_ranking.json
        paper/figures/fig_topsis_radar.png  (+ paper_submission/fig)
"""
import os, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MC = ROOT / 'results_2026_03_03' / 'multiclass'
OUT = MC / 'synthesis'; OUT.mkdir(parents=True, exist_ok=True)
FIGDIRS = [ROOT / 'paper' / 'figures', ROOT / 'paper_submission' / 'fig']

# Per-model values assembled from Tables 1, 3, 6 (8 single models; ensemble/TCN
# excluded for incomplete per-model criteria). benefit=+1, cost=-1.
MODELS = ['GRU', 'ResNet1D', 'FT-Transformer', 'CNN-LSTM', 'BiLSTM-Attn',
          'TabNet', 'Transformer', 'AE-Classifier']
#                bal-acc  f1mac  roc-auc  xdom    params(K) train(s)  infer(ms)
DATA = {
 'GRU':            [0.562, 0.440, 0.991, 0.435,  547.0,  11901, 0.0040],
 'ResNet1D':       [0.580, 0.471, 0.992, 0.506,  325.5,   3106, 0.0054],
 'FT-Transformer': [0.584, 0.475, 0.992, 0.493,  165.2, 248650, 0.0782],
 'CNN-LSTM':       [0.534, 0.435, 0.990, 0.373,  224.2,  43971, 0.0105],
 'BiLSTM-Attn':    [0.528, 0.423, 0.990, 0.570,  835.7,   3824, 0.0155],
 'TabNet':         [0.525, 0.423, 0.989, 0.361,   58.8,   4815, 0.0011],
 'Transformer':    [0.553, 0.454, 0.991, 0.455,   38.5,  19061, 0.0027],
 'AE-Classifier':  [0.475, 0.387, 0.987, 0.501,   27.0,   2407, 0.0003],
}
CRIT = ['Bal-Acc', 'Macro-F1', 'ROC-AUC', 'X-dataset', 'Params', 'Train-t', 'Infer-t']
BENEFIT = np.array([1, 1, 1, 1, -1, -1, -1])   # +1 benefit, -1 cost
WEIGHTS = np.array([0.20, 0.20, 0.10, 0.20, 0.10, 0.10, 0.10])  # emphasise detection+transfer


def topsis(M, benefit, w):
    # 1. vector-normalize columns
    norm = M / np.sqrt((M ** 2).sum(axis=0, keepdims=True))
    # 2. weight
    V = norm * w
    # 3. ideal best/worst (respect benefit/cost)
    best = np.where(benefit > 0, V.max(0), V.min(0))
    worst = np.where(benefit > 0, V.min(0), V.max(0))
    dp = np.sqrt(((V - best) ** 2).sum(1))
    dn = np.sqrt(((V - worst) ** 2).sum(1))
    return dn / (dp + dn)   # closeness coefficient in [0,1], higher=better


def main():
    M = np.array([DATA[m] for m in MODELS], dtype=float)
    score = topsis(M, BENEFIT, WEIGHTS)
    order = np.argsort(score)[::-1]
    ranking = [{'rank': int(i + 1), 'model': MODELS[order[i]],
                'topsis_score': round(float(score[order[i]]), 4)}
               for i in range(len(MODELS))]
    json.dump({'criteria': CRIT, 'weights': WEIGHTS.tolist(), 'ranking': ranking},
              open(OUT / 'topsis_ranking.json', 'w'), indent=2)
    print('TOPSIS ranking (higher = better overall):')
    for r in ranking:
        print(f"  {r['rank']}. {r['model']:<16} {r['topsis_score']:.4f}")

    # ---- radar of the top-3 models on the 4 benefit criteria (normalized 0-1) ----
    ben_idx = [0, 1, 2, 3]
    sub = M[:, ben_idx]
    subn = (sub - sub.min(0)) / (sub.max(0) - sub.min(0) + 1e-9)
    labels = [CRIT[i] for i in ben_idx]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    for i in range(3):
        mi = order[i]
        vals = subn[mi].tolist(); vals += vals[:1]
        ax.plot(ang, vals, linewidth=2, label=f'{MODELS[mi]} (#{i+1})')
        ax.fill(ang, vals, alpha=0.1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(labels)
    ax.set_yticklabels([]); ax.set_title('Top-3 models by TOPSIS (benefit criteria)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    for d in FIGDIRS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'fig_topsis_radar.png', bbox_inches='tight', dpi=300)
        fig.savefig(d / 'fig_topsis_radar.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n[DONE] {OUT / 'topsis_ranking.json'} + fig_topsis_radar")


if __name__ == '__main__':
    main()
