# -*- coding: utf-8 -*-
"""
generate_ieee_figures.py
========================
Regenerate the four headline result figures with IEEEtran-compliant styling:
  - 600 DPI
  - Times New Roman 10pt (with STIX math)
  - Title 11pt bold, labels 10pt bold
  - IEEE double-column width 7.16 x 4.5 in
  - Dashed grid 0.5pt at alpha=0.25
  - Inward ticks, framed legend at alpha=0.9
  - Partial axes spines, 0.05 padding

Generates four NEW figures backed by real experimental data:
  fig_cross_dataset_ieee.png   (cross-dataset collapse, 8 models)
  fig_dl_variance_ieee.png     (5-seed DL variance, 4 models)
  fig_ml_variance_ieee.png     (5-seed ML baseline variance)
  fig_xai_stability_ieee.png   (SHAP vs LIME Lipschitz + Jaccard)
"""

import io, json, os, sys
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ===== IEEE STYLING DEFAULTS =====
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.25,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.6',
    'legend.fancybox': False,
})

IEEE_DOUBLE_COL = (7.16, 4.5)
IEEE_SINGLE_COL = (3.5, 2.6)


# ====================================================================== fig 1
def fig_cross_dataset():
    p = PROJECT_ROOT / 'results_2026_03_03/multiclass/cross_dataset/cross_dataset_real.json'
    with open(p) as f:
        data = json.load(f)
    rows = []
    for name, m in data['models'].items():
        if 'balanced_accuracy' in m:
            rows.append((name.replace('_', '-'),
                          m['accuracy'], m['balanced_accuracy'], m['f1_macro']))
    rows.sort(key=lambda r: r[2], reverse=True)
    names = [r[0] for r in rows]
    acc = [r[1] for r in rows]
    bal = [r[2] for r in rows]
    f1m = [r[3] for r in rows]

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE_COL)
    x = np.arange(len(names))
    w = 0.27
    b1 = ax.bar(x - w, acc, w, label='Accuracy', color='#1f77b4')
    b2 = ax.bar(x, bal, w, label='Balanced Accuracy', color='#d62728')
    b3 = ax.bar(x + w, f1m, w, label='F1 (macro)', color='#2ca02c')

    # Reference line at in-distribution best balanced accuracy (FT-Transformer = 0.584)
    ax.axhline(0.584, linestyle=':', color='gray', linewidth=0.8)
    ax.text(len(names) - 0.5, 0.595, 'In-dist.\\ best (FT-Trans.\\ 0.584)',
             fontsize=8, color='gray', ha='right', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('Score on CICIDS2017 (unseen)')
    ax.set_ylim(0, 0.7)
    ax.set_title('Cross-Dataset Transfer: CIC IoT-DIAD 2024 -> CICIDS2017 (N=2.83M)')
    ax.grid(axis='y')
    ax.legend(loc='upper right', ncol=3)
    out = FIG_DIR / 'fig_cross_dataset_ieee.png'
    plt.savefig(out)
    plt.close(fig)
    print(f'  written {out}')


# ====================================================================== fig 2
def fig_dl_variance():
    p1 = PROJECT_ROOT / 'results_2026_03_03/multiclass/q1_variance/dl_3models_seeds.json'
    p2 = PROJECT_ROOT / 'results_2026_03_03/multiclass/q1_variance/gru_seeds.json'
    with open(p1) as f:
        d3 = json.load(f)
    with open(p2) as f:
        dg = json.load(f)

    models, means, stds, mins, maxs = [], [], [], [], []
    for name, r in d3.items():
        a = r['aggregate']['accuracy']
        models.append(name.replace('_', '-'))
        means.append(a['mean']); stds.append(a['std'])
        mins.append(a['min']); maxs.append(a['max'])
    # add GRU
    a = dg['aggregate']['accuracy']
    models.insert(0, 'GRU')
    means.insert(0, a['mean']); stds.insert(0, a['std'])
    mins.insert(0, a['min']); maxs.insert(0, a['max'])

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE_COL)
    x = np.arange(len(models))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                    color=colors[:len(models)], alpha=0.85,
                    error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    # show min/max as dots
    ax.scatter(x, mins, marker='v', color='k', s=15, zorder=3, label='Min seed')
    ax.scatter(x, maxs, marker='^', color='k', s=15, zorder=3, label='Max seed')
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f'{m:.3f}\n$\\pm${s:.3f}',
                 ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('Accuracy (multi-class)')
    ax.set_ylim(0.3, 1.0)
    ax.set_title('5-Seed Variance for DL Architectures (300K subsample, 8 epochs)')
    ax.grid(axis='y')
    ax.legend(loc='lower left')
    out = FIG_DIR / 'fig_dl_variance_ieee.png'
    plt.savefig(out)
    plt.close(fig)
    print(f'  written {out}')


# ====================================================================== fig 3
def fig_ml_variance():
    p = PROJECT_ROOT / 'results_2026_03_03/multiclass/q1_variance/ml_seeds.json'
    with open(p) as f:
        d = json.load(f)
    models, acc_m, acc_s, bal_m, bal_s = [], [], [], [], []
    for name, r in d.items():
        a = r['aggregate']['accuracy']
        b = r['aggregate']['balanced_accuracy']
        models.append(name)
        acc_m.append(a['mean']); acc_s.append(a['std'])
        bal_m.append(b['mean']); bal_s.append(b['std'])

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE_COL)
    x = np.arange(len(models))
    w = 0.35
    b1 = ax.bar(x - w/2, acc_m, w, yerr=acc_s, capsize=4,
                 color='#1f77b4', label='Accuracy',
                 error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    b2 = ax.bar(x + w/2, bal_m, w, yerr=bal_s, capsize=4,
                 color='#d62728', label='Balanced Accuracy',
                 error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    for i, (m, s) in enumerate(zip(acc_m, acc_s)):
        ax.text(i - w/2, m + 0.02, f'{m:.3f}', ha='center', fontsize=8)
    for i, (m, s) in enumerate(zip(bal_m, bal_s)):
        ax.text(i + w/2, m + 0.02, f'{m:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel('Score on CIC IoT-DIAD 2024 (500K, multi-class)')
    ax.set_ylim(0, 1.05)
    ax.set_title('5-Seed Variance for ML Baselines')
    ax.grid(axis='y')
    ax.legend(loc='upper right')
    out = FIG_DIR / 'fig_ml_variance_ieee.png'
    plt.savefig(out)
    plt.close(fig)
    print(f'  written {out}')


# ====================================================================== fig 4
def fig_xai_stability():
    p = PROJECT_ROOT / 'results_2026_03_03/xai_stability/xai_stability_real.json'
    with open(p) as f:
        d = json.load(f)

    cells = [
        ('FT-Trans. (multi)', d['FT_Transformer_multiclass']['summary']),
        ('GRU (binary)',      d['GRU_binary']['summary']),
    ]

    fig, (axL, axJ) = plt.subplots(1, 2, figsize=IEEE_DOUBLE_COL)

    # Lipschitz (left)
    names = [c[0] for c in cells]
    shap_L = [c[1]['shap']['lipschitz']['mean'] for c in cells]
    lime_L = [c[1]['lime']['lipschitz']['mean'] for c in cells]
    shap_Lerr = [[c[1]['shap']['lipschitz']['mean'] - c[1]['shap']['lipschitz']['ci95_low'] for c in cells],
                  [c[1]['shap']['lipschitz']['ci95_high'] - c[1]['shap']['lipschitz']['mean'] for c in cells]]
    lime_Lerr = [[c[1]['lime']['lipschitz']['mean'] - c[1]['lime']['lipschitz']['ci95_low'] for c in cells],
                  [c[1]['lime']['lipschitz']['ci95_high'] - c[1]['lime']['lipschitz']['mean'] for c in cells]]
    x = np.arange(len(names)); w = 0.35
    axL.bar(x - w/2, shap_L, w, yerr=shap_Lerr, capsize=4,
             color='#1f77b4', label='SHAP',
             error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    axL.bar(x + w/2, lime_L, w, yerr=lime_Lerr, capsize=4,
             color='#ff7f0e', label='LIME',
             error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    axL.set_xticks(x); axL.set_xticklabels(names, fontsize=9)
    axL.set_ylabel(r'Lipschitz $L_\phi$ (lower = more stable)')
    axL.set_title('(a) Local Lipschitz constant')
    axL.grid(axis='y')
    axL.legend(loc='upper left')

    # Jaccard (right)
    shap_J = [c[1]['shap']['jaccard']['mean'] for c in cells]
    lime_J = [c[1]['lime']['jaccard']['mean'] for c in cells]
    shap_Jerr = [[c[1]['shap']['jaccard']['mean'] - c[1]['shap']['jaccard']['ci95_low'] for c in cells],
                  [c[1]['shap']['jaccard']['ci95_high'] - c[1]['shap']['jaccard']['mean'] for c in cells]]
    lime_Jerr = [[c[1]['lime']['jaccard']['mean'] - c[1]['lime']['jaccard']['ci95_low'] for c in cells],
                  [c[1]['lime']['jaccard']['ci95_high'] - c[1]['lime']['jaccard']['mean'] for c in cells]]
    axJ.bar(x - w/2, shap_J, w, yerr=shap_Jerr, capsize=4,
             color='#1f77b4', label='SHAP',
             error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    axJ.bar(x + w/2, lime_J, w, yerr=lime_Jerr, capsize=4,
             color='#ff7f0e', label='LIME',
             error_kw={'elinewidth': 1.0, 'ecolor': 'k'})
    axJ.set_xticks(x); axJ.set_xticklabels(names, fontsize=9)
    axJ.set_ylabel('Top-10 Jaccard $J$ (higher = more stable)')
    axJ.set_ylim(0, 1.15)
    axJ.set_title('(b) Top-10 Jaccard agreement')
    axJ.grid(axis='y')
    axJ.legend(loc='upper right')

    plt.tight_layout()
    out = FIG_DIR / 'fig_xai_stability_ieee.png'
    plt.savefig(out)
    plt.close(fig)
    print(f'  written {out}')


if __name__ == '__main__':
    print('IEEE-compliant figure regeneration')
    print('=' * 50)
    fig_cross_dataset()
    fig_dl_variance()
    fig_ml_variance()
    fig_xai_stability()
    print('done.')
