# -*- coding: utf-8 -*-
"""
generate_ieee_figures_v2.py
===========================
Comprehensive figure regeneration with:
  * STRICT IEEE styling (600 DPI, Times, STIX, etc.)
  * TWO fixed sizes only:
        IEEE_SINGLE = 3.45 x 2.60 in
        IEEE_DOUBLE = 7.16 x 4.40 in
  * Real backing data for every plot (no empty axes)
  * Academic visual conventions (proper axes, units, legends)
"""

import io, json, os, sys
from pathlib import Path

os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                errors='backslashreplace', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
RES_DIR = PROJECT_ROOT / 'results_2026_03_03'

# IEEE styling
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

IEEE_SINGLE = (3.45, 2.60)
IEEE_DOUBLE = (7.16, 4.40)
PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f'  {name}')


# ====================================================== Fig 3: Feature Importance
def fig_feature_importance():
    """Feature importance ranking (top-15) for SHAP, mRMR, IG, Permutation.

    We have only ranked feature names per method, not scores. Plot the
    rank-position of each top-15 feature across methods to show overlap.
    Academic interpretation: lower rank-position = more important.
    """
    methods = {}
    for m in ['SHAP', 'mRMR', 'IG', 'PERM']:
        p = RES_DIR / 'multiclass' / 'stage2_feature_selection' / f'{m}_features.json'
        if not p.exists(): continue
        with open(p) as f:
            data = json.load(f)
        if not isinstance(data, list): continue
        # Normalise to plain feature names
        norm = []
        for item in data:
            if isinstance(item, str):
                norm.append(item)
            elif isinstance(item, dict):
                # mRMR has {'feature': 'X', 'mrmr_score': ...}
                for key in ('feature', 'name', 'feat'):
                    if key in item:
                        norm.append(item[key])
                        break
        methods[m] = norm[:15]

    if len(methods) < 2: return

    # Union of all top-15 features
    all_feats = []
    for feats in methods.values():
        for fname in feats:
            if fname not in all_feats:
                all_feats.append(fname)
    all_feats = all_feats[:20]  # keep visible

    # Build rank matrix: methods x features
    method_names = list(methods.keys())
    rank_matrix = np.full((len(method_names), len(all_feats)), 16, dtype=float)
    for i, m in enumerate(method_names):
        for fname in methods[m]:
            if fname in all_feats:
                rank_matrix[i, all_feats.index(fname)] = methods[m].index(fname) + 1

    # Use rank-15 as inverted importance: importance = max(15 - rank + 1, 0)
    importance = np.maximum(16 - rank_matrix, 0)

    fig, ax = plt.subplots(figsize=IEEE_DOUBLE)
    # Heatmap: x=features, y=methods, color=importance
    im = ax.imshow(importance, aspect='auto', cmap='YlGnBu', vmin=0, vmax=15)
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names)
    ax.set_xticks(range(len(all_feats)))
    short_names = [s.replace('Length', 'Len').replace('Packet', 'Pkt').replace('Flow', 'Fl')[:14]
                    for s in all_feats]
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    # Annotate cells
    for i in range(importance.shape[0]):
        for j in range(importance.shape[1]):
            v = importance[i, j]
            if v > 0:
                color = 'white' if v > 8 else 'black'
                rank = int(16 - v)
                ax.text(j, i, str(rank), ha='center', va='center',
                         fontsize=7, color=color)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('Importance score (16 = best rank)')
    cb.ax.tick_params(labelsize=8)
    ax.set_title('Top Features Across Selection Methods (rank labels in cells)')
    ax.tick_params(top=False, right=False)
    save(fig, 'fig_feature_importance_ieee.png')


# ====================================================== Fig 12: XAI Quality
def fig_xai_quality():
    """SHAP vs LIME: fidelity (left) and explanation time (right, log)."""
    multi_p = RES_DIR / 'multiclass' / 'stage5_xai_quality' / 'xai_quality_summary.json'
    bin_p = RES_DIR / 'binary' / 'stage5_xai_quality' / 'xai_quality_summary.json'

    rows = []
    for label, p in [('Multi-class', multi_p), ('Binary', bin_p)]:
        if not p.exists(): continue
        with open(p) as f:
            d = json.load(f)
        qm = d.get('quality_metrics', d)
        shap = qm.get('SHAP', {})
        lime = qm.get('LIME', {})
        rows.append((label,
                     shap.get('fidelity'), lime.get('fidelity'),
                     shap.get('explanation_time_ms'),
                     lime.get('explanation_time_ms'),
                     qm.get('cross_method_consistency')))
    if not rows: return

    fig, (axF, axT) = plt.subplots(1, 2, figsize=IEEE_DOUBLE)
    modes = [r[0] for r in rows]
    shap_fid = [r[1] for r in rows]
    lime_fid = [r[2] for r in rows]
    shap_t = [r[3] for r in rows]
    lime_t = [r[4] for r in rows]
    consistency = [r[5] for r in rows]

    x = np.arange(len(modes)); w = 0.32

    # Fidelity bars
    b1 = axF.bar(x - w/2, shap_fid, w, label='SHAP', color='#1f77b4',
                  edgecolor='k', linewidth=0.5)
    b2 = axF.bar(x + w/2, lime_fid, w, label='LIME', color='#ff7f0e',
                  edgecolor='k', linewidth=0.5)
    # Cross-method consistency as gray line
    axF2 = axF.twinx()
    axF2.plot(x, consistency, 'o--', color='#555555', linewidth=1.2,
               markersize=6, label='SHAP-LIME\nconsistency')
    axF2.set_ylim(0, 1.0)
    axF2.set_ylabel('Cross-method consistency', color='#555555')
    axF2.tick_params(axis='y', colors='#555555')
    axF2.spines['right'].set_color('#555555')
    axF2.spines['right'].set_visible(True)
    axF2.spines['top'].set_visible(False)

    for bar in (b1 + b2):
        h = bar.get_height()
        axF.text(bar.get_x() + bar.get_width()/2, h + 0.015,
                  f'{h:.3f}', ha='center', va='bottom', fontsize=8)

    axF.set_xticks(x); axF.set_xticklabels(modes)
    axF.set_ylabel('Top-$k$ fidelity')
    axF.set_ylim(0, 1.05)
    axF.set_title('(a) Fidelity and SHAP-LIME consistency')
    axF.grid(axis='y')
    lines_f1, labs_f1 = axF.get_legend_handles_labels()
    lines_f2, labs_f2 = axF2.get_legend_handles_labels()
    axF.legend(lines_f1 + lines_f2, labs_f1 + labs_f2,
                loc='upper left', fontsize=8)

    # Time (log scale)
    b3 = axT.bar(x - w/2, shap_t, w, label='SHAP', color='#1f77b4',
                  edgecolor='k', linewidth=0.5)
    b4 = axT.bar(x + w/2, lime_t, w, label='LIME', color='#ff7f0e',
                  edgecolor='k', linewidth=0.5)
    for bar in (b3 + b4):
        h = bar.get_height()
        axT.text(bar.get_x() + bar.get_width()/2, h * 1.1,
                  f'{int(h)} ms', ha='center', va='bottom', fontsize=8)
    axT.set_xticks(x); axT.set_xticklabels(modes)
    axT.set_yscale('log')
    axT.set_ylabel('Explanation time per sample (ms, log)')
    axT.set_title('(b) Wall-clock cost of explanation')
    axT.grid(axis='y', which='both')
    axT.legend(loc='upper left')

    plt.tight_layout()
    save(fig, 'fig_xai_quality_ieee.png')


# ====================================================== Fig 16: FGSM Attack
def fig_fgsm_pair():
    """FGSM robustness for multi-class and binary as twin panels."""
    fig, (axM, axB) = plt.subplots(1, 2, figsize=IEEE_DOUBLE, sharey=True)
    for ax, mode, color, label in [
            (axM, 'multiclass', '#d62728', 'Multi-class'),
            (axB, 'binary',     '#1f77b4', 'Binary')]:
        p = RES_DIR / mode / 'stage8_robustness' / 'GRU_fgsm.csv'
        if not p.exists(): continue
        df = pd.read_csv(p)
        eps = df['parameter'].str.extract(r'epsilon=([0-9.]+)').astype(float).iloc[:, 0]
        adv = df['adversarial_accuracy']
        ratio = df['robustness_ratio']
        baseline = df['baseline_accuracy'].iloc[0]

        # Plot both adversarial accuracy and robustness ratio
        ax.plot(eps, adv, marker='o', linewidth=1.6, markersize=6,
                 color=color, label='Adv.\\ accuracy')
        ax.plot(eps, ratio, marker='s', linewidth=1.6, markersize=5,
                 color=color, linestyle='--',
                 markerfacecolor='white', label='Robustness ratio')
        ax.axhline(baseline, color='gray', linestyle=':', linewidth=0.8)
        ax.text(0.5, baseline + 0.02, f'baseline = {baseline:.3f}',
                 fontsize=8, color='gray', ha='center')
        ax.set_xlabel(r'FGSM perturbation $\epsilon$')
        if mode == 'multiclass':
            ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.05, 0.55)
        ax.set_xticks([0.1, 0.3, 0.5])
        ax.set_title(f'({"a" if mode == "multiclass" else "b"}) {label}')
        ax.grid(True)
        ax.legend(loc='lower left', fontsize=8)
    plt.tight_layout()
    save(fig, 'fig_fgsm_comparison_ieee.png')


def fig_fgsm_single(mode, fname):
    """Single-mode FGSM figure (single column)."""
    p = RES_DIR / mode / 'stage8_robustness' / 'GRU_fgsm.csv'
    if not p.exists(): return
    df = pd.read_csv(p)
    eps = df['parameter'].str.extract(r'epsilon=([0-9.]+)').astype(float).iloc[:, 0]
    adv = df['adversarial_accuracy']
    ratio = df['robustness_ratio']
    baseline = df['baseline_accuracy'].iloc[0]

    color = '#d62728' if mode == 'multiclass' else '#1f77b4'
    fig, ax = plt.subplots(figsize=IEEE_SINGLE)
    ax.plot(eps, adv, marker='o', linewidth=1.6, markersize=6,
             color=color, label='Adv.\\ accuracy')
    ax.plot(eps, ratio, marker='s', linewidth=1.6, markersize=5,
             color=color, linestyle='--',
             markerfacecolor='white', label='Robustness ratio')
    ax.axhline(baseline, color='gray', linestyle=':', linewidth=0.8)
    ax.text(0.3, baseline + 0.02, f'baseline = {baseline:.3f}',
             fontsize=7, color='gray', ha='center')
    ax.set_xlabel(r'FGSM $\epsilon$')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.1, 0.3, 0.5])
    ax.set_title(f'FGSM Robustness ({mode})')
    ax.grid(True)
    ax.legend(loc='lower left', fontsize=8)
    save(fig, fname)


if __name__ == '__main__':
    print('Targeted IEEE figure fixes')
    print('=' * 60)
    print('Fig 3 (feature importance):')
    fig_feature_importance()
    print('Fig 12 (XAI quality):')
    fig_xai_quality()
    print('Fig 16 (FGSM, pair panels):')
    fig_fgsm_pair()
    print('FGSM single-mode versions:')
    fig_fgsm_single('multiclass', 'fig_fgsm_multiclass_ieee.png')
    fig_fgsm_single('binary',     'fig_fgsm_binary_ieee.png')
    print('=' * 60)
    print('done.')
