# -*- coding: utf-8 -*-
"""
generate_paper_figures.py
=========================
Generate the publication figures for "Beyond In-Distribution Accuracy".
Reads existing result files under results_2026_03_03/ and writes high-DPI PNG+PDF
to paper_figures/.

Figures:
  F1  Accuracy vs macro-F1 vs G-Mean gap (the hook)
  F2  Per-class F1 vs support (the minority cliff)
  F3  Cross-dataset: in-distribution vs CICIDS2017 accuracy
  F4  Adversarial/noise robustness degradation curves (GRU)
  F5  Seed-variance / ResNet1D instability
  F6  XAI fidelity vs explanation time (SHAP vs LIME)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi': 150, 'savefig.dpi': 300, 'font.size': 11,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.axisbelow': True,
    'figure.autolayout': True,
})

ROOT = Path(__file__).resolve().parent.parent
MC = ROOT / 'results_2026_03_03' / 'multiclass'
OUT = ROOT / 'paper_figures'
OUT.mkdir(exist_ok=True)


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] {name}')


# --------------------------------------------------------------- F1
def fig1_accuracy_illusion():
    df = pd.read_csv(MC / 'stage3_ml_models' / 'model_comparison.csv')
    df = df.sort_values('Accuracy', ascending=False)
    x = np.arange(len(df)); w = 0.27
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w, df['Accuracy'], w, label='Accuracy', color='#2c7fb8')
    ax.bar(x, df['F1_Macro'], w, label='Macro-F1', color='#e34a33')
    ax.bar(x + w, df['G_Mean'], w, label='G-Mean', color='#fdae61')
    ax.set_xticks(x); ax.set_xticklabels(df['Model'], rotation=35, ha='right')
    ax.set_ylabel('Score'); ax.set_ylim(0, 1)
    ax.set_title('The accuracy illusion: ~90% accuracy, macro-F1 < 0.5, G-Mean ≈ 0')
    ax.legend()
    ax.annotate('42-pt gap', xy=(0, 0.69), xytext=(1.2, 0.62),
                arrowprops=dict(arrowstyle='->'), fontsize=10)
    save(fig, 'F1_accuracy_illusion')


# --------------------------------------------------------------- F2
def fig2_minority_cliff():
    pc = pd.read_csv(MC / 'stage6_performance' / 'VotingEnsemble_per_class.csv')
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(pc['Support'], pc['F1-Score'], s=90, color='#2c7fb8', zorder=3)
    for _, r in pc.iterrows():
        ax.annotate(r['Class'], (r['Support'], r['F1-Score']),
                    textcoords='offset points', xytext=(6, 4), fontsize=9)
    ax.set_xscale('log')
    ax.set_xlabel('Class support (log scale)'); ax.set_ylabel('Per-class F1')
    ax.set_ylim(0, 1); ax.axhline(0.5, ls='--', color='grey', alpha=0.6)
    ax.set_title('The minority cliff: F1 collapses as class support shrinks')
    save(fig, 'F2_minority_cliff')


# --------------------------------------------------------------- F3
def fig3_cross_dataset():
    cd_path = MC / 'cross_dataset' / 'cross_dataset_real.json'
    comp = pd.read_csv(MC / 'stage3_ml_models' / 'model_comparison.csv')
    indist = dict(zip(comp['Model'], comp['Accuracy']))
    cd = json.load(open(cd_path))['models']
    rows = [(m, indist.get(m, np.nan), v['accuracy'])
            for m, v in cd.items() if 'accuracy' in v and m in indist]
    rows.sort(key=lambda r: r[1], reverse=True)
    names = [r[0] for r in rows]
    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, [r[1] for r in rows], w, label='In-distribution (IoT-DIAD)',
           color='#2c7fb8')
    ax.bar(x + w/2, [r[2] for r in rows], w, label='Cross-dataset (CICIDS2017)',
           color='#e34a33')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=35, ha='right')
    ax.set_ylabel('Accuracy'); ax.set_ylim(0, 1)
    ax.set_title('In-distribution accuracy does not transfer (~90% → 8–32%)')
    ax.legend()
    save(fig, 'F3_cross_dataset')


# --------------------------------------------------------------- F4
def fig4_robustness():
    fig, ax = plt.subplots(figsize=(8, 5))
    for fname, label, marker in [('GRU_fgsm.csv', 'FGSM (ε)', 'o'),
                                  ('GRU_noise.csv', 'Gaussian noise (σ)', 's')]:
        p = MC / 'stage8_robustness' / fname
        if not p.exists():
            continue
        d = pd.read_csv(p)
        levels = [float(s.split('=')[1]) for s in d['parameter']]
        base = d['baseline_accuracy'].iloc[0]
        xs = [0] + levels; ys = [base] + list(d['adversarial_accuracy'])
        ax.plot(xs, ys, marker=marker, label=label, linewidth=2)
    ax.set_xlabel('Perturbation strength'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1); ax.set_title('Robustness degradation (GRU, multi-class)')
    ax.legend()
    save(fig, 'F4_robustness')


# --------------------------------------------------------------- F5
def fig5_seed_variance():
    p = MC / 'q1_variance' / 'dl_3models_seeds.json'
    d = json.load(open(p))
    names = list(d.keys())
    series = [[r['accuracy'] for r in d[m]['per_seed']] for m in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(series, labels=names, showmeans=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#a6bddb')
    for i, s in enumerate(series, 1):
        ax.scatter([i]*len(s), s, color='#e34a33', zorder=3, alpha=0.7)
    ax.set_ylabel('Accuracy across 5 seeds')
    ax.set_title('Seed variance: ResNet1D is unstable (two near-collapses)')
    save(fig, 'F5_seed_variance')


# --------------------------------------------------------------- F6
def fig6_xai():
    p = MC / 'stage5_xai_quality' / 'xai_comparison.csv'
    d = pd.read_csv(p)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    x = np.arange(len(d)); w = 0.35
    ax1.bar(x - w/2, d['Fidelity'], w, color='#2c7fb8', label='Fidelity')
    ax1.set_ylabel('Fidelity', color='#2c7fb8'); ax1.set_ylim(0, 1)
    ax1.set_xticks(x); ax1.set_xticklabels(d['Method'])
    ax2 = ax1.twinx()
    ax2.bar(x + w/2, d['Explanation Time (ms)'], w, color='#e34a33',
            label='Time (ms)')
    ax2.set_ylabel('Explanation time (ms)', color='#e34a33')
    ax2.grid(False)
    ax1.set_title('XAI quality: SHAP fidelity vs LIME speed')
    save(fig, 'F6_xai')


def main():
    for fn in [fig1_accuracy_illusion, fig2_minority_cliff, fig3_cross_dataset,
               fig4_robustness, fig5_seed_variance, fig6_xai]:
        try:
            fn()
        except Exception as exc:
            print(f'[FAIL] {fn.__name__}: {exc}')
    print(f'\nFigures written to {OUT}')


if __name__ == '__main__':
    main()
