# -*- coding: utf-8 -*-
"""
fusion_experiment.py
====================
Compare EARLY (feature/representation-level) vs LATE (decision-level) fusion of
three strong base models (GRU, ResNet1D, FT-Transformer).

  Late fusion (fusion-after): average the softmax probabilities of the three
      independently-trained models, then argmax. (Decision-level.)
  Early fusion (fusion layer): extract each model's penultimate representation
      (input to its final linear layer), CONCATENATE the three, and train a small
      MLP fusion head on top. (Representation-level.)

Reports accuracy / balanced-accuracy / macro-F1 for both, plus each base model.
Output: results_2026_03_03/multiclass/fusion/fusion_comparison.json
"""
import os, sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
os.environ['PYTHONIOENCODING'] = 'utf-8'

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
MC = ROOT / 'results_2026_03_03' / 'multiclass'
MODELS = ROOT / 'models' / 'multiclass' / 'stage3'
OUT = MC / 'fusion'; OUT.mkdir(parents=True, exist_ok=True)
BASE = ['GRU', 'ResNet1D', 'FT_Transformer']
DEV = 'cuda'


def last_linear(net):
    """Return the final nn.Linear module (the classification layer)."""
    lin = None
    for m in net.modules():
        if isinstance(m, nn.Linear):
            lin = m
    return lin


def forward_collect(net, X, bs=16384):
    """Return (probs [N,C], penultimate embeddings [N,D]) for the whole set."""
    net.eval()
    emb_store = {}
    h = last_linear(net).register_forward_pre_hook(
        lambda mod, inp: emb_store.__setitem__('e', inp[0].detach()))
    probs, embs = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device=DEV)
            out = net(xb)
            if isinstance(out, (tuple, list)):
                out = out[-1]
            probs.append(torch.softmax(out, -1).cpu().numpy())
            embs.append(emb_store['e'].cpu().numpy())
    h.remove()
    return np.concatenate(probs), np.concatenate(embs)


def metrics(y, p):
    return {'accuracy': round(float(accuracy_score(y, p)), 4),
            'balanced_accuracy': round(float(balanced_accuracy_score(y, p)), 4),
            'f1_macro': round(float(f1_score(y, p, average='macro', zero_division=0)), 4)}


def main():
    d = np.load(MC / 'stage1_cache' / 'arrays.npz', allow_pickle=True)
    Xtr, ytr = d['X_train'].astype(np.float32), d['y_train'].astype(int)
    Xte, yte = d['X_test'].astype(np.float32), d['y_test'].astype(int)
    # subsample train for the fusion head (keeps it fast; full test for eval)
    rng = np.random.default_rng(42)
    tr_idx = rng.choice(len(Xtr), min(300_000, len(Xtr)), replace=False)
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]
    n_cls = len(np.unique(ytr))
    print(f'[INFO] train={Xtr.shape} test={Xte.shape} classes={n_cls}')

    probs_tr, probs_te, embs_tr, embs_te, base_results = {}, {}, {}, {}, {}
    for name in BASE:
        net = pickle.load(open(MODELS / f'{name}.pkl', 'rb'))
        net = (net.model if hasattr(net, 'model') else net).to(DEV)
        ptr, etr = forward_collect(net, Xtr)
        pte, ete = forward_collect(net, Xte)
        probs_tr[name], probs_te[name] = ptr, pte
        embs_tr[name], embs_te[name] = etr, ete
        base_results[name] = metrics(yte, pte.argmax(1))
        print(f'  {name}: emb_dim={etr.shape[1]}  base {base_results[name]}')
        del net; torch.cuda.empty_cache()

    # ---------- LATE fusion: average probabilities ----------
    late = np.mean([probs_te[n] for n in BASE], axis=0).argmax(1)
    late_m = metrics(yte, late)
    print(f'\n[LATE fusion]  {late_m}')

    # ---------- EARLY fusion: concat embeddings + MLP head ----------
    Ztr = np.concatenate([embs_tr[n] for n in BASE], axis=1).astype(np.float32)
    Zte = np.concatenate([embs_te[n] for n in BASE], axis=1).astype(np.float32)
    print(f'[INFO] fused embedding dim = {Ztr.shape[1]}')

    head = nn.Sequential(nn.Linear(Ztr.shape[1], 128), nn.ReLU(), nn.Dropout(0.3),
                         nn.Linear(128, n_cls)).to(DEV)
    counts = np.bincount(ytr, minlength=n_cls)
    cw = torch.as_tensor(counts.sum() / (n_cls * np.maximum(counts, 1)),
                         dtype=torch.float32, device=DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss(weight=cw)
    Zt = torch.as_tensor(Ztr, device=DEV); yt = torch.as_tensor(ytr, dtype=torch.long, device=DEV)
    bs = 4096
    for epoch in range(15):
        head.train(); perm = torch.randperm(len(Zt), device=DEV)
        for i in range(0, len(Zt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(head(Zt[idx]), yt[idx])
            loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        ep = []
        for i in range(0, len(Zte), bs):
            ep.append(head(torch.as_tensor(Zte[i:i+bs], device=DEV)).argmax(-1).cpu().numpy())
    early = np.concatenate(ep)
    early_m = metrics(yte, early)
    print(f'[EARLY fusion] {early_m}')

    result = {'base_models': base_results, 'late_fusion': late_m,
              'early_fusion': early_m, 'fused_emb_dim': int(Ztr.shape[1]),
              'base': BASE}
    json.dump(result, open(OUT / 'fusion_comparison.json', 'w'), indent=2)

    print('\n' + '=' * 60)
    print(f"{'Method':<26}{'Acc':>8}{'BalAcc':>9}{'F1mac':>8}")
    for n in BASE:
        m = base_results[n]
        print(f"{'base: '+n:<26}{m['accuracy']:>8.3f}{m['balanced_accuracy']:>9.3f}{m['f1_macro']:>8.3f}")
    print(f"{'LATE fusion (voting)':<26}{late_m['accuracy']:>8.3f}{late_m['balanced_accuracy']:>9.3f}{late_m['f1_macro']:>8.3f}")
    print(f"{'EARLY fusion (concat+MLP)':<26}{early_m['accuracy']:>8.3f}{early_m['balanced_accuracy']:>9.3f}{early_m['f1_macro']:>8.3f}")
    print(f"\n[DONE] {OUT / 'fusion_comparison.json'}")


if __name__ == '__main__':
    main()
