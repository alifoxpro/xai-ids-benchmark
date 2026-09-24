"""
================================================================================
STAGE 3: DEEP LEARNING MODELS BENCHMARKING
================================================================================
9 PyTorch DL models for IDS classification:

Existing (3):
1. CNN-LSTM              - Convolutional + Recurrent hybrid
2. Transformer           - Multi-head self-attention encoder
3. AE-Classifier         - Autoencoder + Classification joint training

New (6):
4. TabNet                - Attention-based tabular learning (Google, 2019)
5. ResNet1D              - 1D Residual CNN with skip connections
6. BiLSTM-Attention      - Bidirectional LSTM with self-attention
7. TCN                   - Temporal Convolutional Network (dilated causal)
8. GRU                   - Gated Recurrent Unit (bidirectional)
9. FT-Transformer        - Feature Tokenizer Transformer (Gorishniy 2021)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import logging
import pickle
import json
from abc import ABC, abstractmethod

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, balanced_accuracy_score,
    matthews_corrcoef, log_loss, cohen_kappa_score
)
from sklearn.preprocessing import StandardScaler

from config import DL_MODELS, RANDOM_SEED, get_results_dir, get_models_dir, TRAINING_CONFIG

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        logger.info(f"PyTorch using GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = None
    warnings.warn("PyTorch not available. Deep learning models will be disabled.")
    # Provide a dummy nn.Module so class definitions don't crash at import time.
    # Actual usage is guarded by TORCH_AVAILABLE checks in each Model's fit().
    import types
    nn = types.SimpleNamespace(Module=object)


# ============================================================================
# BASE MODEL
# ============================================================================

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, name: str, random_state: int = RANDOM_SEED):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.training_time = 0
        self.inference_time = 0
        self.best_params = None

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
           X_val: np.ndarray = None, y_val: np.ndarray = None) -> 'BaseModel':
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Evaluate model performance."""
        start_time = time.time()
        y_pred = self.predict(X)
        self.inference_time = (time.time() - start_time) / len(X) * 1000

        cm = confusion_matrix(y, y_pred)
        n_classes = cm.shape[0]

        # Per-class recall for G-Mean
        per_class_recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)

        # Per-class FPR, FNR, Specificity (TNR), MCC
        fpr_per_class = []
        fnr_per_class = []
        specificity_per_class = []
        mcc_per_class = []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            fpr_per_class.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
            fnr_per_class.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
            specificity_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mcc_per_class.append(float((tp*tn - fp*fn) / denom) if denom > 0 else 0.0)

        # Hamming Loss
        hamming = float(np.mean(y_pred != y))

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'mcc': matthews_corrcoef(y, y_pred),
            'cohens_kappa': cohen_kappa_score(y, y_pred),
            'hamming_loss': hamming,
            'g_mean': float(np.sqrt(np.prod(per_class_recall.clip(min=1e-10)))),
            'far': float(np.mean(fpr_per_class)),
            'fpr_per_class': [float(f) for f in fpr_per_class],
            'fnr_per_class': [float(f) for f in fnr_per_class],
            'specificity_per_class': [float(s) for s in specificity_per_class],
            'mcc_per_class': [float(m) for m in mcc_per_class],
            'confusion_matrix': cm.tolist(),
            'inference_time_ms': self.inference_time,
            'training_time_s': self.training_time
        }

        # Probability-dependent metrics
        try:
            y_proba = self.predict_proba(X)
            if y_proba is not None:
                n_unique = len(np.unique(y))
                # ROC-AUC (overall)
                if n_unique == 2:
                    metrics['roc_auc'] = roc_auc_score(y, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y, y_proba, multi_class='ovr', average='weighted')
                # Per-class ROC-AUC (OvR)
                try:
                    from sklearn.preprocessing import label_binarize
                    classes = np.arange(n_classes)
                    y_bin = label_binarize(y, classes=classes)
                    if y_bin.shape[1] == 1:
                        y_bin = np.hstack([1 - y_bin, y_bin])
                    roc_auc_per = []
                    for c in range(n_classes):
                        try:
                            roc_auc_per.append(roc_auc_score(y_bin[:, c], y_proba[:, c]))
                        except:
                            roc_auc_per.append(None)
                    metrics['roc_auc_per_class'] = roc_auc_per
                except:
                    pass
                # Log Loss
                try:
                    metrics['log_loss'] = log_loss(y, y_proba)
                except:
                    metrics['log_loss'] = None
        except:
            metrics['roc_auc'] = None

        # Parameter count (for DL models)
        if hasattr(self, 'model') and hasattr(self.model, 'parameters'):
            try:
                metrics['n_parameters'] = sum(p.numel() for p in self.model.parameters())
            except:
                pass

        return metrics

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path: Path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


# ============================================================================
# CHUNKED GPU INFERENCE HELPER
# ============================================================================

PREDICT_CHUNK = 8192  # rows per GPU forward pass — safe for 8 GB VRAM


def _chunked_predict(model_net, X_scaled, return_proba=False):
    """Run inference on a PyTorch nn.Module in chunks to avoid GPU OOM.

    Args:
        model_net: A PyTorch nn.Module (already on DEVICE)
        X_scaled: numpy array (already scaled)
        return_proba: if True return softmax probabilities, else argmax labels

    Returns:
        numpy array of predictions or probabilities
    """
    model_net.eval()
    outputs = []
    with torch.no_grad():
        for i in range(0, len(X_scaled), PREDICT_CHUNK):
            xb = torch.FloatTensor(X_scaled[i:i + PREDICT_CHUNK]).to(DEVICE)
            logits = model_net(xb)
            # AE-Classifier returns (reconstruction, classification) tuple
            if isinstance(logits, tuple):
                logits = logits[-1]
            if return_proba:
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
            else:
                outputs.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs, axis=0) if not return_proba else np.vstack(outputs)


# ============================================================================
# PYTORCH TRAINING HELPER
# ============================================================================

def _train_pytorch_model(net, X_train, y_train, X_val, y_val,
                         epochs=100, batch_size=512, lr=0.001,
                         patience=10, is_ae=False, ae_loss_weight=0.3,
                         class_weights=None):
    """Train a PyTorch model with AMP, gradient clipping, early stopping, and LR scheduling."""

    use_amp = TRAINING_CONFIG.get('use_amp', True) and DEVICE is not None and DEVICE.type == 'cuda'
    clip_norm = TRAINING_CONFIG.get('gradient_clip_norm', 1.0)

    net = net.to(DEVICE)
    net.train()

    # Keep data on CPU, move batches to GPU on-the-fly (saves VRAM for large datasets)
    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)
    train_ds = TensorDataset(X_t, y_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE is not None and DEVICE.type == 'cuda'))

    has_val = X_val is not None and y_val is not None
    if has_val:
        X_v = torch.FloatTensor(X_val)
        y_v = torch.LongTensor(y_val)

    if class_weights is not None:
        w = torch.FloatTensor(class_weights).to(DEVICE)
        criterion_cls = nn.CrossEntropyLoss(weight=w)
        logger.info(f"Using class-weighted loss: {class_weights}")
    else:
        criterion_cls = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss() if is_ae else None
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # Mixed Precision scaler (AMP)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        logger.info("Mixed Precision (AMP) enabled — faster training, lower VRAM")

    best_val_loss = float('inf')
    best_state = None
    wait = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        net.train()
        total_loss, correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # Forward pass with AMP autocast
            with torch.amp.autocast('cuda', enabled=use_amp):
                if is_ae:
                    recon, logits = net(xb)
                    loss = ae_loss_weight * criterion_mse(recon, xb) + criterion_cls(logits, yb)
                else:
                    logits = net(xb)
                    loss = criterion_cls(logits, yb)

            # Backward pass with AMP + gradient clipping
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_norm=clip_norm)
                optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += xb.size(0)

        train_loss = total_loss / total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        if has_val:
            net.eval()
            with torch.no_grad():
                # Chunked validation to avoid GPU OOM on large val sets
                val_chunk = 8192
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0
                for vi in range(0, X_v.size(0), val_chunk):
                    xvb = X_v[vi:vi+val_chunk].to(DEVICE, non_blocking=True)
                    yvb = y_v[vi:vi+val_chunk].to(DEVICE, non_blocking=True)
                    with torch.amp.autocast('cuda', enabled=use_amp):
                        if is_ae:
                            recon_vb, logits_vb = net(xvb)
                            vl = (ae_loss_weight * criterion_mse(recon_vb, xvb) + criterion_cls(logits_vb, yvb)).item()
                        else:
                            logits_vb = net(xvb)
                            vl = criterion_cls(logits_vb, yvb).item()
                    val_loss_sum += vl * xvb.size(0)
                    val_correct += (logits_vb.argmax(dim=1) == yvb).sum().item()
                    val_total += xvb.size(0)
                val_loss = val_loss_sum / val_total
                val_acc = val_correct / val_total

            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}"
            if has_val:
                msg += f" - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
            logger.info(msg)

    if best_state is not None:
        net.load_state_dict(best_state)

    net.eval()
    return net, history


# ============================================================================
# EXISTING DL MODEL 1: CNN-LSTM
# ============================================================================

class _CNNLSTMNet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(128, 128, batch_first=True, num_layers=1)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, num_layers=1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x[:, -1, :])
        return self.classifier(x)


class CNNLSTMModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("CNN_LSTM", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('CNN_LSTM', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training CNN-LSTM...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _CNNLSTMNet(X_train.shape[1], n_classes)
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"CNN-LSTM trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# EXISTING DL MODEL 2: TRANSFORMER
# ============================================================================

class _TransformerNet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.embed = nn.Linear(input_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=8, dim_feedforward=64,
            dropout=0.1, batch_first=True, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), 16, 16)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


class TransformerModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("Transformer", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('Transformer', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training Transformer...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _TransformerNet(X_train.shape[1], n_classes)
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            lr=0.0001, class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"Transformer trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# EXISTING DL MODEL 3: AUTOENCODER-CLASSIFIER
# ============================================================================

class _AEClassifierNet(nn.Module):
    def __init__(self, input_dim: int, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, input_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_classes),
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        classification = self.classifier(latent)
        return reconstruction, classification


class AutoencoderClassifierModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("AE_Classifier", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('AutoencoderClassifier', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training Autoencoder-Classifier...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _AEClassifierNet(X_train.shape[1], n_classes)
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100), batch_size=512,
            lr=0.001, is_ae=True, ae_loss_weight=0.3,
            class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"AE-Classifier trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# NEW DL MODEL 4: TabNet
# ============================================================================

def _sparsemax(z, dim=-1):
    """Sparsemax activation (Martins & Astudillo, 2016)."""
    sorted_z, _ = torch.sort(z, descending=True, dim=dim)
    cumsum = torch.cumsum(sorted_z, dim=dim)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    support = (sorted_z - (cumsum - 1) / k) > 0
    k_z = support.sum(dim=dim, keepdim=True).float()
    tau = (cumsum.gather(dim, (k_z - 1).long().clamp(min=0)) - 1) / k_z
    return torch.clamp(z - tau, min=0)


class _GLUBlock(nn.Module):
    def __init__(self, in_dim, out_dim, virtual_batch_size=None):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.bn = nn.BatchNorm1d(out_dim * 2)

    def forward(self, x):
        x = self.bn(self.fc(x))
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * torch.sigmoid(x2)


class _TabNetNet(nn.Module):
    def __init__(self, input_dim, n_classes, n_steps=3, feature_dim=64,
                 output_dim=64, relaxation=1.5, gamma=1.0):
        super().__init__()
        self.n_steps = n_steps
        self.relaxation = relaxation

        self.initial_bn = nn.BatchNorm1d(input_dim)

        # Shared and step-specific layers
        self.shared_fc = nn.Linear(input_dim, feature_dim)
        self.shared_bn = nn.BatchNorm1d(feature_dim)

        self.step_attentive = nn.ModuleList()
        self.step_transform = nn.ModuleList()

        for _ in range(n_steps):
            self.step_attentive.append(nn.Sequential(
                nn.Linear(feature_dim, input_dim),
                nn.BatchNorm1d(input_dim),
            ))
            self.step_transform.append(_GLUBlock(input_dim, feature_dim))

        self.final_fc = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, n_classes),
        )

    def forward(self, x):
        x = self.initial_bn(x)
        prior_scales = torch.ones(x.size(0), x.size(1), device=x.device)
        aggregated = torch.zeros(x.size(0), self.shared_fc.out_features, device=x.device)

        h = torch.relu(self.shared_bn(self.shared_fc(x)))

        for step in range(self.n_steps):
            # Attentive transform
            attn_logits = self.step_attentive[step](h)
            attn = _sparsemax(attn_logits * prior_scales)
            prior_scales = prior_scales * (self.relaxation - attn)

            # Masked input
            masked = attn * x

            # Step transform
            h_step = self.step_transform[step](masked)
            aggregated = aggregated + torch.relu(h_step)
            h = h_step

        return self.final_fc(aggregated)


class TabNetModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("TabNet", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('TabNet', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training TabNet...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _TabNetNet(
            X_train.shape[1], n_classes,
            n_steps=self.config.get('n_steps', 3),
            feature_dim=self.config.get('feature_dim', 64),
        )
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            lr=0.002, patience=15, class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"TabNet trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# NEW DL MODEL 5: ResNet1D
# ============================================================================

class _ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class _ResNet1DNet(nn.Module):
    def __init__(self, input_dim, n_classes, channels=128, n_blocks=3):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
        )
        self.blocks = nn.Sequential(*[_ResBlock1D(channels) for _ in range(n_blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)                  # (B, channels)
        x = x.unsqueeze(2)                 # (B, channels, 1)
        x = x.expand(-1, -1, 8)            # (B, channels, 8) fake sequence
        x = self.blocks(x)                 # (B, channels, 8)
        x = self.pool(x).squeeze(2)        # (B, channels)
        return self.classifier(x)


class ResNet1DModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("ResNet1D", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('ResNet1D', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training ResNet1D...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _ResNet1DNet(
            X_train.shape[1], n_classes,
            channels=128, n_blocks=self.config.get('n_blocks', 3)
        )
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"ResNet1D trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# NEW DL MODEL 6: BiLSTM-Attention
# ============================================================================

class _SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size ** 0.5

    def forward(self, x):
        # x: (B, seq, hidden)
        q = self.query(x)
        k = self.key(x)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, x)
        return out


class _BiLSTMAttentionNet(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_size=128, n_layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=n_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.attention = _SelfAttention(hidden_size * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)                      # (B, hidden)
        x = x.unsqueeze(1).expand(-1, 8, -1)   # (B, 8, hidden)
        x, _ = self.lstm(x)                     # (B, 8, hidden*2)
        x = self.attention(x)                   # (B, 8, hidden*2)
        x = x.mean(dim=1)                       # (B, hidden*2)
        return self.classifier(x)


class BiLSTMAttentionModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("BiLSTM_Attention", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('BiLSTM_Attention', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training BiLSTM-Attention...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _BiLSTMAttentionNet(
            X_train.shape[1], n_classes,
            hidden_size=self.config.get('hidden_size', 128),
            n_layers=self.config.get('n_layers', 2)
        )
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"BiLSTM-Attention trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# NEW DL MODEL 7: TCN (Temporal Convolutional Network)
# ============================================================================

class _TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      padding=padding, dilation=dilation)
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.padding = padding

    def forward(self, x):
        res = x
        out = self.dropout(self.relu(self.conv1(x)))
        out = out[:, :, :x.size(2)]  # causal trim
        out = self.dropout(self.relu(self.conv2(out)))
        out = out[:, :, :x.size(2)]  # causal trim
        if self.downsample is not None:
            res = self.downsample(res)
        return self.relu(out + res)


class _TCNNet(nn.Module):
    def __init__(self, input_dim, n_classes, channels=None, kernel_size=3):
        super().__init__()
        if channels is None:
            channels = [64, 128, 128]
        self.embed = nn.Sequential(
            nn.Linear(input_dim, channels[0]),
            nn.ReLU(),
        )
        layers = []
        for i in range(len(channels)):
            in_ch = channels[i - 1] if i > 0 else channels[0]
            out_ch = channels[i]
            layers.append(_TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i))
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(channels[-1], 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)                  # (B, C0)
        x = x.unsqueeze(2).expand(-1, -1, 16)  # (B, C0, 16)
        x = self.tcn(x)                    # (B, C_last, 16)
        x = self.pool(x).squeeze(2)        # (B, C_last)
        return self.classifier(x)


class TCNModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("TCN", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('TCN', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training TCN...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        channels = self.config.get('channels', [64, 128, 128])
        net = _TCNNet(
            X_train.shape[1], n_classes,
            channels=channels,
            kernel_size=self.config.get('kernel_size', 3)
        )
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"TCN trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# NEW DL MODEL 8: GRU (Gated Recurrent Unit)
# ============================================================================

class _GRUNet(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_size=128, n_layers=2):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
        )
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=n_layers,
            batch_first=True, bidirectional=True, dropout=0.3
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.embed(x)                      # (B, hidden)
        x = x.unsqueeze(1).expand(-1, 8, -1)   # (B, 8, hidden)
        x, _ = self.gru(x)                     # (B, 8, hidden*2)
        x = x[:, -1, :]                        # last step: (B, hidden*2)
        return self.classifier(x)


class GRUModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("GRU", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('GRU', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training GRU...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _GRUNet(
            X_train.shape[1], n_classes,
            hidden_size=self.config.get('hidden_size', 128),
            n_layers=self.config.get('n_layers', 2)
        )
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"GRU trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# NEW DL MODEL 9: FT-Transformer (Feature Tokenizer Transformer)
# ============================================================================

class _FTTransformerNet(nn.Module):
    """Feature Tokenizer + Transformer (Gorishniy et al., 2021)."""

    def __init__(self, input_dim, n_classes, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Per-feature linear embeddings: each feature gets its own projection
        self.feature_embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(input_dim)
        ])

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.layer_norm = nn.LayerNorm(d_model)

        # Classification head from [CLS]
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x):
        B = x.size(0)
        # Tokenize each feature
        tokens = []
        for i, embed in enumerate(self.feature_embeddings):
            tokens.append(embed(x[:, i:i+1]))  # (B, d_model)
        tokens = torch.stack(tokens, dim=1)     # (B, n_features, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls, tokens], dim=1)     # (B, n_features+1, d_model)

        x = self.transformer(x)
        x = self.layer_norm(x)

        # Use [CLS] token output
        cls_out = x[:, 0, :]  # (B, d_model)
        return self.classifier(cls_out)


class FTTransformerModel(BaseModel):
    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("FT_Transformer", random_state)
        self.scaler = StandardScaler()
        self.config = DL_MODELS.get('FT_Transformer', {})
        self.history = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            tune_hyperparams=False, class_weights=None):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available")
        logger.info("Training FT-Transformer...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s = self.scaler.transform(X_val) if X_val is not None else None
        n_classes = len(np.unique(y_train))
        net = _FTTransformerNet(
            X_train.shape[1], n_classes,
            d_model=self.config.get('d_model', 64),
            n_heads=self.config.get('n_heads', 4),
            n_layers=self.config.get('n_layers', 3),
        )
        self.model, self.history = _train_pytorch_model(
            net, X_train_s, y_train, X_val_s, y_val,
            epochs=self.config.get('epochs', 100),
            batch_size=self.config.get('batch_size', 512),
            lr=0.0001, patience=15, class_weights=class_weights
        )
        self.training_time = time.time() - start_time
        logger.info(f"FT-Transformer trained in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=False)

    def predict_proba(self, X):
        X_s = self.scaler.transform(X)
        return _chunked_predict(self.model, X_s, return_proba=True)


# ============================================================================
# VOTING ENSEMBLE MODEL
# ============================================================================

class VotingEnsembleModel:
    """
    Soft-Voting Ensemble: averages predicted probabilities of the top-K
    trained models and returns the class with the highest mean probability.
    """

    def __init__(self, model_wrappers, model_names, weights=None):
        """
        Args:
            model_wrappers: list of trained BaseModel subclass instances
            model_names:    list of model name strings
            weights:        optional list of floats (default: equal weighting)
        """
        self.model_wrappers = model_wrappers
        self.model_names = model_names
        self.weights = weights or [1.0 / len(model_wrappers)] * len(model_wrappers)
        self.training_time = sum(m.training_time for m in model_wrappers)
        self.inference_time = 0.0
        self.history = None   # ensemble has no training history

    def predict(self, X):
        """Predict by averaging probabilities."""
        avg_proba = self.predict_proba(X)
        return np.argmax(avg_proba, axis=1)

    def predict_proba(self, X):
        """Return weighted-average class probabilities."""
        probas = []
        for wrapper, w in zip(self.model_wrappers, self.weights):
            p = wrapper.predict_proba(X)
            probas.append(p * w)
        return np.sum(probas, axis=0)

    def evaluate(self, X, y):
        """Evaluate ensemble using same metrics as BaseModel."""
        start = time.time()
        y_pred = self.predict(X)
        self.inference_time = (time.time() - start) * 1000 / max(len(y), 1)

        cm = confusion_matrix(y, y_pred)
        per_class_recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
        fpr_per_class = []
        for i in range(cm.shape[0]):
            fp = cm[:, i].sum() - cm[i, i]
            tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
            fpr_per_class.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)

        return {
            'accuracy': accuracy_score(y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
            'g_mean': float(np.sqrt(np.prod(per_class_recall.clip(min=1e-10)))),
            'far': float(np.mean(fpr_per_class)),
            'fpr_per_class': [float(f) for f in fpr_per_class],
            'confusion_matrix': cm.tolist(),
            'inference_time_ms': self.inference_time,
            'training_time_s': self.training_time
        }


# ============================================================================
# MEMORY-EFFICIENT PRE-SCALING
# ============================================================================

class _PreScaledPassthrough:
    """Drop-in replacement for StandardScaler when data is already scaled.

    Returned by Stage3Pipeline to avoid 3.59 GB copy per model.
    Stores the real scaler so predict/predict_proba work on new data.
    """

    def __init__(self, real_scaler):
        self._real_scaler = real_scaler
        # Copy attributes so anything that inspects the scaler works
        self.mean_ = real_scaler.mean_
        self.scale_ = real_scaler.scale_
        self.var_ = real_scaler.var_
        self.n_features_in_ = real_scaler.n_features_in_

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return X  # Already scaled — no copy needed

    def transform(self, X):
        return X  # Already scaled — no copy needed

    def inverse_transform(self, X):
        return self._real_scaler.inverse_transform(X)


def _inplace_scale(X, mean, scale):
    """Scale array in-place, column by column — zero extra memory."""
    for j in range(X.shape[1]):
        X[:, j] -= mean[j]
        X[:, j] /= scale[j]


def _inplace_unscale(X, mean, scale):
    """Reverse in-place scaling, column by column."""
    for j in range(X.shape[1]):
        X[:, j] *= scale[j]
        X[:, j] += mean[j]


# ============================================================================
# MODEL BENCHMARKING PIPELINE
# ============================================================================

class Stage3Pipeline:
    """Stage 3: DL Models Benchmarking Pipeline."""

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def get_all_models(self) -> Dict[str, BaseModel]:
        """Get all 9 DL model instances."""
        return {
            'CNN_LSTM': CNNLSTMModel(self.random_state),
            'Transformer': TransformerModel(self.random_state),
            'AE_Classifier': AutoencoderClassifierModel(self.random_state),
            'TabNet': TabNetModel(self.random_state),
            'ResNet1D': ResNet1DModel(self.random_state),
            'BiLSTM_Attention': BiLSTMAttentionModel(self.random_state),
            'TCN': TCNModel(self.random_state),
            'GRU': GRUModel(self.random_state),
            'FT_Transformer': FTTransformerModel(self.random_state),
        }

    def run(self, X_train, y_train, X_val, y_val, X_test, y_test,
            models_to_run=None, tune_hyperparams=True,
            mode="multiclass", class_weights=None) -> Dict:
        import gc
        logger.info("=" * 60)
        logger.info("STAGE 3: DEEP LEARNING MODELS BENCHMARKING")
        logger.info("=" * 60)

        all_models = self.get_all_models()
        if models_to_run is not None:
            all_models = {k: v for k, v in all_models.items() if k in models_to_run}

        # ----- Memory optimization: pre-scale data ONCE in-place -----
        # Each model's fit_transform would allocate a 3.59 GB copy.
        # StandardScaler.fit() ALSO allocates a full copy (NaN mask).
        # Instead, compute mean/std column-by-column (zero extra memory).
        logger.info("Pre-scaling data in-place (saves ~3.6 GB per model)...")
        n_feat = X_train.shape[1]
        _mean = np.empty(n_feat, dtype=np.float64)
        _scale = np.empty(n_feat, dtype=np.float64)
        for j in range(n_feat):
            _mean[j] = float(X_train[:, j].mean())
            _scale[j] = float(X_train[:, j].std())
            if _scale[j] == 0:
                _scale[j] = 1.0
        # Build a StandardScaler-compatible object without ever copying X_train
        shared_scaler = StandardScaler()
        shared_scaler.mean_ = _mean.astype(np.float32)
        shared_scaler.scale_ = _scale.astype(np.float32)
        shared_scaler.var_ = (_scale ** 2).astype(np.float32)
        shared_scaler.n_features_in_ = n_feat
        shared_scaler.n_samples_seen_ = np.full(n_feat, X_train.shape[0], dtype=np.int64)
        _inplace_scale(X_train, shared_scaler.mean_, shared_scaler.scale_)
        _inplace_scale(X_val, shared_scaler.mean_, shared_scaler.scale_)
        _inplace_scale(X_test, shared_scaler.mean_, shared_scaler.scale_)
        logger.info("Pre-scaling complete.")

        passthrough = _PreScaledPassthrough(shared_scaler)
        for model in all_models.values():
            if hasattr(model, 'scaler'):
                model.scaler = passthrough

        results = {'models': {}, 'comparison': [], 'best_model': None}
        best_accuracy = 0

        for name, model in all_models.items():
            logger.info(f"\n--- Training {name} ---")
            try:
                model.fit(X_train, y_train, X_val, y_val, tune_hyperparams=tune_hyperparams,
                         class_weights=class_weights)
                val_metrics = model.evaluate(X_val, y_val)
                test_metrics = model.evaluate(X_test, y_test)

                results['models'][name] = {
                    'validation': val_metrics,
                    'test': test_metrics,
                    'best_params': model.best_params,
                    'training_time': model.training_time,
                    'model_object': model,  # Store the full wrapper (has predict/predict_proba)
                    'training_history': model.history if hasattr(model, 'history') else None
                }
                self.models[name] = model

                if test_metrics['accuracy'] > best_accuracy:
                    best_accuracy = test_metrics['accuracy']
                    results['best_model'] = name

                logger.info(f"{name} - Test Accuracy: {test_metrics['accuracy']:.4f}")
            except Exception as e:
                logger.error(f"Error training {name}: {str(e)}")
                results['models'][name] = {'error': str(e)}
            finally:
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # --- Voting Ensemble from top-3 models ---
        trained_models = [(n, self.models[n]) for n in self.models
                          if n in results['models'] and 'error' not in results['models'][n]]
        if len(trained_models) >= 3:
            trained_models.sort(
                key=lambda x: results['models'][x[0]]['test']['accuracy'], reverse=True)
            top3_names = [t[0] for t in trained_models[:3]]
            top3_wrappers = [t[1] for t in trained_models[:3]]
            logger.info(f"\n--- Creating Voting Ensemble from top 3: {top3_names} ---")
            try:
                ensemble = VotingEnsembleModel(top3_wrappers, top3_names)
                ens_test = ensemble.evaluate(X_test, y_test)
                results['models']['VotingEnsemble'] = {
                    'validation': ensemble.evaluate(X_val, y_val),
                    'test': ens_test,
                    'best_params': {'members': top3_names},
                    'training_time': ensemble.training_time,
                    'model_object': ensemble,
                    'training_history': None
                }
                if ens_test['accuracy'] > best_accuracy:
                    best_accuracy = ens_test['accuracy']
                    results['best_model'] = 'VotingEnsemble'
                logger.info(f"VotingEnsemble - Test Accuracy: {ens_test['accuracy']:.4f}")
            except Exception as e:
                logger.warning(f"Ensemble creation failed: {e}")

        # Restore real scaler to models (for saving — so predict works on new data)
        for model in self.models.values():
            if hasattr(model, 'scaler'):
                model.scaler = shared_scaler

        # Inverse-scale data back to original (so other stages aren't affected)
        _inplace_unscale(X_train, shared_scaler.mean_, shared_scaler.scale_)
        _inplace_unscale(X_val, shared_scaler.mean_, shared_scaler.scale_)
        _inplace_unscale(X_test, shared_scaler.mean_, shared_scaler.scale_)
        logger.info("Data restored to original scale.")

        results['comparison'] = self._create_comparison_summary(results)
        self._save_results(results, mode=mode)
        self.results = results

        logger.info("=" * 60)
        logger.info(f"BEST MODEL: {results['best_model']} (accuracy={best_accuracy:.4f})")
        logger.info("STAGE 3 COMPLETE")
        logger.info("=" * 60)
        return results

    def _create_comparison_summary(self, results):
        summary = []
        for name, mr in results['models'].items():
            if 'error' in mr:
                continue
            test = mr['test']
            summary.append({
                'Model': name,
                'Accuracy': test['accuracy'],
                'Balanced_Acc': test.get('balanced_accuracy', 'N/A'),
                'F1_Macro': test.get('f1_macro', 'N/A'),
                'G_Mean': test.get('g_mean', 'N/A'),
                'FAR': test.get('far', 'N/A'),
                'Precision': test['precision'],
                'Recall': test['recall'],
                'F1': test['f1'],
                'ROC-AUC': test.get('roc_auc', 'N/A'),
                'Train Time (s)': test['training_time_s'],
                'Inference (ms)': test['inference_time_ms']
            })
        df = pd.DataFrame(summary)
        if len(df) > 0:
            df = df.sort_values('Accuracy', ascending=False)
        return df

    def _save_results(self, results, mode="multiclass"):
        output_dir = get_results_dir(mode) / 'stage3_ml_models'
        output_dir.mkdir(exist_ok=True)
        results['comparison'].to_csv(output_dir / 'model_comparison.csv', index=False)
        for name, mr in results['models'].items():
            if 'error' not in mr:
                serializable = {
                    'validation': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                                  for k, v in mr['validation'].items()},
                    'test': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                            for k, v in mr['test'].items()},
                    'best_params': mr['best_params'],
                    'training_history': mr.get('training_history')
                }
                with open(output_dir / f'{name}_results.json', 'w') as f:
                    json.dump(serializable, f, indent=2, default=str)
        models_dir = get_models_dir(mode) / 'stage3'
        models_dir.mkdir(exist_ok=True)
        for name, model in self.models.items():
            try:
                model.save(models_dir / f'{name}.pkl')
            except:
                pass
        logger.info(f"Results saved to {output_dir}")

    # ------------------------------------------------------------------
    # STAGE 3 CACHE: save / load trained models + metrics to avoid re-training
    # ------------------------------------------------------------------

    def save_cache(self, results: Dict, mode: str = "multiclass"):
        """Persist Stage 3 results (model wrappers + metrics) to disk."""
        cache_dir = get_results_dir(mode) / 'stage3_cache'
        cache_dir.mkdir(exist_ok=True)

        # 1. Save each model wrapper as pickle (includes scaler, best_params, etc.)
        models_dir = cache_dir / 'model_wrappers'
        models_dir.mkdir(exist_ok=True)
        for name, mr in results['models'].items():
            if 'error' in mr:
                continue
            model_obj = mr.get('model_object')
            if model_obj is not None:
                try:
                    with open(models_dir / f'{name}.pkl', 'wb') as f:
                        pickle.dump(model_obj, f)
                except Exception as e:
                    logger.warning(f"Cache: failed to save {name} wrapper: {e}")

        # 2. Save metrics + metadata as JSON
        meta = {
            'best_model': results['best_model'],
            'model_names': [],
        }
        for name, mr in results['models'].items():
            if 'error' in mr:
                meta['model_names'].append({'name': name, 'error': mr['error']})
                continue
            entry = {
                'name': name,
                'validation': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                               for k, v in mr['validation'].items()},
                'test': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                         for k, v in mr['test'].items()},
                'best_params': mr['best_params'],
                'training_time': mr.get('training_time'),
                'training_history': mr.get('training_history'),
            }
            meta['model_names'].append(entry)

        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        # 3. Save comparison DataFrame
        if results.get('comparison') is not None:
            results['comparison'].to_csv(cache_dir / 'comparison.csv', index=False)

        logger.info(f"Stage 3 cache saved → {cache_dir}")

    @staticmethod
    def load_cache(mode: str = "multiclass"):
        """Load Stage 3 results from cache. Returns dict or None."""
        cache_dir = get_results_dir(mode) / 'stage3_cache'
        meta_path = cache_dir / 'metadata.json'
        models_dir = cache_dir / 'model_wrappers'

        if not meta_path.exists():
            return None

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Stage 3 cache metadata unreadable: {e}")
            return None

        results = {
            'models': {},
            'best_model': meta['best_model'],
            'comparison': None,
        }

        for entry in meta['model_names']:
            name = entry['name']
            if 'error' in entry:
                results['models'][name] = {'error': entry['error']}
                continue

            # Load the model wrapper
            model_obj = None
            pkl_path = models_dir / f'{name}.pkl'
            if pkl_path.exists():
                try:
                    with open(pkl_path, 'rb') as f:
                        model_obj = pickle.load(f)
                except Exception as e:
                    logger.warning(f"Cache: failed to load {name} wrapper: {e}")

            results['models'][name] = {
                'validation': entry['validation'],
                'test': entry['test'],
                'best_params': entry.get('best_params'),
                'training_time': entry.get('training_time'),
                'model_object': model_obj,
                'training_history': entry.get('training_history'),
            }

        # Load comparison
        comp_path = cache_dir / 'comparison.csv'
        if comp_path.exists():
            results['comparison'] = pd.read_csv(comp_path)

        logger.info(f"Stage 3 loaded from cache → {cache_dir} "
                     f"({len([m for m in results['models'].values() if 'error' not in m])} models)")
        return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STAGE 3: Deep Learning Models Benchmarking")
    print("="*70 + "\n")
    from stage1_data_preparation import Stage1Pipeline
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data = stage1.run()
    pipeline = Stage3Pipeline()
    results = pipeline.run(
        X_train=data['splits']['X_train'],
        y_train=data['splits']['y_train'],
        X_val=data['splits']['X_val'],
        y_val=data['splits']['y_val'],
        X_test=data['splits']['X_test'],
        y_test=data['splits']['y_test'],
        tune_hyperparams=False
    )
    print("\n" + "-"*60)
    print("MODEL COMPARISON SUMMARY")
    print("-"*60)
    print(results['comparison'].to_string(index=False))
    print(f"\nBest Model: {results['best_model']}")
