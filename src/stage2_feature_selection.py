"""
================================================================================
STAGE 2: FEATURE SELECTION METHODS BENCHMARKING
================================================================================
This module implements and compares 6 feature selection methods:
1. SHAP-Based Feature Selection (XAI)
2. Principal Component Analysis (PCA)
3. mRMR (Maximum Relevance Minimum Redundancy)
4. Autoencoder Latent Features
5. Permutation Importance
6. Information Gain (Mutual Information)
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler

# PyTorch is optional (for Autoencoder feature selection)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None
    warnings.warn("PyTorch not available. Autoencoder feature selection will be disabled.")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from config import (
    FEATURE_SELECTION, RESULTS_DIR, RANDOM_SEED, VIS_DIR, get_results_dir
)

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SHAPFeatureSelector:
    """
    SHAP-based feature selection using explainable AI.
    Computes SHAP values to rank feature importance.
    """

    def __init__(self, config: Dict = None, random_state: int = RANDOM_SEED):
        self.config = config or FEATURE_SELECTION['SHAP']
        self.random_state = random_state
        self.baseline_model = None
        self.shap_values = None
        self.feature_importance = None
        self.selected_features = None
        self.computation_time = 0

    def fit(self, X: np.ndarray, y: np.ndarray,
           feature_names: List[str] = None) -> 'SHAPFeatureSelector':
        """
        Fit SHAP-based feature selector.

        Args:
            X: Feature array
            y: Target array
            feature_names: Optional list of feature names

        Returns:
            self
        """
        logger.info("Fitting SHAP-based feature selector...")
        start_time = time.time()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        else:
            # Ensure feature_names is a list
            feature_names = list(feature_names)

        # Train baseline Random Forest model
        logger.info("Training baseline Random Forest model...")
        self.baseline_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.baseline_model.fit(X, y)

        if not SHAP_AVAILABLE:
            # Fallback: Use built-in feature importance
            logger.warning("SHAP not available. Using RF feature importance as fallback.")
            importance_values = self.baseline_model.feature_importances_.flatten().tolist()
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
        else:
            # Compute SHAP values
            logger.info("Computing SHAP values...")
            n_samples = min(self.config['n_samples_explain'], len(X))
            sample_idx = np.random.choice(len(X), n_samples, replace=False)
            X_sample = X[sample_idx]

            explainer = shap.TreeExplainer(self.baseline_model)
            self.shap_values = explainer.shap_values(X_sample)

            # Handle multi-class SHAP values - newer SHAP versions return different structures
            if isinstance(self.shap_values, list):
                # Multi-class: list of arrays, one per class
                # Average absolute SHAP values across all classes
                mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in self.shap_values], axis=0)
            elif hasattr(self.shap_values, 'values'):
                # Newer SHAP Explanation object
                shap_array = self.shap_values.values
                if shap_array.ndim == 3:
                    # Shape: (samples, features, classes) - take mean across samples and classes
                    mean_shap = np.abs(shap_array).mean(axis=(0, 2))
                else:
                    mean_shap = np.abs(shap_array).mean(axis=0)
            else:
                # Standard 2D or 3D numpy array
                shap_arr = np.abs(self.shap_values)
                if shap_arr.ndim == 3:
                    # Shape: (samples, features, classes) - average over samples and classes
                    mean_shap = shap_arr.mean(axis=(0, 2))
                else:
                    mean_shap = shap_arr.mean(axis=0)

            # Ensure mean_shap is 1D and matches feature count
            mean_shap = np.asarray(mean_shap).flatten()

            # Verify lengths match
            if len(mean_shap) != len(feature_names):
                logger.warning(f"SHAP values length ({len(mean_shap)}) != feature count ({len(feature_names)}). Using RF importance.")
                importance_values = self.baseline_model.feature_importances_.flatten().tolist()
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_values
                }).sort_values('importance', ascending=False)
            else:
                self.feature_importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': mean_shap.tolist()
                }).sort_values('importance', ascending=False)

        self.computation_time = time.time() - start_time
        logger.info(f"SHAP feature selection completed in {self.computation_time:.2f}s")

        return self

    def select_top_k(self, k: int) -> Tuple[List[str], np.ndarray]:
        """
        Select top-k features based on SHAP importance.

        Args:
            k: Number of features to select

        Returns:
            Tuple of (selected feature names, feature indices)
        """
        if self.feature_importance is None:
            raise ValueError("Must call fit() before select_top_k()")

        top_k = self.feature_importance.head(k)
        self.selected_features = top_k['feature'].tolist()
        indices = [self.feature_importance[self.feature_importance['feature'] == f].index[0]
                  for f in self.selected_features]

        return self.selected_features, np.array(indices)

    def transform(self, X: np.ndarray, k: int = None) -> np.ndarray:
        """
        Transform data to selected features.

        Args:
            X: Feature array
            k: Number of features (uses previously selected if None)

        Returns:
            Transformed array with selected features
        """
        if k is not None:
            self.select_top_k(k)

        if self.selected_features is None:
            raise ValueError("Must call select_top_k() first")

        # Get indices of selected features
        all_features = self.feature_importance['feature'].tolist()
        indices = [all_features.index(f) for f in self.selected_features]

        return X[:, indices]

    def get_stability_score(self, X: np.ndarray, y: np.ndarray,
                           k: int, n_bootstrap: int = 10) -> float:
        """
        Compute stability score using bootstrap sampling.

        Args:
            X: Feature array
            y: Target array
            k: Number of features to select
            n_bootstrap: Number of bootstrap samples

        Returns:
            Stability score (0-1)
        """
        logger.info("Computing SHAP feature stability...")

        selected_sets = []
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        for i in range(n_bootstrap):
            # Bootstrap sample
            idx = np.random.choice(len(X), len(X), replace=True)
            X_boot, y_boot = X[idx], y[idx]

            # Fit and select
            selector = SHAPFeatureSelector(self.config, self.random_state + i)
            selector.fit(X_boot, y_boot, feature_names)
            features, _ = selector.select_top_k(k)
            selected_sets.append(set(features))

        # Compute Jaccard similarity between all pairs
        similarities = []
        for i in range(len(selected_sets)):
            for j in range(i + 1, len(selected_sets)):
                intersection = len(selected_sets[i] & selected_sets[j])
                union = len(selected_sets[i] | selected_sets[j])
                similarities.append(intersection / union if union > 0 else 0)

        return np.mean(similarities)


class PCAFeatureSelector:
    """
    PCA-based dimensionality reduction.
    Extracts principal components that explain most variance.
    """

    def __init__(self, config: Dict = None, random_state: int = RANDOM_SEED):
        self.config = config or FEATURE_SELECTION['PCA']
        self.random_state = random_state
        self.pca = None
        self.n_components = None
        self.explained_variance = None
        self.computation_time = 0

    def fit(self, X: np.ndarray, n_components: int = None,
           variance_threshold: float = None) -> 'PCAFeatureSelector':
        """
        Fit PCA model.

        Args:
            X: Feature array
            n_components: Number of components to keep
            variance_threshold: Variance threshold to determine n_components

        Returns:
            self
        """
        logger.info("Fitting PCA feature selector...")
        start_time = time.time()

        # Standardize data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if variance_threshold is not None:
            # Fit full PCA first
            pca_full = PCA(random_state=self.random_state)
            pca_full.fit(X_scaled)

            # Find number of components for threshold
            cumsum = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= variance_threshold) + 1
            logger.info(f"Components for {variance_threshold*100}% variance: {n_components}")

        if n_components is None:
            n_components = min(X.shape[1], 30)

        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        self.pca.fit(X_scaled)

        self.explained_variance = {
            'individual': self.pca.explained_variance_ratio_.tolist(),
            'cumulative': np.cumsum(self.pca.explained_variance_ratio_).tolist()
        }

        self.computation_time = time.time() - start_time
        logger.info(f"PCA completed in {self.computation_time:.2f}s")
        logger.info(f"Total variance explained: {self.explained_variance['cumulative'][-1]*100:.2f}%")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to principal components.

        Args:
            X: Feature array

        Returns:
            Transformed array
        """
        if self.pca is None:
            raise ValueError("Must call fit() first")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return self.pca.transform(X_scaled)

    def fit_transform(self, X: np.ndarray, n_components: int = None,
                     variance_threshold: float = None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, n_components, variance_threshold)
        return self.transform(X)

    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error.

        Args:
            X: Original feature array

        Returns:
            Mean squared reconstruction error
        """
        if self.pca is None:
            raise ValueError("Must call fit() first")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_transformed = self.pca.transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_transformed)

        mse = np.mean((X_scaled - X_reconstructed) ** 2)
        return mse


class mRMRFeatureSelector:
    """
    Maximum Relevance Minimum Redundancy feature selection.
    Balances feature relevance to target with inter-feature redundancy.
    """

    def __init__(self, config: Dict = None, random_state: int = RANDOM_SEED):
        self.config = config or FEATURE_SELECTION['mRMR']
        self.random_state = random_state
        self.selected_features = None
        self.feature_scores = None
        self.computation_time = 0

    def fit(self, X: np.ndarray, y: np.ndarray,
           feature_names: List[str] = None) -> 'mRMRFeatureSelector':
        """
        Fit mRMR feature selector.

        Args:
            X: Feature array
            y: Target array
            feature_names: Optional list of feature names

        Returns:
            self
        """
        logger.info("Fitting mRMR feature selector...")
        start_time = time.time()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        n_features = X.shape[1]

        # Compute mutual information between each feature and target (relevance)
        logger.info("Computing feature relevance (MI with target)...")
        relevance = mutual_info_classif(X, y, random_state=self.random_state)

        # Compute mutual information between features (redundancy)
        # Subsample to 20K rows for speed — MI doesn't need huge samples
        MAX_REDUNDANCY_SAMPLES = 20_000
        if X.shape[0] > MAX_REDUNDANCY_SAMPLES:
            rng = np.random.RandomState(self.random_state)
            idx_sub = rng.choice(X.shape[0], MAX_REDUNDANCY_SAMPLES, replace=False)
            X_red = X[idx_sub]
        else:
            X_red = X
        logger.info(f"Computing feature redundancy (MI between features) on {X_red.shape[0]} samples...")

        # For efficiency, discretize features
        X_discrete = np.zeros_like(X_red, dtype=int)
        for i in range(n_features):
            X_discrete[:, i] = pd.qcut(X_red[:, i], q=10, labels=False, duplicates='drop')

        # Compute redundancy matrix (simplified version)
        redundancy_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = mutual_info_classif(
                    X_discrete[:, i:i+1], X_discrete[:, j],
                    random_state=self.random_state
                )[0]
                redundancy_matrix[i, j] = mi
                redundancy_matrix[j, i] = mi

        # mRMR selection
        self.feature_scores = []
        selected_indices = []
        remaining_indices = list(range(n_features))

        # Select first feature (highest relevance)
        first_idx = np.argmax(relevance)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        self.feature_scores.append({
            'feature': feature_names[first_idx],
            'relevance': relevance[first_idx],
            'redundancy': 0,
            'mrmr_score': relevance[first_idx]
        })

        # Select remaining features
        max_features = min(50, n_features)
        for _ in range(max_features - 1):
            if not remaining_indices:
                break

            best_score = -np.inf
            best_idx = None

            for idx in remaining_indices:
                # Relevance
                rel = relevance[idx]

                # Average redundancy with selected features
                red = np.mean([redundancy_matrix[idx, s] for s in selected_indices])

                # mRMR score
                score = rel - red

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)

                self.feature_scores.append({
                    'feature': feature_names[best_idx],
                    'relevance': relevance[best_idx],
                    'redundancy': np.mean([redundancy_matrix[best_idx, s]
                                          for s in selected_indices[:-1]]),
                    'mrmr_score': best_score
                })

        self.computation_time = time.time() - start_time
        logger.info(f"mRMR completed in {self.computation_time:.2f}s")

        return self

    def select_top_k(self, k: int) -> Tuple[List[str], List[float]]:
        """
        Select top-k features.

        Args:
            k: Number of features to select

        Returns:
            Tuple of (feature names, mRMR scores)
        """
        if self.feature_scores is None:
            raise ValueError("Must call fit() first")

        k = min(k, len(self.feature_scores))
        top_k = self.feature_scores[:k]

        self.selected_features = [f['feature'] for f in top_k]
        scores = [f['mrmr_score'] for f in top_k]

        return self.selected_features, scores

    def get_feature_importance_df(self) -> pd.DataFrame:
        """Get feature importance as DataFrame."""
        return pd.DataFrame(self.feature_scores)


if TORCH_AVAILABLE:
    class _AENet(nn.Module):
        """PyTorch Autoencoder for feature extraction."""

        def __init__(self, input_dim: int, latent_dim: int, architecture: list):
            super().__init__()
            # Encoder
            enc_layers = []
            prev_dim = input_dim
            for units in architecture[1:]:
                if units > latent_dim:
                    enc_layers.extend([nn.Linear(prev_dim, units), nn.ReLU(), nn.BatchNorm1d(units), nn.Dropout(0.2)])
                    prev_dim = units
            enc_layers.extend([nn.Linear(prev_dim, latent_dim), nn.ReLU()])
            self.encoder = nn.Sequential(*enc_layers)

            # Decoder
            dec_layers = []
            prev_dim = latent_dim
            for units in reversed(architecture[1:]):
                if units > latent_dim:
                    dec_layers.extend([nn.Linear(prev_dim, units), nn.ReLU(), nn.BatchNorm1d(units), nn.Dropout(0.2)])
                    prev_dim = units
            dec_layers.append(nn.Linear(prev_dim, input_dim))
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x):
            latent = self.encoder(x)
            reconstruction = self.decoder(latent)
            return reconstruction

        def encode(self, x):
            return self.encoder(x)


class AutoencoderFeatureSelector:
    """
    Autoencoder-based feature extraction (PyTorch).
    Uses bottleneck layer as learned feature representation.
    """

    def __init__(self, config: Dict = None, random_state: int = RANDOM_SEED):
        self.config = config or FEATURE_SELECTION['Autoencoder']
        self.random_state = random_state
        self.net = None
        self.scaler = StandardScaler()
        self.latent_dim = None
        self.training_history = None
        self.computation_time = 0

    def fit(self, X: np.ndarray, latent_dim: int = 16) -> 'AutoencoderFeatureSelector':
        """Fit autoencoder."""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available. Skipping autoencoder training.")
            self.latent_dim = latent_dim
            self.computation_time = 0
            return self

        logger.info(f"Training autoencoder with latent_dim={latent_dim}...")
        start_time = time.time()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.latent_dim = latent_dim
        input_dim = X.shape[1]

        X_scaled = self.scaler.fit_transform(X)

        # Split train/val
        n_val = int(len(X_scaled) * 0.2)
        X_train_ae = torch.FloatTensor(X_scaled[n_val:]).to(TORCH_DEVICE)
        X_val_ae = torch.FloatTensor(X_scaled[:n_val]).to(TORCH_DEVICE)

        self.net = _AENet(input_dim, latent_dim, self.config['architecture']).to(TORCH_DEVICE)
        optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        train_ds = TensorDataset(X_train_ae, X_train_ae)
        train_loader = DataLoader(train_ds, batch_size=self.config['batch_size'], shuffle=True)

        best_val_loss, best_state, wait = float('inf'), None, 0
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config['epochs']):
            self.net.train()
            total_loss = 0
            for xb, _ in train_loader:
                optimizer.zero_grad()
                recon = self.net(xb)
                loss = criterion(recon, xb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            history['train_loss'].append(total_loss / len(X_train_ae))

            self.net.eval()
            with torch.no_grad():
                val_loss = criterion(self.net(X_val_ae), X_val_ae).item()
            history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= 10:
                    break

        if best_state:
            self.net.load_state_dict(best_state)
        self.net.eval()
        self.training_history = history
        self.computation_time = time.time() - start_time
        logger.info(f"Autoencoder training completed in {self.computation_time:.2f}s")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to latent representation."""
        if not TORCH_AVAILABLE or self.net is None:
            logger.warning("Using PCA fallback for transform")
            pca = PCA(n_components=self.latent_dim)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            return pca.fit_transform(X_scaled)

        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(TORCH_DEVICE)
        with torch.no_grad():
            latent = self.net.encode(X_t)
        return latent.cpu().numpy()

    def fit_transform(self, X: np.ndarray, latent_dim: int = 16) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X, latent_dim)
        return self.transform(X)

    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """Compute reconstruction error."""
        if not TORCH_AVAILABLE or self.net is None:
            return 0.0
        X_scaled = self.scaler.transform(X)
        X_t = torch.FloatTensor(X_scaled).to(TORCH_DEVICE)
        with torch.no_grad():
            recon = self.net(X_t)
        return nn.MSELoss()(recon, X_t).item()


class PermutationFeatureSelector:
    """
    Permutation importance-based feature selection.
    Measures impact of shuffling each feature on model performance.
    """

    def __init__(self, config: Dict = None, random_state: int = RANDOM_SEED):
        self.config = config or FEATURE_SELECTION['PermutationImportance']
        self.random_state = random_state
        self.model = None
        self.importance_result = None
        self.feature_importance = None
        self.computation_time = 0

    def fit(self, X: np.ndarray, y: np.ndarray,
           feature_names: List[str] = None) -> 'PermutationFeatureSelector':
        """
        Fit permutation importance selector.

        Args:
            X: Feature array
            y: Target array
            feature_names: Optional list of feature names

        Returns:
            self
        """
        logger.info("Computing permutation importance...")
        start_time = time.time()

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X, y)

        # Compute permutation importance
        self.importance_result = permutation_importance(
            self.model, X, y,
            n_repeats=self.config['n_repeats'],
            random_state=self.random_state,
            n_jobs=1
        )

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': self.importance_result.importances_mean,
            'importance_std': self.importance_result.importances_std
        }).sort_values('importance_mean', ascending=False)

        self.computation_time = time.time() - start_time
        logger.info(f"Permutation importance completed in {self.computation_time:.2f}s")

        return self

    def select_top_k(self, k: int) -> Tuple[List[str], np.ndarray]:
        """
        Select top-k features.

        Args:
            k: Number of features to select

        Returns:
            Tuple of (feature names, importance values)
        """
        if self.feature_importance is None:
            raise ValueError("Must call fit() first")

        top_k = self.feature_importance.head(k)
        features = top_k['feature'].tolist()
        importances = top_k['importance_mean'].values

        return features, importances

    def get_stability_score(self) -> float:
        """
        Compute stability based on importance std.

        Returns:
            Stability score (inverse of coefficient of variation)
        """
        if self.feature_importance is None:
            raise ValueError("Must call fit() first")

        mean_imp = self.feature_importance['importance_mean'].values
        std_imp = self.feature_importance['importance_std'].values

        # Coefficient of variation (lower is more stable)
        cv = np.mean(std_imp / (mean_imp + 1e-8))

        # Convert to stability score (higher is better)
        stability = 1 / (1 + cv)
        return stability


class InformationGainSelector:
    """
    Feature selection using Mutual Information (Information Gain).
    Uses sklearn.feature_selection.mutual_info_classif.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_scores = None
        self.name = "InformationGain"

    def fit(self, X, y, feature_names=None):
        from sklearn.feature_selection import mutual_info_classif

        logger.info("Computing Information Gain (Mutual Information) scores...")

        mi_scores = mutual_info_classif(
            X, y, discrete_features=False,
            n_neighbors=3, random_state=self.random_state
        )

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_scores = pd.DataFrame({
            'feature': feature_names,
            'mi_score': mi_scores
        }).sort_values('mi_score', ascending=False)

        return self

    def get_top_features(self, k=30):
        if self.feature_scores is None:
            raise ValueError("Must call fit() first")
        return self.feature_scores.head(k)['feature'].tolist()

    def get_scores(self):
        return self.feature_scores


class Stage2Pipeline:
    """
    Main pipeline for Stage 2: Feature Selection Methods Benchmarking.
    Runs all feature selection methods and compares results.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.results = {}

    def run(self, X: np.ndarray, y: np.ndarray,
           feature_names: List[str] = None,
           k_values: List[int] = [10, 20, 30],
           mode: str = "multiclass") -> Dict:
        """
        Run all feature selection methods.

        Args:
            X: Feature array
            y: Target array
            feature_names: Optional list of feature names
            k_values: List of k values to test

        Returns:
            Dictionary with results for each method
        """
        logger.info("=" * 60)
        logger.info("STAGE 2: FEATURE SELECTION METHODS BENCHMARKING")
        logger.info("=" * 60)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Subsample for memory efficiency — MI / SHAP / Permutation
        # don't need 12M+ rows; 500K gives statistically identical rankings
        MAX_SAMPLES = 500_000
        if X.shape[0] > MAX_SAMPLES:
            logger.info(f"Subsampling {X.shape[0]:,} → {MAX_SAMPLES:,} rows for feature selection (memory)")
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X.shape[0], MAX_SAMPLES, replace=False)
            X = X[idx]
            y = y[idx]

        results = {
            'methods': {},
            'comparison': [],
            'best_features': {}
        }

        # 1. SHAP-based selection
        logger.info("\n--- Method 1: SHAP-Based Feature Selection ---")
        shap_selector = SHAPFeatureSelector(random_state=self.random_state)
        shap_selector.fit(X, y, feature_names)

        for k in k_values:
            features, _ = shap_selector.select_top_k(k)
            results['methods'][f'SHAP_k{k}'] = {
                'features': features,
                'computation_time': shap_selector.computation_time,
                'importance': shap_selector.feature_importance.head(k).to_dict('records')
            }

        results['best_features']['SHAP'] = shap_selector.feature_importance.head(30)['feature'].tolist()

        # 2. PCA
        logger.info("\n--- Method 2: Principal Component Analysis ---")
        for variance in [0.90, 0.95]:
            pca_selector = PCAFeatureSelector(random_state=self.random_state)
            pca_selector.fit(X, variance_threshold=variance)

            results['methods'][f'PCA_var{int(variance*100)}'] = {
                'n_components': pca_selector.n_components,
                'explained_variance': pca_selector.explained_variance,
                'computation_time': pca_selector.computation_time,
                'reconstruction_error': pca_selector.get_reconstruction_error(X)
            }

        # 3. mRMR
        logger.info("\n--- Method 3: mRMR Feature Selection ---")
        mrmr_selector = mRMRFeatureSelector(random_state=self.random_state)
        mrmr_selector.fit(X, y, feature_names)

        for k in k_values:
            features, scores = mrmr_selector.select_top_k(k)
            results['methods'][f'mRMR_k{k}'] = {
                'features': features,
                'scores': scores,
                'computation_time': mrmr_selector.computation_time
            }

        results['best_features']['mRMR'] = mrmr_selector.feature_scores[:30]

        # 4. Autoencoder
        logger.info("\n--- Method 4: Autoencoder Latent Features ---")
        for latent_dim in [8, 16]:
            ae_selector = AutoencoderFeatureSelector(random_state=self.random_state)
            ae_selector.fit(X, latent_dim=latent_dim)

            results['methods'][f'AE_dim{latent_dim}'] = {
                'latent_dim': latent_dim,
                'reconstruction_error': ae_selector.get_reconstruction_error(X),
                'computation_time': ae_selector.computation_time,
                'training_history': ae_selector.training_history
            }

        # 5. Permutation Importance
        logger.info("\n--- Method 5: Permutation Importance ---")
        perm_selector = PermutationFeatureSelector(random_state=self.random_state)
        perm_selector.fit(X, y, feature_names)

        for k in k_values:
            features, importances = perm_selector.select_top_k(k)
            results['methods'][f'PERM_k{k}'] = {
                'features': features,
                'importances': importances.tolist(),
                'computation_time': perm_selector.computation_time,
                'stability': perm_selector.get_stability_score()
            }

        results['best_features']['PERM'] = perm_selector.feature_importance.head(30)['feature'].tolist()

        # 6. Information Gain Selection
        logger.info("\n--- Information Gain (Mutual Information) ---")
        try:
            ig_selector = InformationGainSelector(self.random_state)
            ig_selector.fit(X, y, feature_names)
            ig_top = ig_selector.get_top_features(k=30)
            results['methods']['InformationGain'] = {
                'scores': ig_selector.get_scores(),
                'top_features': ig_top,
                'n_selected': len(ig_top)
            }
            results['best_features']['IG'] = ig_top
            logger.info(f"Top 5 IG features: {ig_top[:5]}")
        except Exception as e:
            logger.error(f"Information Gain failed: {e}")

        # Create comparison summary
        results['comparison'] = self._create_comparison_summary(results)

        # Save results
        self._save_results(results, mode=mode)

        self.results = results

        logger.info("=" * 60)
        logger.info(f"STAGE 2 COMPLETE ({mode.upper()} MODE)")
        logger.info("=" * 60)

        return results

    def _create_comparison_summary(self, results: Dict) -> pd.DataFrame:
        """Create summary comparison of all methods."""
        summary = []

        for method_name, method_results in results['methods'].items():
            row = {
                'Method': method_name,
                'Time (s)': method_results.get('computation_time', 0),
                'N_Features/Components': (
                    method_results.get('n_components') or
                    len(method_results.get('features', [])) or
                    method_results.get('latent_dim', 0)
                )
            }

            if 'reconstruction_error' in method_results:
                row['Reconstruction Error'] = method_results['reconstruction_error']

            if 'stability' in method_results:
                row['Stability'] = method_results['stability']

            summary.append(row)

        return pd.DataFrame(summary)

    def _save_results(self, results: Dict, mode: str = "multiclass"):
        """Save feature selection results."""
        output_dir = get_results_dir(mode) / 'stage2_feature_selection'
        output_dir.mkdir(exist_ok=True)

        # Save comparison summary
        results['comparison'].to_csv(output_dir / 'method_comparison.csv', index=False)

        # Save selected features
        for method, features in results['best_features'].items():
            if isinstance(features, list):
                with open(output_dir / f'{method}_features.json', 'w') as f:
                    json.dump(features, f, indent=2)
            else:
                # It's a list of dicts
                with open(output_dir / f'{method}_features.json', 'w') as f:
                    json.dump(features, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")


def run_feature_selection_benchmark(X: np.ndarray, y: np.ndarray,
                                    feature_names: List[str] = None) -> Dict:
    """
    Run complete feature selection benchmark.

    Args:
        X: Feature array
        y: Target array
        feature_names: Optional feature names

    Returns:
        Dictionary with benchmark results
    """
    pipeline = Stage2Pipeline()
    return pipeline.run(X, y, feature_names, k_values=[10, 20, 30, 40, 50])


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 2: Feature Selection Methods Benchmarking")
    print("="*70 + "\n")

    # Import Stage 1 results
    from stage1_data_preparation import Stage1Pipeline

    # Prepare data
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data = stage1.run()

    # Run feature selection benchmark
    pipeline = Stage2Pipeline()
    results = pipeline.run(
        X=data['splits']['X_train'],
        y=data['splits']['y_train'],
        feature_names=data['feature_columns']
    )

    # Print summary
    print("\n" + "-"*50)
    print("FEATURE SELECTION COMPARISON SUMMARY")
    print("-"*50)
    print(results['comparison'].to_string(index=False))
    print("-"*50)
