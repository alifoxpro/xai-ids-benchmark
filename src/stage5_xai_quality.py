"""
================================================================================
STAGE 5: XAI EXPLANATIONS QUALITY BENCHMARKING
================================================================================
This module evaluates:
- SHAP explanation quality
- LIME comparison
- Attention-based explanations
- Gradient-based methods
- Fidelity, Stability, Consistency metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import warnings
import logging
import pickle
import json
from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# PyTorch for gradient-based explanations
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from config import XAI_CONFIG, RESULTS_DIR, VIS_DIR, RANDOM_SEED, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaseExplainer(ABC):
    """Abstract base class for XAI explainers."""

    def __init__(self, name: str, random_state: int = RANDOM_SEED):
        self.name = name
        self.random_state = random_state
        self.explanation_time = 0
        self.feature_importance = None

    @abstractmethod
    def fit(self, model: Any, X_train: np.ndarray, feature_names: List[str] = None) -> 'BaseExplainer':
        pass

    @abstractmethod
    def explain(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        pass


class SHAPExplainer(BaseExplainer):
    """SHAP-based model explainer."""

    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("SHAP", random_state)
        self.config = XAI_CONFIG.get('SHAP', {})
        self.explainer = None
        self.shap_values = None
        self.expected_value = None

    def fit(self, model: Any, X_train: np.ndarray,
           feature_names: List[str] = None) -> 'SHAPExplainer':
        """
        Fit SHAP explainer.

        Args:
            model: Trained model
            X_train: Training data for background
            feature_names: Feature names

        Returns:
            self
        """
        logger.info("Fitting SHAP explainer...")

        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Using fallback.")
            return self

        # Select background samples
        n_background = min(self.config.get('n_samples_background', 100), len(X_train))
        background_idx = np.random.choice(len(X_train), n_background, replace=False)
        background = X_train[background_idx]

        # Choose explainer type based on model
        if hasattr(model, 'estimators_') or hasattr(model, 'feature_importances_'):
            # Tree-based model
            self.explainer = shap.TreeExplainer(model)
        else:
            # Use KernelExplainer for other models
            if hasattr(model, 'predict_proba'):
                self.explainer = shap.KernelExplainer(model.predict_proba, background)
            else:
                self.explainer = shap.KernelExplainer(model.predict, background)

        return self

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Generate SHAP explanations.

        Args:
            X: Samples to explain

        Returns:
            SHAP values array
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            # Fallback: return random importance
            return np.random.randn(*X.shape)

        logger.info(f"Generating SHAP explanations for {len(X)} samples...")
        start_time = time.time()

        self.shap_values = self.explainer.shap_values(X)

        # Handle multi-class output
        if isinstance(self.shap_values, list):
            # list of arrays (one per class) -> average absolute values
            self.shap_values = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
        elif hasattr(self.shap_values, 'values'):
            # Newer SHAP Explanation object
            self.shap_values = self.shap_values.values
        # If still 3D (samples, features, classes), keep as-is for now
        # get_feature_importance() handles both 2D and 3D

        self.explanation_time = time.time() - start_time
        logger.info(f"SHAP explanations generated in {self.explanation_time:.2f}s")

        return self.shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from SHAP values."""
        if self.shap_values is None:
            return pd.DataFrame()

        shap_arr = np.abs(self.shap_values)
        # Handle multi-class: (samples, features, classes) -> average over classes too
        if shap_arr.ndim == 3:
            importance = shap_arr.mean(axis=(0, 2))  # -> (features,)
        else:
            importance = shap_arr.mean(axis=0)  # -> (features,)

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance.flatten()
        }).sort_values('importance', ascending=False)


class LIMEExplainer(BaseExplainer):
    """LIME-based model explainer."""

    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("LIME", random_state)
        self.config = XAI_CONFIG.get('LIME', {})
        self.explainer = None
        self.explanations = []

    def fit(self, model: Any, X_train: np.ndarray,
           feature_names: List[str] = None) -> 'LIMEExplainer':
        """
        Fit LIME explainer.

        Args:
            model: Trained model
            X_train: Training data
            feature_names: Feature names

        Returns:
            self
        """
        logger.info("Fitting LIME explainer...")

        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        if not LIME_AVAILABLE:
            logger.warning("LIME not available. Using fallback.")
            return self

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            mode='classification',
            random_state=self.random_state
        )

        return self

    def explain(self, X: np.ndarray, n_samples: int = None) -> np.ndarray:
        """
        Generate LIME explanations.

        Args:
            X: Samples to explain
            n_samples: Number of samples to explain (default: all)

        Returns:
            Array of feature importances
        """
        if not LIME_AVAILABLE or self.explainer is None:
            return np.random.randn(*X.shape)

        n_samples = n_samples or len(X)
        n_samples = min(n_samples, len(X))

        logger.info(f"Generating LIME explanations for {n_samples} samples...")
        start_time = time.time()

        importance_matrix = np.zeros((n_samples, X.shape[1]))

        for i in range(n_samples):
            try:
                exp = self.explainer.explain_instance(
                    X[i],
                    self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                    num_features=self.config.get('num_features', 30),
                    num_samples=self.config.get('num_samples', 5000)
                )
                self.explanations.append(exp)

                # Extract feature importance
                for feat_idx, importance in exp.local_exp.get(1, exp.local_exp.get(0, [])):
                    if feat_idx < X.shape[1]:
                        importance_matrix[i, feat_idx] = abs(importance)
            except Exception as e:
                logger.warning(f"LIME error for sample {i}: {e}")

        self.explanation_time = time.time() - start_time
        logger.info(f"LIME explanations generated in {self.explanation_time:.2f}s")

        return importance_matrix

    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance from LIME."""
        if not self.explanations:
            return pd.DataFrame()

        # Aggregate importance across all explanations
        importance_dict = {}
        for exp in self.explanations:
            for feat_idx, importance in exp.local_exp.get(1, exp.local_exp.get(0, [])):
                if feat_idx < len(self.feature_names):
                    feat_name = self.feature_names[feat_idx]
                    if feat_name not in importance_dict:
                        importance_dict[feat_name] = []
                    importance_dict[feat_name].append(abs(importance))

        return pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': [np.mean(v) for v in importance_dict.values()]
        }).sort_values('importance', ascending=False)


class GradientExplainer(BaseExplainer):
    """Gradient-based explainer for neural networks."""

    def __init__(self, random_state: int = RANDOM_SEED):
        super().__init__("Gradient", random_state)
        self.model = None
        self.gradients = None

    def fit(self, model: Any, X_train: np.ndarray,
           feature_names: List[str] = None) -> 'GradientExplainer':
        """
        Fit gradient explainer.

        Args:
            model: Trained Keras/TF model
            X_train: Training data
            feature_names: Feature names

        Returns:
            self
        """
        logger.info("Setting up gradient explainer...")

        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

        return self

    def explain(self, X: np.ndarray) -> np.ndarray:
        """
        Generate gradient-based explanations.

        Args:
            X: Samples to explain

        Returns:
            Gradient values array
        """
        if not (TORCH_AVAILABLE and isinstance(self.model, torch.nn.Module)):
            logger.warning("Model is not a PyTorch model. Using fallback.")
            return np.random.randn(*X.shape)

        logger.info(f"Computing gradients for {len(X)} samples...")
        start_time = time.time()

        device = next(self.model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device).requires_grad_(True)

        self.model.eval()
        predictions = self.model(X_tensor)
        # Sum predictions to get scalar for backward
        predictions.sum().backward()

        self.gradients = X_tensor.grad.detach().cpu().numpy()

        self.explanation_time = time.time() - start_time
        logger.info(f"Gradients computed in {self.explanation_time:.2f}s")

        return np.abs(self.gradients)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from gradients."""
        if self.gradients is None:
            return pd.DataFrame()

        importance = np.abs(self.gradients).mean(axis=0)

        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)


class AnchorsExplainer:
    """
    Anchors rule-based explainer for model predictions.
    Provides human-readable IF-THEN rules.
    """

    def __init__(self):
        self.explainer = None
        self.name = "Anchors"

    def fit(self, X_train, feature_names=None, class_names=None):
        try:
            from anchor import anchor_tabular
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            self.explainer = anchor_tabular.AnchorTabularExplainer(
                class_names=class_names if class_names else ['Benign', 'Attack'],
                feature_names=feature_names,
                train_data=X_train
            )
            logger.info("Anchors explainer fitted successfully")
        except ImportError:
            logger.warning("anchor-exp not installed. Anchors explanations unavailable.")
            self.explainer = None
        return self

    def explain(self, predict_fn, X_sample):
        if self.explainer is None:
            return None

        explanations = []
        n_samples = min(len(X_sample), 50)

        for i in range(n_samples):
            try:
                exp = self.explainer.explain_instance(X_sample[i], predict_fn, threshold=0.95)
                explanations.append({
                    'sample_idx': i,
                    'anchor': ' AND '.join(exp.names()),
                    'precision': exp.precision(),
                    'coverage': exp.coverage(),
                    'n_features_used': len(exp.features())
                })
            except Exception as e:
                logger.warning(f"Anchors explanation failed for sample {i}: {e}")

        return explanations


class XAIQualityMetrics:
    """
    Computes quality metrics for XAI explanations:
    - Fidelity: How well explanations match model behavior
    - Stability: Consistency across similar inputs
    - Consistency: Agreement between explanation methods
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def compute_fidelity(
        self,
        model: Any,
        X: np.ndarray,
        explanations: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Compute fidelity score.
        Measures if top-k important features are truly predictive.

        Args:
            model: Trained model
            X: Input samples
            explanations: Explanation values (feature importance)
            k: Number of top features to consider

        Returns:
            Fidelity score (0-1)
        """
        logger.info("Computing fidelity score...")

        n_samples = len(X)
        fidelity_scores = []

        for i in range(min(100, n_samples)):  # Limit for efficiency
            # Get top-k features for this sample
            top_k_indices = np.argsort(explanations[i])[-k:]

            # Create masked version (keep only top-k features)
            X_masked = np.zeros_like(X[i:i+1])
            X_masked[0, top_k_indices] = X[i, top_k_indices]

            # Get predictions
            try:
                pred_original = model.predict(X[i:i+1])
                pred_masked = model.predict(X_masked)

                # Check if prediction is preserved
                if hasattr(pred_original, 'shape') and len(pred_original.shape) > 1:
                    fidelity = 1 - np.abs(pred_original - pred_masked).mean()
                else:
                    fidelity = float(pred_original == pred_masked)

                fidelity_scores.append(fidelity)
            except:
                pass

        return np.mean(fidelity_scores) if fidelity_scores else 0.0

    def compute_stability(
        self,
        explainer: BaseExplainer,
        X: np.ndarray,
        noise_level: float = 0.01,
        n_perturbations: int = 5
    ) -> float:
        """
        Compute stability score.
        Measures if similar inputs get similar explanations.

        Args:
            explainer: Fitted explainer
            X: Input samples
            noise_level: Standard deviation of perturbation noise
            n_perturbations: Number of perturbations per sample

        Returns:
            Stability score (0-1)
        """
        logger.info("Computing stability score...")

        n_samples = min(50, len(X))  # Limit for efficiency
        stabilities = []

        for i in range(n_samples):
            original_exp = explainer.explain(X[i:i+1])

            perturbation_exps = []
            for _ in range(n_perturbations):
                # Add small noise
                X_perturbed = X[i:i+1] + np.random.normal(0, noise_level, X[i:i+1].shape)
                perturbed_exp = explainer.explain(X_perturbed)
                perturbation_exps.append(perturbed_exp)

            # Compute correlation between original and perturbed explanations
            for perturbed_exp in perturbation_exps:
                correlation = np.corrcoef(original_exp.flatten(), perturbed_exp.flatten())[0, 1]
                if not np.isnan(correlation):
                    stabilities.append(correlation)

        return np.mean(stabilities) if stabilities else 0.0

    def compute_consistency(
        self,
        explanations_dict: Dict[str, np.ndarray],
        top_k: int = 10
    ) -> float:
        """
        Compute consistency between different explanation methods.
        Measures overlap in top-k important features.

        Args:
            explanations_dict: Dictionary of {method_name: explanations}
            top_k: Number of top features to compare

        Returns:
            Consistency score (0-1)
        """
        logger.info("Computing consistency score...")

        if len(explanations_dict) < 2:
            return 1.0

        methods = list(explanations_dict.keys())
        n_samples = len(list(explanations_dict.values())[0])

        consistencies = []

        for i in range(min(100, n_samples)):
            top_k_sets = {}

            for method, exps in explanations_dict.items():
                exp_i = np.array(exps[i])
                # If multi-class (2D), average absolute values across classes
                if exp_i.ndim > 1:
                    exp_i = np.abs(exp_i).mean(axis=-1)
                top_k_sets[method] = set(int(x) for x in np.argsort(np.abs(exp_i))[-top_k:])

            # Compute pairwise Jaccard similarity
            for j, m1 in enumerate(methods):
                for m2 in methods[j+1:]:
                    intersection = len(top_k_sets[m1] & top_k_sets[m2])
                    union = len(top_k_sets[m1] | top_k_sets[m2])
                    jaccard = intersection / union if union > 0 else 0
                    consistencies.append(jaccard)

        return np.mean(consistencies) if consistencies else 0.0

    def compute_sparsity(self, explanations, total_features):
        """
        Compute sparsity of explanations.
        Lower sparsity = more concise/interpretable explanations.

        Args:
            explanations: List of explanation arrays (feature importance vectors)
            total_features: Total number of features

        Returns:
            Sparsity score (0-1, lower is sparser/better)
        """
        logger.info("Computing explanation sparsity...")

        sparsity_scores = []
        for exp in explanations:
            exp_array = np.array(exp)
            n_nonzero = np.count_nonzero(exp_array)
            sparsity_scores.append(n_nonzero / total_features)

        avg_sparsity = np.mean(sparsity_scores)
        logger.info(f"Average sparsity: {avg_sparsity:.4f} ({avg_sparsity*100:.1f}% features used)")

        return {
            'mean_sparsity': float(avg_sparsity),
            'std_sparsity': float(np.std(sparsity_scores)),
            'min_sparsity': float(np.min(sparsity_scores)),
            'max_sparsity': float(np.max(sparsity_scores))
        }

    def compute_faithfulness(self, model, X, y, feature_importances, top_k=10):
        """
        Compute faithfulness via sufficiency and necessity.
        Sufficiency: accuracy using only top-k features.
        Necessity: accuracy drop when removing top-k features.

        Args:
            model: Trained model with predict method
            X: Input features
            y: True labels
            feature_importances: Array of feature importance scores
            top_k: Number of top features to evaluate

        Returns:
            Dict with sufficiency and necessity scores
        """
        logger.info(f"Computing faithfulness (top-{top_k} features)...")

        importance_order = np.argsort(np.abs(feature_importances))[::-1]
        top_features = importance_order[:top_k]
        bottom_features = importance_order[top_k:]

        baseline_acc = accuracy_score(y, model.predict(X))

        # Sufficiency: keep only top-k features, zero out rest
        X_sufficient = X.copy()
        if len(bottom_features) > 0:
            X_sufficient[:, bottom_features] = 0
        sufficiency_acc = accuracy_score(y, model.predict(X_sufficient))
        sufficiency = sufficiency_acc / baseline_acc if baseline_acc > 0 else 0

        # Necessity: remove top-k features, keep rest
        X_necessary = X.copy()
        X_necessary[:, top_features] = 0
        necessity_acc = accuracy_score(y, model.predict(X_necessary))
        necessity_drop = (baseline_acc - necessity_acc) / baseline_acc if baseline_acc > 0 else 0

        logger.info(f"Sufficiency: {sufficiency:.4f}, Necessity drop: {necessity_drop:.4f}")

        return {
            'baseline_accuracy': float(baseline_acc),
            'sufficiency_accuracy': float(sufficiency_acc),
            'sufficiency_ratio': float(sufficiency),
            'necessity_accuracy': float(necessity_acc),
            'necessity_drop': float(necessity_drop),
            'top_k': top_k
        }


class Stage5Pipeline:
    """
    Main pipeline for Stage 5: XAI Quality Benchmarking.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.explainers = {}
        self.results = {}

    def run(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str] = None,
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run XAI quality benchmarking.

        Args:
            model: Trained model
            X_train: Training data
            X_test: Test data
            y_test: Test labels
            feature_names: Feature names

        Returns:
            Dictionary with XAI quality results
        """
        logger.info("=" * 60)
        logger.info("STAGE 5: XAI EXPLANATIONS QUALITY BENCHMARKING")
        logger.info("=" * 60)

        feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        n_explain = min(30, len(X_test))  # Limit samples (KernelSHAP is slow on multiclass)

        results = {
            'explainers': {},
            'quality_metrics': {},
            'comparison': []
        }

        # Initialize explainers
        explainers = {
            'SHAP': SHAPExplainer(self.random_state),
            'LIME': LIMEExplainer(self.random_state)
        }

        explanations_dict = {}

        for name, explainer in explainers.items():
            logger.info(f"\n--- Evaluating {name} ---")

            try:
                # Fit explainer
                explainer.fit(model, X_train, feature_names)

                # Generate explanations
                explanations = explainer.explain(X_test[:n_explain])
                explanations_dict[name] = explanations

                # Get feature importance
                importance = explainer.get_feature_importance()

                results['explainers'][name] = {
                    'explanation_time': explainer.explanation_time,
                    'feature_importance': importance.to_dict('records')[:20],  # Top 20
                    'n_samples_explained': n_explain
                }

                self.explainers[name] = explainer

            except Exception as e:
                logger.error(f"Error with {name}: {str(e)}")
                results['explainers'][name] = {'error': str(e)}

        # Compute quality metrics
        metrics = XAIQualityMetrics(self.random_state)

        for name, explanations in explanations_dict.items():
            logger.info(f"\nComputing quality metrics for {name}...")

            try:
                fidelity = metrics.compute_fidelity(model, X_test[:n_explain], explanations)

                exp_time = results['explainers'][name].get('explanation_time', 0)
                results['quality_metrics'][name] = {
                    'fidelity': fidelity,
                    'explanation_time_ms': exp_time * 1000 / n_explain if n_explain > 0 else 0
                }
            except Exception as e:
                logger.warning(f"Error computing metrics for {name}: {e}")

        # Compute consistency between methods
        if len(explanations_dict) >= 2:
            consistency = metrics.compute_consistency(explanations_dict)
            results['quality_metrics']['cross_method_consistency'] = consistency

        # Anchors Explanations
        logger.info("\n--- Anchors Explanations ---")
        try:
            # Subsample X_train for Anchors (beam search is O(n) on train data)
            MAX_ANCHOR_TRAIN = 10_000
            if len(X_train) > MAX_ANCHOR_TRAIN:
                rng = np.random.RandomState(42)
                idx_a = rng.choice(len(X_train), MAX_ANCHOR_TRAIN, replace=False)
                X_anchor_train = X_train[idx_a]
            else:
                X_anchor_train = X_train
            anchors_explainer = AnchorsExplainer()
            anchors_explainer.fit(X_anchor_train, feature_names, class_names=['Benign', 'Attack'])
            if anchors_explainer.explainer is not None:
                anchors_results = anchors_explainer.explain(model.predict, X_test[:20])
                results['anchors'] = anchors_results
        except Exception as e:
            logger.error(f"Anchors explanations failed: {e}")

        # Sparsity
        logger.info("\n--- Explanation Sparsity ---")
        if 'SHAP' in explanations_dict:
            sparsity = metrics.compute_sparsity(
                explanations_dict['SHAP'], X_test.shape[1]
            )
            results['sparsity'] = sparsity

        # Faithfulness
        logger.info("\n--- Explanation Faithfulness ---")
        if 'SHAP' in explanations_dict:
            shap_arr = np.abs(explanations_dict['SHAP'])
            shap_importance = shap_arr.mean(axis=(0, 2)) if shap_arr.ndim == 3 else shap_arr.mean(axis=0)
            faithfulness = metrics.compute_faithfulness(
                model, X_test[:n_explain], y_test[:n_explain],
                shap_importance, top_k=10
            )
            results['faithfulness'] = faithfulness

        # Create comparison summary
        results['comparison'] = self._create_comparison_summary(results)

        # Save results
        self._save_results(results, mode)

        self.results = results

        logger.info("=" * 60)
        logger.info("STAGE 5 COMPLETE")
        logger.info("=" * 60)

        return results

    def _create_comparison_summary(self, results: Dict) -> pd.DataFrame:
        """Create XAI comparison summary."""
        summary = []

        for name, metrics in results['quality_metrics'].items():
            if isinstance(metrics, dict) and 'fidelity' in metrics:
                row = {
                    'Method': name,
                    'Fidelity': metrics.get('fidelity', 'N/A'),
                    'Explanation Time (ms)': metrics.get('explanation_time_ms', 'N/A')
                }
                summary.append(row)

        return pd.DataFrame(summary)

    def _save_results(self, results: Dict, mode: str = "multiclass"):
        """Save XAI quality results."""
        output_dir = get_results_dir(mode) / 'stage5_xai_quality'
        output_dir.mkdir(exist_ok=True)

        # Save comparison
        if not results['comparison'].empty:
            results['comparison'].to_csv(output_dir / 'xai_comparison.csv', index=False)

        # Save detailed results
        summary = {
            'quality_metrics': results['quality_metrics'],
            'explainer_summary': {
                name: {k: v for k, v in exp_results.items() if k != 'feature_importance'}
                for name, exp_results in results['explainers'].items()
            }
        }

        with open(output_dir / 'xai_quality_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 5: XAI Explanations Quality Benchmarking")
    print("="*70 + "\n")

    # Import previous stages
    from stage1_data_preparation import Stage1Pipeline
    from stage3_ml_models import LightGBMModel

    # Prepare data
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data = stage1.run()

    # Train a model
    model = LightGBMModel()
    model.fit(
        data['splits']['X_train'],
        data['splits']['y_train'],
        tune_hyperparams=False
    )

    # Run XAI quality benchmarking
    pipeline = Stage5Pipeline()
    results = pipeline.run(
        model=model.model,
        X_train=data['splits']['X_train'],
        X_test=data['splits']['X_test'],
        y_test=data['splits']['y_test'],
        feature_names=data['feature_columns']
    )

    # Print summary
    print("\n" + "-"*50)
    print("XAI QUALITY COMPARISON")
    print("-"*50)
    print(results['comparison'].to_string(index=False))
    print("-"*50)
