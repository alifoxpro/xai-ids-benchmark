"""
================================================================================
STAGE 8: ROBUSTNESS & ADVERSARIAL TESTING
================================================================================
This module provides:
- FGSM (Fast Gradient Sign Method) adversarial attacks
- Gaussian noise robustness testing
- Feature perturbation analysis
- Model stability evaluation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import time
import warnings
import logging
import json

from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

# PyTorch for gradient-based adversarial attacks
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_DEVICE = None

from config import ADVERSARIAL_CONFIG, RESULTS_DIR, RANDOM_SEED, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FGSMAttack:
    """
    Fast Gradient Sign Method (FGSM) adversarial attack (PyTorch).
    Generates adversarial examples by perturbing inputs in the
    direction of the gradient.
    """

    def __init__(self, model=None):
        self.model = model

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1
    ) -> np.ndarray:
        """Generate FGSM adversarial examples (chunked for large arrays)."""
        logger.info(f"Generating FGSM adversarial examples (epsilon={epsilon}, {X.shape[0]:,} samples)...")

        if self.model is None or not TORCH_AVAILABLE:
            logger.warning("No PyTorch model provided. Using random perturbation.")
            perturbation = epsilon * np.sign(
                np.random.randn(*X.shape).astype(np.float32))
            return X + perturbation

        device = next(self.model.parameters()).device
        # Use train() mode — cuDNN RNN backward requires it
        self.model.train()

        CHUNK = 8192
        adv_chunks = []
        for i in range(0, len(X), CHUNK):
            xb = torch.FloatTensor(X[i:i+CHUNK]).to(device).requires_grad_(True)
            yb = torch.LongTensor(y[i:i+CHUNK]).to(device)
            logits = self.model(xb)
            if isinstance(logits, tuple):
                logits = logits[-1]
            loss = nn.CrossEntropyLoss()(logits, yb)
            loss.backward()
            adv_chunks.append((xb + epsilon * xb.grad.sign()).detach().cpu().numpy())

        self.model.eval()  # Restore eval mode
        return np.concatenate(adv_chunks, axis=0)


class PGDAttack:
    """
    Projected Gradient Descent (PGD) adversarial attack (PyTorch).
    Iterative version of FGSM - stronger attack.
    """

    def __init__(self, model=None):
        self.model = model

    def generate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.3,
        n_steps: int = 10,
        step_size: float = 0.01
    ) -> np.ndarray:
        """Generate PGD adversarial examples (chunked for large arrays)."""
        logger.info(f"Generating PGD adversarial examples (epsilon={epsilon}, steps={n_steps}, {X.shape[0]:,} samples)...")

        if self.model is None or not TORCH_AVAILABLE:
            logger.warning("No PyTorch model provided. Using iterative random perturbation.")
            X_adv = X.copy()
            for _ in range(n_steps):
                perturbation = step_size * np.sign(
                    np.random.randn(*X.shape).astype(np.float32))
                X_adv = X_adv + perturbation
                delta = np.clip(X_adv - X, -epsilon, epsilon)
                X_adv = X + delta
            return X_adv

        device = next(self.model.parameters()).device
        # Use train() mode — cuDNN RNN backward requires it
        self.model.train()

        CHUNK = 8192
        adv_chunks = []
        for i in range(0, len(X), CHUNK):
            xb_orig = torch.FloatTensor(X[i:i+CHUNK]).to(device)
            xb_adv = xb_orig.clone().detach()
            yb = torch.LongTensor(y[i:i+CHUNK]).to(device)
            for step in range(n_steps):
                xb_adv.requires_grad_(True)
                logits = self.model(xb_adv)
                if isinstance(logits, tuple):
                    logits = logits[-1]
                loss = nn.CrossEntropyLoss()(logits, yb)
                loss.backward()
                xb_adv = xb_adv + step_size * xb_adv.grad.sign()
                xb_adv = xb_adv.detach()
                delta = torch.clamp(xb_adv - xb_orig, -epsilon, epsilon)
                xb_adv = xb_orig + delta
            adv_chunks.append(xb_adv.cpu().numpy())

        self.model.eval()  # Restore eval mode
        return np.concatenate(adv_chunks, axis=0)


class GaussianNoiseAttack:
    """
    Gaussian noise perturbation for robustness testing.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        np.random.seed(random_state)

    def generate(
        self,
        X: np.ndarray,
        sigma: float = 0.1
    ) -> np.ndarray:
        """
        Add Gaussian noise to input samples.

        Args:
            X: Input samples
            sigma: Standard deviation of noise

        Returns:
            Noisy samples
        """
        logger.info(f"Adding Gaussian noise (sigma={sigma}, {X.shape[0]:,} samples)...")

        noise = np.random.normal(0, sigma, X.shape).astype(np.float32)
        X_noisy = X + noise

        return X_noisy


class FeaturePerturbation:
    """
    Feature-level perturbation analysis.
    Tests model sensitivity to individual feature changes.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def single_feature_perturbation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        perturbation_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Test impact of perturbing single features.

        Args:
            model: Trained model
            X: Input samples
            y: True labels
            perturbation_ratio: Ratio of samples to perturb

        Returns:
            DataFrame with per-feature sensitivity scores
        """
        logger.info("Analyzing single feature perturbations...")

        n_samples = len(X)
        n_features = X.shape[1]
        n_perturb = int(n_samples * perturbation_ratio)

        baseline_accuracy = accuracy_score(y, model.predict(X))

        sensitivities = []

        for feat_idx in range(n_features):
            # Create perturbed copy
            X_perturbed = X.copy()

            # Shuffle values of this feature for subset of samples
            perturb_idx = np.random.choice(n_samples, n_perturb, replace=False)
            X_perturbed[perturb_idx, feat_idx] = np.random.permutation(
                X_perturbed[perturb_idx, feat_idx]
            )

            # Evaluate
            perturbed_accuracy = accuracy_score(y, model.predict(X_perturbed))

            sensitivities.append({
                'feature_index': feat_idx,
                'baseline_accuracy': baseline_accuracy,
                'perturbed_accuracy': perturbed_accuracy,
                'accuracy_drop': baseline_accuracy - perturbed_accuracy,
                'sensitivity': (baseline_accuracy - perturbed_accuracy) / baseline_accuracy
            })

        df = pd.DataFrame(sensitivities)
        df = df.sort_values('sensitivity', ascending=False)

        return df

    def multi_feature_perturbation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_features_list: List[int] = [1, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        Test impact of perturbing multiple features simultaneously.

        Args:
            model: Trained model
            X: Input samples
            y: True labels
            n_features_list: List of number of features to perturb

        Returns:
            DataFrame with multi-feature perturbation results
        """
        logger.info("Analyzing multi-feature perturbations...")

        baseline_accuracy = accuracy_score(y, model.predict(X))
        total_features = X.shape[1]

        results = []

        for n_features in n_features_list:
            if n_features > total_features:
                continue

            X_perturbed = X.copy()

            # Randomly select features to perturb
            features_to_perturb = np.random.choice(
                total_features, n_features, replace=False
            )

            # Shuffle each selected feature
            for feat_idx in features_to_perturb:
                X_perturbed[:, feat_idx] = np.random.permutation(X_perturbed[:, feat_idx])

            # Evaluate
            perturbed_accuracy = accuracy_score(y, model.predict(X_perturbed))

            results.append({
                'n_features_perturbed': n_features,
                'percentage_perturbed': n_features / total_features * 100,
                'baseline_accuracy': baseline_accuracy,
                'perturbed_accuracy': perturbed_accuracy,
                'accuracy_drop': baseline_accuracy - perturbed_accuracy
            })

        return pd.DataFrame(results)


class ConceptDriftSimulator:
    """
    Simulates concept drift to test model resilience
    to changing data distributions over time.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        np.random.seed(random_state)

    def simulate_gradual_drift(
        self,
        X: np.ndarray,
        magnitude: float = 0.3,
        n_steps: int = 10
    ) -> List[np.ndarray]:
        """
        Simulate gradual concept drift by progressively shifting features.

        Args:
            X: Original data
            magnitude: Final drift magnitude
            n_steps: Number of drift steps

        Returns:
            List of progressively drifted datasets
        """
        logger.info(f"Simulating gradual drift (magnitude={magnitude}, steps={n_steps})...")

        drift_direction = np.random.randn(X.shape[1])
        drift_direction = drift_direction / np.linalg.norm(drift_direction)

        drifted_datasets = []
        for step in range(1, n_steps + 1):
            current_magnitude = magnitude * (step / n_steps)
            shift = current_magnitude * drift_direction * np.std(X, axis=0)
            X_drifted = X + shift
            drifted_datasets.append(X_drifted)

        return drifted_datasets

    def simulate_sudden_drift(
        self,
        X: np.ndarray,
        magnitude: float = 0.5
    ) -> np.ndarray:
        """
        Simulate sudden concept drift with abrupt distribution change.

        Args:
            X: Original data
            magnitude: Drift magnitude

        Returns:
            Drifted dataset
        """
        logger.info(f"Simulating sudden drift (magnitude={magnitude})...")

        shift = magnitude * np.std(X, axis=0) * np.random.choice([-1, 1], size=X.shape[1])
        X_drifted = X + shift

        return X_drifted

    def evaluate_drift_impact(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        magnitude: float = 0.3,
        n_steps: int = 10
    ) -> pd.DataFrame:
        """
        Evaluate model accuracy degradation over gradual drift.
        Creates drifted data one step at a time to avoid storing all copies.

        Args:
            model: Trained model with predict method
            X: Test features
            y: True labels
            magnitude: Final drift magnitude
            n_steps: Number of drift steps

        Returns:
            DataFrame with accuracy at each drift step
        """
        from sklearn.metrics import accuracy_score, f1_score

        baseline_acc = accuracy_score(y, model.predict(X))
        baseline_f1 = f1_score(y, model.predict(X), average='weighted', zero_division=0)

        results = [{
            'step': 0,
            'drift_magnitude': 0.0,
            'accuracy': baseline_acc,
            'f1': baseline_f1,
            'accuracy_drop': 0.0
        }]

        # Compute drift direction once (only 81 floats)
        drift_direction = np.random.randn(X.shape[1]).astype(np.float32)
        drift_direction = drift_direction / np.linalg.norm(drift_direction)
        feature_stds = np.std(X, axis=0).astype(np.float32)

        for step in range(1, n_steps + 1):
            current_mag = magnitude * (step / n_steps)
            shift = (current_mag * drift_direction * feature_stds).astype(np.float32)
            X_drifted = X + shift  # one copy at a time
            y_pred = model.predict(X_drifted)
            acc = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            del X_drifted

            results.append({
                'step': step,
                'drift_magnitude': current_mag,
                'accuracy': acc,
                'f1': f1,
                'accuracy_drop': baseline_acc - acc
            })

        return pd.DataFrame(results)


class RobustnessEvaluator:
    """
    Comprehensive robustness evaluation combining multiple attack types.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.config = ADVERSARIAL_CONFIG

    def evaluate_fgsm_robustness(
        self,
        model: Any,
        pytorch_model,
        X: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate robustness to FGSM attacks.

        Args:
            model: Sklearn-compatible model for prediction
            pytorch_model: Keras model for gradient computation (or None)
            X: Input samples
            y: True labels

        Returns:
            DataFrame with FGSM robustness results
        """
        logger.info("Evaluating FGSM robustness...")

        fgsm = FGSMAttack(pytorch_model)
        baseline_accuracy = accuracy_score(y, model.predict(X))

        results = []

        for epsilon in self.config['FGSM']['epsilon_values']:
            X_adv = fgsm.generate(X, y, epsilon)
            adv_accuracy = accuracy_score(y, model.predict(X_adv))

            results.append({
                'attack': 'FGSM',
                'parameter': f'epsilon={epsilon}',
                'baseline_accuracy': baseline_accuracy,
                'adversarial_accuracy': adv_accuracy,
                'accuracy_drop': baseline_accuracy - adv_accuracy,
                'robustness_ratio': adv_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
            })

        return pd.DataFrame(results)

    def evaluate_noise_robustness(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        """
        Evaluate robustness to Gaussian noise.

        Args:
            model: Trained model
            X: Input samples
            y: True labels

        Returns:
            DataFrame with noise robustness results
        """
        logger.info("Evaluating noise robustness...")

        noise_attack = GaussianNoiseAttack(self.random_state)
        baseline_accuracy = accuracy_score(y, model.predict(X))

        results = []

        for sigma in self.config['GaussianNoise']['sigma_values']:
            X_noisy = noise_attack.generate(X, sigma)
            noisy_accuracy = accuracy_score(y, model.predict(X_noisy))

            results.append({
                'attack': 'GaussianNoise',
                'parameter': f'sigma={sigma}',
                'baseline_accuracy': baseline_accuracy,
                'adversarial_accuracy': noisy_accuracy,
                'accuracy_drop': baseline_accuracy - noisy_accuracy,
                'robustness_ratio': noisy_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
            })

        return pd.DataFrame(results)

    def evaluate_pgd_robustness(
        self,
        model,
        pytorch_model,
        X: np.ndarray,
        y: np.ndarray
    ) -> pd.DataFrame:
        """Evaluate robustness to PGD attacks."""
        logger.info("Evaluating PGD robustness...")

        pgd = PGDAttack(pytorch_model)
        baseline_accuracy = accuracy_score(y, model.predict(X))
        pgd_config = self.config.get('PGD', {'epsilon_values': [0.1, 0.3, 0.5], 'n_steps': 10, 'step_size': 0.01})

        results = []
        for epsilon in pgd_config['epsilon_values']:
            X_adv = pgd.generate(
                X, y, epsilon,
                n_steps=pgd_config.get('n_steps', 10),
                step_size=pgd_config.get('step_size', 0.01)
            )
            adv_accuracy = accuracy_score(y, model.predict(X_adv))

            results.append({
                'attack': 'PGD',
                'parameter': f'epsilon={epsilon},steps={pgd_config.get("n_steps", 10)}',
                'baseline_accuracy': baseline_accuracy,
                'adversarial_accuracy': adv_accuracy,
                'accuracy_drop': baseline_accuracy - adv_accuracy,
                'robustness_ratio': adv_accuracy / baseline_accuracy if baseline_accuracy > 0 else 0
            })

        return pd.DataFrame(results)

    def compute_robustness_score(
        self,
        robustness_results: pd.DataFrame
    ) -> float:
        """
        Compute overall robustness score.

        Args:
            robustness_results: DataFrame with robustness test results

        Returns:
            Robustness score (0-1)
        """
        if robustness_results.empty:
            return 0.0

        avg_robustness_ratio = robustness_results['robustness_ratio'].mean()
        return avg_robustness_ratio


class Stage8Pipeline:
    """
    Main pipeline for Stage 8: Robustness & Adversarial Testing.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.evaluator = RobustnessEvaluator(random_state)
        self.perturbation = FeaturePerturbation(random_state)
        self.results = {}

    def run(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        pytorch_model = None,
        model_name: str = "Model",
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run comprehensive robustness testing.

        Args:
            model: Trained sklearn-compatible model
            X_test: Test features
            y_test: Test labels
            pytorch_model: Optional Keras model for FGSM
            model_name: Name of the model

        Returns:
            Dictionary with all robustness results
        """
        logger.info("=" * 60)
        logger.info("STAGE 8: ROBUSTNESS & ADVERSARIAL TESTING")
        logger.info("=" * 60)

        results = {
            'model_name': model_name,
            'fgsm': None,
            'pgd': None,
            'noise': None,
            'feature_perturbation': None,
            'concept_drift': None,
            'robustness_scores': {}
        }

        # FGSM robustness
        logger.info("\n--- FGSM Attack Testing ---")
        fgsm_results = self.evaluator.evaluate_fgsm_robustness(
            model, pytorch_model, X_test, y_test
        )
        results['fgsm'] = fgsm_results
        logger.info(fgsm_results.to_string(index=False))

        # Noise robustness
        logger.info("\n--- Gaussian Noise Testing ---")
        noise_results = self.evaluator.evaluate_noise_robustness(
            model, X_test, y_test
        )
        results['noise'] = noise_results
        logger.info(noise_results.to_string(index=False))

        # PGD robustness
        logger.info("\n--- PGD Attack Testing ---")
        pgd_results = self.evaluator.evaluate_pgd_robustness(
            model, pytorch_model, X_test, y_test
        )
        results['pgd'] = pgd_results
        logger.info(pgd_results.to_string(index=False))

        # Concept Drift
        logger.info("\n--- Concept Drift Testing ---")
        drift_simulator = ConceptDriftSimulator(self.random_state)
        drift_config = ADVERSARIAL_CONFIG.get('ConceptDrift', {'drift_magnitudes': [0.1, 0.3, 0.5]})
        drift_results = {}
        for mag in drift_config.get('drift_magnitudes', [0.1, 0.3, 0.5]):
            drift_df = drift_simulator.evaluate_drift_impact(
                model, X_test, y_test, magnitude=mag
            )
            drift_results[f'magnitude_{mag}'] = drift_df
            logger.info(f"Drift magnitude {mag}: final accuracy = {drift_df.iloc[-1]['accuracy']:.4f}")
        results['concept_drift'] = drift_results

        # Feature perturbation
        logger.info("\n--- Feature Perturbation Analysis ---")
        feature_results = self.perturbation.multi_feature_perturbation(
            model, X_test, y_test
        )
        results['feature_perturbation'] = feature_results
        logger.info(feature_results.to_string(index=False))

        # Compute robustness scores
        all_robustness = pd.concat([fgsm_results, noise_results, pgd_results], ignore_index=True)
        overall_robustness = self.evaluator.compute_robustness_score(all_robustness)

        results['robustness_scores'] = {
            'fgsm_robustness': self.evaluator.compute_robustness_score(fgsm_results),
            'pgd_robustness': self.evaluator.compute_robustness_score(pgd_results),
            'noise_robustness': self.evaluator.compute_robustness_score(noise_results),
            'overall_robustness': overall_robustness
        }

        logger.info(f"\n--- Robustness Scores ---")
        logger.info(f"FGSM Robustness: {results['robustness_scores']['fgsm_robustness']:.4f}")
        logger.info(f"Noise Robustness: {results['robustness_scores']['noise_robustness']:.4f}")
        logger.info(f"Overall Robustness: {overall_robustness:.4f}")

        # Save results
        self._save_results(results, model_name, mode)

        self.results = results

        logger.info("=" * 60)
        logger.info("STAGE 8 COMPLETE")
        logger.info("=" * 60)

        return results

    def _save_results(self, results: Dict, model_name: str, mode: str = "multiclass"):
        """Save robustness test results."""
        output_dir = get_results_dir(mode) / 'stage8_robustness'
        output_dir.mkdir(exist_ok=True)

        # Save FGSM results
        if results['fgsm'] is not None:
            results['fgsm'].to_csv(output_dir / f'{model_name}_fgsm.csv', index=False)

        # Save noise results
        if results['noise'] is not None:
            results['noise'].to_csv(output_dir / f'{model_name}_noise.csv', index=False)

        # Save feature perturbation
        if results['feature_perturbation'] is not None:
            results['feature_perturbation'].to_csv(
                output_dir / f'{model_name}_feature_perturbation.csv', index=False
            )

        # Save robustness scores
        with open(output_dir / f'{model_name}_robustness_scores.json', 'w') as f:
            json.dump(results['robustness_scores'], f, indent=2)

        logger.info(f"Results saved to {output_dir}")


def evaluate_model_robustness(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pytorch_model = None,
    model_name: str = "Model"
) -> Dict:
    """
    Convenience function to evaluate model robustness.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        pytorch_model: Optional Keras model
        model_name: Model name

    Returns:
        Robustness evaluation results
    """
    pipeline = Stage8Pipeline()
    return pipeline.run(model, X_test, y_test, pytorch_model, model_name)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 8: Robustness & Adversarial Testing")
    print("="*70 + "\n")

    # Import previous stages
    from stage1_data_preparation import Stage1Pipeline
    from stage3_ml_models import LightGBMModel

    # Prepare data
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data = stage1.run()

    # Train model
    model = LightGBMModel()
    model.fit(
        data['splits']['X_train'],
        data['splits']['y_train'],
        tune_hyperparams=False
    )

    # Run robustness testing
    pipeline = Stage8Pipeline()
    results = pipeline.run(
        model=model.model,
        X_test=data['splits']['X_test'],
        y_test=data['splits']['y_test'],
        model_name="LightGBM"
    )

    print("\n" + "-"*50)
    print("ROBUSTNESS SUMMARY")
    print("-"*50)
    print(f"Overall Robustness Score: {results['robustness_scores']['overall_robustness']:.4f}")
    print("-"*50)
