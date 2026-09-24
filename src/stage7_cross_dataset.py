"""
================================================================================
STAGE 7: CROSS-DATASET GENERALIZATION TEST
================================================================================
This module provides:
- Transfer Learning evaluation
- Train on Dataset A, Test on Dataset B
- Generalization performance analysis
- Domain adaptation assessment
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import logging
import json
from itertools import permutations

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import DATASETS, RESULTS_DIR, RANDOM_SEED, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CrossDatasetEvaluator:
    """
    Evaluates model generalization across different datasets.
    Tests transfer learning capabilities.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.results = {}

    def prepare_compatible_features(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray,
        source_features: List[str],
        target_features: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare compatible feature sets between source and target datasets.

        Args:
            X_source: Source dataset features
            X_target: Target dataset features
            source_features: Source feature names
            target_features: Target feature names

        Returns:
            Tuple of (aligned X_source, aligned X_target, common features)
        """
        logger.info("Aligning features between datasets...")

        # Find common features
        common_features = list(set(source_features) & set(target_features))

        if not common_features:
            # If no common feature names, use minimum features
            min_features = min(len(source_features), len(target_features))
            logger.warning(f"No common features found. Using first {min_features} features.")

            X_source_aligned = X_source[:, :min_features]
            X_target_aligned = X_target[:, :min_features]
            common_features = [f"feature_{i}" for i in range(min_features)]
        else:
            # Get indices for common features
            source_indices = [source_features.index(f) for f in common_features if f in source_features]
            target_indices = [target_features.index(f) for f in common_features if f in target_features]

            X_source_aligned = X_source[:, source_indices]
            X_target_aligned = X_target[:, target_indices]

        logger.info(f"Common features: {len(common_features)}")

        return X_source_aligned, X_target_aligned, common_features

    def align_labels(
        self,
        y_source: np.ndarray,
        y_target: np.ndarray,
        source_mapping: Dict,
        target_mapping: Dict
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Align labels between source and target datasets.

        Args:
            y_source: Source labels
            y_target: Target labels
            source_mapping: Source label mapping
            target_mapping: Target label mapping

        Returns:
            Tuple of (aligned y_source, aligned y_target, common mapping)
        """
        logger.info("Aligning labels between datasets...")

        # Create binary classification (Benign vs Attack) for cross-dataset
        # This is more generalizable across different datasets

        # Find benign label in each dataset
        source_benign = None
        target_benign = None

        for label, idx in source_mapping.items():
            if 'benign' in label.lower() or 'normal' in label.lower():
                source_benign = idx
                break

        for label, idx in target_mapping.items():
            if 'benign' in label.lower() or 'normal' in label.lower():
                target_benign = idx
                break

        # Binary classification
        if source_benign is not None:
            y_source_binary = (y_source != source_benign).astype(int)
        else:
            y_source_binary = (y_source > 0).astype(int)

        if target_benign is not None:
            y_target_binary = (y_target != target_benign).astype(int)
        else:
            y_target_binary = (y_target > 0).astype(int)

        common_mapping = {0: 'Benign', 1: 'Attack'}

        logger.info("Labels aligned to binary classification (Benign vs Attack)")

        return y_source_binary, y_target_binary, common_mapping

    def evaluate_transfer(
        self,
        model: Any,
        X_source_train: np.ndarray,
        y_source_train: np.ndarray,
        X_source_test: np.ndarray,
        y_source_test: np.ndarray,
        X_target_test: np.ndarray,
        y_target_test: np.ndarray,
        source_name: str = "Source",
        target_name: str = "Target"
    ) -> Dict:
        """
        Evaluate model transfer from source to target dataset.

        Args:
            model: Model to evaluate
            X_source_train: Source training features
            y_source_train: Source training labels
            X_source_test: Source test features
            y_source_test: Source test labels
            X_target_test: Target test features
            y_target_test: Target test labels
            source_name: Name of source dataset
            target_name: Name of target dataset

        Returns:
            Dictionary with transfer evaluation results
        """
        logger.info(f"\nEvaluating transfer: {source_name} -> {target_name}")

        # Standardize features
        scaler = StandardScaler()
        X_source_train_scaled = scaler.fit_transform(X_source_train)
        X_source_test_scaled = scaler.transform(X_source_test)
        X_target_test_scaled = scaler.transform(X_target_test)

        # Train on source
        logger.info(f"Training on {source_name}...")
        start_time = time.time()

        if hasattr(model, 'fit'):
            model.fit(X_source_train_scaled, y_source_train)
        training_time = time.time() - start_time

        # Test on source (same dataset)
        logger.info(f"Testing on {source_name} (same dataset)...")
        y_pred_source = model.predict(X_source_test_scaled)
        source_accuracy = accuracy_score(y_source_test, y_pred_source)
        source_f1 = f1_score(y_source_test, y_pred_source, average='weighted', zero_division=0)

        # Test on target (different dataset)
        logger.info(f"Testing on {target_name} (different dataset)...")
        y_pred_target = model.predict(X_target_test_scaled)
        target_accuracy = accuracy_score(y_target_test, y_pred_target)
        target_f1 = f1_score(y_target_test, y_pred_target, average='weighted', zero_division=0)

        # Compute generalization gap
        accuracy_gap = source_accuracy - target_accuracy
        f1_gap = source_f1 - target_f1

        results = {
            'source_dataset': source_name,
            'target_dataset': target_name,
            'training_time': training_time,
            'source_accuracy': source_accuracy,
            'source_f1': source_f1,
            'target_accuracy': target_accuracy,
            'target_f1': target_f1,
            'accuracy_gap': accuracy_gap,
            'f1_gap': f1_gap,
            'generalization_ratio': target_accuracy / source_accuracy if source_accuracy > 0 else 0
        }

        logger.info(f"Source accuracy: {source_accuracy:.4f}")
        logger.info(f"Target accuracy: {target_accuracy:.4f}")
        logger.info(f"Generalization gap: {accuracy_gap:.4f}")

        return results


class DomainAdaptation:
    """
    Implements basic domain adaptation techniques for cross-dataset evaluation.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def compute_domain_distance(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray
    ) -> float:
        """
        Compute distance between source and target domains.
        Uses Maximum Mean Discrepancy (MMD) approximation.

        Args:
            X_source: Source domain features
            X_target: Target domain features

        Returns:
            Domain distance score
        """
        logger.info("Computing domain distance...")

        # Use subset for efficiency
        n_samples = min(1000, len(X_source), len(X_target))
        idx_source = np.random.choice(len(X_source), n_samples, replace=False)
        idx_target = np.random.choice(len(X_target), n_samples, replace=False)

        X_s = X_source[idx_source]
        X_t = X_target[idx_target]

        # Simple MMD approximation using mean difference
        mean_source = X_s.mean(axis=0)
        mean_target = X_t.mean(axis=0)

        mmd = np.sqrt(np.sum((mean_source - mean_target) ** 2))

        logger.info(f"Domain distance (MMD approximation): {mmd:.4f}")

        return mmd

    def importance_weighted_adaptation(
        self,
        X_source: np.ndarray,
        X_target: np.ndarray
    ) -> np.ndarray:
        """
        Compute importance weights for source samples.

        Args:
            X_source: Source features
            X_target: Target features

        Returns:
            Importance weights for source samples
        """
        from sklearn.linear_model import LogisticRegression

        logger.info("Computing importance weights...")

        # Create domain classification problem
        X_combined = np.vstack([X_source, X_target])
        y_domain = np.concatenate([
            np.zeros(len(X_source)),
            np.ones(len(X_target))
        ])

        # Train domain classifier
        clf = LogisticRegression(random_state=self.random_state, max_iter=1000)
        clf.fit(X_combined, y_domain)

        # Get probabilities for source samples
        proba = clf.predict_proba(X_source)[:, 1]

        # Compute importance weights
        weights = proba / (1 - proba + 1e-8)
        weights = weights / weights.sum() * len(weights)  # Normalize

        return weights


class Stage7Pipeline:
    """
    Main pipeline for Stage 7: Cross-Dataset Generalization Test.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.evaluator = CrossDatasetEvaluator(random_state)
        self.adapter = DomainAdaptation(random_state)
        self.results = {}

    def run(
        self,
        datasets: Dict[str, Dict],
        models: Dict[str, Any],
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run cross-dataset evaluation.

        Args:
            datasets: Dictionary of {name: {X, y, features, mapping}}
            models: Dictionary of {name: model_instance}

        Returns:
            Dictionary with all cross-dataset results
        """
        logger.info("=" * 60)
        logger.info("STAGE 7: CROSS-DATASET GENERALIZATION TEST")
        logger.info("=" * 60)

        results = {
            'transfer_results': [],
            'domain_distances': {},
            'model_rankings': {}
        }

        dataset_names = list(datasets.keys())

        # Compute domain distances
        logger.info("\n--- Computing Domain Distances ---")
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                # Align features first
                X1_aligned, X2_aligned, _ = self.evaluator.prepare_compatible_features(
                    datasets[name1]['X'],
                    datasets[name2]['X'],
                    datasets[name1].get('features', []),
                    datasets[name2].get('features', [])
                )

                distance = self.adapter.compute_domain_distance(X1_aligned, X2_aligned)
                results['domain_distances'][f"{name1}_to_{name2}"] = distance

        # Evaluate each model on each transfer pair
        logger.info("\n--- Evaluating Transfer Learning ---")

        for source_name in dataset_names:
            for target_name in dataset_names:
                if source_name == target_name:
                    continue

                logger.info(f"\n{'='*40}")
                logger.info(f"Transfer: {source_name} -> {target_name}")
                logger.info(f"{'='*40}")

                source_data = datasets[source_name]
                target_data = datasets[target_name]

                # Align features
                X_source_aligned, X_target_aligned, common_features = \
                    self.evaluator.prepare_compatible_features(
                        source_data['X'],
                        target_data['X'],
                        source_data.get('features', []),
                        target_data.get('features', [])
                    )

                # Align labels
                y_source_aligned, y_target_aligned, _ = \
                    self.evaluator.align_labels(
                        source_data['y'],
                        target_data['y'],
                        source_data.get('mapping', {}),
                        target_data.get('mapping', {})
                    )

                # Split source data
                from sklearn.model_selection import train_test_split
                X_source_train, X_source_test, y_source_train, y_source_test = \
                    train_test_split(
                        X_source_aligned, y_source_aligned,
                        test_size=0.2, random_state=self.random_state,
                        stratify=y_source_aligned
                    )

                # Evaluate each model
                for model_name, model in models.items():
                    try:
                        transfer_result = self.evaluator.evaluate_transfer(
                            model,
                            X_source_train, y_source_train,
                            X_source_test, y_source_test,
                            X_target_aligned, y_target_aligned,
                            source_name, target_name
                        )
                        transfer_result['model'] = model_name
                        results['transfer_results'].append(transfer_result)
                    except Exception as e:
                        logger.error(f"Error with {model_name}: {e}")

        # Create model rankings
        results['model_rankings'] = self._create_rankings(results['transfer_results'])

        # Save results
        self._save_results(results, mode)

        self.results = results

        logger.info("=" * 60)
        logger.info("STAGE 7 COMPLETE")
        logger.info("=" * 60)

        return results

    def _create_rankings(self, transfer_results: List[Dict]) -> pd.DataFrame:
        """Create model rankings based on transfer performance."""
        if not transfer_results:
            return pd.DataFrame()

        df = pd.DataFrame(transfer_results)

        # Aggregate by model
        rankings = df.groupby('model').agg({
            'source_accuracy': 'mean',
            'target_accuracy': 'mean',
            'generalization_ratio': 'mean',
            'accuracy_gap': 'mean'
        }).round(4)

        rankings = rankings.sort_values('target_accuracy', ascending=False)
        rankings['rank'] = range(1, len(rankings) + 1)

        return rankings

    def _save_results(self, results: Dict, mode: str = "multiclass"):
        """Save cross-dataset results."""
        output_dir = get_results_dir(mode) / 'stage7_cross_dataset'
        output_dir.mkdir(exist_ok=True)

        # Save transfer results
        if results['transfer_results']:
            df = pd.DataFrame(results['transfer_results'])
            df.to_csv(output_dir / 'transfer_results.csv', index=False)

        # Save domain distances (convert numpy types to native Python for JSON)
        domain_dist_clean = {
            k: float(v) if hasattr(v, 'item') else v
            for k, v in results['domain_distances'].items()
        }
        with open(output_dir / 'domain_distances.json', 'w') as f:
            json.dump(domain_dist_clean, f, indent=2)

        # Save rankings
        if isinstance(results['model_rankings'], pd.DataFrame):
            results['model_rankings'].to_csv(output_dir / 'model_rankings.csv')

        logger.info(f"Results saved to {output_dir}")


def run_cross_dataset_evaluation(
    datasets: Dict[str, Dict],
    models: Dict[str, Any]
) -> Dict:
    """
    Convenience function to run cross-dataset evaluation.

    Args:
        datasets: Dictionary of dataset information
        models: Dictionary of models to evaluate

    Returns:
        Cross-dataset evaluation results
    """
    pipeline = Stage7Pipeline()
    return pipeline.run(datasets, models)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 7: Cross-Dataset Generalization Test")
    print("="*70 + "\n")

    # Import previous stages
    from stage1_data_preparation import Stage1Pipeline
    from stage3_ml_models import RandomForestModel, LightGBMModel, XGBoostModel

    # Prepare primary dataset
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data1 = stage1.run()

    # Create a second synthetic dataset for demonstration
    stage1_2 = Stage1Pipeline("CICIDS2017")
    data2 = stage1_2.run()

    # Prepare datasets dictionary
    datasets = {
        "CIC_IoT_DIAD": {
            'X': data1['X'],
            'y': data1['y'],
            'features': data1['feature_columns'],
            'mapping': data1['label_mapping']
        },
        "CICIDS2017": {
            'X': data2['X'],
            'y': data2['y'],
            'features': data2['feature_columns'],
            'mapping': data2['label_mapping']
        }
    }

    # Prepare models
    models = {
        'RandomForest': RandomForestModel(),
        'LightGBM': LightGBMModel(),
        'XGBoost': XGBoostModel()
    }

    # Run cross-dataset evaluation
    pipeline = Stage7Pipeline()
    results = pipeline.run(datasets, models)

    # Print summary
    print("\n" + "-"*50)
    print("MODEL GENERALIZATION RANKINGS")
    print("-"*50)
    print(results['model_rankings'].to_string())
    print("-"*50)
