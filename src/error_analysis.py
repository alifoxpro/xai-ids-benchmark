"""
================================================================================
ERROR ANALYSIS MODULE
================================================================================
Analyzes misclassified samples to understand model weaknesses:
- Confusion pair analysis (which classes are confused?)
- Per-class error rate
- Feature distribution comparison (correct vs misclassified)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

from sklearn.metrics import confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Analyze misclassifications for a single model."""

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names

    def analyze(self, y_true, y_pred, X_test=None,
                feature_names=None) -> Dict:
        """
        Full error analysis.

        Args:
            y_true:  ground-truth labels (1-D array)
            y_pred:  predicted labels (1-D array)
            X_test:  feature array (optional, for feature distribution analysis)
            feature_names: list of feature names

        Returns:
            Dictionary with error analysis results
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]

        if self.class_names is None:
            self.class_names = [str(i) for i in range(n_classes)]

        results = {}

        # 1. Per-class error rate
        results['per_class_error'] = self._per_class_error(cm)

        # 2. Confusion pairs (top-K most confused pairs)
        results['confusion_pairs'] = self._confusion_pairs(cm, top_k=15)

        # 3. Overall error statistics
        mask_wrong = y_true != y_pred
        results['total_samples'] = int(len(y_true))
        results['total_errors'] = int(mask_wrong.sum())
        results['error_rate'] = float(mask_wrong.mean())

        # 4. Feature distribution comparison (if X provided)
        if X_test is not None:
            results['feature_analysis'] = self._feature_analysis(
                X_test, y_true, y_pred, feature_names, top_k=10)

        logger.info(f"Error Analysis: {results['total_errors']:,}/{results['total_samples']:,} "
                     f"misclassified ({results['error_rate']*100:.2f}%)")
        return results

    def _per_class_error(self, cm) -> List[Dict]:
        """Compute error rate per class."""
        rows = []
        for i, cls_name in enumerate(self.class_names):
            support = int(cm[i].sum())
            correct = int(cm[i, i])
            errors = support - correct
            error_rate = errors / max(support, 1)
            rows.append({
                'Class': cls_name,
                'Support': support,
                'Correct': correct,
                'Errors': errors,
                'Error_Rate': round(error_rate, 4)
            })
        rows.sort(key=lambda r: r['Error_Rate'], reverse=True)
        return rows

    def _confusion_pairs(self, cm, top_k=15) -> List[Dict]:
        """Find the most confused class pairs (off-diagonal max)."""
        pairs = []
        n = cm.shape[0]
        for i in range(n):
            for j in range(n):
                if i != j and cm[i, j] > 0:
                    pairs.append({
                        'True_Class': self.class_names[i],
                        'Predicted_Class': self.class_names[j],
                        'Count': int(cm[i, j]),
                        'Percentage': round(cm[i, j] / max(cm[i].sum(), 1) * 100, 2)
                    })
        pairs.sort(key=lambda p: p['Count'], reverse=True)
        return pairs[:top_k]

    def _feature_analysis(self, X, y_true, y_pred, feature_names, top_k=10) -> Dict:
        """Compare feature distributions between correct and misclassified samples."""
        mask_correct = y_true == y_pred
        mask_wrong = ~mask_correct

        if mask_wrong.sum() == 0 or mask_correct.sum() == 0:
            return {}

        if feature_names is None:
            feature_names = [f'f_{i}' for i in range(X.shape[1])]

        # Mean difference between correct and wrong predictions
        mean_correct = np.mean(X[mask_correct], axis=0)
        mean_wrong = np.mean(X[mask_wrong], axis=0)
        diff = np.abs(mean_wrong - mean_correct)

        top_indices = np.argsort(diff)[::-1][:top_k]
        top_features = []
        for idx in top_indices:
            top_features.append({
                'feature': feature_names[idx],
                'mean_correct': float(mean_correct[idx]),
                'mean_wrong': float(mean_wrong[idx]),
                'abs_diff': float(diff[idx])
            })

        return {
            'n_correct': int(mask_correct.sum()),
            'n_wrong': int(mask_wrong.sum()),
            'top_discriminative_features': top_features
        }

    def save_results(self, results: Dict, output_dir: Path, model_name: str = 'model'):
        """Save error analysis results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full results as JSON
        with open(output_dir / f'error_analysis_{model_name}.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save per-class errors as CSV
        if 'per_class_error' in results:
            pd.DataFrame(results['per_class_error']).to_csv(
                output_dir / f'per_class_errors_{model_name}.csv', index=False)

        # Save confusion pairs as CSV
        if 'confusion_pairs' in results:
            pd.DataFrame(results['confusion_pairs']).to_csv(
                output_dir / f'confusion_pairs_{model_name}.csv', index=False)

        logger.info(f"Error analysis saved → {output_dir}")


def run_error_analysis_for_stage3(stage3_results: Dict, stage1_results: Dict,
                                   mode: str = 'multiclass') -> Dict:
    """
    Run error analysis for the best model from Stage 3.

    Args:
        stage3_results: results from Stage3Pipeline.run()
        stage1_results: results from Stage1Pipeline.run()
        mode: classification mode

    Returns:
        Dictionary with error analysis results
    """
    from config import get_results_dir

    label_mapping = stage1_results.get('label_mapping', {})
    class_names = list(label_mapping.keys()) if label_mapping else None
    X_test = stage1_results['splits']['X_test']
    y_test = stage1_results['splits']['y_test']
    feature_names = stage1_results.get('feature_columns', None)

    all_error_results = {}
    output_dir = get_results_dir(mode) / 'error_analysis'

    # Analyze best model
    best_name = stage3_results.get('best_model')
    models = stage3_results.get('models', {})

    for model_name in [best_name] if best_name else []:
        if model_name and model_name in models and 'error' not in models[model_name]:
            model_obj = models[model_name].get('model_object')
            if model_obj is not None:
                try:
                    if hasattr(model_obj, 'predict'):
                        y_pred = model_obj.predict(X_test)
                    else:
                        continue

                    analyzer = ErrorAnalyzer(class_names=class_names)
                    results = analyzer.analyze(y_test, y_pred, X_test, feature_names)
                    analyzer.save_results(results, output_dir, model_name)
                    all_error_results[model_name] = results
                except Exception as e:
                    logger.warning(f"Error analysis failed for {model_name}: {e}")

    return all_error_results
