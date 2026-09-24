"""
================================================================================
STAGE 6: PERFORMANCE METRICS BENCHMARKING (COMPREHENSIVE)
================================================================================
This module provides:
- Effectiveness Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Per-class Performance Analysis
- Efficiency Metrics (Training time, Inference time, Throughput)
- Resource Usage (Model size, Memory usage, CPU utilization)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import logging
import json
import sys
import os
import pickle
import psutil
import tracemalloc

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_fscore_support, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, balanced_accuracy_score,
    log_loss
)
from sklearn.preprocessing import LabelBinarizer

from config import METRICS, EFFICIENCY_METRICS, RESULTS_DIR, RANDOM_SEED, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EffectivenessMetrics:
    """
    Computes effectiveness metrics for classification models.
    """

    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names
        self.metrics = {}

    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray = None
    ) -> Dict:
        """
        Compute all effectiveness metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)

        Returns:
            Dictionary with all metrics
        """
        logger.info("Computing effectiveness metrics...")

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Macro averages
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Micro averages
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)

        # G-Mean (geometric mean of per-class recall)
        _, recall_per_class, _, _ = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )
        metrics['g_mean'] = float(np.sqrt(np.prod(recall_per_class.clip(min=1e-10))))

        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba,
                        multi_class='ovr', average='weighted'
                    )
            except Exception as e:
                logger.warning(f"ROC-AUC computation failed: {e}")
                metrics['roc_auc'] = None

        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Hamming Loss
        metrics['hamming_loss'] = float(np.mean(y_pred != y_true))

        # Per-class FPR, FNR, Specificity
        cm = confusion_matrix(y_true, y_pred)
        n_cls = cm.shape[0]
        fpr_per_class, fnr_per_class, spec_per_class, mcc_per = [], [], [], []
        for i in range(n_cls):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            fpr_per_class.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
            fnr_per_class.append(fn / (fn + tp) if (fn + tp) > 0 else 0)
            spec_per_class.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mcc_per.append(float((tp*tn - fp*fn) / denom) if denom > 0 else 0.0)
        metrics['fpr_macro'] = float(np.mean(fpr_per_class))
        metrics['fpr_per_class'] = [float(f) for f in fpr_per_class]
        metrics['fnr_macro'] = float(np.mean(fnr_per_class))
        metrics['fnr_per_class'] = [float(f) for f in fnr_per_class]
        metrics['specificity_macro'] = float(np.mean(spec_per_class))
        metrics['specificity_per_class'] = [float(s) for s in spec_per_class]
        metrics['mcc_per_class'] = [float(m) for m in mcc_per]

        # PR-AUC (Precision-Recall AUC)
        if y_proba is not None:
            try:
                n_classes = len(np.unique(y_true))
                if n_classes == 2:
                    metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
                else:
                    lb = LabelBinarizer()
                    y_true_bin = lb.fit_transform(y_true)
                    if y_true_bin.shape[1] == 1:
                        y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                    metrics['pr_auc'] = average_precision_score(
                        y_true_bin, y_proba, average='weighted'
                    )
            except Exception as e:
                logger.warning(f"PR-AUC computation failed: {e}")
                metrics['pr_auc'] = None

        # Log Loss + Per-class ROC-AUC
        if y_proba is not None:
            try:
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except:
                metrics['log_loss'] = None
            try:
                from sklearn.preprocessing import label_binarize
                classes = np.arange(n_cls)
                y_bin = label_binarize(y_true, classes=classes)
                if y_bin.shape[1] == 1:
                    y_bin = np.hstack([1 - y_bin, y_bin])
                roc_auc_per = []
                for c in range(n_cls):
                    try:
                        roc_auc_per.append(roc_auc_score(y_bin[:, c], y_proba[:, c]))
                    except:
                        roc_auc_per.append(None)
                metrics['roc_auc_per_class'] = roc_auc_per
            except:
                pass

        self.metrics = metrics
        return metrics

    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Compute per-class metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            DataFrame with per-class metrics
        """
        logger.info("Computing per-class metrics...")

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )

        classes = np.unique(np.concatenate([y_true, y_pred]))

        if self.class_names is None:
            class_labels = [f"Class_{c}" for c in classes]
        else:
            class_labels = [self.class_names[c] if c < len(self.class_names) else f"Class_{c}"
                          for c in classes]

        # Per-class FNR, Specificity (TNR), MCC from confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        n_classes = cm.shape[0]
        fnr_list, spec_list, mcc_list = [], [], []
        for i in range(n_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            fnr_list.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
            spec_list.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
            denom = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mcc_list.append(float((tp*tn - fp*fn) / denom) if denom > 0 else 0.0)

        df = pd.DataFrame({
            'Class': class_labels,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'FNR': fnr_list[:len(class_labels)],
            'Specificity': spec_list[:len(class_labels)],
            'MCC': mcc_list[:len(class_labels)],
            'Support': support
        })

        return df

    def generate_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Classification report string
        """
        return classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    def compute_detection_rate(self, y_true, y_pred, class_names=None):
        """
        Compute detection rate (recall) per attack class.
        Highlights classes with detection rate < 90%.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional class name mapping

        Returns:
            DataFrame with per-class detection rates
        """
        logger.info("Computing detection rates per class...")

        from sklearn.metrics import recall_score

        classes = np.unique(np.concatenate([y_true, y_pred]))

        if class_names is None:
            class_labels = [f"Class_{c}" for c in classes]
        else:
            class_labels = [class_names[c] if c < len(class_names) else f"Class_{c}"
                          for c in classes]

        _, recall, _, support = precision_recall_fscore_support(
            y_true, y_pred, zero_division=0
        )

        df = pd.DataFrame({
            'Class': class_labels,
            'Detection_Rate': recall,
            'Support': support,
            'Alert': ['LOW DR!' if r < 0.90 else 'OK' for r in recall]
        })

        df = df.sort_values('Detection_Rate', ascending=True)

        low_dr = df[df['Detection_Rate'] < 0.90]
        if len(low_dr) > 0:
            logger.warning(f"{len(low_dr)} classes have detection rate < 90%!")

        return df


class EfficiencyMetrics:
    """
    Computes efficiency metrics including timing and resource usage.
    """

    def __init__(self):
        self.metrics = {}

    def measure_training_time(
        self,
        train_func: callable,
        *args, **kwargs
    ) -> Tuple[Any, float]:
        """
        Measure training time.

        Args:
            train_func: Training function to measure
            *args, **kwargs: Arguments for training function

        Returns:
            Tuple of (result, training_time)
        """
        start_time = time.time()
        result = train_func(*args, **kwargs)
        training_time = time.time() - start_time

        self.metrics['training_time'] = training_time
        return result, training_time

    def measure_inference_time(
        self,
        predict_func: callable,
        X: np.ndarray,
        n_runs: int = 5
    ) -> Dict:
        """
        Measure inference time statistics.

        Args:
            predict_func: Prediction function
            X: Input data
            n_runs: Number of runs for averaging

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for _ in range(n_runs):
            start_time = time.time()
            predict_func(X)
            elapsed = time.time() - start_time
            times.append(elapsed)

        total_samples = len(X) * n_runs
        total_time = sum(times)

        timing = {
            'total_time': total_time / n_runs,
            'time_per_sample_ms': (total_time / total_samples) * 1000,
            'throughput_samples_per_sec': total_samples / total_time,
            'mean_batch_time': np.mean(times),
            'std_batch_time': np.std(times)
        }

        self.metrics['inference'] = timing
        return timing

    def measure_model_size(self, model: Any) -> Dict:
        """
        Measure model size in memory.

        Args:
            model: Model object

        Returns:
            Dictionary with size information
        """
        import tempfile

        size_info = {}

        # Try to get pickled size
        try:
            with tempfile.NamedTemporaryFile(delete=False) as f:
                pickle.dump(model, f)
                size_bytes = os.path.getsize(f.name)
                os.unlink(f.name)

            size_info['size_bytes'] = size_bytes
            size_info['size_mb'] = size_bytes / (1024 * 1024)
        except:
            size_info['size_bytes'] = None
            size_info['size_mb'] = None

        # Get object size in memory
        size_info['memory_size_bytes'] = sys.getsizeof(model)

        self.metrics['model_size'] = size_info
        return size_info

    def measure_memory_usage(
        self,
        func: callable,
        *args, **kwargs
    ) -> Tuple[Any, Dict]:
        """
        Measure memory usage during function execution.

        Args:
            func: Function to measure
            *args, **kwargs: Function arguments

        Returns:
            Tuple of (result, memory_info)
        """
        tracemalloc.start()

        result = func(*args, **kwargs)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        memory_info = {
            'current_mb': current / (1024 * 1024),
            'peak_mb': peak / (1024 * 1024)
        }

        self.metrics['memory'] = memory_info
        return result, memory_info

    def measure_cpu_usage(
        self,
        func: callable,
        *args,
        interval: float = 0.1,
        **kwargs
    ) -> Tuple[Any, Dict]:
        """
        Measure CPU usage during function execution.

        Args:
            func: Function to measure
            interval: Sampling interval in seconds
            *args, **kwargs: Function arguments

        Returns:
            Tuple of (result, cpu_info)
        """
        import threading
        import queue

        cpu_readings = queue.Queue()
        stop_flag = threading.Event()

        def monitor_cpu():
            while not stop_flag.is_set():
                cpu_readings.put(psutil.cpu_percent(interval=interval))

        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()

        result = func(*args, **kwargs)

        stop_flag.set()
        monitor_thread.join()

        readings = []
        while not cpu_readings.empty():
            readings.append(cpu_readings.get())

        cpu_info = {
            'mean_cpu_percent': np.mean(readings) if readings else 0,
            'max_cpu_percent': max(readings) if readings else 0,
            'min_cpu_percent': min(readings) if readings else 0
        }

        self.metrics['cpu'] = cpu_info
        return result, cpu_info


class Stage6Pipeline:
    """
    Main pipeline for Stage 6: Performance Metrics Benchmarking.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.results = {}

    def run(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
        class_names: List[str] = None,
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run comprehensive performance benchmarking.

        Args:
            model: Model object with fit() and predict() methods
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            class_names: Optional class names for reporting

        Returns:
            Dictionary with all performance metrics
        """
        logger.info("=" * 60)
        logger.info("STAGE 6: PERFORMANCE METRICS BENCHMARKING")
        logger.info("=" * 60)

        results = {
            'model_name': model_name,
            'effectiveness': {},
            'efficiency': {},
            'per_class': None
        }

        # Efficiency metrics
        efficiency = EfficiencyMetrics()

        # Measure training time (skip if X_train is None — model already trained)
        logger.info("\n--- Measuring Training Performance ---")
        if X_train is not None and hasattr(model, 'fit'):
            def train():
                model.fit(X_train, y_train)
            _, training_time = efficiency.measure_training_time(train)
            results['efficiency']['training_time_s'] = training_time
            logger.info(f"Training time: {training_time:.2f}s")
        else:
            logger.info("Skipping re-training (using pre-trained model)")

        # Measure inference time
        logger.info("\n--- Measuring Inference Performance ---")
        if hasattr(model, 'predict'):
            inference_metrics = efficiency.measure_inference_time(
                model.predict, X_test, n_runs=3
            )
            results['efficiency']['inference_time_ms'] = inference_metrics['time_per_sample_ms']
            results['efficiency']['throughput_per_sec'] = inference_metrics['throughput_samples_per_sec']
            logger.info(f"Inference time: {inference_metrics['time_per_sample_ms']:.3f} ms/sample")
            logger.info(f"Throughput: {inference_metrics['throughput_samples_per_sec']:.0f} samples/sec")

        # Measure model size
        logger.info("\n--- Measuring Model Size ---")
        size_metrics = efficiency.measure_model_size(model)
        results['efficiency']['model_size_mb'] = size_metrics.get('size_mb')
        if size_metrics.get('size_mb'):
            logger.info(f"Model size: {size_metrics['size_mb']:.2f} MB")

        # Measure memory usage during prediction
        logger.info("\n--- Measuring Memory Usage ---")
        if hasattr(model, 'predict'):
            _, memory_metrics = efficiency.measure_memory_usage(
                model.predict, X_test[:1000]
            )
            results['efficiency']['memory_peak_mb'] = memory_metrics['peak_mb']
            logger.info(f"Peak memory: {memory_metrics['peak_mb']:.2f} MB")

        # Effectiveness metrics
        logger.info("\n--- Computing Effectiveness Metrics ---")
        effectiveness = EffectivenessMetrics(class_names)

        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_test)
            except:
                pass

        # Compute metrics
        effectiveness_metrics = effectiveness.compute_all(y_test, y_pred, y_proba)
        results['effectiveness'] = effectiveness_metrics

        logger.info(f"Accuracy: {effectiveness_metrics['accuracy']:.4f}")
        logger.info(f"Precision (weighted): {effectiveness_metrics['precision_weighted']:.4f}")
        logger.info(f"Recall (weighted): {effectiveness_metrics['recall_weighted']:.4f}")
        logger.info(f"F1 (weighted): {effectiveness_metrics['f1_weighted']:.4f}")
        if effectiveness_metrics.get('roc_auc'):
            logger.info(f"ROC-AUC: {effectiveness_metrics['roc_auc']:.4f}")
        if effectiveness_metrics.get('mcc') is not None:
            logger.info(f"MCC: {effectiveness_metrics['mcc']:.4f}")
        if effectiveness_metrics.get('cohens_kappa') is not None:
            logger.info(f"Cohen's Kappa: {effectiveness_metrics['cohens_kappa']:.4f}")
        logger.info(f"FPR (macro): {effectiveness_metrics.get('fpr_macro', 'N/A')}")
        if effectiveness_metrics.get('pr_auc') is not None:
            logger.info(f"PR-AUC: {effectiveness_metrics['pr_auc']:.4f}")

        # Log new metrics
        if effectiveness_metrics.get('hamming_loss') is not None:
            logger.info(f"Hamming Loss: {effectiveness_metrics['hamming_loss']:.6f}")
        if effectiveness_metrics.get('log_loss') is not None:
            logger.info(f"Log Loss: {effectiveness_metrics['log_loss']:.4f}")
        if effectiveness_metrics.get('fnr_macro') is not None:
            logger.info(f"FNR (macro): {effectiveness_metrics['fnr_macro']:.6f}")
        if effectiveness_metrics.get('specificity_macro') is not None:
            logger.info(f"Specificity (macro): {effectiveness_metrics['specificity_macro']:.4f}")

        # Parameter count (DL models)
        if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
            try:
                n_params = sum(p.numel() for p in model.model.parameters())
                results['efficiency']['n_parameters'] = n_params
                logger.info(f"Parameters: {n_params:,}")
            except:
                pass

        # 95% Bootstrap Confidence Intervals for key metrics
        logger.info("\n--- Computing 95% Bootstrap Confidence Intervals ---")
        ci_results = {}
        n_bootstrap = 1000
        rng = np.random.RandomState(42)
        n_test = len(y_test)
        for metric_name, metric_fn in [
            ('accuracy', lambda yt, yp: accuracy_score(yt, yp)),
            ('f1_weighted', lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0)),
            ('f1_macro', lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0)),
            ('mcc', lambda yt, yp: matthews_corrcoef(yt, yp)),
            ('balanced_accuracy', lambda yt, yp: balanced_accuracy_score(yt, yp)),
        ]:
            scores = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n_test, n_test, replace=True)
                try:
                    scores.append(metric_fn(y_test[idx], y_pred[idx]))
                except:
                    pass
            if scores:
                ci_results[metric_name] = {
                    'mean': float(np.mean(scores)),
                    'ci_lower': float(np.percentile(scores, 2.5)),
                    'ci_upper': float(np.percentile(scores, 97.5)),
                    'std': float(np.std(scores))
                }
                logger.info(f"  {metric_name}: {ci_results[metric_name]['mean']:.4f} "
                           f"[{ci_results[metric_name]['ci_lower']:.4f}, "
                           f"{ci_results[metric_name]['ci_upper']:.4f}]")
        results['confidence_intervals'] = ci_results

        # Detection Rate
        logger.info("\n--- Detection Rate per Class ---")
        detection_df = effectiveness.compute_detection_rate(y_test, y_pred, class_names)
        results['detection_rate'] = detection_df
        logger.info(detection_df.to_string(index=False))

        # Per-class metrics
        logger.info("\n--- Computing Per-Class Metrics ---")
        per_class_df = effectiveness.compute_per_class_metrics(y_test, y_pred)
        results['per_class'] = per_class_df

        logger.info("\nPer-Class Performance:")
        logger.info(per_class_df.to_string(index=False))

        # Save results
        self._save_results(results, model_name, mode)

        self.results = results

        logger.info("=" * 60)
        logger.info("STAGE 6 COMPLETE")
        logger.info("=" * 60)

        return results

    def _save_results(self, results: Dict, model_name: str, mode: str = "multiclass"):
        """Save performance results."""
        output_dir = get_results_dir(mode) / 'stage6_performance'
        output_dir.mkdir(exist_ok=True)

        # Save effectiveness metrics
        effectiveness_summary = {
            k: v for k, v in results['effectiveness'].items()
            if k != 'confusion_matrix'
        }

        with open(output_dir / f'{model_name}_effectiveness.json', 'w') as f:
            json.dump(effectiveness_summary, f, indent=2, default=str)

        # Save efficiency metrics
        with open(output_dir / f'{model_name}_efficiency.json', 'w') as f:
            json.dump(results['efficiency'], f, indent=2, default=str)

        # Save per-class metrics
        if results['per_class'] is not None:
            results['per_class'].to_csv(
                output_dir / f'{model_name}_per_class.csv', index=False
            )

        # Save confusion matrix
        if 'confusion_matrix' in results['effectiveness']:
            cm = np.array(results['effectiveness']['confusion_matrix'])
            np.save(output_dir / f'{model_name}_confusion_matrix.npy', cm)

        logger.info(f"Results saved to {output_dir}")


def benchmark_model_performance(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
    class_names: List[str] = None
) -> Dict:
    """
    Convenience function to benchmark model performance.

    Args:
        model: Model with fit/predict methods
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for reporting
        class_names: Optional class names

    Returns:
        Performance metrics dictionary
    """
    pipeline = Stage6Pipeline()
    return pipeline.run(
        model, X_train, y_train, X_test, y_test,
        model_name, class_names
    )


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 6: Performance Metrics Benchmarking")
    print("="*70 + "\n")

    # Import previous stages
    from stage1_data_preparation import Stage1Pipeline
    from stage3_ml_models import LightGBMModel

    # Prepare data
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data = stage1.run()

    # Train model
    model = LightGBMModel()

    # Run performance benchmarking
    pipeline = Stage6Pipeline()
    results = pipeline.run(
        model=model,
        X_train=data['splits']['X_train'],
        y_train=data['splits']['y_train'],
        X_test=data['splits']['X_test'],
        y_test=data['splits']['y_test'],
        model_name="LightGBM",
        class_names=list(data['label_mapping'].keys())
    )

    print("\n" + "-"*50)
    print("PERFORMANCE SUMMARY")
    print("-"*50)
    print(f"Accuracy: {results['effectiveness']['accuracy']:.4f}")
    print(f"Training Time: {results['efficiency'].get('training_time_s', 'N/A')}s")
    print(f"Inference Time: {results['efficiency'].get('inference_time_ms', 'N/A')} ms/sample")
    print("-"*50)
