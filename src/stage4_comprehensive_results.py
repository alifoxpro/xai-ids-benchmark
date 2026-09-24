"""
================================================================================
STAGE 4: COMPREHENSIVE RESULTS BENCHMARKING
================================================================================
This module provides:
- Feature Selection Method Comparison
- ML Model Comparison across datasets
- Feature Set × Model Matrix Analysis
- Performance rankings and statistical summaries
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import time
import warnings
import logging
import json
from itertools import product

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import RESULTS_DIR, RANDOM_SEED, DATASETS, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComprehensiveResultsAnalyzer:
    """
    Analyzes and compares results across feature selection methods,
    ML models, and datasets.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.results = {
            'feature_selection': {},
            'model_comparison': {},
            'cross_matrix': {},
            'rankings': {}
        }

    def benchmark_feature_selection_with_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_selection_results: Dict,
        dataset_name: str = "CIC_IoT_DIAD_2024"
    ) -> pd.DataFrame:
        """
        Benchmark different feature selection methods using all models.

        Args:
            X: Full feature array
            y: Target array
            feature_selection_results: Results from Stage 2
            dataset_name: Name of the dataset

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Benchmarking feature selection methods on {dataset_name} ({X.shape[0]:,} rows)...")

        from stage3_ml_models import (
            GRUModel, ResNet1DModel, FTTransformerModel
        )

        # Get feature sets
        feature_methods = {
            'FS_SHAP': feature_selection_results.get('best_features', {}).get('SHAP', []),
            'FS_mRMR': [f['feature'] for f in feature_selection_results.get('best_features', {}).get('mRMR', [])]
                       if isinstance(feature_selection_results.get('best_features', {}).get('mRMR'), list) else [],
            'FS_PERM': feature_selection_results.get('best_features', {}).get('PERM', []),
        }

        # Use top 30 features for each method
        for method in feature_methods:
            feature_methods[method] = feature_methods[method][:30] if feature_methods[method] else []

        results = []

        # DL model set for comparison
        models = {
            'GRU': GRUModel(self.random_state),
            'ResNet1D': ResNet1DModel(self.random_state),
            'FT_Transformer': FTTransformerModel(self.random_state)
        }

        # Create feature name to index mapping
        n_features = X.shape[1]
        all_feature_names = [f"feature_{i}" for i in range(n_features)]

        for fs_method, selected_features in feature_methods.items():
            if not selected_features:
                continue

            logger.info(f"  Testing {fs_method} with {len(selected_features)} features...")

            # Get feature indices
            try:
                feature_indices = [all_feature_names.index(f) for f in selected_features
                                  if f in all_feature_names]
            except:
                feature_indices = list(range(min(30, n_features)))

            if not feature_indices:
                feature_indices = list(range(min(30, n_features)))

            X_selected = X[:, feature_indices]

            # Index-based stratified split (memory-efficient for large arrays)
            from sklearn.model_selection import StratifiedShuffleSplit
            from sklearn.utils.class_weight import compute_class_weight
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=self.random_state)
            train_idx, test_idx = next(sss1.split(X_selected, y))
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            del train_idx, test_idx

            sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=self.random_state)
            tr_idx, val_idx = next(sss2.split(X_train, y_train))
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            del tr_idx, val_idx, X_train, y_train

            # Compute class weights for weighted loss
            cw = compute_class_weight('balanced', classes=np.unique(y_tr), y=y_tr).astype(np.float32)

            for model_name, model in models.items():
                try:
                    start_time = time.time()
                    model.fit(X_tr, y_tr, X_val, y_val, tune_hyperparams=False,
                              class_weights=cw)
                    train_time = time.time() - start_time

                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)

                    results.append({
                        'Dataset': dataset_name,
                        'Feature_Method': fs_method,
                        'N_Features': len(feature_indices),
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'Train_Time': train_time
                    })
                except Exception as e:
                    logger.warning(f"Error with {fs_method} + {model_name}: {e}")

        return pd.DataFrame(results)

    def create_feature_model_matrix(
        self,
        benchmark_results: pd.DataFrame,
        dataset_name: str = "CIC_IoT_DIAD_2024"
    ) -> pd.DataFrame:
        """
        Create Feature Set × Model performance matrix.

        Args:
            benchmark_results: Results from benchmark_feature_selection_with_models
            dataset_name: Name of the dataset

        Returns:
            Pivot table with Feature Methods as rows and Models as columns
        """
        logger.info("Creating Feature × Model matrix...")

        if benchmark_results.empty:
            return pd.DataFrame()

        # Filter by dataset
        df = benchmark_results[benchmark_results['Dataset'] == dataset_name]

        # Create pivot table
        matrix = df.pivot_table(
            values='Accuracy',
            index='Feature_Method',
            columns='Model',
            aggfunc='mean'
        )

        # Add best model column
        matrix['Best_Model'] = matrix.idxmax(axis=1)

        # Add average column
        numeric_cols = [c for c in matrix.columns if c != 'Best_Model']
        matrix['Average'] = matrix[numeric_cols].mean(axis=1)

        return matrix

    def compare_models_across_datasets(
        self,
        all_dataset_results: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Compare model performance across multiple datasets.

        Args:
            all_dataset_results: Dictionary mapping dataset names to result DataFrames

        Returns:
            DataFrame with cross-dataset comparison
        """
        logger.info("Comparing models across datasets...")

        comparison = []

        for dataset_name, results_df in all_dataset_results.items():
            if results_df.empty:
                continue

            # Get best accuracy per model across all feature methods
            best_per_model = results_df.groupby('Model')['Accuracy'].max()

            for model, accuracy in best_per_model.items():
                comparison.append({
                    'Dataset': dataset_name,
                    'Model': model,
                    'Best_Accuracy': accuracy
                })

        df = pd.DataFrame(comparison)

        if df.empty:
            return df

        # Pivot for cross-dataset view
        pivot = df.pivot_table(
            values='Best_Accuracy',
            index='Model',
            columns='Dataset',
            aggfunc='max'
        )

        # Add average and ranking
        pivot['Average'] = pivot.mean(axis=1)
        pivot['Rank'] = pivot['Average'].rank(ascending=False)

        return pivot.sort_values('Average', ascending=False)

    def generate_summary_statistics(
        self,
        benchmark_results: pd.DataFrame
    ) -> Dict:
        """
        Generate comprehensive summary statistics.

        Args:
            benchmark_results: Combined benchmark results

        Returns:
            Dictionary with summary statistics
        """
        logger.info("Generating summary statistics...")

        if benchmark_results.empty:
            return {}

        summary = {
            'overall_stats': {
                'total_experiments': len(benchmark_results),
                'datasets': benchmark_results['Dataset'].unique().tolist(),
                'feature_methods': benchmark_results['Feature_Method'].unique().tolist(),
                'models': benchmark_results['Model'].unique().tolist()
            },
            'best_combinations': {},
            'feature_method_ranking': {},
            'model_ranking': {}
        }

        # Best combination per dataset
        for dataset in benchmark_results['Dataset'].unique():
            df_dataset = benchmark_results[benchmark_results['Dataset'] == dataset]
            best_idx = df_dataset['Accuracy'].idxmax()
            best_row = df_dataset.loc[best_idx]

            summary['best_combinations'][dataset] = {
                'feature_method': best_row['Feature_Method'],
                'model': best_row['Model'],
                'accuracy': float(best_row['Accuracy']),
                'n_features': int(best_row['N_Features'])
            }

        # Feature method ranking
        fm_accuracy = benchmark_results.groupby('Feature_Method')['Accuracy'].agg(['mean', 'std', 'max'])
        fm_accuracy = fm_accuracy.sort_values('mean', ascending=False)
        summary['feature_method_ranking'] = fm_accuracy.to_dict()

        # Model ranking
        model_accuracy = benchmark_results.groupby('Model')['Accuracy'].agg(['mean', 'std', 'max'])
        model_accuracy = model_accuracy.sort_values('mean', ascending=False)
        summary['model_ranking'] = model_accuracy.to_dict()

        return summary

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_selection_results: Dict,
        dataset_name: str = "CIC_IoT_DIAD_2024",
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run comprehensive results analysis.

        Args:
            X: Feature array
            y: Target array
            feature_selection_results: Results from Stage 2
            dataset_name: Dataset name

        Returns:
            Dictionary with all analysis results
        """
        logger.info("=" * 60)
        logger.info("STAGE 4: COMPREHENSIVE RESULTS BENCHMARKING")
        logger.info("=" * 60)

        # Benchmark feature selection methods with models
        benchmark_df = self.benchmark_feature_selection_with_models(
            X, y, feature_selection_results, dataset_name
        )

        # Create feature × model matrix
        matrix = self.create_feature_model_matrix(benchmark_df, dataset_name)

        # Generate summary
        summary = self.generate_summary_statistics(benchmark_df)

        results = {
            'benchmark_results': benchmark_df,
            'feature_model_matrix': matrix,
            'summary': summary,
            'dataset': dataset_name
        }

        # Save results
        self._save_results(results, dataset_name, mode)

        self.results = results

        logger.info("=" * 60)
        logger.info("STAGE 4 COMPLETE")
        logger.info("=" * 60)

        return results

    def _save_results(self, results: Dict, dataset_name: str, mode: str = "multiclass"):
        """Save comprehensive results."""
        output_dir = get_results_dir(mode) / 'stage4_comprehensive'
        output_dir.mkdir(exist_ok=True)

        # Save benchmark results
        if not results['benchmark_results'].empty:
            results['benchmark_results'].to_csv(
                output_dir / f'{dataset_name}_benchmark_results.csv', index=False
            )

        # Save matrix
        if not results['feature_model_matrix'].empty:
            results['feature_model_matrix'].to_csv(
                output_dir / f'{dataset_name}_feature_model_matrix.csv'
            )

        # Save summary
        with open(output_dir / f'{dataset_name}_summary.json', 'w') as f:
            # Convert numpy types to Python types
            summary = results['summary']
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {output_dir}")


class ResultsVisualization:
    """
    Creates visualizations for comprehensive results.
    """

    def __init__(self):
        self.figures = {}

    def plot_feature_method_comparison(
        self,
        benchmark_df: pd.DataFrame,
        save_path: Path = None
    ):
        """
        Plot comparison of feature selection methods.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if benchmark_df.empty:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy by feature method
        fm_acc = benchmark_df.groupby('Feature_Method')['Accuracy'].mean().sort_values(ascending=False)

        ax1 = axes[0]
        bars = ax1.bar(range(len(fm_acc)), fm_acc.values, color='steelblue')
        ax1.set_xticks(range(len(fm_acc)))
        ax1.set_xticklabels(fm_acc.index, rotation=45, ha='right')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_title('Feature Selection Method Comparison')
        ax1.set_ylim([fm_acc.min() - 0.01, 1.0])

        # Add value labels
        for bar, val in zip(bars, fm_acc.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        # Boxplot by model
        ax2 = axes[1]
        benchmark_df.boxplot(column='Accuracy', by='Model', ax=ax2)
        ax2.set_title('Accuracy Distribution by Model')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('Accuracy')
        plt.suptitle('')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.figures['feature_comparison'] = fig
        return fig

    def plot_heatmap_matrix(
        self,
        matrix: pd.DataFrame,
        save_path: Path = None
    ):
        """
        Plot heatmap of Feature × Model matrix.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        if matrix.empty:
            return

        # Remove non-numeric columns
        numeric_matrix = matrix.select_dtypes(include=[np.number])

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.heatmap(
            numeric_matrix,
            annot=True,
            fmt='.4f',
            cmap='RdYlGn',
            center=numeric_matrix.mean().mean(),
            ax=ax
        )

        ax.set_title('Feature Selection × Model Performance Matrix')
        ax.set_xlabel('Model / Metric')
        ax.set_ylabel('Feature Selection Method')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        self.figures['heatmap'] = fig
        return fig

    def create_summary_table(
        self,
        summary: Dict
    ) -> str:
        """
        Create formatted summary table as string.
        """
        if not summary:
            return "No summary data available."

        lines = []
        lines.append("=" * 60)
        lines.append("COMPREHENSIVE BENCHMARKING SUMMARY")
        lines.append("=" * 60)

        # Overall stats
        overall = summary.get('overall_stats', {})
        lines.append(f"\nTotal Experiments: {overall.get('total_experiments', 0)}")
        lines.append(f"Datasets: {', '.join(overall.get('datasets', []))}")
        lines.append(f"Feature Methods: {', '.join(overall.get('feature_methods', []))}")
        lines.append(f"Models: {', '.join(overall.get('models', []))}")

        # Best combinations
        lines.append("\n" + "-" * 40)
        lines.append("BEST COMBINATIONS PER DATASET")
        lines.append("-" * 40)

        for dataset, combo in summary.get('best_combinations', {}).items():
            lines.append(f"\n{dataset}:")
            lines.append(f"  Feature Method: {combo.get('feature_method')}")
            lines.append(f"  Model: {combo.get('model')}")
            lines.append(f"  Accuracy: {combo.get('accuracy', 0):.4f}")

        # Rankings
        lines.append("\n" + "-" * 40)
        lines.append("FEATURE METHOD RANKING (by mean accuracy)")
        lines.append("-" * 40)

        fm_ranking = summary.get('feature_method_ranking', {}).get('mean', {})
        for rank, (method, acc) in enumerate(sorted(fm_ranking.items(), key=lambda x: -x[1]), 1):
            lines.append(f"  {rank}. {method}: {acc:.4f}")

        lines.append("\n" + "-" * 40)
        lines.append("MODEL RANKING (by mean accuracy)")
        lines.append("-" * 40)

        model_ranking = summary.get('model_ranking', {}).get('mean', {})
        for rank, (model, acc) in enumerate(sorted(model_ranking.items(), key=lambda x: -x[1]), 1):
            lines.append(f"  {rank}. {model}: {acc:.4f}")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)


def run_comprehensive_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    feature_selection_results: Dict,
    dataset_name: str = "CIC_IoT_DIAD_2024",
    mode: str = "multiclass"
) -> Dict:
    """
    Run comprehensive benchmarking pipeline.

    Args:
        X: Feature array
        y: Target array
        feature_selection_results: Results from Stage 2
        dataset_name: Dataset name

    Returns:
        Dictionary with all results
    """
    analyzer = ComprehensiveResultsAnalyzer()
    results = analyzer.run(X, y, feature_selection_results, dataset_name, mode)

    # Create visualizations
    viz = ResultsVisualization()

    vis_dir = get_results_dir(mode) / 'stage4_comprehensive' / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    viz.plot_feature_method_comparison(
        results['benchmark_results'],
        save_path=vis_dir / f'{dataset_name}_feature_comparison.png'
    )

    viz.plot_heatmap_matrix(
        results['feature_model_matrix'],
        save_path=vis_dir / f'{dataset_name}_heatmap.png'
    )

    # Print summary
    summary_text = viz.create_summary_table(results['summary'])
    print(summary_text)

    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 4: Comprehensive Results Benchmarking")
    print("="*70 + "\n")

    # Import previous stages
    from stage1_data_preparation import Stage1Pipeline
    from stage2_feature_selection import Stage2Pipeline

    # Prepare data
    stage1 = Stage1Pipeline("CIC_IoT_DIAD_2024")
    data = stage1.run()

    # Feature selection
    stage2 = Stage2Pipeline()
    fs_results = stage2.run(
        X=data['splits']['X_train'],
        y=data['splits']['y_train'],
        feature_names=data['feature_columns']
    )

    # Run comprehensive benchmark
    results = run_comprehensive_benchmark(
        X=data['X'],
        y=data['y'],
        feature_selection_results=fs_results,
        dataset_name="CIC_IoT_DIAD_2024"
    )

    print("\n" + "-"*50)
    print("STAGE 4 COMPLETE")
    print("-"*50)


# ============================================================================
# ABLATION STUDY — Feature Count Impact
# ============================================================================

def run_ablation_study(X, y, feature_names, best_features_ranked,
                       model_class=None, k_values=None,
                       mode='multiclass', class_weights=None) -> pd.DataFrame:
    """
    Ablation study: measure accuracy with increasing feature counts.

    Args:
        X:                    Full feature array (n_samples × n_features)
        y:                    Target array
        feature_names:        List of all feature names (matches X columns)
        best_features_ranked: Ordered list of feature names (most important first)
        model_class:          A BaseModel subclass to use (default: GRUModel)
        k_values:             Number of features to test (default: [10, 20, 30, 50, ALL])
        mode:                 'binary' or 'multiclass'
        class_weights:        Optional class weights

    Returns:
        DataFrame with columns: n_features, accuracy, f1, balanced_accuracy, training_time
    """
    import gc
    from sklearn.model_selection import StratifiedShuffleSplit

    if model_class is None:
        from stage3_ml_models import GRUModel
        model_class = GRUModel

    # Replace None with actual feature count
    if k_values is None:
        k_values = [10, 20, 30, 50, len(feature_names)]
    else:
        k_values = [k if k is not None else len(feature_names) for k in k_values]

    # Build feature name -> column index mapping
    feat_to_idx = {name: i for i, name in enumerate(feature_names)}

    # Subsample large datasets to prevent OOM (1M rows is enough for ablation)
    MAX_ABLATION_SAMPLES = 1_000_000
    if X.shape[0] > MAX_ABLATION_SAMPLES:
        rng = np.random.RandomState(RANDOM_SEED)
        sub_idx = rng.choice(X.shape[0], MAX_ABLATION_SAMPLES, replace=False)
        X = X[sub_idx]
        y = y[sub_idx]
        logger.info(f"Ablation: subsampled to {MAX_ABLATION_SAMPLES:,} rows for memory efficiency")

    # Index-based stratified split (memory-efficient for large arrays)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_SEED)
    train_idx, temp_idx = next(sss1.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]
    del train_idx, temp_idx

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]
    del val_idx, test_idx, X_temp, y_temp
    gc.collect()

    rows = []
    for k in k_values:
        k = min(k, len(feature_names))
        # Select top-k features
        selected = best_features_ranked[:k]
        col_idx = [feat_to_idx[f] for f in selected if f in feat_to_idx]
        if not col_idx:
            continue

        Xtr = X_train[:, col_idx]
        Xv = X_val[:, col_idx]
        Xte = X_test[:, col_idx]

        logger.info(f"Ablation: k={k} features — training {model_class.__name__}...")
        try:
            model = model_class()
            t0 = time.time()
            model.fit(Xtr, y_train, Xv, y_val, tune_hyperparams=False,
                      class_weights=class_weights)
            train_time = time.time() - t0
            metrics = model.evaluate(Xte, y_test)

            rows.append({
                'n_features': k,
                'accuracy': metrics['accuracy'],
                'balanced_accuracy': metrics.get('balanced_accuracy', 0),
                'f1': metrics['f1'],
                'f1_macro': metrics.get('f1_macro', 0),
                'training_time': train_time
            })
            logger.info(f"  k={k}: acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        except Exception as e:
            logger.warning(f"Ablation k={k} failed: {e}")
        finally:
            # Free model memory between iterations
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    df = pd.DataFrame(rows)
    if len(df) > 0:
        output_dir = get_results_dir(mode) / 'stage4_ablation'
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / 'ablation_results.csv', index=False)
        logger.info(f"Ablation results saved → {output_dir}")

    return df
