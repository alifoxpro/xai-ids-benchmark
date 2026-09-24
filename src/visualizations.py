"""
================================================================================
COMPREHENSIVE VISUALIZATION MODULE
================================================================================
Provides academic-grade visualizations for the XAI-IDS benchmarking study:
- SHAP summary plots
- ROC / PR curves
- Confusion matrix heatmaps
- Feature importance comparison
- Robustness degradation charts
- Cross-dataset generalization charts
- Critical Difference diagrams (Nemenyi)
- Radar charts for multi-metric comparison
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
import logging

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from config import VIS_DIR, RESULTS_DIR

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Academic publication style settings
ACADEMIC_STYLE = {
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
}
plt.rcParams.update(ACADEMIC_STYLE)

# Consistent color palette
METRIC_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def _create_figure(figsize=(10, 6)):
    """Create a single academic-style figure."""
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save_and_close(fig, save_path, dpi=300):
    """Save figure and close to free memory."""
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved: {save_path.name}")
    plt.close(fig)
    return fig


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray,
                      feature_names: List[str] = None,
                      top_k: int = 20,
                      save_path: Path = None) -> plt.Figure:
    """
    Create SHAP beeswarm/summary plot.

    Args:
        shap_values: SHAP values array (n_samples x n_features)
        X: Feature matrix
        feature_names: Feature names
        top_k: Number of top features to show
        save_path: Path to save figure
    """
    try:
        import shap
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            max_display=top_k,
            show=False
        )
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("SHAP summary plot created")
        return fig
    except Exception as e:
        logger.warning(f"SHAP summary plot failed: {e}")
        return _plot_shap_bar_fallback(shap_values, feature_names, top_k, save_path)


def _plot_shap_bar_fallback(shap_values, feature_names, top_k, save_path):
    """Fallback bar chart when shap.summary_plot fails."""
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    if feature_names is None:
        feature_names = [f"f_{i}" for i in range(len(mean_abs))]

    indices = np.argsort(mean_abs)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = range(len(indices))
    ax.barh(y_pos, mean_abs[indices], color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'Top {top_k} Features by SHAP Importance')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_roc_curves(model_results: Dict[str, Dict],
                    save_path: Path = None) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        model_results: {model_name: {'fpr': array, 'tpr': array, 'auc': float}}
        save_path: Path to save figure
    """
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

    for (name, data), color in zip(model_results.items(), colors):
        if 'fpr' in data and 'tpr' in data:
            fpr, tpr = data['fpr'], data['tpr']
            roc_auc = data.get('auc', auc(fpr, tpr))
        elif 'y_true' in data and 'y_proba' in data:
            fpr, tpr, _ = roc_curve(data['y_true'], data['y_proba'])
            roc_auc = auc(fpr, tpr)
        else:
            continue

        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{name} (AUC = {roc_auc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("ROC curves plot created")
    return fig


def plot_pr_curves(model_results: Dict[str, Dict],
                   save_path: Path = None) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.

    Args:
        model_results: {model_name: {'precision': array, 'recall': array, 'ap': float}}
        save_path: Path to save figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_results)))

    for (name, data), color in zip(model_results.items(), colors):
        if 'precision' in data and 'recall' in data:
            prec, rec = data['precision'], data['recall']
            ap = data.get('ap', 0)
        elif 'y_true' in data and 'y_proba' in data:
            prec, rec, _ = precision_recall_curve(data['y_true'], data['y_proba'])
            ap = average_precision_score(data['y_true'], data['y_proba'])
        else:
            continue

        ax.plot(rec, prec, color=color, lw=2,
                label=f'{name} (AP = {ap:.4f})')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves Comparison')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("PR curves plot created")
    return fig


def plot_confusion_matrix_heatmap(cm: np.ndarray,
                                   class_names: List[str] = None,
                                   model_name: str = "Model",
                                   normalize: bool = True,
                                   save_path: Path = None) -> plt.Figure:
    """
    Plot confusion matrix as a heatmap.

    Args:
        cm: Confusion matrix array
        class_names: Class label names
        model_name: Model name for title
        normalize: Whether to normalize the matrix
        save_path: Path to save figure
    """
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_display = cm_norm
    else:
        cm_display = cm

    n_classes = cm.shape[0]

    # --- Adaptive sizing for readability ---
    figsize = max(8, n_classes * 1.3)
    annot_fontsize = max(7, 14 - n_classes) if n_classes >= 6 else 12
    label_fontsize = max(7, 12 - n_classes // 2)

    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.9))

    # Build custom annotation text: percentage + raw count
    if normalize:
        annot_text = np.empty_like(cm_display, dtype=object)
        for i in range(n_classes):
            for j in range(n_classes):
                pct = cm_display[i, j] * 100
                raw = cm[i, j]
                if pct >= 1:
                    annot_text[i, j] = f"{pct:.1f}%\n({raw:,})"
                elif pct > 0:
                    annot_text[i, j] = f"{pct:.2f}%\n({raw:,})"
                else:
                    annot_text[i, j] = "0"
    else:
        annot_text = True

    sns.heatmap(
        cm_display, annot=annot_text, fmt='' if normalize else 'd',
        cmap='Blues', xticklabels=class_names, yticklabels=class_names,
        ax=ax, square=True, linewidths=1, linecolor='white',
        cbar_kws={'shrink': 0.75, 'label': 'Proportion' if normalize else 'Count'},
        annot_kws={'fontsize': annot_fontsize}
    )

    # Readable tick labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=label_fontsize)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=label_fontsize)
    ax.set_xlabel('Predicted Label', fontsize=label_fontsize + 2, labelpad=10)
    ax.set_ylabel('True Label', fontsize=label_fontsize + 2, labelpad=10)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=label_fontsize + 4, pad=15)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"Confusion matrix heatmap created for {model_name}")
    return fig


def plot_feature_importance_comparison(
        importance_dict: Dict[str, pd.DataFrame],
        top_k: int = 15,
        save_path: Path = None) -> plt.Figure:
    """
    Side-by-side bar charts comparing feature importance across methods.

    Args:
        importance_dict: {method_name: DataFrame with 'feature' and 'importance' columns}
        top_k: Number of top features to show per method
        save_path: Path to save figure
    """
    n_methods = len(importance_dict)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8))
    if n_methods == 1:
        axes = [axes]

    colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod']

    for idx, (method, df) in enumerate(importance_dict.items()):
        ax = axes[idx]
        df_top = df.head(top_k)

        y_pos = range(len(df_top))
        ax.barh(y_pos, df_top['importance'], color=colors[idx % len(colors)])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_top['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'{method}')

    plt.suptitle(f'Top {top_k} Features by Selection Method', fontsize=14, y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Feature importance comparison plot created")
    return fig


def plot_robustness_degradation(robustness_df: pd.DataFrame,
                                 x_col: str = 'parameter',
                                 y_col: str = 'adversarial_accuracy',
                                 group_col: str = 'attack',
                                 save_path: Path = None) -> plt.Figure:
    """
    Plot accuracy degradation under adversarial attacks.

    Args:
        robustness_df: DataFrame with attack results
        x_col: Column for x-axis (parameter values)
        y_col: Column for y-axis (accuracy)
        group_col: Column to group by (attack type)
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for attack_name, group in robustness_df.groupby(group_col):
        ax.plot(range(len(group)), group[y_col].values,
                marker='o', linewidth=2, markersize=8,
                label=attack_name)
        ax.set_xticks(range(len(group)))
        ax.set_xticklabels(group[x_col].values, rotation=45, ha='right')

    ax.set_xlabel('Attack Parameter')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Robustness Under Adversarial Attacks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if robustness_df[y_col].min() > 0.5:
        ax.set_ylim([robustness_df[y_col].min() - 0.05, 1.0])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Robustness degradation plot created")
    return fig


def plot_concept_drift_impact(drift_df: pd.DataFrame,
                               save_path: Path = None) -> plt.Figure:
    """
    Plot accuracy degradation over concept drift steps.

    Args:
        drift_df: DataFrame with columns: step, drift_magnitude, accuracy, f1
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(drift_df['drift_magnitude'], drift_df['accuracy'],
             'b-o', linewidth=2, markersize=6, label='Accuracy')
    ax1.plot(drift_df['drift_magnitude'], drift_df['f1'],
             'r-s', linewidth=2, markersize=6, label='F1-Score')
    ax1.set_xlabel('Drift Magnitude')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance vs Concept Drift Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(drift_df['step'], drift_df['accuracy_drop'], color='salmon')
    ax2.set_xlabel('Drift Step')
    ax2.set_ylabel('Accuracy Drop')
    ax2.set_title('Cumulative Accuracy Drop per Drift Step')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Concept drift impact plot created")
    return fig


def plot_cross_dataset_generalization(transfer_df: pd.DataFrame,
                                       save_path: Path = None) -> plt.Figure:
    """
    Grouped bar chart: source vs target accuracy per model.

    Args:
        transfer_df: DataFrame with transfer learning results
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    models = transfer_df['model'].unique() if 'model' in transfer_df.columns else []
    if len(models) == 0:
        return fig

    x = np.arange(len(models))
    width = 0.35

    source_acc = []
    target_acc = []
    for model in models:
        model_data = transfer_df[transfer_df['model'] == model]
        source_acc.append(model_data['source_accuracy'].mean())
        target_acc.append(model_data['target_accuracy'].mean())

    bars1 = ax.bar(x - width/2, source_acc, width, label='Source Dataset', color='steelblue')
    bars2 = ax.bar(x + width/2, target_acc, width, label='Target Dataset', color='coral')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Cross-Dataset Generalization Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Cross-dataset generalization plot created")
    return fig


def plot_critical_difference_diagram(
        model_ranks: Dict[str, float],
        cd: float = None,
        title: str = "Critical Difference Diagram (Nemenyi)",
        save_path: Path = None) -> plt.Figure:
    """
    Critical Difference diagram for Nemenyi post-hoc test.

    Args:
        model_ranks: {model_name: average_rank}
        cd: Critical difference value
        title: Plot title
        save_path: Path to save figure
    """
    sorted_models = sorted(model_ranks.items(), key=lambda x: x[1])
    names = [m[0] for m in sorted_models]
    ranks = [m[1] for m in sorted_models]
    n = len(names)

    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 3))

    ax.set_xlim(0.5, max(ranks) + 0.5)
    ax.set_ylim(0, 1)

    # Draw rank line
    ax.hlines(0.5, 0.5, max(ranks) + 0.5, color='black', linewidth=1)

    # Draw ticks and labels
    for i, (name, rank) in enumerate(zip(names, ranks)):
        y_offset = 0.7 if i % 2 == 0 else 0.3
        ax.plot(rank, 0.5, 'ko', markersize=8)
        ax.vlines(rank, 0.5, y_offset, color='black', linewidth=1)
        ax.text(rank, y_offset + (0.05 if y_offset > 0.5 else -0.05),
                f'{name}\n({rank:.2f})',
                ha='center', va='bottom' if y_offset > 0.5 else 'top',
                fontsize=9)

    # Draw CD bar if provided
    if cd is not None:
        ax.hlines(0.9, 0.5, 0.5 + cd, color='red', linewidth=3)
        ax.text(0.5 + cd / 2, 0.95, f'CD={cd:.2f}', ha='center',
                fontsize=10, color='red', fontweight='bold')

        # Connect models that are NOT significantly different
        for i in range(n):
            for j in range(i + 1, n):
                if abs(ranks[j] - ranks[i]) < cd:
                    y_line = 0.5 - 0.05 * (j - i)
                    ax.hlines(y_line, ranks[i], ranks[j],
                              color='gray', linewidth=2, alpha=0.5)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('Average Rank')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Critical difference diagram created")
    return fig


def plot_radar_chart(model_metrics: Dict[str, Dict[str, float]],
                     metric_names: List[str] = None,
                     save_path: Path = None) -> plt.Figure:
    """
    Radar/spider chart for multi-metric comparison across models.

    Args:
        model_metrics: {model_name: {metric_name: value}}
        metric_names: List of metric names to include
        save_path: Path to save figure
    """
    if metric_names is None:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 'Robustness']

    n_metrics = len(metric_names)
    angles = [n / float(n_metrics) * 2 * np.pi for n in range(n_metrics)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_metrics)))

    for (model_name, metrics), color in zip(model_metrics.items(), colors):
        values = [metrics.get(m, 0) for m in metric_names]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.set_title('Multi-Metric Model Comparison', size=14, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Radar chart created")
    return fig


def plot_training_convergence(history_dict: Dict[str, Dict],
                               save_path: Path = None) -> plt.Figure:
    """
    Plot training/validation loss convergence for deep learning models.

    Args:
        history_dict: {model_name: {'loss': [...], 'val_loss': [...]}}
        save_path: Path to save figure
    """
    n_models = len(history_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for idx, (name, history) in enumerate(history_dict.items()):
        ax = axes[idx]
        epochs = range(1, len(history.get('loss', [])) + 1)

        if 'loss' in history:
            ax.plot(epochs, history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{name} - Training Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Training convergence plot created")
    return fig


def generate_all_visualizations(all_results: Dict,
                                 output_dir: Path = None) -> Dict[str, plt.Figure]:
    """
    Generate all visualizations from aggregated results.

    Args:
        all_results: Dictionary with results from all stages
        output_dir: Directory to save visualizations

    Returns:
        Dictionary of generated figures
    """
    if output_dir is None:
        output_dir = VIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # Confusion matrix heatmaps
    stage6 = all_results.get('stage6', {})
    if 'effectiveness' in stage6:
        cm = stage6['effectiveness'].get('confusion_matrix')
        if cm is not None:
            cm_array = np.array(cm)
            fig = plot_confusion_matrix_heatmap(
                cm_array,
                model_name=stage6.get('model_name', 'Model'),
                save_path=output_dir / 'confusion_matrix.png'
            )
            figures['confusion_matrix'] = fig

    # Robustness degradation
    stage8 = all_results.get('stage8', {})
    if 'fgsm' in stage8 and stage8['fgsm'] is not None:
        all_attacks = pd.concat(
            [df for df in [stage8.get('fgsm'), stage8.get('noise'), stage8.get('pgd')]
             if df is not None],
            ignore_index=True
        )
        fig = plot_robustness_degradation(
            all_attacks,
            save_path=output_dir / 'robustness_degradation.png'
        )
        figures['robustness'] = fig

    # Concept drift
    drift_results = stage8.get('concept_drift', {})
    for mag_key, drift_df in drift_results.items():
        if isinstance(drift_df, pd.DataFrame):
            fig = plot_concept_drift_impact(
                drift_df,
                save_path=output_dir / f'concept_drift_{mag_key}.png'
            )
            figures[f'drift_{mag_key}'] = fig
            break  # Just plot one magnitude

    # Cross-dataset generalization
    stage7 = all_results.get('stage7', {})
    if 'transfer_results' in stage7 and stage7['transfer_results']:
        transfer_df = pd.DataFrame(stage7['transfer_results'])
        fig = plot_cross_dataset_generalization(
            transfer_df,
            save_path=output_dir / 'cross_dataset_generalization.png'
        )
        figures['cross_dataset'] = fig

    logger.info(f"Generated {len(figures)} visualizations in {output_dir}")
    return figures


###############################################################################
# NEW STAGE-SPECIFIC VISUALIZATION FUNCTIONS
###############################################################################

def plot_class_distribution(class_counts_before: Dict[str, int],
                            class_counts_after: Dict[str, int] = None,
                            mode: str = "multiclass",
                            save_path: Path = None) -> plt.Figure:
    """
    Bar chart of class distribution before and after balancing.
    """
    fig, axes = plt.subplots(1, 2 if class_counts_after else 1,
                              figsize=(14 if class_counts_after else 8, 6))
    if not class_counts_after:
        axes = [axes]

    # Before balancing
    ax = axes[0]
    classes = list(class_counts_before.keys())
    counts = list(class_counts_before.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), counts, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Class Distribution - Before Balancing ({mode})')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count:,}', ha='center', va='bottom', fontsize=8)

    # After balancing
    if class_counts_after:
        ax = axes[1]
        classes_a = list(class_counts_after.keys())
        counts_a = list(class_counts_after.values())
        colors_a = plt.cm.Set3(np.linspace(0, 1, len(classes_a)))
        bars = ax.bar(range(len(classes_a)), counts_a, color=colors_a)
        ax.set_xticks(range(len(classes_a)))
        ax.set_xticklabels(classes_a, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Number of Samples')
        ax.set_title(f'Class Distribution - After Balancing ({mode})')
        ax.grid(True, alpha=0.3, axis='y')
        for bar, count in zip(bars, counts_a):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Class distribution plot created")
    return fig


def plot_correlation_heatmap(X: np.ndarray, feature_names: List[str],
                              top_k: int = 20,
                              save_path: Path = None) -> plt.Figure:
    """
    Correlation heatmap of top-k features by variance.
    """
    # Select top-k features by variance
    variances = np.var(X, axis=0)
    top_indices = np.argsort(variances)[::-1][:top_k]
    top_names = [feature_names[i] for i in top_indices]

    corr = np.corrcoef(X[:, top_indices], rowvar=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, xticklabels=top_names, yticklabels=top_names,
                cmap='RdBu_r', center=0, annot=False, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title(f'Feature Correlation Heatmap (Top {top_k} by Variance)')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Correlation heatmap created")
    return fig


def plot_model_comparison_bars(comparison_df: pd.DataFrame,
                                metric: str = 'Accuracy',
                                save_path: Path = None) -> plt.Figure:
    """
    Horizontal bar chart of all model performances.
    """
    if comparison_df is None or len(comparison_df) == 0:
        return plt.figure()

    df = comparison_df.sort_values(metric, ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(range(len(df)), df[metric].values, color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index if 'Model' not in df.columns
                       else df['Model'].values)
    ax.set_xlabel(metric)
    ax.set_title(f'Model Comparison - {metric}')
    ax.grid(True, alpha=0.3, axis='x')

    for bar, val in zip(bars, df[metric].values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Model comparison bars created")
    return fig


def plot_xai_comparison(quality_metrics: Dict[str, Dict],
                         save_path: Path = None) -> plt.Figure:
    """
    Bar chart comparing XAI methods on fidelity and explanation time.
    """
    methods = list(quality_metrics.keys())
    fidelities = [quality_metrics[m].get('fidelity', 0) for m in methods]
    times = [quality_metrics[m].get('explanation_time_ms', 0) for m in methods]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    ax1.bar(methods, fidelities, color=colors)
    ax1.set_ylabel('Fidelity Score')
    ax1.set_title('XAI Method Fidelity')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(fidelities):
        ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

    ax2.bar(methods, times, color=colors)
    ax2.set_ylabel('Explanation Time (ms)')
    ax2.set_title('XAI Method Speed')
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(times):
        ax2.text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("XAI comparison plot created")
    return fig


def plot_per_class_performance(per_class_df: pd.DataFrame,
                                save_path: Path = None) -> plt.Figure:
    """
    Grouped bar chart of per-class Precision, Recall, F1.
    """
    if per_class_df is None or len(per_class_df) == 0:
        return plt.figure()

    classes = per_class_df['Class'].values if 'Class' in per_class_df.columns \
              else per_class_df.index.values
    n = len(classes)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 6))

    prec = per_class_df['Precision'].values if 'Precision' in per_class_df.columns else []
    rec = per_class_df['Recall'].values if 'Recall' in per_class_df.columns else []
    f1 = per_class_df['F1-Score'].values if 'F1-Score' in per_class_df.columns else []

    if len(prec) > 0:
        ax.bar(x - width, prec, width, label='Precision', color='steelblue')
    if len(rec) > 0:
        ax.bar(x, rec, width, label='Recall', color='coral')
    if len(f1) > 0:
        ax.bar(x + width, f1, width, label='F1-Score', color='seagreen')

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Per-class performance plot created")
    return fig


def plot_efficiency_scatter(model_results: Dict[str, Dict],
                             save_path: Path = None) -> plt.Figure:
    """
    Scatter plot: training time vs accuracy for all models.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    names = []
    accs = []
    times = []
    for name, res in model_results.items():
        acc = res.get('test', {}).get('accuracy', res.get('accuracy', None))
        t = res.get('training_time', res.get('training_time_s', None))
        if acc is not None and t is not None:
            names.append(name)
            accs.append(acc)
            times.append(t)

    if not names:
        return fig

    colors = plt.cm.Set1(np.linspace(0, 1, len(names)))
    scatter = ax.scatter(times, accs, c=colors, s=150, edgecolors='black', zorder=5)

    for i, name in enumerate(names):
        ax.annotate(name, (times[i], accs[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Efficiency vs Effectiveness Trade-off')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Efficiency scatter plot created")
    return fig


def plot_feature_model_heatmap(benchmark_df: pd.DataFrame,
                                save_path: Path = None) -> plt.Figure:
    """
    Heatmap of accuracy for Feature Method x Model combinations.
    """
    if benchmark_df is None or len(benchmark_df) == 0:
        return plt.figure()

    try:
        pivot = benchmark_df.pivot_table(
            values='Accuracy', index='Feature_Method',
            columns='Model', aggfunc='mean'
        )
    except Exception:
        return plt.figure()

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.5),
                                     max(6, len(pivot.index) * 0.5)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, vmin=0.5, vmax=1.0)
    ax.set_title('Feature Selection Method x Model Accuracy')
    ax.set_xlabel('Model')
    ax.set_ylabel('Feature Method')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Feature-Model heatmap created")
    return fig


def plot_pvalue_heatmap(pairwise_results: List[Dict],
                         save_path: Path = None) -> plt.Figure:
    """
    Heatmap of p-values from pairwise statistical tests.
    """
    if not pairwise_results:
        return plt.figure()

    # Collect all model names
    models = set()
    for r in pairwise_results:
        if 'model_a' in r:
            models.add(r['model_a'])
            models.add(r['model_b'])
        elif 'groups' in r:
            pair = r['groups'].split(' vs ')
            if len(pair) == 2:
                models.add(pair[0].strip())
                models.add(pair[1].strip())
    models = sorted(models)

    if len(models) < 2:
        return plt.figure()

    n = len(models)
    pval_matrix = np.ones((n, n))

    for r in pairwise_results:
        if 'model_a' in r:
            a, b = r['model_a'], r['model_b']
        elif 'groups' in r:
            pair = r['groups'].split(' vs ')
            if len(pair) != 2:
                continue
            a, b = pair[0].strip(), pair[1].strip()
        else:
            continue

        if a in models and b in models:
            i, j = models.index(a), models.index(b)
            pval = r.get('p_value', 1.0)
            pval_matrix[i][j] = pval
            pval_matrix[j][i] = pval

    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(6, n)))
    mask = np.eye(n, dtype=bool)
    sns.heatmap(pval_matrix, xticklabels=models, yticklabels=models,
                cmap='RdYlGn_r', annot=True, fmt='.4f', ax=ax,
                mask=mask, vmin=0, vmax=0.1, linewidths=0.5)
    ax.set_title('Pairwise Statistical Test P-Values\n(green = significant, red = not significant)')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("P-value heatmap created")
    return fig


def plot_final_rankings(rankings: Dict, save_path: Path = None) -> plt.Figure:
    """
    Summary bar chart of final rankings across all dimensions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = []
    winners = []
    scores = []

    if 'ml_models' in rankings:
        categories.append('Best ML Model')
        r = rankings['ml_models']
        winners.append(r.get('best_model', 'N/A'))
        scores.append(r.get('best_accuracy', 0))

    if 'feature_selection' in rankings:
        categories.append('Best Feature Method')
        r = rankings['feature_selection']
        winners.append(r.get('best_method', 'N/A'))
        scores.append(1.0)  # placeholder

    if 'robustness' in rankings:
        categories.append('Robustness Score')
        r = rankings['robustness']
        winners.append('Overall')
        scores.append(r.get('overall', 0))

    if 'xai' in rankings:
        categories.append('Best XAI Method')
        r = rankings['xai']
        winners.append(r.get('best_method', 'N/A'))
        scores.append(r.get('fidelity', 0))

    if not categories:
        return fig

    colors = ['steelblue', 'coral', 'seagreen', 'orchid'][:len(categories)]
    bars = ax.barh(range(len(categories)), scores, color=colors)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([f'{cat}\n({win})' for cat, win in zip(categories, winners)])
    ax.set_xlabel('Score')
    ax.set_title('Final Benchmarking Rankings Summary')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1.1)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Final rankings plot created")
    return fig


# ============================================================================
# NEW INDIVIDUAL ACADEMIC-STYLE VISUALIZATION FUNCTIONS
# ============================================================================

# --- Stage 1: Individual Class Distribution ---

def plot_class_distribution_before(class_counts, mode='multiclass', save_path=None):
    """Individual bar chart: class distribution BEFORE balancing."""
    fig, ax = _create_figure(figsize=(max(10, len(class_counts) * 1.2), 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), counts, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Class Distribution — Original ({mode.title()})')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{c:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_class_distribution_after(class_counts, mode='multiclass', save_path=None):
    """Individual bar chart: class distribution AFTER processing."""
    fig, ax = _create_figure(figsize=(max(10, len(class_counts) * 1.2), 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), counts, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Class Distribution — Final ({mode.title()})')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{c:,}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 3: Model Comparison Individual Plots ---

def plot_metric_comparison_bar(comparison_df, metric, title=None, save_path=None):
    """Horizontal bar chart for a single metric across all models, sorted."""
    if comparison_df is None or comparison_df.empty or metric not in comparison_df.columns:
        return None
    fig, ax = _create_figure(figsize=(10, max(5, len(comparison_df) * 0.6)))
    df = comparison_df.dropna(subset=[metric]).sort_values(metric, ascending=True)
    if df.empty:
        plt.close(fig)
        return None
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(df)))
    bars = ax.barh(range(len(df)), df[metric].values, color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Model'].values)
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title or f'Model Comparison — {metric.replace("_", " ").title()}')
    for bar, val in zip(bars, df[metric].values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_time_comparison_bar(comparison_df, time_col, title=None, save_path=None):
    """Horizontal bar chart for training/inference time, sorted."""
    if comparison_df is None or comparison_df.empty or time_col not in comparison_df.columns:
        return None
    fig, ax = _create_figure(figsize=(10, max(5, len(comparison_df) * 0.6)))
    df = comparison_df.sort_values(time_col, ascending=True)
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(df)))
    bars = ax.barh(range(len(df)), df[time_col].values, color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['Model'].values)
    ax.set_xlabel(time_col)
    ax.set_title(title or f'Model Comparison — {time_col}')
    for bar, val in zip(bars, df[time_col].values):
        fmt = f'{val:.2f}' if val < 100 else f'{val:.0f}'
        ax.text(bar.get_width() + 0.01 * max(df[time_col].values),
                bar.get_y() + bar.get_height()/2, fmt, va='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_individual_training_loss(history, model_name, save_path=None):
    """Individual training loss curve for one model."""
    if not history or 'train_loss' not in history:
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', lw=2, label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r--', lw=2, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{model_name} — Training Loss Convergence')
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_individual_training_accuracy(history, model_name, save_path=None):
    """Individual training accuracy curve for one model."""
    if not history or 'train_acc' not in history:
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    epochs = range(1, len(history['train_acc']) + 1)
    ax.plot(epochs, history['train_acc'], 'b-', lw=2, label='Training Accuracy')
    if 'val_acc' in history and history['val_acc']:
        ax.plot(epochs, history['val_acc'], 'r--', lw=2, label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{model_name} — Training Accuracy')
    ax.legend(framealpha=0.9)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_metrics_heatmap(comparison_df, metrics_cols, save_path=None):
    """Heatmap: models × metrics (accuracy, F1, balanced_acc, G-Mean, etc.)."""
    if comparison_df is None or comparison_df.empty:
        return None
    fig, ax = _create_figure(figsize=(max(8, len(metrics_cols) * 2), max(6, len(comparison_df) * 0.7)))
    valid_cols = [c for c in metrics_cols if c in comparison_df.columns]
    if not valid_cols:
        plt.close(fig)
        return None
    data = comparison_df.set_index('Model')[valid_cols].apply(pd.to_numeric, errors='coerce')
    sns.heatmap(data, annot=True, fmt='.4f', cmap='YlOrRd', linewidths=0.5,
                ax=ax, vmin=data.min().min() * 0.95, vmax=1.0, cbar_kws={'label': 'Score'})
    ax.set_title('Model Performance Metrics Heatmap')
    ax.set_ylabel('')
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_model_ranking_table(comparison_df, save_path=None):
    """Table figure with color-coded rankings."""
    if comparison_df is None or comparison_df.empty:
        return None
    fig, ax = _create_figure(figsize=(14, max(4, len(comparison_df) * 0.6 + 2)))
    ax.axis('off')
    cols = [c for c in comparison_df.columns if c != 'Model']
    display_df = comparison_df[['Model'] + cols[:8]].copy()
    for c in display_df.columns[1:]:
        display_df[c] = display_df[c].apply(lambda x: f'{x:.4f}' if isinstance(x, (int, float)) else str(x))
    table = ax.table(cellText=display_df.values, colLabels=display_df.columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold')
        elif row == 1:
            cell.set_facecolor('#D6E4F0')
        elif row % 2 == 0:
            cell.set_facecolor('#F2F2F2')
    ax.set_title('Model Performance Ranking', fontsize=14, pad=20, fontweight='bold')
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 5: XAI Individual Plots ---

def plot_xai_fidelity_comparison(quality_metrics, save_path=None):
    """Bar chart: fidelity per XAI method."""
    if not quality_metrics:
        return None
    methods = [m for m, v in quality_metrics.items()
               if isinstance(v, dict) and 'fidelity' in v]
    if not methods:
        return None
    fidelities = [quality_metrics[m]['fidelity'] for m in methods]
    fig, ax = _create_figure(figsize=(8, 5))
    colors = METRIC_PALETTE[:len(methods)]
    bars = ax.bar(methods, fidelities, color=colors, width=0.6)
    ax.set_ylabel('Fidelity Score')
    ax.set_title('XAI Method — Fidelity Comparison')
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, fidelities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=11)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_xai_explanation_time(quality_metrics, save_path=None):
    """Bar chart: explanation time per XAI method."""
    if not quality_metrics:
        return None
    methods = [m for m, v in quality_metrics.items()
               if isinstance(v, dict) and 'explanation_time_ms' in v]
    if not methods:
        return None
    times = [quality_metrics[m]['explanation_time_ms'] for m in methods]
    fig, ax = _create_figure(figsize=(8, 5))
    colors = METRIC_PALETTE[:len(methods)]
    bars = ax.bar(methods, times, color=colors, width=0.6)
    ax.set_ylabel('Time per Sample (ms)')
    ax.set_title('XAI Method — Explanation Time')
    for bar, v in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{v:.1f}', ha='center', fontsize=11)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_xai_faithfulness(faithfulness_results, save_path=None):
    """Grouped bar: sufficiency ratio and necessity drop."""
    if not faithfulness_results:
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    metrics_names = ['Sufficiency Ratio', 'Necessity Drop']
    values = [faithfulness_results.get('sufficiency_ratio', 0),
              faithfulness_results.get('necessity_drop', 0)]
    colors = ['#2ca02c', '#d62728']
    bars = ax.bar(metrics_names, values, color=colors, width=0.5)
    ax.set_ylabel('Score')
    ax.set_title('XAI Faithfulness — Sufficiency vs Necessity')
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=11)
    baseline = faithfulness_results.get('baseline_accuracy', 0)
    ax.axhline(y=baseline, color='gray', ls='--', lw=1, alpha=0.7, label=f'Baseline Acc: {baseline:.3f}')
    ax.legend()
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 6: Per-Class Individual Plots ---

def plot_per_class_single_metric(per_class_df, metric_col, title=None, save_path=None):
    """Bar chart for a single per-class metric (Precision/Recall/F1)."""
    if per_class_df is None or per_class_df.empty or metric_col not in per_class_df.columns:
        return None
    fig, ax = _create_figure(figsize=(max(10, len(per_class_df) * 1.2), 6))
    classes = per_class_df['Class'].values if 'Class' in per_class_df.columns else per_class_df.index
    values = per_class_df[metric_col].values
    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))
    bars = ax.bar(range(len(classes)), values, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel(metric_col)
    ax.set_title(title or f'Per-Class {metric_col}')
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_detection_rate_per_class(detection_df, threshold=0.90, save_path=None):
    """Detection rate per class with threshold line."""
    if detection_df is None or detection_df.empty:
        return None
    fig, ax = _create_figure(figsize=(max(10, len(detection_df) * 1.2), 6))
    classes = detection_df['Class'].values
    rates = detection_df['Detection_Rate'].values
    colors = ['#d62728' if r < threshold else '#2ca02c' for r in rates]
    bars = ax.bar(range(len(classes)), rates, color=colors)
    ax.axhline(y=threshold, color='red', linestyle='--', lw=2,
               label=f'{threshold*100:.0f}% Threshold', alpha=0.7)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Detection Rate')
    ax.set_title('Detection Rate per Attack Class')
    ax.set_ylim(0, 1.1)
    ax.legend()
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_far_per_class(fpr_per_class, class_names=None, save_path=None):
    """False Alarm Rate per class bar chart."""
    if not fpr_per_class:
        return None
    fig, ax = _create_figure(figsize=(max(10, len(fpr_per_class) * 1.2), 6))
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(fpr_per_class))]
    colors = ['#d62728' if f > 0.05 else '#2ca02c' for f in fpr_per_class]
    bars = ax.bar(range(len(fpr_per_class)), fpr_per_class, color=colors)
    ax.axhline(y=0.05, color='orange', ls='--', lw=2, label='5% Threshold', alpha=0.7)
    ax.set_xticks(range(len(fpr_per_class)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('False Alarm Rate (FPR)')
    ax.set_title('False Alarm Rate per Class')
    ax.legend()
    for bar, v in zip(bars, fpr_per_class):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 7: Generalization Individual Plots ---

def plot_generalization_gap_bar(transfer_results, save_path=None):
    """Bar chart: accuracy gap per model."""
    if not transfer_results:
        return None
    if isinstance(transfer_results, list):
        df = pd.DataFrame(transfer_results)
    else:
        df = transfer_results
    if df.empty or 'accuracy_gap' not in df.columns:
        return None
    fig, ax = _create_figure(figsize=(10, 6))
    gap_per_model = df.groupby('model')['accuracy_gap'].mean().sort_values()
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(gap_per_model)))
    bars = ax.barh(range(len(gap_per_model)), gap_per_model.values, color=colors)
    ax.set_yticks(range(len(gap_per_model)))
    ax.set_yticklabels(gap_per_model.index)
    ax.set_xlabel('Accuracy Gap (Source − Target)')
    ax.set_title('Cross-Dataset Generalization Gap')
    ax.axvline(x=0, color='black', lw=0.5)
    for bar, v in zip(bars, gap_per_model.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_generalization_ratio_bar(transfer_results, save_path=None):
    """Bar chart: generalization ratio per model."""
    if not transfer_results:
        return None
    if isinstance(transfer_results, list):
        df = pd.DataFrame(transfer_results)
    else:
        df = transfer_results
    if df.empty or 'generalization_ratio' not in df.columns:
        return None
    fig, ax = _create_figure(figsize=(10, 6))
    ratio_per_model = df.groupby('model')['generalization_ratio'].mean().sort_values()
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(ratio_per_model)))
    bars = ax.barh(range(len(ratio_per_model)), ratio_per_model.values, color=colors)
    ax.set_yticks(range(len(ratio_per_model)))
    ax.set_yticklabels(ratio_per_model.index)
    ax.set_xlabel('Generalization Ratio (Target / Source)')
    ax.set_title('Cross-Dataset Generalization Ratio')
    ax.axvline(x=1.0, color='red', ls='--', lw=1.5, label='Perfect (1.0)', alpha=0.7)
    ax.legend()
    for bar, v in zip(bars, ratio_per_model.values):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{v:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_source_vs_target_accuracy(transfer_results, save_path=None):
    """Grouped bar chart: source vs target accuracy per model."""
    if not transfer_results:
        return None
    if isinstance(transfer_results, list):
        df = pd.DataFrame(transfer_results)
    else:
        df = transfer_results
    if df.empty:
        return None
    fig, ax = _create_figure(figsize=(12, 6))
    models = df['model'].unique()
    x = np.arange(len(models))
    w = 0.35
    src = [df[df['model'] == m]['source_accuracy'].mean() for m in models]
    tgt = [df[df['model'] == m]['target_accuracy'].mean() for m in models]
    ax.bar(x - w/2, src, w, label='Source', color='#1f77b4')
    ax.bar(x + w/2, tgt, w, label='Target', color='#ff7f0e')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Accuracy')
    ax.set_title('Source vs Target Accuracy per Model')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 8: Robustness Individual Plots ---

def plot_attack_robustness_curve(robustness_df, attack_type, x_label='Parameter', save_path=None):
    """Line plot: accuracy vs parameter for a single attack type."""
    if robustness_df is None or (isinstance(robustness_df, pd.DataFrame) and robustness_df.empty):
        return None
    if isinstance(robustness_df, pd.DataFrame):
        df = robustness_df
    else:
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    baseline = df['baseline_accuracy'].iloc[0] if 'baseline_accuracy' in df.columns else None
    if baseline is not None:
        ax.axhline(y=baseline, color='green', ls='--', lw=1.5, label=f'Baseline: {baseline:.4f}', alpha=0.7)
    ax.plot(range(len(df)), df['adversarial_accuracy'].values, 'ro-', lw=2, markersize=8, label='Under Attack')
    ax.set_xticks(range(len(df)))
    labels = df['parameter'].values if 'parameter' in df.columns else range(len(df))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel(x_label)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{attack_type} Attack — Accuracy Degradation')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_overall_robustness_bar(robustness_scores, save_path=None):
    """Bar chart: overall robustness scores (FGSM, PGD, Noise, Overall)."""
    if not robustness_scores:
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    keys = list(robustness_scores.keys())
    vals = list(robustness_scores.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(keys)]
    bars = ax.bar(range(len(keys)), vals, color=colors, width=0.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace('_', ' ').title() for k in keys], rotation=30, ha='right')
    ax.set_ylabel('Robustness Ratio')
    ax.set_title('Overall Robustness Scores')
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{v:.3f}', ha='center', fontsize=11)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_concept_drift_accuracy(drift_df, save_path=None):
    """Line: accuracy over drift magnitude (individual)."""
    if drift_df is None or (isinstance(drift_df, pd.DataFrame) and drift_df.empty):
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    ax.plot(drift_df['drift_magnitude'], drift_df['accuracy'], 'b-o', lw=2, markersize=6)
    ax.set_xlabel('Drift Magnitude')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Concept Drift Magnitude')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_concept_drift_f1(drift_df, save_path=None):
    """Line: F1 over drift magnitude (individual)."""
    if drift_df is None or (isinstance(drift_df, pd.DataFrame) and drift_df.empty):
        return None
    if 'f1' not in drift_df.columns:
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    ax.plot(drift_df['drift_magnitude'], drift_df['f1'], 'r-s', lw=2, markersize=6)
    ax.set_xlabel('Drift Magnitude')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score vs Concept Drift Magnitude')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_feature_perturbation_curve(perturbation_df, save_path=None):
    """Line: accuracy vs % features perturbed."""
    if perturbation_df is None or (isinstance(perturbation_df, pd.DataFrame) and perturbation_df.empty):
        return None
    fig, ax = _create_figure(figsize=(8, 5))
    ax.plot(perturbation_df['percentage_perturbed'], perturbation_df['perturbed_accuracy'],
            'ro-', lw=2, markersize=8)
    baseline = perturbation_df['baseline_accuracy'].iloc[0] if 'baseline_accuracy' in perturbation_df.columns else None
    if baseline:
        ax.axhline(y=baseline, color='green', ls='--', lw=1.5, label=f'Baseline: {baseline:.4f}', alpha=0.7)
        ax.legend()
    ax.set_xlabel('% Features Perturbed')
    ax.set_ylabel('Accuracy')
    ax.set_title('Feature Perturbation Impact on Accuracy')
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 9: Statistical Individual Plots ---

def plot_confidence_intervals(ci_data, save_path=None):
    """Error bar chart with bootstrap confidence intervals per model."""
    if not ci_data:
        return None
    fig, ax = _create_figure(figsize=(10, max(5, len(ci_data) * 0.6)))
    models = list(ci_data.keys())
    means = [ci_data[m].get('statistic', 0) for m in models]
    lowers = [ci_data[m].get('lower_bound', 0) for m in models]
    uppers = [ci_data[m].get('upper_bound', 0) for m in models]
    errors_low = [m - l for m, l in zip(means, lowers)]
    errors_high = [u - m for m, u in zip(means, uppers)]
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(models)))
    ax.barh(range(len(models)), means, xerr=[errors_low, errors_high],
            capsize=5, color=colors, alpha=0.85, ecolor='black')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel('Accuracy')
    ax.set_title('Bootstrap 95% Confidence Intervals')
    for i, (m, lo, hi) in enumerate(zip(means, lowers, uppers)):
        ax.text(hi + 0.005, i, f'{m:.4f} [{lo:.4f}, {hi:.4f}]', va='center', fontsize=8)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_effect_sizes(t_test_results, save_path=None):
    """Cohen's d bar chart for model pair comparisons."""
    if not t_test_results:
        return None
    fig, ax = _create_figure(figsize=(12, max(5, len(t_test_results) * 0.5)))
    pairs = []
    d_vals = []
    for r in t_test_results:
        groups = r.get('groups', ['?', '?'])
        pairs.append(f'{groups[0]} vs {groups[1]}')
        d_vals.append(abs(r.get('cohens_d', 0)))
    colors = ['#d62728' if d > 0.8 else '#ff7f0e' if d > 0.5 else '#2ca02c' for d in d_vals]
    bars = ax.barh(range(len(pairs)), d_vals, color=colors)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pairs, fontsize=8)
    ax.set_xlabel("|Cohen's d|")
    ax.set_title("Effect Sizes — Pairwise Model Comparisons")
    ax.axvline(x=0.2, color='green', ls=':', alpha=0.5, label='Small (0.2)')
    ax.axvline(x=0.5, color='orange', ls=':', alpha=0.5, label='Medium (0.5)')
    ax.axvline(x=0.8, color='red', ls=':', alpha=0.5, label='Large (0.8)')
    ax.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# --- Stage 10: Recommendation Summary ---

def plot_recommendation_summary(recommendations, save_path=None):
    """Table-style figure with recommendation results."""
    if not recommendations:
        return None
    fig, ax = _create_figure(figsize=(14, max(4, len(recommendations) * 1.2 + 2)))
    ax.axis('off')
    rows = []
    for key, rec in recommendations.items():
        category = rec.get('category', key)
        model = rec.get('model', rec.get('method', '—'))
        reason = rec.get('reason', '')
        score_val = ''
        for k in ['accuracy', 'inference_ms', 'score', 'fidelity', 'value']:
            if k in rec:
                score_val = f'{rec[k]:.4f}' if isinstance(rec[k], float) else str(rec[k])
                break
        rows.append([category, model, score_val, reason[:60]])
    col_labels = ['Category', 'Model/Method', 'Score', 'Reason']
    table = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2E75B6')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#D6E4F0')
        cell.set_edgecolor('#CCCCCC')
    ax.set_title('Benchmarking Recommendations', fontsize=16, pad=25, fontweight='bold')
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# ============================================================================
# LATEX TABLE EXPORT
# ============================================================================

def export_latex_table(df, caption='', label='', save_path=None,
                       bold_max_cols=None):
    """
    Export a DataFrame as a LaTeX table file.

    Args:
        df:             DataFrame to export
        caption:        Table caption
        label:          LaTeX label for cross-reference
        save_path:      .tex file path
        bold_max_cols:  list of column names where max value should be bold
    """
    if df is None or df.empty:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Optionally bold best values
    df_display = df.copy()
    if bold_max_cols:
        for col in bold_max_cols:
            if col in df_display.columns:
                try:
                    numeric_vals = pd.to_numeric(df_display[col], errors='coerce')
                    max_idx = numeric_vals.idxmax()
                    if pd.notna(max_idx):
                        val = df_display.at[max_idx, col]
                        df_display.at[max_idx, col] = f"\\textbf{{{val}}}"
                except Exception:
                    pass

    latex_str = df_display.to_latex(index=False, escape=False, caption=caption, label=label)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    logger.info(f"LaTeX table saved: {save_path.name}")


# ============================================================================
# ERROR ANALYSIS PLOTS
# ============================================================================

def plot_confusion_pairs_bar(confusion_pairs, top_k=10, save_path=None):
    """Bar chart of most confused class pairs."""
    if not confusion_pairs:
        return None
    pairs = confusion_pairs[:top_k]
    fig, ax = _create_figure(figsize=(12, max(5, len(pairs) * 0.5)))
    labels = [f"{p['True_Class']} → {p['Predicted_Class']}" for p in pairs]
    counts = [p['Count'] for p in pairs]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(pairs)))
    bars = ax.barh(range(len(pairs)), counts, color=colors)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Number of Misclassifications')
    ax.set_title(f'Top {len(pairs)} Most Confused Class Pairs')
    ax.invert_yaxis()
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{c:,}', va='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


def plot_error_rate_per_class(per_class_errors, save_path=None):
    """Bar chart of error rate per class."""
    if not per_class_errors:
        return None
    fig, ax = _create_figure(figsize=(max(10, len(per_class_errors) * 1.2), 6))
    classes = [r['Class'] for r in per_class_errors]
    rates = [r['Error_Rate'] for r in per_class_errors]
    colors = ['#d62728' if r > 0.10 else '#ff7f0e' if r > 0.05 else '#2ca02c' for r in rates]
    bars = ax.bar(range(len(classes)), rates, color=colors)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel('Error Rate')
    ax.set_title('Per-Class Error Rate')
    ax.axhline(y=0.05, color='orange', ls='--', lw=1.5, label='5% Threshold', alpha=0.7)
    ax.axhline(y=0.10, color='red', ls='--', lw=1.5, label='10% Threshold', alpha=0.7)
    ax.legend()
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{v:.3f}', ha='center', fontsize=9)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# ============================================================================
# ABLATION STUDY PLOT
# ============================================================================

def plot_ablation_curve(ablation_df, save_path=None):
    """Line plot: accuracy and F1 vs number of features."""
    if ablation_df is None or ablation_df.empty:
        return None
    fig, ax = _create_figure(figsize=(10, 6))
    x = ablation_df['n_features'].values
    ax.plot(x, ablation_df['accuracy'].values, 'bo-', lw=2, markersize=8, label='Accuracy')
    if 'f1' in ablation_df.columns:
        ax.plot(x, ablation_df['f1'].values, 'rs-', lw=2, markersize=8, label='F1 (weighted)')
    if 'balanced_accuracy' in ablation_df.columns:
        ax.plot(x, ablation_df['balanced_accuracy'].values, 'g^-', lw=2, markersize=6, label='Balanced Acc')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study — Performance vs Feature Count')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


# ============================================================================
# BINARY vs MULTICLASS COMPARISON
# ============================================================================

def plot_binary_vs_multiclass_comparison(binary_df, multi_df, metric='Accuracy',
                                          save_path=None):
    """Grouped bar chart comparing models between binary and multiclass modes."""
    if binary_df is None or multi_df is None:
        return None
    if binary_df.empty or multi_df.empty:
        return None
    if metric not in binary_df.columns or metric not in multi_df.columns:
        return None

    # Find common models
    b_models = set(binary_df['Model'].values) if 'Model' in binary_df.columns else set()
    m_models = set(multi_df['Model'].values) if 'Model' in multi_df.columns else set()
    common = sorted(b_models & m_models)
    if not common:
        return None

    fig, ax = _create_figure(figsize=(max(10, len(common) * 1.5), 6))
    x = np.arange(len(common))
    w = 0.35

    b_vals = [float(binary_df[binary_df['Model'] == m][metric].values[0]) for m in common]
    m_vals = [float(multi_df[multi_df['Model'] == m][metric].values[0]) for m in common]

    ax.bar(x - w/2, b_vals, w, label='Binary', color='#1f77b4')
    ax.bar(x + w/2, m_vals, w, label='Multiclass', color='#ff7f0e')

    ax.set_xticks(x)
    ax.set_xticklabels(common, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_title(f'Binary vs Multiclass — {metric}')
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    return _save_and_close(fig, save_path)


if __name__ == "__main__":
    print("Visualization module - run via main.py or import individual functions")
