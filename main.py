"""
================================================================================
XAI-BASED IDS BENCHMARKING STUDY v3.0
Comprehensive Comparison & Evaluation Framework
================================================================================

This is the main entry point for running the complete benchmarking pipeline.
It orchestrates all 10 stages of the study.

Usage:
    python main.py                              # Run all stages, both modes
    python main.py --mode binary                # Binary classification only
    python main.py --mode multiclass            # Multi-class (8 categories) only
    python main.py --mode binary multiclass     # Both modes (default)
    python main.py --stage 1                    # Run specific stage
    python main.py --stage 1 2 3                # Run multiple stages
    python main.py --quick                      # Quick run with reduced data
    python main.py --no-cache                   # Force reload (ignore Stage 1 cache)
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd
import warnings
import logging
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print study banner."""
    banner = """
    ======================================================================
      XAI-BASED IDS BENCHMARKING STUDY v3.0
      Comprehensive Comparison & Evaluation Framework
    ======================================================================
      Stages:
       1. Dataset Preparation & Baseline
       2. Feature Selection Methods Benchmarking
       3. Machine Learning Models Benchmarking
       4. Comprehensive Results Benchmarking
       5. XAI Explanations Quality Benchmarking
       6. Performance Metrics Benchmarking
       7. Cross-Dataset Generalization Test
       8. Robustness & Adversarial Testing
       9. Statistical Significance Testing
      10. Final Benchmarking Summary & Report
    ======================================================================
      Modes: binary (Benign vs Attack) | multiclass (8 categories)
    ======================================================================
    """
    print(banner)


###############################################################################
# VISUALIZATION FUNCTIONS — one per stage
###############################################################################

def visualize_stage_1(stage1_results: dict, mode: str):
    """Generate and save Stage 1 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import (
        plot_class_distribution, plot_correlation_heatmap,
        plot_class_distribution_before, plot_class_distribution_after
    )

    vis_dir = get_vis_dir(mode) / "stage1"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 1 visualizations → {vis_dir}")

    # Map numeric labels to real class names
    label_mapping = stage1_results.get('label_mapping', {})
    reverse_map = {v: k for k, v in label_mapping.items()} if label_mapping else {}

    def _to_named(counts_dict):
        return {reverse_map.get(k, str(k)): int(v) for k, v in counts_dict.items()}

    dist_before = stage1_results.get('class_distribution_before', {})
    dist_after = stage1_results.get('class_distribution', {})
    counts_before = _to_named(dist_before.get('counts', {}))
    counts_after = _to_named(dist_after.get('counts', {}))

    # --- 1a. Combined class distribution (legacy) ---
    try:
        if counts_after:
            fig = plot_class_distribution(
                class_counts_before=counts_before if counts_before else counts_after,
                class_counts_after=counts_after,
                mode=mode,
                save_path=vis_dir / "class_distribution.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ class_distribution.png")
    except Exception as e:
        logger.warning(f"[VIS] class distribution failed: {e}")

    # --- 1b. Individual: before processing ---
    try:
        src = counts_before if counts_before else counts_after
        if src:
            plot_class_distribution_before(src, mode=mode,
                                           save_path=vis_dir / "class_distribution_before.png")
            logger.info("[VIS] ✓ class_distribution_before.png")
    except Exception as e:
        logger.warning(f"[VIS] class_distribution_before failed: {e}")

    # --- 1c. Individual: after processing ---
    try:
        if counts_after:
            plot_class_distribution_after(counts_after, mode=mode,
                                          save_path=vis_dir / "class_distribution_after.png")
            logger.info("[VIS] ✓ class_distribution_after.png")
    except Exception as e:
        logger.warning(f"[VIS] class_distribution_after failed: {e}")

    # --- 1d. Feature correlation heatmap ---
    try:
        X = stage1_results.get('X')
        feat_names = stage1_results.get('feature_columns', [])
        if X is not None and len(feat_names) > 0:
            n = min(50_000, X.shape[0])
            rng = np.random.RandomState(42)
            idx = rng.choice(X.shape[0], n, replace=False)
            fig = plot_correlation_heatmap(
                X[idx], feat_names, top_k=20,
                save_path=vis_dir / "feature_correlation.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ feature_correlation.png")
    except Exception as e:
        logger.warning(f"[VIS] correlation heatmap failed: {e}")


def visualize_stage_2(stage2_results: dict, mode: str):
    """Generate and save Stage 2 visualizations."""
    from config import get_vis_dir
    from visualizations import plot_feature_importance_comparison

    vis_dir = get_vis_dir(mode) / "stage2"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 2 visualizations → {vis_dir}")

    try:
        methods_dict = stage2_results.get('methods', {})
        importance_dict = {}
        for method_key, mdata in methods_dict.items():
            imp_list = mdata.get('importance', [])
            if imp_list and isinstance(imp_list, list) and isinstance(imp_list[0], dict):
                df = pd.DataFrame(imp_list)
                if 'feature' in df.columns and 'importance' in df.columns:
                    importance_dict[method_key] = df

        if importance_dict:
            fig = plot_feature_importance_comparison(
                importance_dict, top_k=15,
                save_path=vis_dir / "feature_importance_comparison.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ feature_importance_comparison.png")
        else:
            logger.info("[VIS] No importance data found for Stage 2 plot")
    except Exception as e:
        logger.warning(f"[VIS] Stage 2 visualization failed: {e}")


def visualize_stage_3(stage3_results: dict, stage1_results: dict, mode: str):
    """Generate and save Stage 3 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import (
        plot_model_comparison_bars, plot_confusion_matrix_heatmap,
        plot_roc_curves, plot_training_convergence,
        plot_metric_comparison_bar, plot_time_comparison_bar,
        plot_individual_training_loss, plot_individual_training_accuracy,
        plot_metrics_heatmap, plot_model_ranking_table
    )

    vis_dir = get_vis_dir(mode) / "stage3"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 3 visualizations → {vis_dir}")

    comp_df = stage3_results.get('comparison')
    models = stage3_results.get('models', {})
    label_mapping = stage1_results.get('label_mapping', {})
    class_names = list(label_mapping.keys()) if label_mapping else None

    # --- 3a. Legacy model comparison bar ---
    try:
        if comp_df is not None and len(comp_df) > 0:
            metric_col = 'Accuracy' if 'Accuracy' in comp_df.columns else comp_df.columns[1]
            fig = plot_model_comparison_bars(
                comp_df, metric=metric_col,
                save_path=vis_dir / "model_comparison.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ model_comparison.png")
    except Exception as e:
        logger.warning(f"[VIS] model comparison bars failed: {e}")

    # --- 3b. Individual metric bars (Accuracy, Balanced_Acc, F1_Macro, G_Mean) ---
    try:
        if comp_df is not None and len(comp_df) > 0:
            for metric in ['Accuracy', 'Balanced_Acc', 'F1_Macro', 'G_Mean', 'FAR']:
                if metric in comp_df.columns:
                    plot_metric_comparison_bar(
                        comp_df, metric,
                        save_path=vis_dir / f"metric_{metric.lower()}.png"
                    )
                    logger.info(f"[VIS] ✓ metric_{metric.lower()}.png")
    except Exception as e:
        logger.warning(f"[VIS] individual metric bars failed: {e}")

    # --- 3c. Time comparison bars ---
    try:
        if comp_df is not None and len(comp_df) > 0:
            for tcol in ['Training_Time_s', 'Inference_Time_ms']:
                if tcol in comp_df.columns:
                    plot_time_comparison_bar(
                        comp_df, tcol,
                        save_path=vis_dir / f"time_{tcol.lower()}.png"
                    )
                    logger.info(f"[VIS] ✓ time_{tcol.lower()}.png")
    except Exception as e:
        logger.warning(f"[VIS] time comparison bars failed: {e}")

    # --- 3d. Metrics heatmap ---
    try:
        if comp_df is not None and len(comp_df) > 0:
            metric_cols = [c for c in ['Accuracy', 'Balanced_Acc', 'F1_Macro',
                                        'G_Mean', 'F1', 'Precision', 'Recall']
                           if c in comp_df.columns]
            if metric_cols:
                plot_metrics_heatmap(comp_df, metric_cols,
                                     save_path=vis_dir / "metrics_heatmap.png")
                logger.info("[VIS] ✓ metrics_heatmap.png")
    except Exception as e:
        logger.warning(f"[VIS] metrics heatmap failed: {e}")

    # --- 3e. Model ranking table ---
    try:
        if comp_df is not None and len(comp_df) > 0:
            plot_model_ranking_table(comp_df, save_path=vis_dir / "model_ranking_table.png")
            logger.info("[VIS] ✓ model_ranking_table.png")
    except Exception as e:
        logger.warning(f"[VIS] model ranking table failed: {e}")

    # --- 3f. Confusion matrices for ALL models ---
    try:
        for mname, mdata in models.items():
            if 'error' not in mdata:
                cm = mdata.get('test', {}).get('confusion_matrix')
                if cm is not None:
                    cm_arr = np.array(cm)
                    fig = plot_confusion_matrix_heatmap(
                        cm_arr, class_names=class_names, model_name=mname,
                        save_path=vis_dir / f"confusion_matrix_{mname}.png"
                    )
                    plt.close(fig)
                    logger.info(f"[VIS] ✓ confusion_matrix_{mname}.png")
    except Exception as e:
        logger.warning(f"[VIS] confusion matrices failed: {e}")

    # --- 3g. ROC curves ---
    try:
        roc_data = {}
        for mname, mdata in models.items():
            if 'error' not in mdata:
                test = mdata.get('test', {})
                if 'fpr' in test and 'tpr' in test:
                    roc_data[mname] = {
                        'fpr': np.array(test['fpr']),
                        'tpr': np.array(test['tpr']),
                        'auc': test.get('roc_auc', 0)
                    }
        if roc_data:
            fig = plot_roc_curves(roc_data, save_path=vis_dir / "roc_curves.png")
            plt.close(fig)
            logger.info("[VIS] ✓ roc_curves.png")
    except Exception as e:
        logger.warning(f"[VIS] ROC curves failed: {e}")

    # --- 3h. Individual training loss & accuracy per model ---
    try:
        for mname, mdata in models.items():
            if 'error' not in mdata:
                hist = mdata.get('training_history', {})
                if hist:
                    plot_individual_training_loss(
                        hist, mname,
                        save_path=vis_dir / f"training_loss_{mname}.png"
                    )
                    logger.info(f"[VIS] ✓ training_loss_{mname}.png")
                    plot_individual_training_accuracy(
                        hist, mname,
                        save_path=vis_dir / f"training_acc_{mname}.png"
                    )
                    logger.info(f"[VIS] ✓ training_acc_{mname}.png")
    except Exception as e:
        logger.warning(f"[VIS] individual training curves failed: {e}")

    # --- 3i. Legacy combined training convergence ---
    try:
        history_dict = {}
        for mname in models:
            if 'error' not in models[mname]:
                hist = models[mname].get('training_history', {})
                if hist and ('loss' in hist or 'train_loss' in hist):
                    history_dict[mname] = hist
        if history_dict:
            fig = plot_training_convergence(
                history_dict, save_path=vis_dir / "training_convergence.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ training_convergence.png")
    except Exception as e:
        logger.warning(f"[VIS] training convergence failed: {e}")


def visualize_stage_4(stage4_results: dict, mode: str):
    """Generate and save Stage 4 visualizations."""
    from config import get_vis_dir
    from visualizations import plot_feature_model_heatmap

    vis_dir = get_vis_dir(mode) / "stage4"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 4 visualizations → {vis_dir}")

    try:
        bench_df = stage4_results.get('benchmark_results')
        if bench_df is not None and len(bench_df) > 0:
            fig = plot_feature_model_heatmap(
                bench_df, save_path=vis_dir / "feature_model_heatmap.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ feature_model_heatmap.png")
    except Exception as e:
        logger.warning(f"[VIS] Stage 4 heatmap failed: {e}")


def visualize_stage_5(stage5_results: dict, mode: str):
    """Generate and save Stage 5 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import (
        plot_xai_comparison, plot_xai_fidelity_comparison,
        plot_xai_explanation_time, plot_xai_faithfulness
    )

    vis_dir = get_vis_dir(mode) / "stage5"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 5 visualizations → {vis_dir}")

    qm = stage5_results.get('quality_metrics', {})
    xai_metrics = {k: v for k, v in qm.items()
                   if isinstance(v, dict) and 'fidelity' in v}

    # --- 5a. Legacy combined XAI comparison ---
    try:
        if xai_metrics:
            fig = plot_xai_comparison(
                xai_metrics, save_path=vis_dir / "xai_comparison.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ xai_comparison.png")
    except Exception as e:
        logger.warning(f"[VIS] XAI comparison failed: {e}")

    # --- 5b. Individual: Fidelity comparison ---
    try:
        if xai_metrics:
            plot_xai_fidelity_comparison(xai_metrics,
                                          save_path=vis_dir / "xai_fidelity.png")
            logger.info("[VIS] ✓ xai_fidelity.png")
    except Exception as e:
        logger.warning(f"[VIS] xai_fidelity failed: {e}")

    # --- 5c. Individual: Explanation time ---
    try:
        if xai_metrics:
            plot_xai_explanation_time(xai_metrics,
                                       save_path=vis_dir / "xai_explanation_time.png")
            logger.info("[VIS] ✓ xai_explanation_time.png")
    except Exception as e:
        logger.warning(f"[VIS] xai_explanation_time failed: {e}")

    # --- 5d. Individual: Faithfulness ---
    try:
        faith = stage5_results.get('faithfulness', {})
        if faith:
            plot_xai_faithfulness(faith,
                                   save_path=vis_dir / "xai_faithfulness.png")
            logger.info("[VIS] ✓ xai_faithfulness.png")
    except Exception as e:
        logger.warning(f"[VIS] xai_faithfulness failed: {e}")


def visualize_stage_6(stage6_results: dict, stage3_results: dict, mode: str):
    """Generate and save Stage 6 visualizations (individual academic plots).

    Handles both single-model results (legacy) and multi-model dict format.
    """
    from config import get_vis_dir
    from visualizations import (
        plot_per_class_performance, plot_efficiency_scatter, plot_radar_chart,
        plot_per_class_single_metric, plot_detection_rate_per_class, plot_far_per_class
    )

    vis_dir = get_vis_dir(mode) / "stage6"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 6 visualizations → {vis_dir}")

    # Detect multi-model format: keys are model names, values are result dicts
    is_multi = (isinstance(stage6_results, dict)
                and 'per_class' not in stage6_results
                and 'effectiveness' not in stage6_results)

    if is_multi:
        # Pick the best model (highest accuracy) for per-class plots
        best_name, best_res = None, None
        best_acc = -1
        for mname, mres in stage6_results.items():
            if isinstance(mres, dict):
                acc = mres.get('effectiveness', {}).get('accuracy', 0)
                if acc > best_acc:
                    best_acc = acc
                    best_name = mname
                    best_res = mres
        if best_res is None:
            logger.warning("[VIS] No valid Stage 6 model results for visualization")
            return
        single_results = best_res
    else:
        single_results = stage6_results
        best_name = single_results.get('model_name', 'Model')

    pc_df = single_results.get('per_class')

    # --- 6a. Per-class performance ---
    try:
        if pc_df is not None and len(pc_df) > 0:
            fig = plot_per_class_performance(
                pc_df, save_path=vis_dir / "per_class_performance.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ per_class_performance.png")
    except Exception as e:
        logger.warning(f"[VIS] per-class performance failed: {e}")

    # --- 6b. Individual: per-class Precision, Recall, F1 ---
    try:
        if pc_df is not None and len(pc_df) > 0:
            for metric in ['Precision', 'Recall', 'F1-Score']:
                if metric in pc_df.columns:
                    plot_per_class_single_metric(
                        pc_df, metric,
                        save_path=vis_dir / f"per_class_{metric.lower().replace('-', '_')}.png"
                    )
                    logger.info(f"[VIS] ✓ per_class_{metric.lower().replace('-', '_')}.png")
    except Exception as e:
        logger.warning(f"[VIS] per-class single metric failed: {e}")

    # --- 6c. Detection rate per class ---
    try:
        dr_df = single_results.get('detection_rate')
        if dr_df is not None and 'Detection_Rate' in dr_df.columns:
            plot_detection_rate_per_class(dr_df, threshold=0.90,
                                           save_path=vis_dir / "detection_rate_per_class.png")
            logger.info("[VIS] ✓ detection_rate_per_class.png")
    except Exception as e:
        logger.warning(f"[VIS] detection rate failed: {e}")

    # --- 6d. FAR per class ---
    try:
        eff = single_results.get('effectiveness', {})
        fpr_list = eff.get('fpr_per_class', [])
        class_names = None
        if pc_df is not None and 'Class' in pc_df.columns:
            class_names = pc_df['Class'].tolist()
        if fpr_list:
            plot_far_per_class(fpr_list, class_names=class_names,
                                save_path=vis_dir / "far_per_class.png")
            logger.info("[VIS] ✓ far_per_class.png")
    except Exception as e:
        logger.warning(f"[VIS] FAR per class failed: {e}")

    # --- 6e. Efficiency scatter ---
    try:
        models_data = stage3_results.get('models', {})
        if models_data:
            fig = plot_efficiency_scatter(
                models_data, save_path=vis_dir / "efficiency_scatter.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ efficiency_scatter.png")
    except Exception as e:
        logger.warning(f"[VIS] efficiency scatter failed: {e}")

    # --- 6f. Radar chart (all models if multi-model) ---
    try:
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
        radar_data = {}

        if is_multi:
            for mname, mres in stage6_results.items():
                if isinstance(mres, dict) and 'effectiveness' in mres:
                    eff = mres['effectiveness']
                    radar_data[mname] = {
                        'Accuracy': eff.get('accuracy', 0),
                        'Precision': eff.get('precision_weighted', 0),
                        'Recall': eff.get('recall_weighted', 0),
                        'F1': eff.get('f1_weighted', 0),
                        'MCC': max(0, eff.get('mcc', 0))
                    }
        else:
            eff = single_results.get('effectiveness', {})
            radar_data[best_name] = {
                'Accuracy': eff.get('accuracy', 0),
                'Precision': eff.get('precision_weighted', 0),
                'Recall': eff.get('recall_weighted', 0),
                'F1': eff.get('f1_weighted', 0),
                'MCC': max(0, eff.get('mcc', 0))
            }

        if radar_data:
            fig = plot_radar_chart(
                radar_data, metric_names=metric_names,
                save_path=vis_dir / "radar_chart.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ radar_chart.png")
    except Exception as e:
        logger.warning(f"[VIS] radar chart failed: {e}")


def visualize_stage_7(stage7_results: dict, mode: str):
    """Generate and save Stage 7 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import (
        plot_cross_dataset_generalization,
        plot_generalization_gap_bar, plot_generalization_ratio_bar,
        plot_source_vs_target_accuracy
    )

    vis_dir = get_vis_dir(mode) / "stage7"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 7 visualizations → {vis_dir}")

    transfer_list = stage7_results.get('transfer_results', [])

    # --- 7a. Legacy combined chart ---
    try:
        if transfer_list:
            transfer_df = pd.DataFrame(transfer_list)
            fig = plot_cross_dataset_generalization(
                transfer_df, save_path=vis_dir / "cross_dataset_generalization.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ cross_dataset_generalization.png")
    except Exception as e:
        logger.warning(f"[VIS] cross-dataset generalization failed: {e}")

    # --- 7b. Individual: generalization gap ---
    try:
        if transfer_list:
            plot_generalization_gap_bar(transfer_list,
                                         save_path=vis_dir / "generalization_gap.png")
            logger.info("[VIS] ✓ generalization_gap.png")
    except Exception as e:
        logger.warning(f"[VIS] generalization gap failed: {e}")

    # --- 7c. Individual: generalization ratio ---
    try:
        if transfer_list:
            plot_generalization_ratio_bar(transfer_list,
                                            save_path=vis_dir / "generalization_ratio.png")
            logger.info("[VIS] ✓ generalization_ratio.png")
    except Exception as e:
        logger.warning(f"[VIS] generalization ratio failed: {e}")

    # --- 7d. Individual: source vs target accuracy ---
    try:
        if transfer_list:
            plot_source_vs_target_accuracy(transfer_list,
                                             save_path=vis_dir / "source_vs_target_accuracy.png")
            logger.info("[VIS] ✓ source_vs_target_accuracy.png")
    except Exception as e:
        logger.warning(f"[VIS] source vs target accuracy failed: {e}")


def visualize_stage_8(stage8_results: dict, mode: str):
    """Generate and save Stage 8 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import (
        plot_robustness_degradation, plot_concept_drift_impact,
        plot_attack_robustness_curve, plot_overall_robustness_bar,
        plot_concept_drift_accuracy, plot_concept_drift_f1,
        plot_feature_perturbation_curve
    )

    vis_dir = get_vis_dir(mode) / "stage8"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 8 visualizations → {vis_dir}")

    # --- 8a. Legacy combined robustness degradation ---
    try:
        dfs = []
        for key in ['fgsm', 'pgd', 'noise']:
            df = stage8_results.get(key)
            if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                dfs.append(df)
        if dfs:
            all_attacks = pd.concat(dfs, ignore_index=True)
            fig = plot_robustness_degradation(
                all_attacks, save_path=vis_dir / "robustness_degradation.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ robustness_degradation.png")
    except Exception as e:
        logger.warning(f"[VIS] robustness degradation failed: {e}")

    # --- 8b. Individual attack curves (FGSM, PGD, Noise) ---
    try:
        for attack_key, attack_label in [('fgsm', 'FGSM'), ('pgd', 'PGD'), ('noise', 'Gaussian Noise')]:
            df = stage8_results.get(attack_key)
            if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                plot_attack_robustness_curve(
                    df, attack_label, x_label='Epsilon / Sigma',
                    save_path=vis_dir / f"attack_{attack_key}.png"
                )
                logger.info(f"[VIS] ✓ attack_{attack_key}.png")
    except Exception as e:
        logger.warning(f"[VIS] individual attack curves failed: {e}")

    # --- 8c. Overall robustness bar ---
    try:
        rob_scores = stage8_results.get('robustness_scores', {})
        if rob_scores:
            plot_overall_robustness_bar(rob_scores,
                                         save_path=vis_dir / "overall_robustness.png")
            logger.info("[VIS] ✓ overall_robustness.png")
    except Exception as e:
        logger.warning(f"[VIS] overall robustness bar failed: {e}")

    # --- 8d. Legacy concept drift ---
    try:
        drift_results = stage8_results.get('concept_drift', {})
        for mag_key, drift_df in drift_results.items():
            if isinstance(drift_df, pd.DataFrame) and len(drift_df) > 0:
                fig = plot_concept_drift_impact(
                    drift_df, save_path=vis_dir / f"concept_drift_{mag_key}.png"
                )
                plt.close(fig)
                logger.info(f"[VIS] ✓ concept_drift_{mag_key}.png")
    except Exception as e:
        logger.warning(f"[VIS] concept drift failed: {e}")

    # --- 8e. Individual concept drift accuracy & F1 ---
    try:
        drift_results = stage8_results.get('concept_drift', {})
        for mag_key, drift_df in drift_results.items():
            if isinstance(drift_df, pd.DataFrame) and len(drift_df) > 0:
                plot_concept_drift_accuracy(drift_df,
                                              save_path=vis_dir / f"drift_accuracy_{mag_key}.png")
                logger.info(f"[VIS] ✓ drift_accuracy_{mag_key}.png")
                plot_concept_drift_f1(drift_df,
                                        save_path=vis_dir / f"drift_f1_{mag_key}.png")
                logger.info(f"[VIS] ✓ drift_f1_{mag_key}.png")
    except Exception as e:
        logger.warning(f"[VIS] concept drift individual failed: {e}")

    # --- 8f. Feature perturbation curve ---
    try:
        perturb_df = stage8_results.get('feature_perturbation')
        if perturb_df is not None and isinstance(perturb_df, pd.DataFrame) and len(perturb_df) > 0:
            plot_feature_perturbation_curve(perturb_df,
                                              save_path=vis_dir / "feature_perturbation.png")
            logger.info("[VIS] ✓ feature_perturbation.png")
    except Exception as e:
        logger.warning(f"[VIS] feature perturbation failed: {e}")


def visualize_stage_9(stage9_results: dict, mode: str):
    """Generate and save Stage 9 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import (
        plot_critical_difference_diagram, plot_pvalue_heatmap,
        plot_confidence_intervals, plot_effect_sizes
    )

    vis_dir = get_vis_dir(mode) / "stage9"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 9 visualizations → {vis_dir}")

    # --- 9a. Critical difference diagram ---
    try:
        nemenyi = stage9_results.get('nemenyi', {})
        mean_ranks = nemenyi.get('mean_ranks', {})
        cd = nemenyi.get('critical_difference')
        if mean_ranks:
            fig = plot_critical_difference_diagram(
                mean_ranks, cd=cd,
                save_path=vis_dir / "critical_difference.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ critical_difference.png")
    except Exception as e:
        logger.warning(f"[VIS] critical difference diagram failed: {e}")

    # --- 9b. P-value heatmap ---
    try:
        pairwise = stage9_results.get('wilcoxon', [])
        if not pairwise:
            pairwise = stage9_results.get('t_tests', [])
        if pairwise:
            fig = plot_pvalue_heatmap(
                pairwise, save_path=vis_dir / "pvalue_heatmap.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ pvalue_heatmap.png")
    except Exception as e:
        logger.warning(f"[VIS] p-value heatmap failed: {e}")

    # --- 9c. Individual: Confidence intervals ---
    try:
        ci_data = stage9_results.get('bootstrap_ci', {})
        if ci_data:
            plot_confidence_intervals(ci_data,
                                       save_path=vis_dir / "confidence_intervals.png")
            logger.info("[VIS] ✓ confidence_intervals.png")
    except Exception as e:
        logger.warning(f"[VIS] confidence intervals failed: {e}")

    # --- 9d. Individual: Effect sizes ---
    try:
        t_tests = stage9_results.get('t_tests', [])
        if t_tests:
            plot_effect_sizes(t_tests,
                                save_path=vis_dir / "effect_sizes.png")
            logger.info("[VIS] ✓ effect_sizes.png")
    except Exception as e:
        logger.warning(f"[VIS] effect sizes failed: {e}")


def visualize_stage_10(stage10_results: dict, mode: str):
    """Generate and save Stage 10 visualizations (individual academic plots)."""
    from config import get_vis_dir
    from visualizations import plot_final_rankings, plot_recommendation_summary

    vis_dir = get_vis_dir(mode) / "stage10"
    vis_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[VIS] Generating Stage 10 visualizations → {vis_dir}")

    # --- 10a. Final rankings bar ---
    try:
        rankings = stage10_results.get('rankings', {})
        if rankings:
            fig = plot_final_rankings(
                rankings, save_path=vis_dir / "final_rankings.png"
            )
            plt.close(fig)
            logger.info("[VIS] ✓ final_rankings.png")
    except Exception as e:
        logger.warning(f"[VIS] final rankings failed: {e}")

    # --- 10b. Recommendation summary table ---
    try:
        recs = stage10_results.get('recommendations', {})
        if recs:
            plot_recommendation_summary(recs,
                                          save_path=vis_dir / "recommendation_summary.png")
            logger.info("[VIS] ✓ recommendation_summary.png")
    except Exception as e:
        logger.warning(f"[VIS] recommendation summary failed: {e}")


###############################################################################
# STAGE EXECUTION FUNCTIONS
###############################################################################

def run_stage_1(mode: str = "multiclass", dataset_name: str = "CIC_IoT_DIAD_2024",
                quick: bool = False, use_cache: bool = True):
    """Run Stage 1: Dataset Preparation & Baseline (with disk caching)."""
    from stage1_data_preparation import Stage1Pipeline
    from config import CACHE_CONFIG

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 1: Dataset Preparation ({mode.upper()})")
    logger.info("=" * 60)

    # Try loading from cache first
    if use_cache and CACHE_CONFIG.get('enabled', True):
        cached = Stage1Pipeline.load_cache(mode)
        if cached is not None:
            logger.info(f"Stage 1 Complete (from cache) - {cached['n_samples']} samples, "
                        f"{cached['n_features']} features, {cached['n_classes']} classes")
            return cached

    # No cache — run full pipeline
    pipeline = Stage1Pipeline(dataset_name, mode=mode)
    results = pipeline.run(
        balance_classes=False,
        balance_method='none',
        engineer_features=True
    )

    # Save to cache for future runs
    if CACHE_CONFIG.get('enabled', True):
        try:
            pipeline.save_cache(results)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    logger.info(f"Stage 1 Complete - {results['n_samples']} samples, {results['n_features']} features, {results['n_classes']} classes")
    return results


def run_stage_2(stage1_results: dict, mode: str = "multiclass", quick: bool = False):
    """Run Stage 2: Feature Selection Methods Benchmarking."""
    from stage2_feature_selection import Stage2Pipeline

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 2: Feature Selection ({mode.upper()})")
    logger.info("=" * 60)

    pipeline = Stage2Pipeline()
    k_values = [10, 20, 30] if quick else [10, 20, 30, 40, 50]

    results = pipeline.run(
        X=stage1_results['splits']['X_train'],
        y=stage1_results['splits']['y_train'],
        feature_names=stage1_results['feature_columns'],
        k_values=k_values,
        mode=mode
    )

    logger.info("Stage 2 Complete")
    return results


def run_stage_3(stage1_results: dict, mode: str = "multiclass", quick: bool = False,
                use_cache: bool = True):
    """Run Stage 3: Machine Learning Models Benchmarking."""
    from stage3_ml_models import Stage3Pipeline
    from config import CACHE_CONFIG

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 3: ML Models Benchmarking ({mode.upper()})")
    logger.info("=" * 60)

    # Try loading from cache first
    if use_cache and CACHE_CONFIG.get('enabled', True):
        cached = Stage3Pipeline.load_cache(mode)
        if cached is not None:
            n_models = len([m for m in cached['models'].values() if 'error' not in m])
            logger.info(f"Stage 3 Complete (from cache) - {n_models} models, "
                        f"Best: {cached['best_model']}")
            return cached

    pipeline = Stage3Pipeline()

    if quick:
        models_to_run = ['GRU', 'FT_Transformer', 'TabNet']
    else:
        models_to_run = None  # Run all 9 DL models

    # Subsample training/val data to avoid OOM on 11M+ rows
    MAX_TRAIN = 500_000
    MAX_VAL = 100_000
    X_tr = stage1_results['splits']['X_train']
    y_tr = stage1_results['splits']['y_train']
    X_vl = stage1_results['splits']['X_val']
    y_vl = stage1_results['splits']['y_val']

    if len(X_tr) > MAX_TRAIN:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        logger.info(f"Subsampled training data: {len(stage1_results['splits']['X_train']):,} → {MAX_TRAIN:,}")
    if len(X_vl) > MAX_VAL:
        rng = np.random.RandomState(42)
        idx_v = rng.choice(len(X_vl), MAX_VAL, replace=False)
        X_vl, y_vl = X_vl[idx_v], y_vl[idx_v]
        logger.info(f"Subsampled validation data: {len(stage1_results['splits']['X_val']):,} → {MAX_VAL:,}")

    results = pipeline.run(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_vl,
        y_val=y_vl,
        X_test=stage1_results['splits']['X_test'],
        y_test=stage1_results['splits']['y_test'],
        models_to_run=models_to_run,
        tune_hyperparams=not quick,
        mode=mode,
        class_weights=stage1_results.get('class_weights')
    )
    del X_tr, y_tr, X_vl, y_vl

    # Save to cache for future runs
    try:
        pipeline.save_cache(results, mode=mode)
    except Exception as e:
        logger.warning(f"Stage 3 cache save failed: {e}")

    logger.info(f"Stage 3 Complete - Best Model: {results['best_model']}")
    return results


def run_stage_4(stage1_results: dict, stage2_results: dict,
                mode: str = "multiclass", quick: bool = False):
    """Run Stage 4: Comprehensive Results Benchmarking."""
    from stage4_comprehensive_results import run_comprehensive_benchmark

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 4: Comprehensive Results ({mode.upper()})")
    logger.info("=" * 60)

    # Reconstruct full X/y from splits if not loaded (memory optimization)
    X_full = stage1_results.get('X')
    y_full = stage1_results.get('y')
    if X_full is None:
        X_full = np.concatenate([stage1_results['splits']['X_train'],
                                 stage1_results['splits']['X_val'],
                                 stage1_results['splits']['X_test']])
        y_full = np.concatenate([stage1_results['splits']['y_train'],
                                 stage1_results['splits']['y_val'],
                                 stage1_results['splits']['y_test']])

    results = run_comprehensive_benchmark(
        X=X_full,
        y=y_full,
        feature_selection_results=stage2_results,
        dataset_name="CIC_IoT_DIAD_2024",
        mode=mode
    )
    del X_full, y_full

    logger.info("Stage 4 Complete")
    return results


def run_stage_5(stage1_results: dict, stage3_results: dict,
                mode: str = "multiclass", quick: bool = False):
    """Run Stage 5: XAI Explanations Quality Benchmarking."""
    from stage5_xai_quality import Stage5Pipeline
    from stage3_ml_models import FTTransformerModel

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 5: XAI Quality ({mode.upper()})")
    logger.info("=" * 60)

    # Use best DL model from Stage 3 if available, otherwise train FT-Transformer
    best_model_obj = None
    if stage3_results and 'models' in stage3_results:
        best_name = stage3_results.get('best_model')
        if best_name and best_name in stage3_results['models']:
            best_model_obj = stage3_results['models'][best_name].get('model_object')

    if best_model_obj is None:
        # Subsample to fit in 8GB VRAM — 500K rows is enough for XAI evaluation
        MAX_TRAIN = 500_000
        X_tr = stage1_results['splits']['X_train']
        y_tr = stage1_results['splits']['y_train']
        if len(X_tr) > MAX_TRAIN:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            logger.info(f"Subsampled training data to {MAX_TRAIN:,} rows for XAI model")

        # Subsample validation too
        MAX_VAL = 100_000
        X_vl = stage1_results['splits']['X_val']
        y_vl = stage1_results['splits']['y_val']
        if len(X_vl) > MAX_VAL:
            idx_v = rng.choice(len(X_vl), MAX_VAL, replace=False)
            X_vl, y_vl = X_vl[idx_v], y_vl[idx_v]

        logger.info("Training FT-Transformer for XAI evaluation...")
        model = FTTransformerModel()
        model.fit(
            X_tr, y_tr, X_vl, y_vl,
            tune_hyperparams=False,
            class_weights=stage1_results.get('class_weights')
        )
        best_model_obj = model
        del X_tr, y_tr, X_vl, y_vl

    # Subsample for XAI explanations (SHAP/LIME background + test)
    MAX_XAI = 50_000
    X_xai_train = stage1_results['splits']['X_train']
    X_xai_test = stage1_results['splits']['X_test']
    y_xai_test = stage1_results['splits']['y_test']
    if len(X_xai_train) > MAX_XAI:
        rng2 = np.random.RandomState(42)
        idx_xai = rng2.choice(len(X_xai_train), MAX_XAI, replace=False)
        X_xai_train = X_xai_train[idx_xai]
    if len(X_xai_test) > 10_000:
        rng2 = np.random.RandomState(42)
        idx_xai_te = rng2.choice(len(X_xai_test), 10_000, replace=False)
        X_xai_test = X_xai_test[idx_xai_te]
        y_xai_test = y_xai_test[idx_xai_te]

    pipeline = Stage5Pipeline()
    results = pipeline.run(
        model=best_model_obj,
        X_train=X_xai_train,
        X_test=X_xai_test,
        y_test=y_xai_test,
        feature_names=stage1_results['feature_columns'],
        mode=mode
    )
    del X_xai_train, X_xai_test, y_xai_test

    logger.info("Stage 5 Complete")
    return results


def _rebuild_model_wrappers(stage1_results: dict, mode: str = "multiclass"):
    """Rebuild model wrapper objects from Stage 3 cache.

    Binary .pkl files contain full wrapper objects (use directly).
    Multiclass .pkl files contain raw nn.Module objects (reconstruct wrapper).
    """
    from stage3_ml_models import (
        GRUModel, ResNet1DModel, FTTransformerModel, CNNLSTMModel,
        BiLSTMAttentionModel, TabNetModel, TCNModel, TransformerModel,
        AutoencoderClassifierModel, VotingEnsembleModel, Stage3Pipeline
    )
    from config import get_results_dir
    from sklearn.preprocessing import StandardScaler
    import pickle

    # Try loading via Stage3Pipeline.load_cache first (it returns model_object)
    cache_results = Stage3Pipeline.load_cache(mode)
    if cache_results is None:
        logger.warning("Stage 3 cache not found")
        return {}

    # Map model names to wrapper classes
    wrapper_classes = {
        'GRU': GRUModel, 'ResNet1D': ResNet1DModel,
        'FT_Transformer': FTTransformerModel, 'CNN_LSTM': CNNLSTMModel,
        'BiLSTM_Attention': BiLSTMAttentionModel, 'TabNet': TabNetModel,
        'TCN': TCNModel, 'Transformer': TransformerModel,
        'AE_Classifier': AutoencoderClassifierModel,
    }

    # Fit a scaler on training data (needed by all wrapper predict methods)
    X_train = stage1_results['splits']['X_train']
    # Use a subsample for scaler fitting (scaler only needs stats)
    MAX_FIT = 100_000
    if len(X_train) > MAX_FIT:
        rng = np.random.RandomState(42)
        X_fit = X_train[rng.choice(len(X_train), MAX_FIT, replace=False)]
    else:
        X_fit = X_train
    shared_scaler = StandardScaler().fit(X_fit)
    del X_fit

    model_wrappers = {}
    for name, data in cache_results['models'].items():
        if 'error' in data:
            continue
        raw_obj = data.get('model_object')
        if raw_obj is None:
            continue

        if hasattr(raw_obj, 'predict') and hasattr(raw_obj, 'predict_proba'):
            # Already a proper wrapper object (e.g. binary cache saves full wrappers)
            model_wrappers[name] = raw_obj
            logger.info(f"Loaded wrapper directly: {name}")
        elif name in wrapper_classes:
            # Raw nn.Module, reconstruct wrapper
            wrapper = wrapper_classes[name].__new__(wrapper_classes[name])
            wrapper.name = name
            wrapper.random_state = 42
            wrapper.scaler = shared_scaler
            wrapper.model = raw_obj
            wrapper.best_params = data.get('best_params', {})
            wrapper.training_time = data.get('training_time', 0)
            wrapper.history = data.get('training_history')
            wrapper.config = {}
            model_wrappers[name] = wrapper
            logger.info(f"Rebuilt wrapper: {name}")
        elif name == 'VotingEnsemble':
            # VotingEnsemble stores the full ensemble object
            if hasattr(raw_obj, 'predict'):
                model_wrappers[name] = raw_obj
                logger.info(f"Loaded VotingEnsemble directly")
        else:
            logger.warning(f"Unknown model type: {name}")

    # Build VotingEnsemble from top wrappers if not already loaded
    if 'VotingEnsemble' not in model_wrappers and len(model_wrappers) >= 3:
        try:
            comp_csv = get_results_dir(mode) / 'stage3_ml_models' / 'model_comparison.csv'
            if comp_csv.exists():
                comp_df = pd.read_csv(comp_csv)
                top3 = comp_df.nlargest(3, 'Accuracy')['Model'].tolist()
                top3 = [n for n in top3 if n in model_wrappers and n != 'VotingEnsemble']
                if len(top3) >= 2:
                    members = [model_wrappers[n] for n in top3]
                    ensemble = VotingEnsembleModel(members, top3)
                    model_wrappers['VotingEnsemble'] = ensemble
                    logger.info(f"Rebuilt VotingEnsemble from: {top3}")
        except Exception as e:
            logger.warning(f"Failed to rebuild VotingEnsemble: {e}")

    return model_wrappers


def run_stage_6(stage1_results: dict, stage3_results: dict = None,
                mode: str = "multiclass", quick: bool = False):
    """Run Stage 6: Performance Metrics Benchmarking for ALL models."""
    from stage6_performance_metrics import Stage6Pipeline
    import gc

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 6: Performance Metrics - ALL MODELS ({mode.upper()})")
    logger.info("=" * 60)

    from config import get_results_dir
    class_names = list(stage1_results['label_mapping'].keys())
    X_test = stage1_results['splits']['X_test']
    y_test = stage1_results['splits']['y_test']

    # Subsample test set for inference benchmarking
    MAX_TEST = 50_000
    if len(X_test) > MAX_TEST:
        rng = np.random.RandomState(42)
        idx_te = rng.choice(len(X_test), MAX_TEST, replace=False)
        X_test_sub, y_test_sub = X_test[idx_te], y_test[idx_te]
    else:
        X_test_sub, y_test_sub = X_test, y_test

    # Rebuild wrapper objects with predict() from Stage 3 cache
    model_sources = _rebuild_model_wrappers(stage1_results, mode)

    if not model_sources:
        logger.error("No models found! Check stage3_cache/ directory.")
        return {}

    logger.info(f"Evaluating {len(model_sources)} models: {list(model_sources.keys())}")

    # Load stage3 comparison CSV for training times
    stage3_csv = get_results_dir(mode) / 'stage3_ml_models' / 'model_comparison.csv'
    stage3_timing = {}
    if stage3_csv.exists():
        comp_df = pd.read_csv(stage3_csv)
        for _, row in comp_df.iterrows():
            stage3_timing[row['Model']] = {
                'train_time_s': row.get('Train Time (s)', 0),
                'inference_ms': row.get('Inference (ms)', 0),
            }

    all_results = {}
    for model_name, model in model_sources.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Stage 6 benchmarking: {model_name}")
        logger.info(f"{'='*50}")

        pipeline = Stage6Pipeline()
        try:
            result = pipeline.run(
                model=model,
                X_train=None,  # skip re-training
                y_train=None,
                X_test=X_test_sub,
                y_test=y_test_sub,
                model_name=model_name,
                class_names=class_names,
                mode=mode
            )
            # Patch in Stage 3 training time
            timing = stage3_timing.get(model_name, {})
            if timing and 'efficiency' in result:
                result['efficiency']['training_time_s_stage3'] = timing.get('train_time_s', 0)
            all_results[model_name] = result
        except Exception as e:
            logger.error(f"Stage 6 failed for {model_name}: {e}")
            import traceback; traceback.print_exc()

        gc.collect()

    logger.info(f"\nStage 6 Complete — evaluated {len(all_results)} models")
    return all_results


def run_stage_7(stage1_results: dict, mode: str = "multiclass", quick: bool = False):
    """Run Stage 7: Cross-Dataset Generalization Test.

    Since we only have one dataset (CIC IoT-DIAD 2024), we simulate
    cross-dataset generalization by splitting the balanced data into
    two disjoint halves (Split_A and Split_B) and testing transfer
    between them. This avoids reloading the full 19.5M rows.
    """
    from stage7_cross_dataset import Stage7Pipeline
    from stage3_ml_models import GRUModel, ResNet1DModel, FTTransformerModel
    from sklearn.model_selection import StratifiedShuffleSplit

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 7: Cross-Dataset Generalization ({mode.upper()})")
    logger.info("=" * 60)

    # Subsample for speed — 200K rows total (100K per split)
    MAX_TOTAL = 200_000
    X = np.concatenate([stage1_results['splits']['X_train'],
                        stage1_results['splits']['X_val'],
                        stage1_results['splits']['X_test']])
    y = np.concatenate([stage1_results['splits']['y_train'],
                        stage1_results['splits']['y_val'],
                        stage1_results['splits']['y_test']])
    if len(X) > MAX_TOTAL:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), MAX_TOTAL, replace=False)
        X, y = X[idx], y[idx]
        logger.info(f"Subsampled to {MAX_TOTAL:,} rows for cross-dataset test")

    features = stage1_results['feature_columns']
    mapping = stage1_results['label_mapping']

    # Index-based stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    idx_a, idx_b = next(sss.split(X, y))
    X_a, y_a = X[idx_a], y[idx_a]
    X_b, y_b = X[idx_b], y[idx_b]
    del idx_a, idx_b, X, y
    logger.info(f"Split_A: {len(y_a)} samples, Split_B: {len(y_b)} samples")

    datasets = {
        "CIC_IoT_DIAD_SplitA": {
            'X': X_a, 'y': y_a,
            'features': features, 'mapping': mapping
        },
        "CIC_IoT_DIAD_SplitB": {
            'X': X_b, 'y': y_b,
            'features': features, 'mapping': mapping
        }
    }

    # Use fewer epochs for cross-dataset test (convergence is fast on 500K rows)
    models = {
        'GRU': GRUModel(),
        'ResNet1D': ResNet1DModel()
    }
    for m in models.values():
        m.config = {**m.config, 'epochs': 30}

    if not quick:
        ft = FTTransformerModel()
        ft.config = {**ft.config, 'epochs': 30}
        models['FT_Transformer'] = ft

    pipeline = Stage7Pipeline()
    results = pipeline.run(datasets, models, mode=mode)

    logger.info("Stage 7 Complete")
    return results


def run_stage_8(stage1_results: dict, stage3_results: dict = None,
                mode: str = "multiclass", quick: bool = False):
    """Run Stage 8: Robustness & Adversarial Testing."""
    from stage8_robustness_testing import Stage8Pipeline
    from stage3_ml_models import GRUModel

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 8: Robustness Testing ({mode.upper()})")
    logger.info("=" * 60)

    # Use best model from Stage 3 if available (PyTorch models support gradient-based FGSM/PGD)
    model_obj = None
    model_name = "GRU"
    if stage3_results and 'models' in stage3_results:
        best_name = stage3_results.get('best_model')
        if best_name and best_name in stage3_results['models']:
            model_obj = stage3_results['models'][best_name].get('model_object')
            model_name = best_name

    if model_obj is None:
        # Subsample to fit in 8GB VRAM
        MAX_TRAIN = 500_000
        X_tr = stage1_results['splits']['X_train']
        y_tr = stage1_results['splits']['y_train']
        if len(X_tr) > MAX_TRAIN:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]
            logger.info(f"Subsampled training data to {MAX_TRAIN:,} for Stage 8 model")
        MAX_VAL = 100_000
        X_vl = stage1_results['splits']['X_val']
        y_vl = stage1_results['splits']['y_val']
        if len(X_vl) > MAX_VAL:
            idx_v = rng.choice(len(X_vl), MAX_VAL, replace=False)
            X_vl, y_vl = X_vl[idx_v], y_vl[idx_v]
        model = GRUModel()
        model.fit(
            X_tr, y_tr, X_vl, y_vl,
            tune_hyperparams=False,
            class_weights=stage1_results.get('class_weights')
        )
        model_obj = model
        del X_tr, y_tr, X_vl, y_vl

    # Extract raw nn.Module for gradient-based attacks (FGSM/PGD)
    pytorch_net = getattr(model_obj, 'model', None)

    # Subsample test set for adversarial attacks (gradient ops on 2.5M rows is too slow)
    MAX_TEST = 50_000
    X_te = stage1_results['splits']['X_test']
    y_te = stage1_results['splits']['y_test']
    if len(X_te) > MAX_TEST:
        rng2 = np.random.RandomState(42)
        idx_te = rng2.choice(len(X_te), MAX_TEST, replace=False)
        X_te, y_te = X_te[idx_te], y_te[idx_te]
        logger.info(f"Subsampled test data to {MAX_TEST:,} for adversarial testing")

    pipeline = Stage8Pipeline()
    results = pipeline.run(
        model=model_obj,
        X_test=X_te,
        y_test=y_te,
        pytorch_model=pytorch_net,
        model_name=model_name,
        mode=mode
    )

    logger.info("Stage 8 Complete")
    return results


def run_stage_9(stage3_results: dict, stage1_results: dict = None,
                mode: str = "multiclass", quick: bool = False):
    """Run Stage 9: Statistical Significance Testing.

    Uses bootstrap resampling on test-set predictions to generate
    per-model score distributions (30 resamples). This gives meaningful
    variance for Wilcoxon/Friedman tests without retraining DL models.
    """
    from stage9_statistical_testing import Stage9Pipeline
    from sklearn.metrics import accuracy_score
    import gc

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 9: Statistical Testing ({mode.upper()})")
    logger.info("=" * 60)

    # --- Rebuild model wrappers with predict() from Stage 3 cache ---
    model_sources = _rebuild_model_wrappers(stage1_results, mode)

    X_test = stage1_results['splits']['X_test']
    y_test = stage1_results['splits']['y_test']

    # Subsample test set to 50K for speed
    MAX_TEST = 50_000
    rng = np.random.RandomState(42)
    if len(X_test) > MAX_TEST:
        idx_sub = rng.choice(len(X_test), MAX_TEST, replace=False)
        X_test_s, y_test_s = X_test[idx_sub], y_test[idx_sub]
    else:
        X_test_s, y_test_s = X_test, y_test

    # --- Get predictions for each model ---
    predictions = {}
    for name, model in model_sources.items():
        try:
            y_pred = model.predict(X_test_s)
            predictions[name] = y_pred
            acc = accuracy_score(y_test_s, y_pred)
            logger.info(f"{name}: test accuracy = {acc:.4f}")
        except Exception as e:
            logger.warning(f"Predict failed for {name}: {e}")
    gc.collect()

    if len(predictions) < 2:
        logger.error("Need at least 2 models with predictions for statistical testing")
        return {}

    # --- Bootstrap resampling to create score distributions ---
    n_bootstrap = 30 if not quick else 10
    model_results = {}
    for name, y_pred in predictions.items():
        scores = []
        for b in range(n_bootstrap):
            idx_b = rng.choice(len(y_test_s), len(y_test_s), replace=True)
            scores.append(accuracy_score(y_test_s[idx_b], y_pred[idx_b]))
        model_results[name] = {'fold_scores': scores}
        logger.info(f"{name}: bootstrap mean={np.mean(scores):.4f} std={np.std(scores):.4f}")

    pipeline = Stage9Pipeline()
    results = pipeline.run(model_results, predictions=predictions,
                           y_true=y_test_s, mode=mode)

    logger.info("Stage 9 Complete")
    return results


def run_stage_10(mode: str = "multiclass", quick: bool = False):
    """Run Stage 10: Final Benchmarking Summary & Report."""
    from stage10_final_report import Stage10Pipeline

    logger.info("=" * 60)
    logger.info(f"RUNNING STAGE 10: Final Report ({mode.upper()})")
    logger.info("=" * 60)

    pipeline = Stage10Pipeline()
    results = pipeline.run(mode=mode)

    logger.info("Stage 10 Complete - Final Report Generated")
    return results


def _run_pipeline_for_mode(mode: str, quick: bool = False,
                            stages: list = None) -> dict:
    """Run all (or selected) stages for a single classification mode."""
    logger.info(f"\n{'#'*70}")
    logger.info(f"# RUNNING IN {mode.upper()} MODE")
    logger.info(f"{'#'*70}\n")

    all_results = {}

    # Determine which stages to run
    if stages is None:
        stages = list(range(1, 11))

    # Stage 1 is needed for most other stages
    stage1_results = None
    stage2_results = None
    stage3_results = None

    if any(s >= 1 for s in stages):
        stage1_results = run_stage_1(mode=mode, quick=quick)
        all_results['stage1'] = stage1_results
        visualize_stage_1(stage1_results, mode)

    if 2 in stages and stage1_results:
        stage2_results = run_stage_2(stage1_results, mode=mode, quick=quick)
        all_results['stage2'] = stage2_results
        visualize_stage_2(stage2_results, mode)

    if 3 in stages and stage1_results:
        stage3_results = run_stage_3(stage1_results, mode=mode, quick=quick)
        all_results['stage3'] = stage3_results
        visualize_stage_3(stage3_results, stage1_results, mode)

        # --- Error Analysis (after Stage 3) ---
        try:
            from error_analysis import run_error_analysis_for_stage3
            from visualizations import plot_confusion_pairs_bar, plot_error_rate_per_class
            from config import get_vis_dir

            logger.info("Running Error Analysis on best model...")
            error_results = run_error_analysis_for_stage3(
                stage3_results, stage1_results, mode=mode)
            all_results['error_analysis'] = error_results

            # Visualize error analysis
            ea_vis_dir = get_vis_dir(mode) / "error_analysis"
            ea_vis_dir.mkdir(parents=True, exist_ok=True)
            for ea_model, ea_data in error_results.items():
                if 'confusion_pairs' in ea_data:
                    plot_confusion_pairs_bar(
                        ea_data['confusion_pairs'], top_k=10,
                        save_path=ea_vis_dir / f"confusion_pairs_{ea_model}.png")
                    logger.info(f"[VIS] ✓ confusion_pairs_{ea_model}.png")
                if 'per_class_error' in ea_data:
                    plot_error_rate_per_class(
                        ea_data['per_class_error'],
                        save_path=ea_vis_dir / f"error_rate_per_class_{ea_model}.png")
                    logger.info(f"[VIS] ✓ error_rate_per_class_{ea_model}.png")
            logger.info("Error Analysis Complete")
        except Exception as e:
            logger.warning(f"Error Analysis failed: {e}")

    if 4 in stages and stage1_results:
        if stage2_results is None and stage1_results:
            stage2_results = run_stage_2(stage1_results, mode=mode, quick=quick)
            all_results['stage2'] = stage2_results
            visualize_stage_2(stage2_results, mode)
        stage4_results = run_stage_4(stage1_results, stage2_results, mode=mode, quick=quick)
        all_results['stage4'] = stage4_results
        visualize_stage_4(stage4_results, mode)

        # --- Ablation Study (after Stage 4) ---
        try:
            from stage4_comprehensive_results import run_ablation_study
            from visualizations import plot_ablation_curve
            from config import get_vis_dir, get_results_dir

            logger.info("Running Ablation Study...")
            # Get best features from Stage 2
            best_features_ranked = None
            if stage2_results:
                best_method = stage2_results.get('best_method', '')
                methods = stage2_results.get('methods', {})
                if best_method and best_method in methods:
                    imp_list = methods[best_method].get('importance', [])
                    if imp_list and isinstance(imp_list, list):
                        best_features_ranked = [f['feature'] for f in imp_list
                                                if isinstance(f, dict) and 'feature' in f]

            if best_features_ranked is None:
                best_features_ranked = stage1_results.get('feature_columns', [])

            ablation_df = run_ablation_study(
                X=stage1_results['splits']['X_train'],
                y=stage1_results['splits']['y_train'],
                feature_names=stage1_results.get('feature_columns', []),
                best_features_ranked=best_features_ranked,
                k_values=[10, 20, 30, 50, None],
                mode=mode,
                class_weights=stage1_results.get('class_weights')
            )
            all_results['ablation'] = ablation_df

            # Save ablation CSV
            abl_dir = get_results_dir(mode) / 'ablation'
            abl_dir.mkdir(parents=True, exist_ok=True)
            ablation_df.to_csv(abl_dir / 'ablation_results.csv', index=False)

            # Visualize ablation curve
            abl_vis_dir = get_vis_dir(mode) / "ablation"
            abl_vis_dir.mkdir(parents=True, exist_ok=True)
            plot_ablation_curve(ablation_df,
                                save_path=abl_vis_dir / "ablation_curve.png")
            logger.info("[VIS] ✓ ablation_curve.png")
            logger.info("Ablation Study Complete")
        except Exception as e:
            logger.warning(f"Ablation Study failed: {e}")

    if 5 in stages and stage1_results:
        if stage3_results is None:
            stage3_results = run_stage_3(stage1_results, mode=mode, quick=quick)
            all_results['stage3'] = stage3_results
            visualize_stage_3(stage3_results, stage1_results, mode)
        stage5_results = run_stage_5(stage1_results, stage3_results, mode=mode, quick=quick)
        all_results['stage5'] = stage5_results
        visualize_stage_5(stage5_results, mode)

    if 6 in stages and stage1_results:
        stage6_results = run_stage_6(stage1_results, stage3_results=stage3_results,
                                      mode=mode, quick=quick)
        all_results['stage6'] = stage6_results
        visualize_stage_6(stage6_results, stage3_results or {}, mode)

    if 7 in stages and stage1_results:
        stage7_results = run_stage_7(stage1_results, mode=mode, quick=quick)
        all_results['stage7'] = stage7_results
        visualize_stage_7(stage7_results, mode)

    if 8 in stages and stage1_results:
        stage8_results = run_stage_8(stage1_results, stage3_results=stage3_results,
                                      mode=mode, quick=quick)
        all_results['stage8'] = stage8_results
        visualize_stage_8(stage8_results, mode)

    if 9 in stages and stage1_results:
        if stage3_results is None:
            stage3_results = run_stage_3(stage1_results, mode=mode, quick=quick)
            all_results['stage3'] = stage3_results
            visualize_stage_3(stage3_results, stage1_results, mode)
        stage9_results = run_stage_9(stage3_results, stage1_results, mode=mode, quick=quick)
        all_results['stage9'] = stage9_results
        visualize_stage_9(stage9_results, mode)

    if 10 in stages:
        stage10_results = run_stage_10(mode=mode, quick=quick)
        all_results['stage10'] = stage10_results
        visualize_stage_10(stage10_results, mode)

    # --- LaTeX Table Export (at the end of all stages) ---
    try:
        from visualizations import export_latex_table
        from config import get_results_dir

        latex_dir = get_results_dir(mode) / 'latex_tables'
        latex_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting LaTeX tables → {latex_dir}")

        # Stage 3 comparison table
        if stage3_results and stage3_results.get('comparison') is not None:
            comp_df = stage3_results['comparison']
            if len(comp_df) > 0:
                export_latex_table(
                    comp_df,
                    caption=f"Model Comparison Results ({mode.title()} Classification)",
                    label=f"tab:model_comparison_{mode}",
                    save_path=latex_dir / "model_comparison.tex",
                    bold_max_cols=['Accuracy', 'F1_Macro', 'Balanced_Acc', 'G_Mean']
                )
                logger.info("[LaTeX] ✓ model_comparison.tex")

        # Stage 6 per-class table
        stage6_results = all_results.get('stage6', {})
        if stage6_results:
            pc_df = stage6_results.get('per_class')
            if pc_df is not None and len(pc_df) > 0:
                export_latex_table(
                    pc_df,
                    caption=f"Per-Class Performance ({mode.title()} Classification)",
                    label=f"tab:per_class_{mode}",
                    save_path=latex_dir / "per_class_performance.tex",
                    bold_max_cols=['Precision', 'Recall', 'F1-Score']
                )
                logger.info("[LaTeX] ✓ per_class_performance.tex")

        # Stage 9 statistical testing table
        stage9_results = all_results.get('stage9', {})
        if stage9_results:
            t_tests = stage9_results.get('t_tests', [])
            if t_tests:
                tt_df = pd.DataFrame(t_tests)
                export_latex_table(
                    tt_df,
                    caption=f"Pairwise Statistical Significance ({mode.title()})",
                    label=f"tab:stat_tests_{mode}",
                    save_path=latex_dir / "statistical_tests.tex"
                )
                logger.info("[LaTeX] ✓ statistical_tests.tex")

        # Ablation table
        ablation_df = all_results.get('ablation')
        if ablation_df is not None and isinstance(ablation_df, pd.DataFrame) and len(ablation_df) > 0:
            export_latex_table(
                ablation_df,
                caption=f"Ablation Study — Accuracy vs Feature Count ({mode.title()})",
                label=f"tab:ablation_{mode}",
                save_path=latex_dir / "ablation_study.tex",
                bold_max_cols=['accuracy', 'f1']
            )
            logger.info("[LaTeX] ✓ ablation_study.tex")

        logger.info("LaTeX export complete")
    except Exception as e:
        logger.warning(f"LaTeX export failed: {e}")

    return all_results


def run_all_stages(quick: bool = False, modes: list = None):
    """Run all 10 stages for each classification mode."""
    if modes is None:
        modes = ['binary', 'multiclass']

    print_banner()
    start_time = datetime.now()

    all_mode_results = {}
    for mode in modes:
        mode_results = _run_pipeline_for_mode(mode, quick)
        all_mode_results[mode] = mode_results

    # --- Binary vs Multiclass Comparison ---
    if len(modes) == 2 and 'binary' in all_mode_results and 'multiclass' in all_mode_results:
        try:
            from visualizations import plot_binary_vs_multiclass_comparison, export_latex_table
            from config import BASE_DIR

            logger.info("Generating Binary vs Multiclass Comparison...")
            comp_dir = BASE_DIR / 'results' / 'comparison'
            comp_dir.mkdir(parents=True, exist_ok=True)

            # Load comparison DataFrames from Stage 3
            binary_comp = all_mode_results['binary'].get('stage3', {}).get('comparison')
            multi_comp = all_mode_results['multiclass'].get('stage3', {}).get('comparison')

            if binary_comp is not None and multi_comp is not None and \
               len(binary_comp) > 0 and len(multi_comp) > 0:
                # Plot comparison for key metrics
                for metric in ['Accuracy', 'F1_Macro', 'Balanced_Acc', 'G_Mean']:
                    if metric in binary_comp.columns and metric in multi_comp.columns:
                        plot_binary_vs_multiclass_comparison(
                            binary_comp, multi_comp, metric=metric,
                            save_path=comp_dir / f"binary_vs_multi_{metric.lower()}.png"
                        )
                        logger.info(f"[VIS] ✓ binary_vs_multi_{metric.lower()}.png")

                # Create combined summary table
                summary_rows = []
                for metric in ['Accuracy', 'F1_Macro', 'Balanced_Acc', 'G_Mean']:
                    if metric in binary_comp.columns and metric in multi_comp.columns:
                        summary_rows.append({
                            'Metric': metric,
                            'Binary_Best': f"{binary_comp[metric].max():.4f}",
                            'Binary_Mean': f"{binary_comp[metric].mean():.4f}",
                            'Multi_Best': f"{multi_comp[metric].max():.4f}",
                            'Multi_Mean': f"{multi_comp[metric].mean():.4f}",
                        })
                if summary_rows:
                    summary_df = pd.DataFrame(summary_rows)
                    summary_df.to_csv(comp_dir / 'binary_vs_multiclass_summary.csv', index=False)
                    export_latex_table(
                        summary_df,
                        caption="Binary vs Multiclass Classification Comparison",
                        label="tab:binary_vs_multiclass",
                        save_path=comp_dir / "binary_vs_multiclass.tex"
                    )
                    logger.info("[LaTeX] ✓ binary_vs_multiclass.tex")

                logger.info("Binary vs Multiclass Comparison Complete")
            else:
                logger.info("Skipping comparison: Stage 3 results not available for both modes")
        except Exception as e:
            logger.warning(f"Binary vs Multiclass comparison failed: {e}")

    end_time = datetime.now()
    duration = end_time - start_time

    logger.info("=" * 70)
    logger.info("BENCHMARKING STUDY COMPLETE!")
    logger.info(f"Modes: {modes}")
    logger.info(f"Total Duration: {duration}")
    logger.info("=" * 70)

    return all_mode_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="XAI-Based IDS Benchmarking Study",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--stage', '-s',
        type=int,
        nargs='+',
        help='Specific stage(s) to run (1-10)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick run with reduced data and models'
    )

    parser.add_argument(
        '--mode', '-m',
        type=str,
        nargs='+',
        default=['binary', 'multiclass'],
        choices=['binary', 'multiclass'],
        help='Classification mode(s): binary, multiclass, or both (default: both)'
    )

    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='CIC_IoT_DIAD_2024',
        help='Dataset to use (default: CIC_IoT_DIAD_2024)'
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable Stage 1 caching (force reload from CSV files)'
    )

    args = parser.parse_args()

    # Apply cache override
    if args.no_cache:
        from config import CACHE_CONFIG
        CACHE_CONFIG['enabled'] = False
        logger.info("Cache disabled via --no-cache flag")

    print_banner()

    if args.stage:
        # Run specific stages for each mode
        logger.info(f"Running stages: {args.stage}, modes: {args.mode}")

        for mode in args.mode:
            _run_pipeline_for_mode(mode, args.quick, stages=sorted(args.stage))
    else:
        # Run all stages for each mode
        run_all_stages(args.quick, args.mode)


if __name__ == "__main__":
    main()
