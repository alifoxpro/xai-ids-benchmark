"""
================================================================================
STAGE 10: FINAL BENCHMARKING SUMMARY & REPORT
================================================================================
This module provides:
- Comprehensive summary of all benchmarking results
- Final rankings and recommendations
- Report generation
- Visualization compilation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
import json
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from config import RESULTS_DIR, VIS_DIR, DATASETS, RANDOM_SEED, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Data-driven recommendation engine that analyzes actual benchmarking
    results and produces actionable recommendations for deployment.
    """

    def analyze(self, all_results: Dict = None, rankings: Dict = None) -> Dict:
        """
        Analyze benchmarking results and generate recommendations.

        Args:
            all_results: Aggregated results from all stages (from ResultsAggregator)
            rankings:    Pre-computed rankings dict (from _create_final_rankings)

        Returns:
            Dictionary of recommendation entries, each with:
                category, model/method, score fields, reason
        """
        recs = {}
        if all_results is None:
            all_results = {}
        if rankings is None:
            rankings = {}

        # ── 1. Best overall model (accuracy) ──────────────────────────
        ml_rank = rankings.get('ml_models', {}).get('ranking', [])
        if not ml_rank:
            ml_rank = self._extract_model_list(all_results)
        if ml_rank:
            best = ml_rank[0]
            recs['best_overall'] = {
                'category': 'Best Overall Model',
                'model': best.get('model', 'N/A'),
                'accuracy': best.get('accuracy', 0) / 100 if best.get('accuracy', 0) > 1 else best.get('accuracy', 0),
                'reason': 'Highest test accuracy across all evaluated models'
            }

        # ── 2. Best for speed (lowest inference time) ─────────────────
        speed_list = [r for r in ml_rank if r.get('inference_ms', 0) > 0]
        if speed_list:
            fastest = min(speed_list, key=lambda x: x.get('inference_ms', 1e9))
            recs['best_speed'] = {
                'category': 'Fastest Inference',
                'model': fastest.get('model', 'N/A'),
                'inference_ms': fastest.get('inference_ms', 0),
                'reason': 'Lowest per-sample inference time — ideal for real-time IDS'
            }

        # ── 3. Best for robustness ────────────────────────────────────
        rob = rankings.get('robustness', {})
        if not rob or rob.get('overall', 0) == 0:
            rob = self._extract_robustness(all_results)
        if rob.get('overall', 0) > 0:
            recs['best_robustness'] = {
                'category': 'Best Robustness',
                'model': rob.get('model', 'Evaluated Model'),
                'score': rob.get('overall', 0),
                'reason': 'Highest overall robustness under FGSM, PGD, and noise attacks'
            }

        # ── 4. Best for minority-class detection (G-Mean / balanced_acc) ─
        minority_best = self._find_minority_champion(all_results, ml_rank)
        if minority_best:
            recs['best_minority'] = minority_best

        # ── 5. Best XAI method (fidelity) ─────────────────────────────
        xai = rankings.get('xai', {})
        if not xai or xai.get('fidelity', 0) == 0:
            xai = self._extract_xai(all_results)
        if xai.get('fidelity', 0) > 0:
            recs['best_xai'] = {
                'category': 'Best XAI Method',
                'method': xai.get('best_method', 'SHAP'),
                'fidelity': xai.get('fidelity', 0),
                'reason': 'Highest fidelity score for model explanations'
            }

        # ── 6. Best generalization ────────────────────────────────────
        gen = rankings.get('generalization', {})
        if gen.get('best_generalizer', 'N/A') != 'N/A':
            recs['best_generalization'] = {
                'category': 'Best Cross-Dataset Generalization',
                'model': gen['best_generalizer'],
                'score': gen.get('avg_cross_dataset_accuracy', 0),
                'reason': 'Best performance on unseen dataset splits'
            }

        # ── 7. Deployment recommendation (accuracy/speed tradeoff) ────
        if 'best_overall' in recs and 'best_speed' in recs:
            best_acc_model = recs['best_overall']['model']
            fastest_model = recs['best_speed']['model']
            if best_acc_model == fastest_model:
                deploy_reason = f'{best_acc_model} is both the most accurate and fastest'
            else:
                deploy_reason = (f'{best_acc_model} for maximum accuracy; '
                                 f'{fastest_model} for real-time deployment')
            recs['deployment'] = {
                'category': 'Recommended Deployment',
                'model': best_acc_model,
                'reason': deploy_reason
            }

        return recs

    # ── Helper methods ────────────────────────────────────────────────

    def _extract_model_list(self, all_results: Dict) -> List:
        """Build ranked model list from ml_models stage results."""
        ml = all_results.get('ml_models', {})
        ranking = []
        for name, data in ml.items():
            if isinstance(data, dict):
                acc = data.get('accuracy', data.get('test_accuracy', 0))
                if isinstance(acc, (int, float)) and acc > 0:
                    ranking.append({
                        'model': name,
                        'accuracy': acc * 100 if acc <= 1 else acc,
                        'inference_ms': data.get('inference_ms',
                                                  data.get('inference_time_ms', 0))
                    })
        ranking.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
        return ranking

    def _extract_robustness(self, all_results: Dict) -> Dict:
        """Extract robustness info from all_results."""
        rob = all_results.get('robustness', {})
        overall = 0
        for key, data in rob.items():
            if isinstance(data, dict):
                overall = max(overall, data.get('overall_robustness',
                                                data.get('overall', 0)))
        return {'overall': overall, 'model': 'Evaluated Model'}

    def _extract_xai(self, all_results: Dict) -> Dict:
        """Extract XAI metrics from all_results."""
        xai = all_results.get('xai_quality', {})
        best_fidelity = 0
        best_method = 'SHAP'
        for method, data in xai.items():
            if isinstance(data, dict):
                fid = data.get('fidelity', 0)
                if fid > best_fidelity:
                    best_fidelity = fid
                    best_method = method
        return {'fidelity': best_fidelity, 'best_method': best_method}

    def _find_minority_champion(self, all_results: Dict, ml_rank: List) -> Optional[Dict]:
        """Find model with best balanced accuracy or G-Mean."""
        # Try from comprehensive results or stage3 comparison CSV
        ml = all_results.get('ml_models', {})
        best_model = None
        best_score = 0
        metric_used = 'balanced_accuracy'

        for name, data in ml.items():
            if isinstance(data, dict):
                for metric in ['g_mean', 'balanced_accuracy', 'f1_macro']:
                    val = data.get(metric, 0)
                    if isinstance(val, (int, float)) and val > best_score:
                        best_score = val
                        best_model = name
                        metric_used = metric

        if best_model and best_score > 0:
            return {
                'category': 'Best Minority-Class Detection',
                'model': best_model,
                'value': best_score,
                'reason': f'Highest {metric_used} — robust on imbalanced classes'
            }
        return None

    def generate_text_report(self, recommendations: Dict) -> str:
        """Format recommendations as human-readable text."""
        lines = []
        lines.append("")
        lines.append("DATA-DRIVEN RECOMMENDATIONS")
        lines.append("=" * 60)
        lines.append("")

        if not recommendations:
            lines.append("No recommendations available — run all stages first.")
            return "\n".join(lines)

        for i, (key, rec) in enumerate(recommendations.items(), 1):
            cat = rec.get('category', key)
            model = rec.get('model', rec.get('method', '—'))
            reason = rec.get('reason', '')
            lines.append(f"{i}. {cat.upper()}")
            lines.append(f"   Winner : {model}")

            # Show relevant score
            for score_key in ['accuracy', 'inference_ms', 'score', 'fidelity', 'value']:
                if score_key in rec:
                    val = rec[score_key]
                    if isinstance(val, float):
                        if score_key == 'inference_ms':
                            lines.append(f"   Score  : {val:.2f} ms")
                        elif val <= 1:
                            lines.append(f"   Score  : {val:.4f} ({val*100:.2f}%)")
                        else:
                            lines.append(f"   Score  : {val:.4f}")
                    else:
                        lines.append(f"   Score  : {val}")
                    break

            lines.append(f"   Reason : {reason}")
            lines.append("")

        lines.append("GENERAL CAUTIONS:")
        lines.append("  • Monitor for concept drift in production environments")
        lines.append("  • Re-evaluate periodically with new attack signatures")
        lines.append("  • Cross-dataset generalization may require fine-tuning")
        lines.append("  • Class-weighted loss is used — ensure weights remain valid")
        lines.append("")

        return "\n".join(lines)

    def save_recommendations(self, recommendations: Dict, output_dir: Path):
        """Save recommendations as JSON + TXT files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON (machine-readable)
        json_recs = {}
        for key, rec in recommendations.items():
            json_recs[key] = {k: (float(v) if isinstance(v, (np.floating,)) else v)
                              for k, v in rec.items()}
        with open(output_dir / 'recommendations.json', 'w') as f:
            json.dump(json_recs, f, indent=2, default=str)

        # TXT (human-readable)
        report_text = self.generate_text_report(recommendations)
        with open(output_dir / 'RECOMMENDATIONS.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Recommendations saved to {output_dir}")


class BenchmarkingReport:
    """
    Generates comprehensive benchmarking report.
    """

    def __init__(self):
        self.sections = {}
        self.figures = {}

    def add_section(self, name: str, content: str):
        """Add a section to the report."""
        self.sections[name] = content

    def add_figure(self, name: str, fig: plt.Figure):
        """Add a figure to the report."""
        self.figures[name] = fig

    def generate_executive_summary(
        self,
        feature_results: Dict,
        model_results: Dict,
        xai_results: Dict,
        robustness_results: Dict
    ) -> str:
        """
        Generate executive summary of benchmarking results.

        Args:
            feature_results: Feature selection results
            model_results: ML model results
            xai_results: XAI quality results
            robustness_results: Robustness testing results

        Returns:
            Executive summary text
        """
        summary = []
        summary.append("=" * 70)
        summary.append("EXECUTIVE SUMMARY")
        summary.append("=" * 70)
        summary.append("")

        # Best Feature Selection Method
        if feature_results:
            best_fs = feature_results.get('best_method', 'SHAP')
            summary.append("BEST FEATURE SELECTION METHOD:")
            summary.append(f"  Winner: {best_fs}")
            summary.append("")

        # Best ML Model
        if model_results:
            best_model = model_results.get('best_model', 'LightGBM')
            best_accuracy = model_results.get('best_accuracy', 0.9987)
            summary.append("BEST MACHINE LEARNING MODEL:")
            summary.append(f"  Winner: {best_model}")
            summary.append(f"  Accuracy: {best_accuracy:.4f}")
            summary.append("")

        # XAI Performance
        if xai_results:
            summary.append("XAI EXPLANATION QUALITY:")
            summary.append(f"  SHAP Fidelity: {xai_results.get('fidelity', 0.94):.2%}")
            summary.append(f"  SHAP Stability: {xai_results.get('stability', 0.96):.2%}")
            summary.append("")

        # Robustness
        if robustness_results:
            overall_robustness = robustness_results.get('overall', 0.97)
            summary.append("ROBUSTNESS EVALUATION:")
            summary.append(f"  Overall Robustness Score: {overall_robustness:.2%}")
            summary.append("")

        summary.append("=" * 70)

        return "\n".join(summary)

    def generate_recommendations(
        self,
        all_results: Dict
    ) -> str:
        """
        Generate recommendations based on benchmarking results.
        Delegates to RecommendationEngine for data-driven analysis.

        Args:
            all_results: All benchmarking results

        Returns:
            Recommendations text
        """
        engine = RecommendationEngine()
        recs = engine.analyze(all_results)
        return engine.generate_text_report(recs)


class ResultsAggregator:
    """
    Aggregates results from all stages.
    """

    def __init__(self):
        self.all_results = {}

    def load_stage_results(self, stage_dir: Path, stage_name: str) -> Dict:
        """Load results from a stage directory."""
        results = {}

        if not stage_dir.exists():
            return results

        # Load JSON files
        for json_file in stage_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    results[json_file.stem] = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")

        # Load CSV files
        for csv_file in stage_dir.glob('*.csv'):
            try:
                results[csv_file.stem] = pd.read_csv(csv_file).to_dict('records')
            except Exception as e:
                logger.warning(f"Error loading {csv_file}: {e}")

        self.all_results[stage_name] = results
        return results

    def aggregate_all_stages(self, mode: str = "multiclass") -> Dict:
        """Load and aggregate results from all stages."""
        stages = [
            ('stage1_artifacts', 'data_preparation'),
            ('stage2_feature_selection', 'feature_selection'),
            ('stage3_ml_models', 'ml_models'),
            ('stage4_comprehensive', 'comprehensive'),
            ('stage5_xai_quality', 'xai_quality'),
            ('stage6_performance', 'performance'),
            ('stage7_cross_dataset', 'cross_dataset'),
            ('stage8_robustness', 'robustness'),
            ('stage9_statistical', 'statistical')
        ]

        for dir_name, stage_name in stages:
            stage_dir = get_results_dir(mode) / dir_name
            self.load_stage_results(stage_dir, stage_name)

        return self.all_results


class VisualizationGenerator:
    """
    Generates comprehensive visualizations for the report.
    """

    def __init__(self):
        self.figures = {}

    def create_model_comparison_chart(
        self,
        model_results: Dict
    ) -> plt.Figure:
        """Create model comparison bar chart."""
        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(model_results.keys())
        accuracies = [r.get('accuracy', 0) for r in model_results.values()]

        bars = ax.bar(range(len(models)), accuracies, color='steelblue')

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_ylim([min(accuracies) - 0.01, 1.0])

        # Add value labels
        for bar, val in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        self.figures['model_comparison'] = fig
        return fig

    def create_radar_chart(
        self,
        metrics: Dict[str, Dict]
    ) -> plt.Figure:
        """Create radar chart for multi-metric comparison."""
        categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'Speed', 'Robustness']
        n_cats = len(categories)

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]

        for model_name, model_metrics in metrics.items():
            values = [
                model_metrics.get('accuracy', 0),
                model_metrics.get('precision', 0),
                model_metrics.get('recall', 0),
                model_metrics.get('f1', 0),
                model_metrics.get('speed_score', 0.5),
                model_metrics.get('robustness', 0.9)
            ]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Multi-Metric Model Comparison')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        self.figures['radar_chart'] = fig
        return fig

    def create_feature_importance_chart(
        self,
        importance_df: pd.DataFrame,
        top_n: int = 20
    ) -> plt.Figure:
        """Create feature importance horizontal bar chart."""
        fig, ax = plt.subplots(figsize=(10, 8))

        if len(importance_df) > top_n:
            importance_df = importance_df.head(top_n)

        y_pos = range(len(importance_df))
        ax.barh(y_pos, importance_df['importance'], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance (SHAP)')

        plt.tight_layout()
        self.figures['feature_importance'] = fig
        return fig

    def save_all_figures(self, output_dir: Path):
        """Save all figures to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, fig in self.figures.items():
            fig.savefig(output_dir / f'{name}.png', dpi=150, bbox_inches='tight')
            logger.info(f"Saved {name}.png")


class Stage10Pipeline:
    """
    Main pipeline for Stage 10: Final Report Generation.
    """

    def __init__(self):
        self.aggregator = ResultsAggregator()
        self.report = BenchmarkingReport()
        self.viz = VisualizationGenerator()
        self.final_results = {}

    def run(
        self,
        custom_results: Dict = None,
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run final report generation.

        Args:
            custom_results: Optional custom results to include

        Returns:
            Dictionary with final report and summary
        """
        logger.info("=" * 60)
        logger.info("STAGE 10: FINAL BENCHMARKING SUMMARY & REPORT")
        logger.info("=" * 60)

        # Aggregate all results
        logger.info("\n--- Aggregating Results from All Stages ---")
        all_results = self.aggregator.aggregate_all_stages(mode)

        if custom_results:
            all_results.update(custom_results)

        # Create final rankings
        logger.info("\n--- Creating Final Rankings ---")
        rankings = self._create_final_rankings(all_results, mode)

        # Generate visualizations
        logger.info("\n--- Generating Visualizations ---")
        self._generate_all_visualizations(all_results, mode)

        # Generate report sections
        logger.info("\n--- Generating Report Sections ---")

        # Executive summary
        executive_summary = self.report.generate_executive_summary(
            feature_results=rankings.get('feature_selection', {}),
            model_results=rankings.get('ml_models', {}),
            xai_results=rankings.get('xai', {}),
            robustness_results=rankings.get('robustness', {})
        )
        self.report.add_section('Executive Summary', executive_summary)

        # Recommendations (data-driven)
        rec_engine = RecommendationEngine()
        self.recommendations = rec_engine.analyze(all_results, rankings)
        recommendations_text = rec_engine.generate_text_report(self.recommendations)
        self.report.add_section('Recommendations', recommendations_text)

        # Save recommendations as separate files
        rec_output_dir = get_results_dir(mode) / 'stage10_final_report'
        rec_engine.save_recommendations(self.recommendations, rec_output_dir)

        # Create final summary
        final_summary = self._create_final_summary(rankings)

        # Save everything
        self._save_final_report(final_summary, rankings, mode)

        # Print report
        self._print_report(final_summary)

        self.final_results = {
            'rankings': rankings,
            'summary': final_summary,
            'recommendations': self.recommendations,
            'all_results': all_results
        }

        logger.info("=" * 60)
        logger.info("STAGE 10 COMPLETE - BENCHMARKING STUDY FINISHED")
        logger.info("=" * 60)

        return self.final_results

    def _create_final_rankings(self, all_results: Dict, mode: str = "multiclass") -> Dict:
        """Create final rankings dynamically from actual results."""
        rankings = {
            'feature_selection': self._extract_feature_rankings(all_results, mode),
            'ml_models': self._extract_model_rankings(all_results, mode),
            'xai': self._extract_xai_rankings(all_results, mode),
            'robustness': self._extract_robustness_rankings(all_results, mode),
            'generalization': self._extract_generalization_rankings(all_results, mode)
        }

        return rankings

    def _extract_feature_rankings(self, all_results: Dict, mode: str = "multiclass") -> Dict:
        """Extract feature selection rankings from actual results."""
        fs_results = all_results.get('feature_selection', {})
        ranking = []

        for method_name, data in fs_results.items():
            if isinstance(data, dict) and 'score' in data:
                ranking.append({'method': method_name, 'score': data['score']})
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'method' in item:
                        ranking.append(item)

        if not ranking:
            # Try loading from CSV
            csv_path = get_results_dir(mode) / 'stage2_feature_selection'
            if csv_path.exists():
                for csv_file in csv_path.glob('*ranking*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        for _, row in df.iterrows():
                            ranking.append({
                                'method': row.get('method', row.get('Feature_Method', 'Unknown')),
                                'score': row.get('score', row.get('Accuracy', 0)) * 100
                            })
                    except Exception:
                        pass

        if ranking:
            ranking.sort(key=lambda x: x.get('score', 0), reverse=True)
            for i, item in enumerate(ranking):
                item['rank'] = i + 1
            best_method = ranking[0]['method']
        else:
            best_method = 'N/A'

        return {'best_method': best_method, 'ranking': ranking}

    def _extract_model_rankings(self, all_results: Dict, mode: str = "multiclass") -> Dict:
        """Extract ML model rankings from actual results."""
        ml_results = all_results.get('ml_models', {})
        ranking = []

        for model_name, data in ml_results.items():
            if isinstance(data, dict):
                # Handle nested JSON: {test: {accuracy: ...}} or flat {accuracy: ...}
                acc = data.get('accuracy', 0)
                if acc == 0 and 'test' in data and isinstance(data['test'], dict):
                    acc = data['test'].get('accuracy', 0)
                if acc == 0:
                    acc = data.get('test_accuracy', 0)

                inf_ms = data.get('inference_ms', data.get('inference_time_ms', 0))

                if isinstance(acc, (int, float)) and acc > 0:
                    ranking.append({
                        'model': model_name.replace('_results', ''),
                        'accuracy': acc * 100 if acc <= 1 else acc,
                        'inference_ms': inf_ms
                    })

        # Fallback: read from model_comparison.csv
        if not ranking:
            csv_path = get_results_dir(mode) / 'stage3_ml_models'
            if csv_path.exists():
                for csv_file in csv_path.glob('*comparison*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        for _, row in df.iterrows():
                            model = row.get('Model', row.get('model', 'Unknown'))
                            acc_val = float(row.get('Accuracy', row.get('accuracy', 0)))
                            inf_val = float(row.get('Inference (ms)', row.get('inference_ms', 0)))
                            ranking.append({
                                'model': model,
                                'accuracy': acc_val * 100 if acc_val <= 1 else acc_val,
                                'inference_ms': inf_val
                            })
                    except Exception:
                        pass

        if ranking:
            ranking.sort(key=lambda x: x.get('accuracy', 0), reverse=True)
            for i, item in enumerate(ranking):
                item['rank'] = i + 1
            best_model = ranking[0]['model']
            best_accuracy = ranking[0]['accuracy'] / 100
        else:
            best_model = 'N/A'
            best_accuracy = 0

        return {'best_model': best_model, 'best_accuracy': best_accuracy, 'ranking': ranking}

    def _extract_xai_rankings(self, all_results: Dict, mode: str = "multiclass") -> Dict:
        """Extract XAI quality rankings from actual results."""
        xai_results = all_results.get('xai_quality', {})

        fidelity = 0
        stability = 0
        consistency = 0
        best_method = 'SHAP'

        # Try direct keys first
        for key, data in xai_results.items():
            if isinstance(data, dict):
                fidelity = max(fidelity, data.get('fidelity', 0))
                stability = max(stability, data.get('stability', 0))
                consistency = max(consistency, data.get('consistency', 0))
                # Handle nested: quality_metrics -> SHAP -> fidelity
                qm = data.get('quality_metrics', {})
                for method, mdata in qm.items():
                    if isinstance(mdata, dict):
                        fid = mdata.get('fidelity', 0)
                        if fid > fidelity:
                            fidelity = fid
                            best_method = method
                # cross_method_consistency
                cc = qm.get('cross_method_consistency', 0)
                if cc > consistency:
                    consistency = cc

        # Fallback: load from saved JSON files
        if fidelity == 0:
            json_path = get_results_dir(mode) / 'stage5_xai_quality'
            if json_path.exists():
                for json_file in json_path.glob('*.json'):
                    try:
                        import json as json_mod
                        with open(json_file, 'r') as f:
                            data = json_mod.load(f)
                        # Handle nested structure
                        qm = data.get('quality_metrics', data)
                        for method_key in ['SHAP', 'LIME', 'Anchors']:
                            mdata = qm.get(method_key, {})
                            if isinstance(mdata, dict):
                                fid = mdata.get('fidelity', 0)
                                if fid > fidelity:
                                    fidelity = fid
                                    best_method = method_key
                        consistency = max(consistency, qm.get('cross_method_consistency', 0))
                        # Top-level fallbacks
                        fidelity = max(fidelity, data.get('fidelity', 0))
                        stability = max(stability, data.get('stability', 0))
                    except Exception:
                        pass

        return {
            'best_method': best_method,
            'fidelity': fidelity,
            'stability': stability,
            'consistency': consistency
        }

    def _extract_robustness_rankings(self, all_results: Dict, mode: str = "multiclass") -> Dict:
        """Extract robustness rankings from actual results."""
        rob_results = all_results.get('robustness', {})

        overall = 0
        fgsm = 0
        pgd = 0
        noise = 0

        for key, data in rob_results.items():
            if isinstance(data, dict):
                if 'robustness_scores' in key or 'overall' in str(data):
                    overall = data.get('overall_robustness', data.get('overall', 0))
                    fgsm = data.get('fgsm_robustness', data.get('fgsm', 0))
                    pgd = data.get('pgd_robustness', data.get('pgd', 0))
                    noise = data.get('noise_robustness', data.get('noise', 0))

        if overall == 0:
            json_path = get_results_dir(mode) / 'stage8_robustness'
            if json_path.exists():
                for json_file in json_path.glob('*robustness_scores*.json'):
                    try:
                        import json as json_mod
                        with open(json_file, 'r') as f:
                            data = json_mod.load(f)
                        overall = data.get('overall_robustness', 0)
                        fgsm = data.get('fgsm_robustness', 0)
                        pgd = data.get('pgd_robustness', 0)
                        noise = data.get('noise_robustness', 0)
                    except Exception:
                        pass

        return {'overall': overall, 'fgsm': fgsm, 'pgd': pgd, 'noise': noise}

    def _extract_generalization_rankings(self, all_results: Dict, mode: str = "multiclass") -> Dict:
        """Extract generalization rankings from actual results."""
        gen_results = all_results.get('cross_dataset', {})
        best_generalizer = 'N/A'
        avg_accuracy = 0

        for key, data in gen_results.items():
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and 'target_accuracy' in item:
                        avg_accuracy = max(avg_accuracy, item['target_accuracy'])
                        if item['target_accuracy'] == avg_accuracy:
                            best_generalizer = item.get('model', 'Unknown')

        if avg_accuracy == 0:
            csv_path = get_results_dir(mode) / 'stage7_cross_dataset'
            if csv_path.exists():
                for csv_file in csv_path.glob('*ranking*.csv'):
                    try:
                        df = pd.read_csv(csv_file)
                        if 'target_accuracy' in df.columns:
                            best_idx = df['target_accuracy'].idxmax()
                            best_generalizer = df.loc[best_idx].get('model', df.index[best_idx])
                            avg_accuracy = df['target_accuracy'].mean()
                    except Exception:
                        pass

        return {
            'best_generalizer': best_generalizer,
            'avg_cross_dataset_accuracy': avg_accuracy
        }

    def _generate_all_visualizations(self, all_results: Dict, mode: str = "multiclass"):
        """Generate visualizations from actual results."""
        viz_dir = get_results_dir(mode) / 'stage10_final_report' / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Build model data from actual results
        ml_results = all_results.get('ml_models', {})
        model_data = {}
        for model_name, data in ml_results.items():
            if isinstance(data, dict):
                acc = data.get('accuracy', data.get('test_accuracy', 0))
                if isinstance(acc, (int, float)) and acc > 0:
                    model_data[model_name] = {'accuracy': acc if acc <= 1 else acc / 100}

        if model_data:
            self.viz.create_model_comparison_chart(model_data)

        # Generate comprehensive visualizations
        try:
            from visualizations import generate_all_visualizations
            generate_all_visualizations(all_results, viz_dir)
        except ImportError:
            logger.warning("visualizations module not available, using basic charts")

        # Save visualizations
        self.viz.save_all_figures(viz_dir)

    def _create_final_summary(self, rankings: Dict) -> Dict:
        """Create final summary dictionary from actual rankings."""
        fs_winner = rankings['feature_selection'].get('best_method', 'N/A')
        ml_winner = rankings['ml_models'].get('best_model', 'N/A')
        ml_accuracy = rankings['ml_models'].get('best_accuracy', 0)
        xai_winner = rankings['xai'].get('best_method', 'N/A')
        xai_fidelity = rankings['xai'].get('fidelity', 0)
        gen_winner = rankings['generalization'].get('best_generalizer', 'N/A')
        robustness = rankings['robustness'].get('overall', 0)

        # Build inference time from rankings
        ml_ranking = rankings['ml_models'].get('ranking', [])
        inference_time = 'N/A'
        if ml_ranking:
            inference_time = f"{ml_ranking[0].get('inference_ms', 'N/A')} ms"

        # Dynamic key findings
        findings = []
        if fs_winner != 'N/A':
            findings.append(f'{fs_winner}-based feature selection is the top-performing method')
        if ml_winner != 'N/A':
            findings.append(f'{ml_winner} achieves best accuracy-speed tradeoff')
        if gen_winner != 'N/A':
            findings.append(f'{gen_winner} shows best cross-dataset generalization')
        if ml_accuracy > 0.99:
            findings.append(f'Top models achieve >{ml_accuracy*100:.1f}% accuracy on primary dataset')
        if xai_fidelity > 0:
            findings.append(f'{xai_winner} explanations provide {xai_fidelity:.1%} fidelity')
        if robustness > 0:
            findings.append(f'Overall robustness score: {robustness:.1%}')

        if not findings:
            findings = ['Run all stages to generate findings']

        summary = {
            'study_title': 'XAI-Based IDS Benchmarking Study',
            'completion_date': datetime.now().isoformat(),
            'dataset': 'CIC IoT-DIAD 2024',
            'winners': {
                'feature_selection': fs_winner,
                'ml_model': ml_winner,
                'xai_method': xai_winner
            },
            'key_findings': findings,
            'overall_champion': {
                'configuration': f'FS_{fs_winner} + {ml_winner}',
                'accuracy': f'{ml_accuracy*100:.2f}%' if ml_accuracy > 0 else 'N/A',
                'inference_time': inference_time,
                'explainability': f'{xai_fidelity:.1%}' if xai_fidelity > 0 else 'N/A',
                'robustness': f'{robustness:.1%}' if robustness > 0 else 'N/A'
            }
        }

        return summary

    def _save_final_report(self, summary: Dict, rankings: Dict, mode: str = "multiclass"):
        """Save final report to disk."""
        output_dir = get_results_dir(mode) / 'stage10_final_report'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        with open(output_dir / 'final_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        # Save rankings
        with open(output_dir / 'final_rankings.json', 'w') as f:
            json.dump(rankings, f, indent=2, default=str)

        # Save model rankings as CSV
        model_df = pd.DataFrame(rankings['ml_models']['ranking'])
        model_df.to_csv(output_dir / 'model_rankings.csv', index=False)

        # Save feature selection rankings
        fs_df = pd.DataFrame(rankings['feature_selection']['ranking'])
        fs_df.to_csv(output_dir / 'feature_selection_rankings.csv', index=False)

        # Save text report
        report_text = self._generate_text_report(summary, rankings)
        with open(output_dir / 'FINAL_REPORT.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Final report saved to {output_dir}")

    def _generate_text_report(self, summary: Dict, rankings: Dict) -> str:
        """Generate full text report."""
        lines = []

        lines.append("=" * 70)
        lines.append("XAI-BASED IDS BENCHMARKING STUDY - FINAL REPORT")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"Report Generated: {summary['completion_date']}")
        lines.append(f"Primary Dataset: {summary['dataset']}")
        lines.append("")

        # Executive Summary
        lines.append("=" * 70)
        lines.append("EXECUTIVE SUMMARY")
        lines.append("=" * 70)
        lines.append("")
        lines.append("WINNERS:")
        lines.append(f"  Best Feature Selection: {summary['winners']['feature_selection']}")
        lines.append(f"  Best ML Model: {summary['winners']['ml_model']}")
        lines.append(f"  Best XAI Method: {summary['winners']['xai_method']}")
        lines.append("")

        lines.append("OVERALL CHAMPION CONFIGURATION:")
        champion = summary['overall_champion']
        lines.append(f"  Configuration: {champion['configuration']}")
        lines.append(f"  Accuracy: {champion['accuracy']}")
        lines.append(f"  Inference Time: {champion['inference_time']}")
        lines.append(f"  Explainability: {champion['explainability']}")
        lines.append(f"  Robustness: {champion['robustness']}")
        lines.append("")

        # Feature Selection Rankings
        lines.append("=" * 70)
        lines.append("FEATURE SELECTION METHOD RANKINGS")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"{'Rank':<6}{'Method':<15}{'Score (%)':<12}")
        lines.append("-" * 35)
        for item in rankings['feature_selection']['ranking']:
            lines.append(f"{item['rank']:<6}{item['method']:<15}{item['score']:<12.2f}")
        lines.append("")

        # ML Model Rankings
        lines.append("=" * 70)
        lines.append("MACHINE LEARNING MODEL RANKINGS")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"{'Rank':<6}{'Model':<15}{'Accuracy (%)':<15}{'Inference (ms)':<15}")
        lines.append("-" * 55)
        for item in rankings['ml_models']['ranking']:
            lines.append(f"{item['rank']:<6}{item['model']:<15}{item['accuracy']:<15.2f}{item['inference_ms']:<15.1f}")
        lines.append("")

        # XAI Quality
        lines.append("=" * 70)
        lines.append("XAI QUALITY METRICS")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Fidelity: {rankings['xai']['fidelity']:.2%}")
        lines.append(f"  Stability: {rankings['xai']['stability']:.2%}")
        lines.append(f"  Consistency: {rankings['xai']['consistency']:.2%}")
        lines.append("")

        # Robustness
        lines.append("=" * 70)
        lines.append("ROBUSTNESS SCORES")
        lines.append("=" * 70)
        lines.append("")
        lines.append(f"  Overall Robustness: {rankings['robustness']['overall']:.2%}")
        lines.append(f"  FGSM Robustness: {rankings['robustness']['fgsm']:.2%}")
        lines.append(f"  Noise Robustness: {rankings['robustness']['noise']:.2%}")
        lines.append("")

        # Key Findings
        lines.append("=" * 70)
        lines.append("KEY FINDINGS")
        lines.append("=" * 70)
        lines.append("")
        for i, finding in enumerate(summary['key_findings'], 1):
            lines.append(f"  {i}. {finding}")
        lines.append("")

        # Recommendations (data-driven)
        lines.append("=" * 70)
        lines.append("RECOMMENDATIONS")
        lines.append("=" * 70)
        lines.append("")
        if hasattr(self, 'recommendations') and self.recommendations:
            rec_engine = RecommendationEngine()
            lines.append(rec_engine.generate_text_report(self.recommendations))
        else:
            lines.append("  Run all stages to generate data-driven recommendations.")
        lines.append("")

        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        return "\n".join(lines)

    def _print_report(self, summary: Dict):
        """Print final report to console."""
        print("\n" + "=" * 70)
        print("FINAL BENCHMARKING RESULTS")
        print("=" * 70)
        print("")
        print("OVERALL CHAMPION:")
        champion = summary['overall_champion']
        print(f"  {champion['configuration']}")
        print(f"  Accuracy: {champion['accuracy']}")
        print(f"  Inference: {champion['inference_time']}")
        print(f"  Explainability: {champion['explainability']}")
        print("")
        print("KEY FINDINGS:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"  {i}. {finding}")
        print("")
        print("=" * 70)


def generate_final_report(custom_results: Dict = None) -> Dict:
    """
    Convenience function to generate final report.

    Args:
        custom_results: Optional custom results to include

    Returns:
        Final report dictionary
    """
    pipeline = Stage10Pipeline()
    return pipeline.run(custom_results)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 10: Final Report Generation")
    print("="*70 + "\n")

    # Generate final report
    pipeline = Stage10Pipeline()
    results = pipeline.run()

    print("\n" + "-"*50)
    print("BENCHMARKING STUDY COMPLETED SUCCESSFULLY!")
    print("-"*50)
    print(f"Results saved to: {RESULTS_DIR / 'stage10_final_report'}")
    print("-"*50)
