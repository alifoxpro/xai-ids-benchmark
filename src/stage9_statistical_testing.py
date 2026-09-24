"""
================================================================================
STAGE 9: STATISTICAL SIGNIFICANCE TESTING
================================================================================
This module provides:
- ANOVA tests for multiple group comparison
- t-tests for pairwise comparison
- Wilcoxon signed-rank test for paired non-parametric comparison
- McNemar's test for classifier error rate comparison
- Friedman test for ranked comparisons
- Nemenyi post-hoc test
- Bootstrap confidence intervals
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import logging
import json

from scipy import stats
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon,
    f_oneway, kruskal, friedmanchisquare
)

try:
    from scikit_posthocs import posthoc_nemenyi_friedman
    POSTHOC_AVAILABLE = True
except ImportError:
    POSTHOC_AVAILABLE = False

from config import STATISTICAL_TESTS, RESULTS_DIR, RANDOM_SEED, get_results_dir

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StatisticalTester:
    """
    Performs statistical significance tests for model comparison.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.results = {}

    def anova_test(
        self,
        groups: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Perform one-way ANOVA test.

        Args:
            groups: Dictionary of {group_name: values_array}

        Returns:
            Dictionary with ANOVA results
        """
        logger.info("Performing ANOVA test...")

        group_values = list(groups.values())
        group_names = list(groups.keys())

        # Check assumptions
        if len(group_values) < 2:
            return {'error': 'Need at least 2 groups for ANOVA'}

        # Perform ANOVA
        f_statistic, p_value = f_oneway(*group_values)

        # Effect size (eta-squared)
        all_values = np.concatenate(group_values)
        grand_mean = np.mean(all_values)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_values)
        ss_total = np.sum((all_values - grand_mean)**2)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0

        result = {
            'test': 'One-way ANOVA',
            'groups': group_names,
            'f_statistic': float(f_statistic),
            'p_value': float(p_value),
            'eta_squared': float(eta_squared),
            'significant': p_value < self.alpha,
            'conclusion': (
                f"Significant difference found (p={p_value:.4f} < {self.alpha})"
                if p_value < self.alpha
                else f"No significant difference (p={p_value:.4f} >= {self.alpha})"
            )
        }

        logger.info(f"F-statistic: {f_statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(result['conclusion'])

        return result

    def t_test(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        name1: str = "Group1",
        name2: str = "Group2",
        paired: bool = False
    ) -> Dict:
        """
        Perform t-test between two groups.

        Args:
            group1: First group values
            group2: Second group values
            name1: Name of first group
            name2: Name of second group
            paired: Whether to use paired t-test

        Returns:
            Dictionary with t-test results
        """
        logger.info(f"Performing {'paired' if paired else 'independent'} t-test: {name1} vs {name2}")

        if paired:
            t_statistic, p_value = ttest_rel(group1, group2)
        else:
            t_statistic, p_value = ttest_ind(group1, group2)

        # Cohen's d effect size
        pooled_std = np.sqrt(((len(group1)-1)*np.std(group1)**2 +
                              (len(group2)-1)*np.std(group2)**2) /
                             (len(group1) + len(group2) - 2))
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

        result = {
            'test': 'Paired t-test' if paired else 'Independent t-test',
            'groups': [name1, name2],
            't_statistic': float(t_statistic),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'mean_difference': float(np.mean(group1) - np.mean(group2)),
            'significant': p_value < self.alpha,
            'conclusion': (
                f"{name1} significantly different from {name2} (p={p_value:.4f})"
                if p_value < self.alpha
                else f"No significant difference between {name1} and {name2} (p={p_value:.4f})"
            )
        }

        logger.info(f"t-statistic: {t_statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(result['conclusion'])

        return result

    def friedman_test(
        self,
        groups: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Perform Friedman test (non-parametric alternative to repeated measures ANOVA).

        Args:
            groups: Dictionary of {group_name: values_array}

        Returns:
            Dictionary with Friedman test results
        """
        logger.info("Performing Friedman test...")

        group_values = list(groups.values())
        group_names = list(groups.keys())

        if len(group_values) < 3:
            return {'error': 'Need at least 3 groups for Friedman test'}

        # Ensure equal sample sizes
        min_len = min(len(g) for g in group_values)
        group_values = [g[:min_len] for g in group_values]

        # Perform Friedman test
        try:
            statistic, p_value = friedmanchisquare(*group_values)
        except Exception as e:
            return {'error': str(e)}

        result = {
            'test': 'Friedman test',
            'groups': group_names,
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'conclusion': (
                f"Significant difference found (p={p_value:.4f} < {self.alpha})"
                if p_value < self.alpha
                else f"No significant difference (p={p_value:.4f} >= {self.alpha})"
            )
        }

        logger.info(f"Friedman statistic: {statistic:.4f}, p-value: {p_value:.4f}")
        logger.info(result['conclusion'])

        return result

    def nemenyi_test(
        self,
        groups: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Perform Nemenyi post-hoc test after Friedman test.

        Args:
            groups: Dictionary of {group_name: values_array}

        Returns:
            Dictionary with Nemenyi test results
        """
        logger.info("Performing Nemenyi post-hoc test...")

        if not POSTHOC_AVAILABLE:
            logger.warning("scikit-posthocs not available. Computing critical difference manually.")
            return self._nemenyi_manual(groups)

        group_values = list(groups.values())
        group_names = list(groups.keys())

        # Ensure equal sample sizes
        min_len = min(len(g) for g in group_values)
        group_values = [g[:min_len] for g in group_values]

        # Create DataFrame for post-hoc test
        data = np.column_stack(group_values)
        df = pd.DataFrame(data, columns=group_names)

        try:
            posthoc = posthoc_nemenyi_friedman(df)
            result = {
                'test': 'Nemenyi post-hoc',
                'pairwise_p_values': posthoc.to_dict(),
                'significant_pairs': []
            }

            # Find significant pairs
            for i, name1 in enumerate(group_names):
                for j, name2 in enumerate(group_names[i+1:], i+1):
                    if posthoc.iloc[i, j] < self.alpha:
                        result['significant_pairs'].append({
                            'pair': (name1, name2),
                            'p_value': float(posthoc.iloc[i, j])
                        })

            logger.info(f"Significant pairs found: {len(result['significant_pairs'])}")

        except Exception as e:
            result = {'error': str(e)}

        return result

    def _nemenyi_manual(
        self,
        groups: Dict[str, np.ndarray]
    ) -> Dict:
        """Manual Nemenyi critical difference computation."""
        k = len(groups)  # Number of groups
        n = min(len(g) for g in groups.values())  # Sample size

        # Critical value from studentized range distribution (approximation)
        q_alpha = stats.studentized_range.ppf(1 - self.alpha, k, np.inf)

        # Critical difference
        cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))

        # Compute mean ranks
        group_values = list(groups.values())
        group_names = list(groups.keys())

        # Rank all values together
        all_values = np.concatenate([g[:n] for g in group_values])
        ranks = stats.rankdata(all_values)

        # Compute mean rank per group
        mean_ranks = {}
        start = 0
        for name, g in groups.items():
            end = start + n
            mean_ranks[name] = np.mean(ranks[start:end])
            start = end

        result = {
            'test': 'Nemenyi (manual)',
            'critical_difference': float(cd),
            'mean_ranks': {k: float(v) for k, v in mean_ranks.items()},
            'significant_pairs': []
        }

        # Find pairs with rank difference > CD
        for i, (name1, rank1) in enumerate(mean_ranks.items()):
            for name2, rank2 in list(mean_ranks.items())[i+1:]:
                if abs(rank1 - rank2) > cd:
                    result['significant_pairs'].append({
                        'pair': (name1, name2),
                        'rank_difference': abs(rank1 - rank2)
                    })

        logger.info(f"Critical difference: {cd:.4f}")
        logger.info(f"Significant pairs: {len(result['significant_pairs'])}")

        return result

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func: callable = np.mean,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Compute bootstrap confidence interval.

        Args:
            data: Data array
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (0-1)

        Returns:
            Dictionary with confidence interval
        """
        logger.info("Computing bootstrap confidence interval...")

        n = len(data)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, n, replace=True)
            bootstrap_stats.append(statistic_func(sample))

        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_stats, alpha/2 * 100)
        upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)

        result = {
            'statistic': float(statistic_func(data)),
            'confidence_level': confidence_level,
            'lower_bound': float(lower),
            'upper_bound': float(upper),
            'margin_of_error': float((upper - lower) / 2)
        }

        logger.info(f"CI: [{lower:.4f}, {upper:.4f}]")

        return result

    def wilcoxon_test(self, scores_a, scores_b, alpha=0.05):
        """
        Wilcoxon signed-rank test for paired samples.
        Non-parametric alternative to paired t-test.

        Args:
            scores_a: Scores from model A (e.g., fold accuracies)
            scores_b: Scores from model B
            alpha: Significance level

        Returns:
            Dict with test results
        """
        from scipy.stats import wilcoxon

        logger.info("Running Wilcoxon signed-rank test...")

        scores_a = np.array(scores_a)
        scores_b = np.array(scores_b)

        try:
            statistic, p_value = wilcoxon(scores_a, scores_b, alternative='two-sided')
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            return {
                'test': 'Wilcoxon',
                'statistic': None,
                'p_value': None,
                'significant': False,
                'error': str(e)
            }

        significant = p_value < alpha

        result = {
            'test': 'Wilcoxon',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'alpha': alpha,
            'mean_a': float(np.mean(scores_a)),
            'mean_b': float(np.mean(scores_b)),
            'interpretation': f"{'Significant' if significant else 'No significant'} difference (p={p_value:.6f})"
        }

        logger.info(f"Wilcoxon: statistic={statistic:.4f}, p={p_value:.6f}, significant={significant}")
        return result

    def mcnemar_test(self, y_true, y_pred_a, y_pred_b, alpha=0.05):
        """
        McNemar's test for comparing two classifiers on the same test set.
        Tests if the classifiers have the same error rate.

        Args:
            y_true: True labels
            y_pred_a: Predictions from classifier A
            y_pred_b: Predictions from classifier B
            alpha: Significance level

        Returns:
            Dict with test results
        """
        logger.info("Running McNemar's test...")

        y_true = np.array(y_true)
        y_pred_a = np.array(y_pred_a)
        y_pred_b = np.array(y_pred_b)

        # Build contingency table
        correct_a = (y_pred_a == y_true)
        correct_b = (y_pred_b == y_true)

        # n01: A correct, B wrong
        n01 = np.sum(correct_a & ~correct_b)
        # n10: A wrong, B correct
        n10 = np.sum(~correct_a & correct_b)

        # McNemar statistic with continuity correction
        if (n01 + n10) == 0:
            return {
                'test': 'McNemar',
                'statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'n01': int(n01),
                'n10': int(n10),
                'interpretation': 'Both classifiers make identical errors'
            }

        statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)

        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(statistic, df=1)

        significant = p_value < alpha

        result = {
            'test': 'McNemar',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'alpha': alpha,
            'n01_a_correct_b_wrong': int(n01),
            'n10_a_wrong_b_correct': int(n10),
            'interpretation': f"{'Significant' if significant else 'No significant'} difference in error rates (p={p_value:.6f})"
        }

        logger.info(f"McNemar: statistic={statistic:.4f}, p={p_value:.6f}, n01={n01}, n10={n10}")
        return result


class Stage9Pipeline:
    """
    Main pipeline for Stage 9: Statistical Significance Testing.
    """

    def __init__(self, significance_level: float = 0.05):
        self.alpha = significance_level
        self.tester = StatisticalTester(significance_level)
        self.results = {}

    def run(
        self,
        model_results: Dict[str, Dict],
        n_folds: int = 5,
        predictions: Optional[Dict[str, np.ndarray]] = None,
        y_true: Optional[np.ndarray] = None,
        mode: str = "multiclass"
    ) -> Dict:
        """
        Run comprehensive statistical testing.

        Args:
            model_results: Dictionary of {model_name: {fold_scores}}
            n_folds: Number of CV folds
            predictions: Optional dict of {model_name: predicted_labels} for McNemar test
            y_true: Optional array of true labels for McNemar test

        Returns:
            Dictionary with all statistical test results
        """
        logger.info("=" * 60)
        logger.info("STAGE 9: STATISTICAL SIGNIFICANCE TESTING")
        logger.info("=" * 60)

        results = {
            'anova': None,
            't_tests': [],
            'wilcoxon': [],
            'mcnemar': [],
            'friedman': None,
            'nemenyi': None,
            'confidence_intervals': {},
            'summary': {}
        }

        # Convert model results to appropriate format
        groups = self._prepare_groups(model_results)

        if len(groups) < 2:
            logger.warning("Need at least 2 groups for statistical testing")
            return results

        # ANOVA test
        logger.info("\n--- ANOVA Test ---")
        results['anova'] = self.tester.anova_test(groups)

        # Pairwise t-tests
        logger.info("\n--- Pairwise t-Tests ---")
        model_names = list(groups.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name1, name2 = model_names[i], model_names[j]
                t_result = self.tester.t_test(
                    groups[name1], groups[name2],
                    name1, name2
                )
                results['t_tests'].append(t_result)

        # Friedman test (if 3+ groups)
        if len(groups) >= 3:
            logger.info("\n--- Friedman Test ---")
            results['friedman'] = self.tester.friedman_test(groups)

            # Nemenyi post-hoc test
            if results['friedman'].get('significant', False):
                logger.info("\n--- Nemenyi Post-hoc Test ---")
                results['nemenyi'] = self.tester.nemenyi_test(groups)

        # Bootstrap confidence intervals
        logger.info("\n--- Bootstrap Confidence Intervals ---")
        for name, values in groups.items():
            results['confidence_intervals'][name] = self.tester.bootstrap_confidence_interval(
                values, np.mean, n_bootstrap=1000
            )

        # Wilcoxon signed-rank tests (pairwise)
        logger.info("\n--- Wilcoxon Signed-Rank Tests ---")
        wilcoxon_results = []
        model_names_list = list(model_results.keys())
        for i, model_a in enumerate(model_names_list):
            for model_b in model_names_list[i+1:]:
                scores_a = model_results[model_a].get('fold_scores', [])
                scores_b = model_results[model_b].get('fold_scores', [])
                if len(scores_a) >= 5 and len(scores_b) >= 5:
                    w_result = self.tester.wilcoxon_test(scores_a, scores_b)
                    w_result['model_a'] = model_a
                    w_result['model_b'] = model_b
                    wilcoxon_results.append(w_result)
        # Apply Holm-Bonferroni correction to Wilcoxon p-values
        if wilcoxon_results:
            p_values = [r['p_value'] for r in wilcoxon_results if r['p_value'] is not None]
            if p_values:
                n_comparisons = len(p_values)
                sorted_indices = np.argsort(p_values)
                for rank, idx in enumerate(sorted_indices):
                    corrected_alpha = self.alpha / (n_comparisons - rank)
                    orig_p = wilcoxon_results[idx]['p_value']
                    wilcoxon_results[idx]['holm_significant'] = orig_p < corrected_alpha
                    wilcoxon_results[idx]['holm_corrected_alpha'] = corrected_alpha
                logger.info(f"Applied Holm-Bonferroni correction to {n_comparisons} Wilcoxon comparisons")
        results['wilcoxon'] = wilcoxon_results

        # McNemar tests (if predictions available)
        logger.info("\n--- McNemar Tests ---")
        mcnemar_results = []
        if predictions and y_true is not None:
            pred_models = list(predictions.keys())
            for i, model_a in enumerate(pred_models):
                for model_b in pred_models[i+1:]:
                    mc_result = self.tester.mcnemar_test(
                        y_true,
                        predictions[model_a],
                        predictions[model_b]
                    )
                    mc_result['model_a'] = model_a
                    mc_result['model_b'] = model_b
                    mcnemar_results.append(mc_result)
        results['mcnemar'] = mcnemar_results

        # Create summary
        results['summary'] = self._create_summary(results, groups)

        # Save results
        self._save_results(results, mode)

        self.results = results

        logger.info("=" * 60)
        logger.info("STAGE 9 COMPLETE")
        logger.info("=" * 60)

        return results

    def _prepare_groups(self, model_results: Dict) -> Dict[str, np.ndarray]:
        """Prepare groups for statistical testing."""
        groups = {}

        for model_name, result in model_results.items():
            if isinstance(result, dict):
                # Extract accuracy values from CV folds
                if 'fold_scores' in result:
                    groups[model_name] = np.array(result['fold_scores'])
                elif 'accuracy' in result:
                    # Single value - create small sample
                    groups[model_name] = np.array([result['accuracy']] * 5)
                elif 'test' in result and 'accuracy' in result['test']:
                    groups[model_name] = np.array([result['test']['accuracy']] * 5)
            elif isinstance(result, (list, np.ndarray)):
                groups[model_name] = np.array(result)

        return groups

    def _create_summary(
        self,
        results: Dict,
        groups: Dict[str, np.ndarray]
    ) -> Dict:
        """Create summary of statistical tests."""
        summary = {
            'n_models': len(groups),
            'model_names': list(groups.keys()),
            'best_model': None,
            'significant_differences': []
        }

        # Find best model
        mean_scores = {name: np.mean(values) for name, values in groups.items()}
        summary['best_model'] = max(mean_scores, key=mean_scores.get)
        summary['mean_scores'] = {k: float(v) for k, v in mean_scores.items()}

        # Collect significant differences
        if results['anova'] and results['anova'].get('significant'):
            summary['significant_differences'].append('ANOVA: Overall difference significant')

        for t_test in results['t_tests']:
            if t_test.get('significant'):
                pair = t_test['groups']
                summary['significant_differences'].append(f"t-test: {pair[0]} vs {pair[1]}")

        for w_test in results.get('wilcoxon', []):
            if w_test.get('significant'):
                summary['significant_differences'].append(
                    f"Wilcoxon: {w_test['model_a']} vs {w_test['model_b']}"
                )

        for mc_test in results.get('mcnemar', []):
            if mc_test.get('significant'):
                summary['significant_differences'].append(
                    f"McNemar: {mc_test['model_a']} vs {mc_test['model_b']}"
                )

        return summary

    def _save_results(self, results: Dict, mode: str = "multiclass"):
        """Save statistical test results."""
        output_dir = get_results_dir(mode) / 'stage9_statistical'
        output_dir.mkdir(exist_ok=True)

        # Save main results
        main_results = {
            'anova': results['anova'],
            'friedman': results['friedman'],
            'nemenyi': results['nemenyi'] if isinstance(results['nemenyi'], dict) else None,
            'wilcoxon': results.get('wilcoxon', []),
            'mcnemar': results.get('mcnemar', []),
            'summary': results['summary']
        }

        with open(output_dir / 'statistical_tests.json', 'w') as f:
            json.dump(main_results, f, indent=2, default=str)

        # Save t-test results
        if results['t_tests']:
            t_test_df = pd.DataFrame(results['t_tests'])
            t_test_df.to_csv(output_dir / 't_tests.csv', index=False)

        # Save Wilcoxon results
        if results.get('wilcoxon'):
            wilcoxon_df = pd.DataFrame(results['wilcoxon'])
            wilcoxon_df.to_csv(output_dir / 'wilcoxon_tests.csv', index=False)

        # Save McNemar results
        if results.get('mcnemar'):
            mcnemar_df = pd.DataFrame(results['mcnemar'])
            mcnemar_df.to_csv(output_dir / 'mcnemar_tests.csv', index=False)

        # Save confidence intervals
        ci_data = []
        for name, ci in results['confidence_intervals'].items():
            ci_data.append({
                'model': name,
                **ci
            })
        if ci_data:
            ci_df = pd.DataFrame(ci_data)
            ci_df.to_csv(output_dir / 'confidence_intervals.csv', index=False)

        logger.info(f"Results saved to {output_dir}")


def run_statistical_tests(
    model_results: Dict[str, Dict],
    significance_level: float = 0.05,
    predictions: Optional[Dict[str, np.ndarray]] = None,
    y_true: Optional[np.ndarray] = None
) -> Dict:
    """
    Convenience function to run statistical tests.

    Args:
        model_results: Model results dictionary
        significance_level: Alpha level for significance
        predictions: Optional dict of {model_name: predicted_labels} for McNemar test
        y_true: Optional array of true labels for McNemar test

    Returns:
        Statistical test results
    """
    pipeline = Stage9Pipeline(significance_level)
    return pipeline.run(model_results, predictions=predictions, y_true=y_true)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 9: Statistical Significance Testing")
    print("="*70 + "\n")

    # Simulated model results for demonstration
    np.random.seed(RANDOM_SEED)

    model_results = {
        'LightGBM': {
            'fold_scores': [0.998, 0.997, 0.999, 0.998, 0.997]
        },
        'XGBoost': {
            'fold_scores': [0.996, 0.995, 0.997, 0.996, 0.995]
        },
        'RandomForest': {
            'fold_scores': [0.994, 0.993, 0.995, 0.994, 0.993]
        },
        'SVM': {
            'fold_scores': [0.985, 0.984, 0.986, 0.985, 0.984]
        },
        'KNN': {
            'fold_scores': [0.980, 0.979, 0.981, 0.980, 0.979]
        }
    }

    # Run statistical testing
    pipeline = Stage9Pipeline()
    results = pipeline.run(model_results)

    # Print summary
    print("\n" + "-"*50)
    print("STATISTICAL TESTING SUMMARY")
    print("-"*50)
    print(f"Best Model: {results['summary']['best_model']}")
    print(f"Significant Differences Found: {len(results['summary']['significant_differences'])}")
    for diff in results['summary']['significant_differences']:
        print(f"  - {diff}")
    print("-"*50)
