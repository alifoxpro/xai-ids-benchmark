#!/usr/bin/env bash
#
# Master script that runs the full Q1-upgrade experimental suite.
# Total runtime on the reference RTX 5070: approximately 9-12 GPU-days.
#
# Phases run sequentially. Comment out a phase to skip.
# Each phase writes results to results_2026_03_03/<mode>/<phase>/...
#
# Usage:
#   bash experiments/run_all_q1_experiments.sh
#   bash experiments/run_all_q1_experiments.sh --quick    # uses train_quick.yaml

set -e
cd "$(dirname "$0")/.."

QUICK_FLAG=""
if [ "$1" = "--quick" ]; then
    QUICK_FLAG="--quick"
    echo "[INFO] Running in QUICK mode (~6 GPU-hours total)."
fi

echo "============================================================"
echo "  Phase 1: Multiple-seed variance (5 seeds × 10 models × 2 modes)"
echo "============================================================"
python experiments/run_multiple_seeds.py --mode multiclass $QUICK_FLAG
python experiments/run_multiple_seeds.py --mode binary    $QUICK_FLAG

echo "============================================================"
echo "  Phase 2: Design-choice ablation (GRU binary, FT-Transformer multi)"
echo "============================================================"
python experiments/run_ablation_design.py --mode binary    --model GRU
python experiments/run_ablation_design.py --mode multiclass --model FT-Transformer

echo "============================================================"
echo "  Phase 3: XAI stability (Lipschitz + Jaccard)"
echo "============================================================"
python experiments/run_xai_stability.py --mode multiclass --model FT-Transformer
python experiments/run_xai_stability.py --mode binary    --model GRU

echo "============================================================"
echo "  Phase 4: Cross-dataset evaluation (CIC IoT-DIAD -> CICIDS2017)"
echo "============================================================"
python experiments/run_cross_dataset.py \
       --source-mode multiclass \
       --target-dataset CICIDS2017 \
       --models GRU,ResNet1D,FT-Transformer

echo ""
echo "[ALL DONE] Q1 experimental suite completed."
echo "Find results under results_2026_03_03/<mode>/{seed_variance,ablation_design,xai_stability,cross_dataset}/"
