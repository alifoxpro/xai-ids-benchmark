# XAI-IDS Benchmark: Leakage-Free, Multi-Criteria Evaluation of Deep-Learning IoT Intrusion Detection

Code and result files for the paper

> **Minority-Class Detection in Deep-Learning IoT Intrusion Detection: A Leakage-Free Multi-Criteria Evaluation**
> A. H. Fadel, University of Diyala. Submitted to *Cybernetics and Information Technologies* (2026).

Ten deep-learning architectures are evaluated on the full, deduplicated CIC IoT-DIAD 2024
dataset (about 17 million flows) under a leakage-free protocol (stratified split before any
re-weighting, class-weighted loss, no oversampling). The models are compared along seven
criteria: effectiveness, per-class detection, computational cost, explainability,
adversarial robustness, cross-dataset generalization (CICIDS2017) and statistical
significance. The central finding is a large gap between accuracy (90.2%) and
macro-F1 (0.48), caused by the rare attack classes; it reappears on UNSW-NB15 and NSL-KDD.

Architectures: GRU, ResNet1D, FT-Transformer, CNN-LSTM, BiLSTM-Attention, TabNet, TCN,
Transformer, AE-Classifier and a soft-voting ensemble.

## Installation

Reference environment: Python 3.11.14, PyTorch 2.10.0 + CUDA 12.8, NVIDIA RTX 5070 Laptop GPU
(8 GB, compute capability 12.0), Windows 11.

```bash
conda create -n xai-ids python=3.11.14
conda activate xai-ids
pip install -r requirements.txt
```

or with Docker:

```bash
docker build -t xai-ids-benchmark .
docker run --gpus all -v "$(pwd)/data:/app/data" -v "$(pwd)/results:/app/results" xai-ids-benchmark python main.py --mode multiclass
```

## Datasets

The datasets are not included in this repository. Download them and place them under `data/`:

| Dataset | Source | Folder |
|---|---|---|
| CIC IoT-DIAD 2024 (primary) | https://www.unb.ca/cic/datasets/iot-diad-2024.html | `data/CIC IoT-DIAD 2024/` |
| CICIDS2017 (cross-dataset transfer) | https://www.unb.ca/cic/datasets/ids-2017.html | `data/cic-ids-2017/` |
| UNSW-NB15 (generalization) | https://research.unsw.edu.au/projects/unsw-nb15-dataset | `data/UNSW-NB15/` |
| NSL-KDD (generalization) | https://www.unb.ca/cic/datasets/nsl.html | `data/nsl-kdd/` |

For CIC IoT-DIAD 2024 the eight folder-level classes are used (Benign, BruteForce, DDOS, DOS,
Mirai, Recon, Spoofing, Web-Based); labels are taken from the folder names.

## Running the pipeline

`main.py` runs the ten stages in `src/`:

```bash
python main.py --mode multiclass            # all stages, multi-class
python main.py --mode binary                # all stages, binary
python main.py --mode multiclass --stage 3  # a single stage
python main.py --mode multiclass --quick    # reduced smoke test
```

| Stage | File | Content |
|---|---|---|
| 1 | `src/stage1_data_preparation.py` | loading, deduplication, stratified split, class weights |
| 2 | `src/stage2_feature_selection.py` | feature selection |
| 3 | `src/stage3_ml_models.py` | the ten deep-learning architectures and training |
| 4 | `src/stage4_comprehensive_results.py` | effectiveness metrics |
| 5 | `src/stage5_xai_quality.py` | SHAP / LIME fidelity, latency and agreement |
| 6 | `src/stage6_performance_metrics.py` | parameters, FLOPs, training and inference time |
| 7 | `src/stage7_cross_dataset.py` | transfer to CICIDS2017 |
| 8 | `src/stage8_robustness_testing.py` | FGSM, PGD, Gaussian noise, drift |
| 9 | `src/stage9_statistical_testing.py` | Friedman, Wilcoxon + Holm, McNemar, Nemenyi, bootstrap |
| 10 | `src/stage10_final_report.py` | final report |

Primary seed: 42. Seed-variance runs use 17, 42, 123, 2024 and 31337 (`configs/seeds.txt`).

## Additional experiments (`experiments/`)

| Script | Paper content |
|---|---|
| `recompute_fulltest_metrics.py` | per-class metrics of the best model on the full test set |
| `audit_binary_metrics.py` | checkpoint-verified binary results |
| `compute_params_flops.py` | parameter counts and FLOPs |
| `run_xai_stability_real.py` | Lipschitz sensitivity and top-10 Jaccard stability |
| `generate_xai_misclassified.py` | explanations of misclassified minority flows |
| `rerun_robustness_fixed.py` | FGSM / PGD / noise / drift robustness |
| `run_real_cross_dataset.py`, `run_cross_dataset_binary.py` | eight-class and binary transfer to CICIDS2017 |
| `run_multiple_seeds.py`, `run_3model_multi_seed.py`, `run_q1_ml_variance.py` | seed variance |
| `run_ablation_design.py` | ablation (class weighting, data size, depth) |
| `fusion_experiment.py` | early versus late fusion |
| `recipe_experiment.py` | balanced training recipe on three datasets |
| `run_smote_leakage_demo.py` | effect of SMOTE before the split |
| `generalize_illusion.py` | accuracy / macro-F1 gap on UNSW-NB15 and NSL-KDD |
| `topsis_synthesis.py` | TOPSIS multi-criteria ranking |

## Results

`results_2026_03_03/` contains the result files (CSV, JSON, TXT) of the run reported in the
paper, for the `multiclass` and `binary` modes. Cached arrays and trained model weights are
too large for git; they are provided as assets of the GitHub Releases of this repository.

## Citation

```bibtex
@article{fadel2026minority,
  title   = {Minority-Class Detection in Deep-Learning IoT Intrusion Detection:
             A Leakage-Free Multi-Criteria Evaluation},
  author  = {Fadel, Ali Hussein},
  journal = {Cybernetics and Information Technologies},
  year    = {2026},
  note    = {Submitted}
}
```

## Licence

Code: Apache License 2.0. Result files: CC BY 4.0.

## Contact

Ali Hussein Fadel, Department of Computer Science, College of Science, University of Diyala,
Iraq (ali_hussein_fadel@uodiyala.edu.iq). Questions and issues are welcome via GitHub Issues.
