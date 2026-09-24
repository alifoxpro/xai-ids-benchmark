"""
Configuration file for XAI-Based IDS Benchmarking Study
========================================================
Contains all hyperparameters, paths, and settings
"""


from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results_2026_03_03"
VIS_DIR = BASE_DIR / "visualizations"
DATASET_ROOT = BASE_DIR / "data" / "CIC IoT-DIAD 2024"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, VIS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CLASSIFICATION MODE CONFIGURATION
# ============================================================================
CLASSIFICATION_MODES = ["binary", "multiclass"]

CLASSIFICATION_CONFIG = {
    "binary": {
        "classes": {"Benign": 0, "Attack": 1},
        "n_classes": 2,
        "description": "Binary classification: Benign vs Attack"
    },
    "multiclass": {
        "classes": {
            "Benign": 0, "BruteForce": 1, "DDOS": 2, "DOS": 3,
            "Mirai": 4, "Recon": 5, "Spoofing": 6, "Web-Based": 7
        },
        "n_classes": 8,
        "description": "Multi-class classification: 8 attack categories"
    }
}

# Map each top-level category to its sub-folders containing CSVs
DATASET_FOLDERS = {
    "Benign":     ["Benign"],
    "BruteForce": ["BruteForce"],
    "DDOS":       ["DDOS/DDOS ACK Fragmentation", "DDOS/DDOS HTTP FLOOD",
                   "DDOS/DDOS ICMP FLOOD", "DDOS/DDOS ICMP Fragmentation"],
    "DOS":        ["DOS/DOS HTTP FLOOD", "DOS/DOS SYN FLOOD",
                   "DOS/DOS TCP FLOOD", "DOS/DOS UDP FLOOD"],
    "Mirai":      ["Mirai"],
    "Recon":      ["Recon"],
    "Spoofing":   ["Spoofing/ARP Spoofing", "Spoofing/DNS Spoofing"],
    "Web-Based":  ["Web-Based/Sqlinjection", "Web-Based/Uploading Attack",
                   "Web-Based/XSS"]
}

# Columns to drop from CSVs (metadata, not features)
NON_FEATURE_COLUMNS = [
    "Flow ID", "Src IP", "Dst IP", "Src Port", "Dst Port",
    "Protocol", "Timestamp", "Label"
]

# Balancing strategy per mode
# No SMOTE/Undersample — use class-weighted CrossEntropyLoss instead
# Keeps all 17M rows (after deduplication) with real distribution
BALANCING_STRATEGY = {
    "multiclass": {
        "method": "class_weighted",
        "remove_duplicates": True,
    },
    "binary": {
        "method": "class_weighted",
        "remove_duplicates": True,
    }
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    "use_amp": True,             # Mixed Precision (AMP) — ~30% less VRAM, 1.5-2x faster
    "gradient_clip_norm": 1.0,   # Max gradient norm (prevents exploding gradients)
}

# ============================================================================
# CACHE CONFIGURATION
# ============================================================================
CACHE_CONFIG = {
    "enabled": True,                # Cache Stage 1 results to disk
    "cache_dir_name": "stage1_cache",  # Subdirectory under results/{mode}/
    "compress": True,               # Use compressed npz (saves ~60% disk)
}


def get_results_dir(mode: str) -> Path:
    """Return mode-specific results directory."""
    mode_dir = RESULTS_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    return mode_dir


def get_models_dir(mode: str) -> Path:
    """Return mode-specific models directory."""
    mode_dir = MODELS_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    return mode_dir


def get_vis_dir(mode: str) -> Path:
    """Return mode-specific visualizations directory."""
    mode_dir = VIS_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    return mode_dir

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================
DATASETS = {
    "CIC_IoT_DIAD_2024": {
        "name": "CIC IoT-DIAD 2024",
        "n_samples": 1338304,
        "n_features": 78,
        "n_attack_types": 14,
        "is_primary": True,
        "url": None,  # Add download URL if available
        "file_pattern": "*.csv"
    },
    "CICIDS2017": {
        "name": "CICIDS 2017",
        "n_samples": 2830743,
        "n_features": 41,
        "n_attack_types": 11,
        "is_primary": False,
        "url": None,
        "file_pattern": "*.csv"
    },
    "UNSW_NB15": {
        "name": "UNSW-NB15",
        "n_samples": 2540044,
        "n_features": 42,
        "n_attack_types": 10,
        "is_primary": False,
        "url": None,
        "file_pattern": "*.csv"
    },
    "CICIOT2023": {
        "name": "CICIOT 2023",
        "n_samples": 1000000,
        "n_features": 23,
        "n_attack_types": 7,
        "is_primary": False,
        "url": None,
        "file_pattern": "*.csv"
    }
}

# ============================================================================
# DATA SPLIT CONFIGURATION
# ============================================================================
DATA_SPLIT = {
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,
    "random_state": 42,
    "stratify": True,
    "n_folds": 5  # For cross-validation
}

# ============================================================================
# FEATURE SELECTION CONFIGURATIONS
# ============================================================================
FEATURE_SELECTION = {
    "SHAP": {
        "top_k_options": [10, 20, 30, 40, 50],
        "baseline_model": "RandomForest",
        "n_samples_explain": 1000
    },
    "PCA": {
        "n_components_options": [5, 10, 15, 20, 25, 30],
        "variance_thresholds": [0.80, 0.85, 0.90, 0.95]
    },
    "mRMR": {
        "top_k_options": [10, 20, 30, 40, 50],
        "method": "MIQ"  # or "MID"
    },
    "Autoencoder": {
        "latent_dims": [8, 10, 16, 20],
        "architecture": [78, 64, 32, 16, 8],  # Encoder layers
        "epochs": 50,
        "batch_size": 256
    },
    "PermutationImportance": {
        "top_k_options": [10, 20, 30, 40, 50],
        "n_repeats": 10
    },
    "InformationGain": {
        "top_k_options": [10, 20, 30, 40, 50],
        "n_neighbors": 3,
        "random_state": 42
    }
}

# ============================================================================
# TRADITIONAL ML MODELS HYPERPARAMETERS
# ============================================================================
ML_MODELS = {
    "DecisionTree": {
        "max_depth": [5, 10, 20, None],
        "criterion": ["gini", "entropy"],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "RandomForest": {
        "n_estimators": [50, 100, 200, 500],
        "max_depth": [5, 10, 20, None],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
        "n_jobs": -1
    },
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"]
    },
    "SVM": {
        "kernel": ["rbf", "linear", "poly", "sigmoid"],
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "n_jobs": -1
    },
    "LightGBM": {
        "num_leaves": [20, 31, 50, 100],
        "max_depth": [5, 10, 15, -1],
        "learning_rate": [0.01, 0.1, 0.3],
        "feature_fraction": [0.7, 0.8, 0.9, 1.0],
        "bagging_fraction": [0.7, 0.8, 0.9, 1.0],
        "n_jobs": -1
    },
    "CatBoost": {
        "iterations": [100, 200, 500],
        "depth": [4, 6, 8, 10],
        "learning_rate": [0.01, 0.1, 0.3],
        "l2_leaf_reg": [1, 3, 5, 7]
    },
    "MLP": {
        "hidden_layer_sizes": [(128, 64), (256, 128, 64), (512, 256, 128)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
        "max_iter": 500
    },
    "Ensemble": {
        "method": "stacking",  # or "voting"
        "base_estimators": ["RandomForest", "LightGBM", "XGBoost"],
        "meta_learner": "LogisticRegression",
        "cv": 3
    }
}

# ============================================================================
# DEEP LEARNING MODELS CONFIGURATIONS
# ============================================================================
DL_MODELS = {
    "CNN_LSTM": {
        "conv_layers": [1, 2],
        "conv_filters": [32, 64],
        "lstm_units": [64, 128, 256],
        "dropout_rates": [0.2, 0.3, 0.5],
        "dense_layers": [1, 2],
        "dense_units": [128, 256],
        "batch_size": 256,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "Transformer": {
        "attention_heads": [4, 8, 16],
        "hidden_dim": [256, 512],
        "encoder_layers": [2, 4, 6],
        "dropout_rates": [0.1, 0.2],
        "batch_size": 256,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "AutoencoderClassifier": {
        "encoder_architecture": [78, 64, 32, 16, 8],
        "decoder_architecture": [8, 16, 32, 64, 78],
        "classifier_head": [128, 64],
        "loss_weights": [0.5, 1.0],  # AE loss vs Classification loss
        "dropout_rates": [0.2, 0.3],
        "batch_size": 512,
        "epochs": 100
    },
    "TabNet": {
        "n_steps": 3,
        "feature_dim": 64,
        "output_dim": 64,
        "relaxation_factor": 1.5,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping_patience": 15
    },
    "ResNet1D": {
        "n_blocks": 3,
        "channels": [128, 128, 256],
        "embed_dim": 128,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "BiLSTM_Attention": {
        "hidden_size": 128,
        "n_layers": 2,
        "attention_dim": 64,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "TCN": {
        "channels": [64, 128, 128],
        "kernel_size": 3,
        "dropout": 0.2,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "GRU": {
        "hidden_size": 128,
        "n_layers": 2,
        "bidirectional": True,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping_patience": 10
    },
    "FT_Transformer": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.1,
        "batch_size": 512,
        "epochs": 100,
        "early_stopping_patience": 15
    }
}

# ============================================================================
# XAI CONFIGURATIONS
# ============================================================================
XAI_CONFIG = {
    "SHAP": {
        "explainer_type": "GradientExplainer",  # For DL models; KernelExplainer as fallback
        "n_samples_background": 100,
        "n_samples_explain": 1000
    },
    "LIME": {
        "num_features": 30,
        "num_samples": 5000
    },
    "Attention": {
        "extract_weights": True
    },
    "GradientBased": {
        "methods": ["IntegratedGradients", "GradientSHAP"]
    },
    "Anchors": {
        "threshold": 0.95,
        "beam_size": 4,
        "coverage_samples": 10000
    }
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================
METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
    "f1_macro",
    "g_mean",
    "roc_auc",
    "confusion_matrix",
    "classification_report",
    "mcc",
    "cohens_kappa",
    "fpr",
    "far",
    "pr_auc",
    "detection_rate"
]

EFFICIENCY_METRICS = [
    "training_time",
    "inference_time",
    "model_size",
    "memory_usage",
    "throughput"
]

XAI_QUALITY_METRICS = [
    "fidelity",
    "stability",
    "consistency",
    "explanation_time",
    "sparsity",
    "faithfulness_sufficiency",
    "faithfulness_necessity"
]

# ============================================================================
# ADVERSARIAL TESTING CONFIGURATIONS
# ============================================================================
ADVERSARIAL_CONFIG = {
    "FGSM": {
        "epsilon_values": [0.1, 0.3, 0.5]
    },
    "PGD": {
        "epsilon_values": [0.1, 0.3, 0.5],
        "n_steps": 10,
        "step_size": 0.01
    },
    "GaussianNoise": {
        "sigma_values": [0.05, 0.10, 0.15]
    },
    "ConceptDrift": {
        "drift_magnitudes": [0.1, 0.3, 0.5],
        "drift_types": ["gradual", "sudden"],
        "n_drift_steps": 10
    }
}

# ============================================================================
# STATISTICAL TESTING
# ============================================================================
STATISTICAL_TESTS = {
    "significance_level": 0.05,
    "n_folds": 5,
    "tests": ["ANOVA", "t-test", "Friedman", "Nemenyi", "Wilcoxon", "McNemar"]
}

# ============================================================================
# RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
RANDOM_SEED = 42

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": RESULTS_DIR / "experiment.log"
}
