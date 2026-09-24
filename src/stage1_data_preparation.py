"""
================================================================================
STAGE 1: DATASET PREPARATION & BASELINE
================================================================================
This module handles:
- Loading CIC IoT-DIAD 2024 from real folder structure
- Labeling from folder names (binary or multiclass)
- Data preprocessing (normalization, missing values, duplicates)
- Hybrid class balancing (undersample large + SMOTE small)
- Feature engineering
- Train/Val/Test splitting with stratification
- 5-Fold Cross-Validation setup
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime
import pickle
import json

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

from config import (
    DATA_DIR, RESULTS_DIR, DATASETS, DATA_SPLIT, RANDOM_SEED,
    DATASET_ROOT, DATASET_FOLDERS, NON_FEATURE_COLUMNS,
    CLASSIFICATION_CONFIG, BALANCING_STRATEGY, get_results_dir
)

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Loads CIC IoT-DIAD 2024 dataset from real folder structure.
    Assigns labels based on folder names.
    Uses chunked reading to load ALL data without sampling.
    Memory-optimized with float32 dtypes.
    """

    CHUNK_SIZE = 50_000  # rows per chunk during CSV reading

    def __init__(self, dataset_root: Optional[Path] = None,
                 mode: str = "multiclass"):
        self.dataset_root = dataset_root or DATASET_ROOT
        self.mode = mode
        self.folder_map = DATASET_FOLDERS
        self.non_feature_cols = NON_FEATURE_COLUMNS
        self.class_config = CLASSIFICATION_CONFIG[mode]
        self.df = None
        self.label_column = "label"
        self.feature_columns = None

    def _get_feature_columns_from_header(self, csv_file: Path) -> List[str]:
        """Read only the header row and return feature column names."""
        header = pd.read_csv(csv_file, nrows=0).columns.tolist()
        # Strip whitespace from column names
        header = [c.strip() for c in header]
        return [c for c in header if c not in self.non_feature_cols]

    def _read_csv_chunked(self, csv_file: Path, feature_cols: List[str]
                          ) -> Optional[pd.DataFrame]:
        """
        Read a single CSV in chunks, keeping only feature columns.
        Converts everything to float32 for memory efficiency.
        Returns None if the file has no data rows.
        """
        chunks = []
        # Read only feature columns (skip metadata like IPs, Timestamp)
        # Plus keep Label column for verification logging only
        usecols_set = set(feature_cols)

        try:
            reader = pd.read_csv(
                csv_file,
                usecols=lambda c: c.strip() in usecols_set,
                chunksize=self.CHUNK_SIZE,
                low_memory=False
            )
            for chunk in reader:
                # Strip column name whitespace
                chunk.columns = [c.strip() for c in chunk.columns]
                # Force numeric and downcast to float32
                for col in chunk.columns:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
                chunk = chunk.astype(np.float32)
                chunks.append(chunk)
        except Exception as e:
            logger.error(f"  Error reading {csv_file.name}: {e}")
            return None

        if not chunks:
            return None

        return pd.concat(chunks, ignore_index=True)

    def load_data(self) -> pd.DataFrame:
        """
        Walk the 8 top-level category folders, read ALL CSVs in chunks,
        assign label from the top-level folder name.

        Memory strategy:
          - Read only feature columns (drop IPs, Timestamp, etc. at read time)
          - Use float32 instead of float64 (halves memory)
          - Process each CSV in 50K-row chunks
          - ~19.5M rows x 76 features x 4 bytes = ~5.6 GB
        """
        import gc

        logger.info(f"Loading FULL CIC IoT-DIAD 2024 dataset (mode={self.mode})...")
        logger.info(f"Dataset root: {self.dataset_root}")
        logger.info(f"Chunk size: {self.CHUNK_SIZE} rows")

        # Phase 1: Discover all CSVs and determine feature columns from first file
        all_csv_files = []  # list of (category, csv_path)
        for category, subfolders in self.folder_map.items():
            for subfolder in subfolders:
                folder_path = self.dataset_root / subfolder
                csv_files = sorted(folder_path.glob("*.csv"))
                if not csv_files:
                    logger.warning(f"  No CSVs in {folder_path}")
                    continue
                for csv_file in csv_files:
                    all_csv_files.append((category, csv_file))

        if not all_csv_files:
            raise FileNotFoundError(
                f"No CSV files found under {self.dataset_root}. "
                "Check DATASET_ROOT in config.py."
            )

        logger.info(f"Found {len(all_csv_files)} CSV files across {len(self.folder_map)} categories")

        # Get feature columns from first CSV header (all CSVs share same 84 columns)
        feature_cols = self._get_feature_columns_from_header(all_csv_files[0][1])
        logger.info(f"Feature columns from header: {len(feature_cols)}")

        # Phase 2: Read all CSVs category by category using chunked reading
        all_dfs = []
        for category, subfolders in self.folder_map.items():
            csv_files_for_cat = [(cat, p) for cat, p in all_csv_files if cat == category]
            if not csv_files_for_cat:
                logger.warning(f"  {category}: No CSV files found!")
                continue

            category_dfs = []
            cat_rows = 0
            for _, csv_file in csv_files_for_cat:
                df_chunk = self._read_csv_chunked(csv_file, feature_cols)
                if df_chunk is not None and len(df_chunk) > 0:
                    cat_rows += len(df_chunk)
                    category_dfs.append(df_chunk)

            if not category_dfs:
                logger.warning(f"  {category}: No data rows found (all CSVs empty)")
                continue

            cat_df = pd.concat(category_dfs, ignore_index=True)
            del category_dfs
            gc.collect()

            # Assign label based on mode
            if self.mode == "binary":
                cat_df[self.label_column] = "Benign" if category == "Benign" else "Attack"
            else:
                cat_df[self.label_column] = category

            logger.info(f"  {category}: {len(cat_df):,} rows loaded")
            all_dfs.append(cat_df)

        # Phase 3: Concatenate all categories
        logger.info("Concatenating all categories...")
        self.df = pd.concat(all_dfs, ignore_index=True)
        del all_dfs
        gc.collect()

        logger.info(f"Total dataset shape: {self.df.shape}")
        mem_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info(f"DataFrame memory usage: {mem_mb:.1f} MB")

        # Log class distribution
        dist = self.df[self.label_column].value_counts()
        for cls, count in dist.items():
            logger.info(f"  {cls}: {count:,}")

        return self.df

    def identify_columns(self) -> Tuple[List[str], str]:
        """
        Identify feature columns by removing non-feature metadata columns.
        Preserves real CICFlowMeter feature names.
        """
        if self.df is None:
            raise ValueError("Call load_data() first")

        # Drop non-feature columns that exist in the DataFrame
        existing_cols = set(self.df.columns)
        cols_to_drop = [c for c in self.non_feature_cols if c in existing_cols]

        self.feature_columns = [
            c for c in self.df.columns
            if c not in cols_to_drop and c != self.label_column
        ]

        logger.info(f"Label column: {self.label_column}")
        logger.info(f"Feature columns: {len(self.feature_columns)}")
        logger.info(f"Dropped metadata columns: {cols_to_drop}")

        return self.feature_columns, self.label_column


class DataPreprocessor:
    """
    Handles data preprocessing including:
    - Missing value imputation
    - Duplicate removal
    - Normalization/Standardization
    - Feature type handling
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_stats = {}

    def handle_missing_values(self, df: pd.DataFrame,
                              strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in the dataset. Memory-efficient for large datasets."""
        logger.info("Handling missing values...")

        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]

        if len(missing_cols) > 0:
            logger.info(f"Columns with missing values: {len(missing_cols)}")
            for col_name in missing_cols.index:
                logger.info(f"  {col_name}: {missing_cols[col_name]:,} missing")

            # Drop columns that are entirely NaN
            all_nan_cols = df.columns[df.isnull().all()].tolist()
            if all_nan_cols:
                logger.info(f"Dropping {len(all_nan_cols)} all-NaN columns")
                df = df.drop(columns=all_nan_cols)

            # Fill missing values column-by-column (avoids SimpleImputer memory spike)
            for col in missing_cols.index:
                if col in all_nan_cols:
                    continue  # already dropped
                if col not in df.columns:
                    continue
                if np.issubdtype(df[col].dtype, np.number):
                    if strategy == 'median':
                        fill_val = df[col].median()
                    elif strategy == 'mean':
                        fill_val = df[col].mean()
                    else:
                        fill_val = 0
                    df[col] = df[col].fillna(fill_val)
                    logger.info(f"  Filled {col} with {strategy}={fill_val:.4f}")
                else:
                    fill_val = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "unknown"
                    df[col] = df[col].fillna(fill_val)
                    logger.info(f"  Filled {col} with mode={fill_val}")
        else:
            logger.info("No missing values found")

        return df

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows from the dataset."""
        import gc
        initial_shape = df.shape[0]
        logger.info(f"Checking for duplicates in {initial_shape:,} rows...")
        df = df.drop_duplicates()
        removed = initial_shape - df.shape[0]
        logger.info(f"Removed {removed:,} duplicate rows ({removed/initial_shape*100:.2f}%)")
        gc.collect()
        return df

    def handle_infinite_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace infinite values with NaN column-by-column (memory-efficient)."""
        import gc
        logger.info("Checking for infinite values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        total_inf = 0
        for col in numeric_cols:
            inf_count = np.isinf(df[col].values).sum()
            if inf_count > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                total_inf += inf_count
        if total_inf > 0:
            logger.info(f"Replaced {total_inf:,} infinite values with NaN")
        else:
            logger.info("No infinite values found")
        gc.collect()
        return self.handle_missing_values(df)

    def normalize_features(self, df: pd.DataFrame,
                          feature_columns: List[str],
                          method: str = 'standard') -> pd.DataFrame:
        """Normalize/Standardize numerical features. Memory-efficient for large datasets."""
        logger.info(f"Normalizing features using {method} method...")

        numeric_features = [c for c in feature_columns
                          if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        if not numeric_features:
            return df

        n_rows = len(df)
        logger.info(f"  Normalizing {len(numeric_features)} numeric features, "
                   f"{n_rows:,} rows...")

        if method == 'standard':
            # Compute mean and std column-by-column (no full copy needed)
            means = {}
            stds = {}
            for col in numeric_features:
                means[col] = float(df[col].mean())
                stds[col] = float(df[col].std())
                if stds[col] == 0 or np.isnan(stds[col]):
                    stds[col] = 1.0  # avoid division by zero
                df[col] = ((df[col] - means[col]) / stds[col]).astype(np.float32)

            # Store a StandardScaler-compatible object for later stages
            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array([means[c] for c in numeric_features])
            self.scaler.var_ = np.array([stds[c]**2 for c in numeric_features])
            self.scaler.scale_ = np.array([stds[c] for c in numeric_features])
            self.scaler.n_features_in_ = len(numeric_features)
            self.scaler.feature_names_in_ = np.array(numeric_features)

            self.feature_stats['mean'] = means
            self.feature_stats['std'] = stds

        elif method == 'minmax':
            mins = {}
            maxs = {}
            for col in numeric_features:
                mins[col] = float(df[col].min())
                maxs[col] = float(df[col].max())
                denom = maxs[col] - mins[col]
                if denom == 0:
                    denom = 1.0
                df[col] = ((df[col] - mins[col]) / denom).astype(np.float32)

            self.scaler = MinMaxScaler()
            self.scaler.data_min_ = np.array([mins[c] for c in numeric_features])
            self.scaler.data_max_ = np.array([maxs[c] for c in numeric_features])
            self.scaler.data_range_ = self.scaler.data_max_ - self.scaler.data_min_
            self.scaler.scale_ = 1.0 / np.where(self.scaler.data_range_ == 0, 1.0, self.scaler.data_range_)
            self.scaler.min_ = -self.scaler.data_min_ * self.scaler.scale_
            self.scaler.n_features_in_ = len(numeric_features)
            self.scaler.feature_names_in_ = np.array(numeric_features)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return df

    def encode_labels(self, df: pd.DataFrame,
                     label_column: str) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical labels to numeric values."""
        logger.info("Encoding labels...")

        self.label_encoder = LabelEncoder()
        df[label_column + '_encoded'] = self.label_encoder.fit_transform(df[label_column])

        label_mapping = dict(zip(self.label_encoder.classes_,
                                 range(len(self.label_encoder.classes_))))

        logger.info(f"Label classes: {list(label_mapping.keys())}")

        return df, label_mapping

    def preprocess_pipeline(self, df: pd.DataFrame,
                           feature_columns: List[str],
                           label_column: str,
                           normalize: bool = True,
                           normalize_method: str = 'standard') -> Tuple[pd.DataFrame, Dict]:
        """Run full preprocessing pipeline."""
        import gc
        logger.info(f"Running preprocessing pipeline on {len(df):,} rows...")

        # Feature columns should already be float32 from chunked loading
        # Just ensure no stray object dtypes
        non_numeric = [c for c in feature_columns
                      if c in df.columns and not np.issubdtype(df[c].dtype, np.number)]
        if non_numeric:
            logger.info(f"Converting {len(non_numeric)} non-numeric feature columns...")
            for col in non_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float32)

        # Step 1: Handle infinite values
        df = self.handle_infinite_values(df)

        # Step 2: Handle missing values
        df = self.handle_missing_values(df)

        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        gc.collect()

        # Step 4: Normalize features
        if normalize:
            df = self.normalize_features(df, feature_columns, normalize_method)
            gc.collect()

        # Step 5: Encode labels
        df, label_mapping = self.encode_labels(df, label_column)

        logger.info(f"Preprocessing complete. Final shape: {df.shape}")

        return df, label_mapping


class ClassBalancer:
    """
    Handles class imbalance using:
    - Hybrid approach: undersample large + SMOTE small
    - SMOTE, ADASYN, RandomUnderSampler
    - Combination methods (SMOTETomek, SMOTEENN)
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.balancer = None

    def get_class_distribution(self, y: np.ndarray) -> Dict:
        """Get class distribution statistics."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        distribution = {
            'counts': dict(zip(unique, counts)),
            'percentages': dict(zip(unique, counts / total * 100))
        }
        return distribution

    def balance_hybrid(self, X: np.ndarray, y: np.ndarray,
                       mode: str = "multiclass") -> Tuple[np.ndarray, np.ndarray]:
        """
        Hybrid balancing: undersample large classes, SMOTE small classes.

        Multiclass strategy:
          - Classes > undersample_cap (100K): random undersample to 100K
          - Classes < smote_threshold (50K): SMOTE up to 50K
          - Classes between 50K-100K: keep as-is

        Binary strategy:
          - Both classes capped at undersample_cap (200K)
        """
        strategy = BALANCING_STRATEGY[mode]

        initial_dist = self.get_class_distribution(y)
        logger.info(f"Initial distribution: {initial_dist['counts']}")

        unique_classes = np.unique(y)
        undersample_cap = strategy["undersample_cap"]

        # Phase 1: Undersample large classes
        indices_to_keep = []
        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            if len(cls_indices) > undersample_cap:
                sampled = np.random.RandomState(self.random_state).choice(
                    cls_indices, undersample_cap, replace=False
                )
                indices_to_keep.append(sampled)
                logger.info(f"  Class {cls}: undersampled {len(cls_indices)} -> {undersample_cap}")
            else:
                indices_to_keep.append(cls_indices)
                logger.info(f"  Class {cls}: kept {len(cls_indices)} (no undersample needed)")

        import gc
        indices_to_keep = np.concatenate(indices_to_keep)
        np.random.RandomState(self.random_state).shuffle(indices_to_keep)
        logger.info(f"  Selecting {len(indices_to_keep):,} rows from {len(y):,}...")
        X_under = X[indices_to_keep]
        y_under = y[indices_to_keep]
        # Free original large arrays if possible
        gc.collect()

        # Phase 2: SMOTE small classes
        smote_threshold = strategy.get("smote_threshold")
        smote_target = strategy.get("smote_target")

        if smote_threshold is not None and smote_target is not None:
            sampling_strategy = {}
            min_class_count = min(np.sum(y_under == cls) for cls in np.unique(y_under))

            for cls in np.unique(y_under):
                cls_count = np.sum(y_under == cls)
                if cls_count < smote_threshold:
                    sampling_strategy[cls] = smote_target

            if sampling_strategy:
                logger.info(f"  SMOTE targets: {sampling_strategy}")
                k_neighbors = min(5, min_class_count - 1)
                k_neighbors = max(1, k_neighbors)

                smote = SMOTE(
                    sampling_strategy=sampling_strategy,
                    random_state=self.random_state,
                    k_neighbors=k_neighbors
                )
                X_balanced, y_balanced = smote.fit_resample(X_under, y_under)
                logger.info(f"  SMOTE applied: {len(y_under)} -> {len(y_balanced)}")
            else:
                X_balanced, y_balanced = X_under, y_under
                logger.info("  No classes need SMOTE")
        else:
            X_balanced, y_balanced = X_under, y_under

        final_dist = self.get_class_distribution(y_balanced)
        logger.info(f"Final distribution: {final_dist['counts']}")
        logger.info(f"Total samples: {len(y)} -> {len(y_balanced)}")

        return X_balanced, y_balanced

    def balance_classes(self, X: np.ndarray, y: np.ndarray,
                       method: str = 'smote',
                       sampling_strategy: Union[str, Dict] = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """Balance classes using specified method (legacy support)."""
        logger.info(f"Balancing classes using {method}...")

        initial_dist = self.get_class_distribution(y)
        logger.info(f"Initial class distribution: {initial_dist['counts']}")

        if method == 'smote':
            self.balancer = SMOTE(sampling_strategy=sampling_strategy,
                                  random_state=self.random_state)
        elif method == 'adasyn':
            self.balancer = ADASYN(sampling_strategy=sampling_strategy,
                                   random_state=self.random_state)
        elif method == 'under':
            self.balancer = RandomUnderSampler(sampling_strategy=sampling_strategy,
                                               random_state=self.random_state)
        elif method == 'smote_tomek':
            self.balancer = SMOTETomek(random_state=self.random_state)
        elif method == 'smote_enn':
            self.balancer = SMOTEENN(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown balancing method: {method}")

        X_balanced, y_balanced = self.balancer.fit_resample(X, y)

        final_dist = self.get_class_distribution(y_balanced)
        logger.info(f"Final class distribution: {final_dist['counts']}")
        logger.info(f"Samples: {len(y)} -> {len(y_balanced)}")

        return X_balanced, y_balanced


class DataSplitter:
    """
    Handles data splitting for training, validation, and testing.
    Supports stratified splitting and k-fold cross-validation.
    """

    def __init__(self, config: Dict = DATA_SPLIT, random_state: int = RANDOM_SEED):
        self.config = config
        self.random_state = random_state

    def train_val_test_split(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Split data into train, validation, and test sets."""
        logger.info("Splitting data into train/val/test sets...")

        train_ratio = self.config['train_ratio']
        val_ratio = self.config['val_ratio']
        test_ratio = self.config['test_ratio']

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            random_state=self.random_state,
            stratify=y if self.config['stratify'] else None
        )

        # Second split: separate validation from training
        val_adjusted_ratio = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_adjusted_ratio,
            random_state=self.random_state,
            stratify=y_temp if self.config['stratify'] else None
        )

        splits = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

        logger.info(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

        return splits

    def get_cv_folds(self, X: np.ndarray, y: np.ndarray) -> List[Tuple]:
        """Generate stratified k-fold cross-validation indices."""
        logger.info(f"Generating {self.config['n_folds']}-fold CV splits...")

        skf = StratifiedKFold(
            n_splits=self.config['n_folds'],
            shuffle=True,
            random_state=self.random_state
        )

        folds = list(skf.split(X, y))

        for i, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"Fold {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")

        return folds


class FeatureEngineer:
    """
    Performs feature engineering including:
    - Statistical features
    - Ratio features
    """

    def __init__(self):
        self.new_features = []

    def create_statistical_features(self, df: pd.DataFrame,
                                    feature_columns: List[str]) -> pd.DataFrame:
        """Create statistical aggregate features. Uses numpy for memory efficiency."""
        logger.info("Creating statistical features...")

        numeric_cols = [c for c in feature_columns
                       if df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]

        if len(numeric_cols) >= 5:
            n_rows = len(df)
            logger.info(f"  Computing stats across {len(numeric_cols)} columns, "
                       f"{n_rows:,} rows (chunked)...")

            # Use numpy directly on the underlying array to avoid pandas float64 upcast
            vals = df[numeric_cols].values  # already float32

            # Compute in chunks to avoid memory spike
            chunk_size = 500_000
            feat_mean = np.empty(n_rows, dtype=np.float32)
            feat_std = np.empty(n_rows, dtype=np.float32)
            feat_min = np.empty(n_rows, dtype=np.float32)
            feat_max = np.empty(n_rows, dtype=np.float32)

            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                chunk = vals[start:end]
                feat_mean[start:end] = np.nanmean(chunk, axis=1)
                feat_std[start:end] = np.nanstd(chunk, axis=1)
                feat_min[start:end] = np.nanmin(chunk, axis=1)
                feat_max[start:end] = np.nanmax(chunk, axis=1)

            df['feat_mean'] = feat_mean
            df['feat_std'] = feat_std
            df['feat_min'] = feat_min
            df['feat_max'] = feat_max
            df['feat_range'] = (feat_max - feat_min).astype(np.float32)

            del vals, feat_mean, feat_std, feat_min, feat_max

            self.new_features.extend(['feat_mean', 'feat_std', 'feat_min',
                                      'feat_max', 'feat_range'])

        return df

    def create_ratio_features(self, df: pd.DataFrame,
                             feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """Create ratio features between feature pairs."""
        logger.info("Creating ratio features...")

        for num_col, denom_col in feature_pairs:
            if num_col in df.columns and denom_col in df.columns:
                ratio_name = f"ratio_{num_col}_{denom_col}"
                df[ratio_name] = df[num_col] / (df[denom_col] + 1e-8)
                self.new_features.append(ratio_name)

        return df

    def get_all_features(self, original_features: List[str]) -> List[str]:
        """Get combined list of original and engineered features."""
        return original_features + self.new_features


class Stage1Pipeline:
    """
    Main pipeline for Stage 1: Dataset Preparation & Baseline.
    Supports binary and multiclass classification modes.
    """

    def __init__(self, dataset_name: str = "CIC_IoT_DIAD_2024",
                 mode: str = "multiclass"):
        self.dataset_name = dataset_name
        self.mode = mode
        self.loader = DatasetLoader(mode=mode)
        self.preprocessor = DataPreprocessor()
        self.balancer = ClassBalancer()
        self.splitter = DataSplitter()
        self.engineer = FeatureEngineer()

        self.df = None
        self.X = None
        self.y = None
        self.splits = None
        self.cv_folds = None
        self.label_mapping = None
        self.feature_columns = None

    def run(self, balance_classes: bool = True,
           balance_method: str = 'hybrid',
           engineer_features: bool = True) -> Dict:
        """
        Run the complete Stage 1 pipeline.

        Args:
            balance_classes: Whether to apply class balancing
            balance_method: 'hybrid' (recommended), 'smote', 'under', etc.
            engineer_features: Whether to create engineered features

        Returns:
            Dictionary containing all processed data and metadata
        """
        logger.info("=" * 60)
        logger.info(f"STAGE 1: DATASET PREPARATION ({self.mode.upper()} MODE)")
        logger.info("=" * 60)

        # Step 1: Load data from real folder structure
        self.df = self.loader.load_data()
        self.feature_columns, label_column = self.loader.identify_columns()

        # Step 2: Feature engineering (before preprocessing)
        if engineer_features:
            # Convert to numeric first for statistical features
            for col in self.feature_columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

            self.df = self.engineer.create_statistical_features(
                self.df, self.feature_columns
            )
            self.feature_columns = self.engineer.get_all_features(self.feature_columns)

        # Step 3: Preprocess data
        self.df, self.label_mapping = self.preprocessor.preprocess_pipeline(
            self.df,
            self.feature_columns,
            label_column
        )

        # Step 4: Prepare arrays (float32 for memory efficiency)
        import gc
        logger.info("Extracting feature arrays...")
        self.X = self.df[self.feature_columns].values.astype(np.float32)
        self.y = self.df[label_column + '_encoded'].values

        # Free the DataFrame - we only need arrays from here
        del self.df
        self.df = None
        gc.collect()

        logger.info(f"Arrays: X={self.X.shape}, y={self.y.shape}")
        mem_mb = self.X.nbytes / (1024 * 1024)
        logger.info(f"X array memory: {mem_mb:.1f} MB")

        # Step 5: Class distribution (no balancing — use class-weighted loss instead)
        self._class_dist_before = self.balancer.get_class_distribution(self.y)
        # Compute class weights for weighted CrossEntropyLoss
        from sklearn.utils.class_weight import compute_class_weight
        unique_classes = np.unique(self.y)
        cw = compute_class_weight('balanced', classes=unique_classes, y=self.y)
        self._class_weights = cw.astype(np.float32)
        logger.info(f"Class weights computed (no balancing): {dict(zip(unique_classes, self._class_weights))}")
        gc.collect()

        # Step 6: Split data
        self.splits = self.splitter.train_val_test_split(self.X, self.y)

        # Step 7: Generate CV folds
        self.cv_folds = self.splitter.get_cv_folds(
            self.splits['X_train'],
            self.splits['y_train']
        )

        # Prepare results
        results = {
            'dataset_name': self.dataset_name,
            'mode': self.mode,
            'X': self.X,
            'y': self.y,
            'splits': self.splits,
            'cv_folds': self.cv_folds,
            'feature_columns': self.feature_columns,
            'label_mapping': self.label_mapping,
            'n_features': len(self.feature_columns),
            'n_samples': len(self.y),
            'n_classes': len(self.label_mapping),
            'class_distribution': self.balancer.get_class_distribution(self.y),
            'class_distribution_before': self._class_dist_before,
            'class_weights': self._class_weights,
            'scaler': self.preprocessor.scaler,
            'label_encoder': self.preprocessor.label_encoder
        }

        # Save preprocessing artifacts
        self._save_artifacts(results)

        logger.info("=" * 60)
        logger.info(f"STAGE 1 COMPLETE ({self.mode.upper()} MODE)")
        logger.info(f"  Samples: {results['n_samples']}")
        logger.info(f"  Features: {results['n_features']}")
        logger.info(f"  Classes: {results['n_classes']}")
        logger.info("=" * 60)

        return results

    # ── Cache save / load ──────────────────────────────────────────

    def save_cache(self, results: Dict):
        """Persist Stage 1 results to disk (compressed npz + JSON metadata)."""
        from config import CACHE_CONFIG
        cache_dir = get_results_dir(self.mode) / CACHE_CONFIG.get('cache_dir_name', 'stage1_cache')
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving Stage 1 cache → {cache_dir}")

        # Save arrays (compressed)
        np.savez_compressed(cache_dir / 'arrays.npz',
                            X=results['X'], y=results['y'],
                            X_train=results['splits']['X_train'],
                            y_train=results['splits']['y_train'],
                            X_val=results['splits']['X_val'],
                            y_val=results['splits']['y_val'],
                            X_test=results['splits']['X_test'],
                            y_test=results['splits']['y_test'],
                            class_weights=results.get('class_weights', np.array([])))

        # Save metadata (JSON-safe)
        def _to_native(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): _to_native(v) for k, v in obj.items()}
            return obj

        meta = {
            'dataset_name': results['dataset_name'],
            'mode': results['mode'],
            'feature_columns': results['feature_columns'],
            'label_mapping': _to_native(results['label_mapping']),
            'n_features': int(results['n_features']),
            'n_samples': int(results['n_samples']),
            'n_classes': int(results['n_classes']),
            'class_distribution': _to_native(results.get('class_distribution', {})),
            'class_distribution_before': _to_native(results.get('class_distribution_before', {})),
        }
        with open(cache_dir / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2, default=str)

        # Save scaler + label encoder
        if results.get('scaler') is not None:
            with open(cache_dir / 'scaler.pkl', 'wb') as f:
                pickle.dump(results['scaler'], f)
        if results.get('label_encoder') is not None:
            with open(cache_dir / 'label_encoder.pkl', 'wb') as f:
                pickle.dump(results['label_encoder'], f)

        logger.info(f"Cache saved ({(cache_dir / 'arrays.npz').stat().st_size / 1e6:.0f} MB)")

    @staticmethod
    def load_cache(mode: str):
        """Load Stage 1 results from disk cache. Returns dict or None if not found."""
        from config import CACHE_CONFIG
        cache_dir = get_results_dir(mode) / CACHE_CONFIG.get('cache_dir_name', 'stage1_cache')
        arrays_path = cache_dir / 'arrays.npz'
        meta_path = cache_dir / 'metadata.json'

        if not arrays_path.exists() or not meta_path.exists():
            return None

        logger.info(f"Loading Stage 1 from cache → {cache_dir}")

        import gc
        gc.collect()  # Free any lingering memory before big load

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Load arrays — load one at a time with gc between to control RAM peak
        data = np.load(arrays_path, allow_pickle=False, mmap_mode='r')

        X_train = np.array(data['X_train']); gc.collect()
        y_train = np.array(data['y_train']); gc.collect()
        X_val = np.array(data['X_val'])
        y_val = np.array(data['y_val'])
        X_test = np.array(data['X_test'])
        y_test = np.array(data['y_test'])
        class_weights = np.array(data['class_weights'])

        # Skip loading full X/y — reconstruct lazily only if needed (saves ~5 GB)
        # Stages use splits directly; ablation/analysis can reconstruct from splits
        results = {
            'dataset_name': meta['dataset_name'],
            'mode': meta['mode'],
            'X': None,  # Lazy — use splits instead
            'y': None,  # Lazy — use splits instead
            'splits': {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test,
            },
            'cv_folds': None,
            'feature_columns': meta['feature_columns'],
            'label_mapping': meta['label_mapping'],
            'n_features': meta['n_features'],
            'n_samples': meta['n_samples'],
            'n_classes': meta['n_classes'],
            'class_distribution': meta.get('class_distribution', {}),
            'class_distribution_before': meta.get('class_distribution_before', {}),
            'class_weights': class_weights if class_weights.size > 0 else None,
            'scaler': None,
            'label_encoder': None,
        }
        del data  # Close mmap file handle
        gc.collect()

        # Restore scaler / label encoder
        scaler_path = cache_dir / 'scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                results['scaler'] = pickle.load(f)
        le_path = cache_dir / 'label_encoder.pkl'
        if le_path.exists():
            with open(le_path, 'rb') as f:
                results['label_encoder'] = pickle.load(f)

        logger.info(f"Cache loaded: {results['n_samples']:,} samples, {results['n_features']} features")
        return results

    def _save_artifacts(self, results: Dict):
        """Save preprocessing artifacts to mode-specific directory."""
        artifacts_dir = get_results_dir(self.mode) / 'stage1_artifacts'
        artifacts_dir.mkdir(exist_ok=True)

        # Save scaler
        if results['scaler'] is not None:
            with open(artifacts_dir / f"{self.dataset_name}_scaler.pkl", 'wb') as f:
                pickle.dump(results['scaler'], f)

        # Save label encoder
        if results['label_encoder'] is not None:
            with open(artifacts_dir / f"{self.dataset_name}_label_encoder.pkl", 'wb') as f:
                pickle.dump(results['label_encoder'], f)

        # Save metadata
        def convert_to_native(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {str(k): convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            return obj

        metadata = {
            'dataset_name': results['dataset_name'],
            'mode': results['mode'],
            'n_features': int(results['n_features']),
            'n_samples': int(results['n_samples']),
            'n_classes': int(results['n_classes']),
            'feature_columns': results['feature_columns'],
            'label_mapping': {str(k): int(v) for k, v in results['label_mapping'].items()},
            'class_distribution': {str(k): int(v) for k, v in results['class_distribution']['counts'].items()}
        }

        with open(artifacts_dir / f"{self.dataset_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Artifacts saved to {artifacts_dir}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("XAI-Based IDS Benchmarking Study")
    print("STAGE 1: Dataset Preparation & Baseline")
    print("="*70 + "\n")

    for mode in ["binary", "multiclass"]:
        print(f"\n{'#'*70}")
        print(f"# MODE: {mode.upper()}")
        print(f"{'#'*70}\n")

        pipeline = Stage1Pipeline("CIC_IoT_DIAD_2024", mode=mode)
        results = pipeline.run(
            balance_classes=True,
            balance_method='hybrid',
            engineer_features=True
        )

        print(f"\n{'-'*50}")
        print(f"SUMMARY ({mode.upper()})")
        print("-"*50)
        print(f"Dataset: {results['dataset_name']}")
        print(f"Mode: {results['mode']}")
        print(f"Total samples: {results['n_samples']}")
        print(f"Number of features: {results['n_features']}")
        print(f"Number of classes: {results['n_classes']}")
        print(f"Training samples: {len(results['splits']['y_train'])}")
        print(f"Validation samples: {len(results['splits']['y_val'])}")
        print(f"Test samples: {len(results['splits']['y_test'])}")
        print(f"CV Folds: {len(results['cv_folds'])}")
        print("-"*50)
