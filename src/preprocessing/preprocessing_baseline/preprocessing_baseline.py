import gc
import json
import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Set up logger
logger = logging.getLogger("PULSE_logger")


class PreprocessorBaseline:
    """
    A modular class for preprocessing ICU dataset for convML/convDL models.

    This class encapsulates the preprocessing pipeline for ICU data, including:
    - Outlier removal
    - NA flagging
    - Data splitting (train/val/test)
    - Feature standardization
    - Missing value imputation
    - Data reshaping for mortality prediction

    Attributes:
        base_path (str): Base path for data loading and saving
        task (str): Current task (e.g., 'mortality', 'aki', 'sepsis')
        dataset_name (str): Current dataset name (e.g., 'hirid', 'miiv', 'eicu')
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        base_path: str,
        random_seed: int = 42,
        config: Dict[str, Any] = None,
        original_base_path: str = None,
    ):
        """
        Initialize the PreprocessorBaseline with configuration parameters.

        Args:
            base_path (str): Base path for data loading and saving
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            config (Dict[str, Any], optional): Configuration options. Defaults to None.
        """
        self.base_path = base_path
        self.random_seed = random_seed
        self.task = None
        self.dataset_name = None

        # Store original base path if provided (for permanent storage)
        self.original_base_path = original_base_path

        # Set default configuration
        self.config = {
            "replace_outliers": True,
            "flag_na": True,
            "standardize": True,
            "static_imputation": True,
            "dynamic_imputation": True,
            "save_data": True,
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        }

        # Update with user config if provided
        if config:
            self.config.update(config)

        # Log the active configuration
        logger.info(
            "PreprocessorBaseline initialized with configuration: %s", self.config
        )

        # Dictionary of feature limits for outlier detection
        self.features_dict = {
            "hr": ("Heart Rate [bpm]", (60, 100), (20, 320)),
            "sbp": ("Systolic Blood Pressure\n[mmHg]", (90, 120), (30, 300)),
            "dbp": ("Diastolic Blood Pressure\n[mmHg]", (60, 80), (10, 200)),
            "map": ("Mean Arterial Pressure (MAP) [mmHg]", (65, 100), (20, 250)),
            "o2sat": ("Oxygen Saturation [%]", (95, 100), (50, 100)),
            "resp": ("Respiratory Rate\n[breaths/min]", (12, 20), (4, 80)),
            "temp": ("Temperature [°C]", (36.5, 37.5), (30, 42)),
            "ph": ("pH Level [-]", (7.35, 7.45), (6.7, 8.0)),
            "po2": ("Partial Pressure of\nOxygen (PaO2) [mmHg]", (75, 100), (40, 600)),
            "pco2": (
                "Partial Pressure of\nCarbon Dioxide (PaCO2) [mmHg]",
                (35, 45),
                (10, 150),
            ),
            "be": ("Base Excess [mmol/L]", (-2, 2), (-25, 25)),
            "bicar": ("Bicarbonate [mmol/L]", (22, 29), (5, 50)),
            "fio2": ("Fraction of Inspired Oxygen\n(FiO2) [%]", (21, 100), (21, 100)),
            "inr_pt": (
                "International Normalized Ratio\n(INR) [-]",
                (0.8, 1.2),
                (0.5, 20),
            ),
            "ptt": ("Partial Thromboplastin Time\n(PTT) [sec]", (25, 35), (10, 250)),
            "fgn": ("Fibrinogen [mg/dL]", (200, 400), (30, 1100)),
            "na": ("Sodium [mmol/L]", (135, 145), (90, 170)),
            "k": ("Potassium [mmol/L]", (3.5, 5), (1, 9)),
            "cl": ("Chloride [mmol/L]", (96, 106), (70, 140)),
            "ca": ("Calcium [mg/dL]", (8.5, 10.5), (4, 20)),
            "cai": ("Ionized Calcium [mmol/L]", (1.1, 1.3), (0.4, 2.2)),
            "mg": ("Magnesium [mg/dL]", (1.7, 2.2), (0.5, 5)),
            "phos": ("Phosphate [mg/dL]", (2.5, 4.5), (0.5, 15)),
            "glu": ("Glucose [mg/dL]", (70, 140), (25, 1000)),
            "lact": ("Lactate [mmol/L]", (0.5, 2), (0.1, 20)),
            "alb": ("Albumin [g/dL]", (3.5, 5), (0.5, 6)),
            "alp": ("Alkaline Phosphatase [U/L]", (44, 147), (10, 1200)),
            "alt": ("Alanine Aminotransferase\n(ALT) [U/L]", (7, 56), (10, 5000)),
            "ast": ("Aspartate Aminotransferase\n(AST) [U/L]", (10, 40), (10, 8000)),
            "bili": ("Total Bilirubin [mg/dL]", (0.1, 1.2), (0.1, 50)),
            "bili_dir": ("Direct Bilirubin [mg/dL]", (0, 0.3), (0, 30)),
            "bun": ("Blood Urea Nitrogen\n(BUN) [mg/dL]", (7, 20), (1, 180)),
            "crea": ("Creatinine [mg/dL]", (0.6, 1.3), (0.1, 20)),
            "urine": ("Urine Output [mL/h]", (30, 50), (0, 2000)),
            "hgb": ("Hemoglobin [g/dL]", (12.5, 17.5), (3, 20)),
            "mch": ("Mean Corpuscular\nHemoglobin (MCH) [pg]", (27, 33), (15, 45)),
            "mchc": (
                "Mean Corpuscular Hemoglobin\nConcentration (MCHC) [g/dL]",
                (32, 36),
                (20, 45),
            ),
            "mcv": ("Mean Corpuscular\nVolume (MCV) [fL]", (80, 100), (50, 130)),
            "plt": ("Platelets [1000/µL]", (150, 450), (10, 1500)),
            "wbc": ("White Blood Cell Count\n(WBC) [1000/µL]", (4, 11), (0.1, 500)),
            "neut": ("Neutrophils [%]", (55, 70), (0, 100)),
            "bnd": ("Band Neutrophils [%]", (0, 6), (0, 50)),
            "lymph": ("Lymphocytes [%]", (20, 40), (0, 90)),
            "crp": ("C-Reactive Protein\n(CRP) [mg/L]", (0, 10), (0, 500)),
            "methb": ("Methemoglobin [%]", (0, 2), (0, 60)),
            "ck": ("Creatine Kinase\n(CK) [U/L]", (30, 200), (10, 100000)),
            "ckmb": ("Creatine Kinase-MB\n(CK-MB) [ng/mL]", (0, 5), (0, 500)),
            "tnt": ("Troponin T [ng/mL]", (0, 14), (0, 1000)),
            "height": ("Height [cm]", (), (135, 220)),
            "weight": ("Weight [kg]", (), (40, 250)),
        }

        # Lists of column types for processing
        self.dynamic_columns = [
            "alb",
            "alp",
            "alt",
            "ast",
            "be",
            "bicar",
            "bili",
            "bili_dir",
            "bnd",
            "bun",
            "ca",
            "cai",
            "ck",
            "ckmb",
            "cl",
            "crea",
            "crp",
            "dbp",
            "fgn",
            "fio2",
            "glu",
            "hgb",
            "hr",
            "inr_pt",
            "k",
            "lact",
            "lymph",
            "map",
            "mch",
            "mchc",
            "mcv",
            "methb",
            "mg",
            "na",
            "neut",
            "o2sat",
            "pco2",
            "ph",
            "phos",
            "plt",
            "po2",
            "ptt",
            "resp",
            "sbp",
            "temp",
            "tnt",
            "urine",
            "wbc",
        ]
        self.static_columns = ["age", "sex", "weight", "height"]
        self.static_numeric_columns = ["age", "weight", "height"]

    def load_data(
        self, task: str, dataset_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load parquet files for a dataset and task.

        Args:
            task (str): Task name ('mortality', 'aki', 'sepsis')
            dataset_name (str): Dataset name ('hirid', 'miiv', 'eicu')

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple of dynamic, outcome, and static dataframes

        Raises:
            Exception: If there's an error reading the data files
        """
        full_path = os.path.join(
            self.base_path, f"datasets/original_harmonized/{task}/{dataset_name}"
        )
        try:
            dyn_df = pd.read_parquet(f"{full_path}/dyn.parquet", engine="pyarrow")
            outc_df = pd.read_parquet(f"{full_path}/outc.parquet", engine="pyarrow")
            sta_df = pd.read_parquet(f"{full_path}/sta.parquet", engine="pyarrow")

            return dyn_df, outc_df, sta_df

        except Exception as e:
            logger.error("Error reading original from %s/%s: %s", task, dataset_name, e)
            raise

    def replace_outliers_with_nans(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Replace outlier values with NaNs based on predefined feature limits.

        Args:
            df (pd.DataFrame): DataFrame with features to be checked for outliers

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - DataFrame with outliers replaced by NaN
                - DataFrame with outlier counts per feature
        """
        outlier_counts = pd.DataFrame(columns=["Lower Outliers", "Upper Outliers"])

        for col, (label, _, (lower, upper)) in self.features_dict.items():
            if col in df.columns:
                margin = 0.05 * (upper - lower)
                new_lower, new_upper = lower - margin, upper + margin

                lower_outliers = (df[col] < new_lower).sum()
                upper_outliers = (df[col] > new_upper).sum()

                outlier_counts.loc[col] = [lower_outliers, upper_outliers]
                df.loc[~df[col].between(new_lower, new_upper), col] = np.nan

        return df, outlier_counts

    def flag_na_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Flag NA values in a dataframe with new columns.

        Args:
            X (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with additional binary columns indicating NA values
        """
        na_flags = pd.DataFrame()

        for col in X.columns:
            if col not in ["stay_id", "time", "age", "sex", "height", "weight"]:
                na_flags[col + "_na"] = X[col].isna().astype(int)

        X = pd.concat([X, na_flags], axis=1)
        return X

    def split_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Split data into training, validation, and testing sets based on temporal order of stay_ids.
        The first a% of stay_ids are assigned to the training set, the next b% to validation,
        and the remaining c% to the test set. This creates a temporal split suitable for ICU data.

        Args:
            X (pd.DataFrame): Features DataFrame
            y (pd.DataFrame): Labels DataFrame

        Returns:
            Tuple containing:
                X_train (pd.DataFrame): Training features
                X_val (pd.DataFrame): Validation features
                X_test (pd.DataFrame): Testing features
                y_train (pd.DataFrame): Training labels
                y_val (pd.DataFrame): Validation labels
                y_test (pd.DataFrame): Testing labels
        """
        # Get split ratios from config
        train_ratio = self.config["split_ratios"]["train"]
        val_ratio = self.config["split_ratios"]["val"]
        test_ratio = self.config["split_ratios"]["test"]

        # Validate ratios sum to 1
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.001:  # Allow small floating point errors
            logger.warning(
                "Split ratios don't sum to 1.0: %s. Normalizing...", total_ratio
            )
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        # Get unique stay_ids in sorted order (temporal ordering)
        unique_stay_ids = sorted(X["stay_id"].unique())
        total_stays = len(unique_stay_ids)

        # Calculate cutoff indices
        train_cutoff = int(total_stays * train_ratio)
        val_cutoff = train_cutoff + int(total_stays * val_ratio)

        # Split stay_ids into train, val, and test sets
        train_stay_ids = unique_stay_ids[:train_cutoff]
        val_stay_ids = unique_stay_ids[train_cutoff:val_cutoff]
        test_stay_ids = unique_stay_ids[val_cutoff:]

        # Create masks for each set
        train_mask = X["stay_id"].isin(train_stay_ids)
        val_mask = X["stay_id"].isin(val_stay_ids)
        test_mask = X["stay_id"].isin(test_stay_ids)

        # Split X and y data using masks
        X_train = X[train_mask]
        y_train = y[train_mask]

        X_val = X[val_mask]
        y_val = y[val_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        # Log the split information
        total_stays = X["stay_id"].nunique()

        logger.info(
            "  Training set: %s stays (%s), %s rows (%s)",
            X_train["stay_id"].nunique(),
            f"{X_train['stay_id'].nunique()/total_stays:.1%}",
            len(X_train),
            f"{len(X_train)/len(X):.1%}",
        )
        logger.info(
            "  Validation set: %s stays (%s), %s rows (%s)",
            X_val["stay_id"].nunique(),
            f"{X_val['stay_id'].nunique()/total_stays:.1%}",
            len(X_val),
            f"{len(X_val)/len(X):.1%}",
        )
        logger.info(
            "  Testing set: %s stays (%s), %s rows (%s)",
            X_test["stay_id"].nunique(),
            f"{X_test['stay_id'].nunique()/total_stays:.1%}",
            len(X_test),
            f"{len(X_test)/len(X):.1%}",
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def standardize_all_sets(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Standardize features for train, validation, and test sets using StandardScaler.
        Fits on training data only, then transforms all sets.

        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            X_test (pd.DataFrame): Testing features

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Standardized X_train, X_val, X_test
        """
        # Identify columns not for standardization
        columns_not_for_standardization = [
            col for col in X_train.columns if "_na" in col
        ] + ["stay_id", "time", "sex"]

        # Identify columns for standardization
        available_dynamic_columns = [
            col for col in self.dynamic_columns if col in X_train.columns
        ]
        columns_for_standardization = (
            available_dynamic_columns + self.static_numeric_columns
        )
        columns_for_standardization = [
            col
            for col in columns_for_standardization
            if col not in columns_not_for_standardization
        ]

        # Fit scaler on training data only
        scaler = StandardScaler()
        X_train.loc[:, columns_for_standardization] = scaler.fit_transform(
            X_train.loc[:, columns_for_standardization]
        )

        # Save scaler parameters for all standardized columns (static and dynamic)
        scaler_params = {}
        for i, col in enumerate(columns_for_standardization):
            scaler_params[col] = {"mean": scaler.mean_[i], "std": scaler.scale_[i]}

        if self.config["save_data"]:
            self.save_scaler_parameters(scaler_params)

        # Transform validation and test sets with the same scaler
        X_val.loc[:, columns_for_standardization] = scaler.transform(
            X_val.loc[:, columns_for_standardization]
        )
        X_test.loc[:, columns_for_standardization] = scaler.transform(
            X_test.loc[:, columns_for_standardization]
        )

        # Map sex categories consistently across all datasets
        X_train["sex"] = X_train["sex"].map({"Female": 1, "Male": -1})
        X_val["sex"] = X_val["sex"].map({"Female": 1, "Male": -1})
        X_test["sex"] = X_test["sex"].map({"Female": 1, "Male": -1})

        return X_train, X_val, X_test

    def save_scaler_parameters(self, scaler_params: dict) -> None:
        """Save scaler parameters for future use in destandardization."""
        config_dirname = self.generate_preprocessing_dirname()

        # Determine the base path to use (original_base_path if available, otherwise base_path)
        target_base_path = (
            self.original_base_path
            if hasattr(self, "original_base_path") and self.original_base_path
            else self.base_path
        )

        # Save to notebooks/scaler_parameters directory
        directory = os.path.join(target_base_path, "notebooks/scaler_parameters")
        os.makedirs(directory, exist_ok=True)

        scaler_file = os.path.join(
            directory,
            f"scaler_parameters_{self.task}_{self.dataset_name}_{config_dirname}.json",
        )
        with open(scaler_file, "w") as f:
            json.dump(scaler_params, f, indent=2)

        logger.info(f"Saved scaler parameters to {scaler_file}")

    def impute_static_all_sets(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
        """
        Impute missing values in static features for train, validation, and test sets.
        Uses statistics from training data for imputation across all sets.

        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            X_test (pd.DataFrame): Testing features

        Returns:
            Tuple containing:
                X_train (pd.DataFrame): Training features with imputed static values
                X_val (pd.DataFrame): Validation features with imputed static values
                X_test (pd.DataFrame): Testing features with imputed static values
                global_means_static_dict (Dict): Dictionary of global means used for imputation
        """
        global_means_static_dict = {}

        # Forward and backward fill within each stay_id for all sets
        for col in self.static_columns:
            if col in X_train.columns:
                for df in [X_train, X_val, X_test]:
                    df[col] = df.groupby("stay_id")[col].transform(
                        lambda x: x.ffill().bfill()
                    )

        # Calculate medians from training data only for numeric columns
        for col in self.static_numeric_columns:
            if col in X_train.columns:
                global_mean = X_train.groupby("stay_id")[col].mean().mean().round(2)
                for df in [X_train, X_val, X_test]:
                    df[col] = df[col].fillna(global_mean)
                global_means_static_dict[col] = global_mean

        # Debug log for global medians
        logger.debug(
            "Global means used as imputation fallback for static features (if forward-/backward fill fails): %s",
            global_means_static_dict,
        )

        # Calculate mode from training data only for sex
        if "sex" in X_train.columns:
            mode_sex = X_train["sex"].mode()[0]
            for df in [X_train, X_val, X_test]:
                df["sex"] = df["sex"].fillna(mode_sex)

        return X_train, X_val, X_test, global_means_static_dict

    def median_before_imputation(self, X_train: pd.DataFrame) -> pd.Series:
        """
        Calculate global medians for dynamic features before imputation.

        Args:
            X_train (pd.DataFrame): Training features

        Returns:
            pd.Series: Series of global median for dynamic features
        """
        exclude_columns = [col for col in X_train.columns if "_na" in col] + [
            "time",
            "sex",
            "age",
            "height",
            "weight",
            "stay_id",
        ]
        global_median_train_series = (
            X_train.drop(columns=exclude_columns, errors="ignore").median().round(2)
        )

        # Debug log for global medians
        logger.debug(
            "Global medians used as imputation fallback for dynamic features: %s",
            global_median_train_series.to_dict(),
        )

        return global_median_train_series

    def dynamic_imputation(
        self, df: pd.DataFrame, global_median_train_series: pd.Series
    ) -> pd.DataFrame:
        """
        Impute missing values in dynamic features using forward fill and global medians.

        Args:
            df (pd.DataFrame): DataFrame with missing values
            global_median_train_series (pd.Series): Series of global medians for imputation

        Returns:
            pd.DataFrame: DataFrame with imputed values
        """
        df = df.groupby("stay_id").apply(lambda group: group.ffill(axis=0))
        df = df.reset_index(drop=True)
        df = df.fillna(global_median_train_series)
        return df

    def reshape_mortality_data(
        self, X: pd.DataFrame, y: pd.DataFrame, set_name: str = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reshape data for mortality task by combining all rows for each stay_id into a single row.
        Static features are kept once, while dynamic features are grouped by feature with consecutive timesteps (ascending suffixes).

        Args:
            X (pd.DataFrame): Features dataframe with stay_id
            y (pd.DataFrame): Labels dataframe with stay_id
            set_name (str, optional): Name of the dataset (train, val, test) for logging

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Reshaped X and y
        """
        # Handle static features first - take first value for each stay_id
        static_features = self.static_columns.copy()

        if "stay_id" not in static_features:
            static_features.append("stay_id")

        # Get static data using group by with first() - much faster
        static_data = X[static_features].groupby("stay_id").first().reset_index()

        # For dynamic features, use pivot_table
        dynamic_features = [col for col in X.columns if col not in static_features]

        # Create row index within each stay_id
        X = X.copy()
        X["time_idx"] = X.groupby("stay_id").cumcount() + 1

        # Use pivot_table to reshape - vastly more efficient
        dynamic_data_list = []

        for feature in dynamic_features:
            pivot_df = X.pivot_table(
                index="stay_id", columns="time_idx", values=feature, aggfunc="first"
            )
            # Rename columns to feature_1, feature_2, etc.
            pivot_df.columns = [f"{feature}_{i}" for i in pivot_df.columns]
            dynamic_data_list.append(pivot_df)

        # Combine all dynamic feature pivots
        if dynamic_data_list:
            dynamic_data = pd.concat(dynamic_data_list, axis=1)
            # Join with static data
            X_reshaped = static_data.merge(
                dynamic_data.reset_index(), on="stay_id", how="left"
            )
        else:
            X_reshaped = static_data

        # For y, take the label of the last row for each stay_id
        y_reshaped = y.groupby("stay_id")["label"].last().reset_index()

        return X_reshaped, y_reshaped

    def calculate_dataset_statistics(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        set_type: str,
        task: str,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """
        Calculate statistics for the dataset.

        Args:
            X (pd.DataFrame): Features dataframe
            y (pd.DataFrame): Labels dataframe
            set_type (str, optional): Type of dataset ('train', 'validation', 'test'). Defaults to "train".
            task (str): Task name (e.g., 'mortality', 'aki', 'sepsis')
            dataset_name (str): Dataset name (e.g., 'hirid', 'miiv', 'eicu')

        Returns:
            Dict[str, Any]: Dictionary of dataset statistics
        """
        try:
            # For mortality task (after reshaping), use simplified statistics
            if task == "mortality":
                # Count total stays (each row is now a stay)
                total_stays = len(X)

                # Count cases (stays with positive label)
                positive_stays = y["label"].sum()
                negative_stays = total_stays - positive_stays

                # Format the statistics in a dictionary
                stats = {
                    "Task": task,
                    "Dataset": dataset_name,
                    "Set": set_type,
                    "Total Stays": total_stays,
                    "Cases": positive_stays,
                    "Cases %": f"{(positive_stays/total_stays)*100:.1f}%",
                    "Controls": negative_stays,
                    "Controls %": f"{(negative_stays/total_stays)*100:.1f}%",
                    "Total Rows": total_stays,  # Same as total_stays after reshaping
                    "Rows of Cases": positive_stays,  # Same as positive_stays after reshaping
                    "Rows of Cases %": f"{(positive_stays/total_stays)*100:.1f}%",
                    "Rows of Controls": negative_stays,  # Same as negative_stays after reshaping
                    "Rows of Controls %": f"{(negative_stays/total_stays)*100:.1f}%",
                    "Positive Labels": positive_stays,  # Same as positive_stays after reshaping
                    "Positive Labels %": f"{(positive_stays/total_stays)*100:.1f}%",
                    "Negative Labels": negative_stays,  # Same as negative_stays after reshaping
                    "Negative Labels %": f"{(negative_stays/total_stays)*100:.1f}%",
                }

                return stats

            # For other tasks, use the original statistics calculation
            else:
                # Add row index for each stay_id to facilitate proper merging
                X_train_indexed = X.copy()
                y_train_indexed = y.copy()
                X_train_indexed["row_idx"] = X_train_indexed.groupby(
                    "stay_id"
                ).cumcount()
                y_train_indexed["row_idx"] = y_train_indexed.groupby(
                    "stay_id"
                ).cumcount()

                # Merge X and y temporarily for analysis
                merged_full = pd.merge(
                    X_train_indexed, y_train_indexed, on=["stay_id", "row_idx"]
                )

                # Count total rows
                total_rows = len(merged_full)

                # Analyze by stay_id
                # Assuming 'label' is 1/True for positive cases and 0/False for negative cases across all tasks
                stay_id_has_positive = merged_full.groupby("stay_id")["label"].any()

                # Count of stay_ids with and without positive labels
                positive_stays = stay_id_has_positive.sum()
                negative_stays = len(stay_id_has_positive) - positive_stays
                total_stays = len(stay_id_has_positive)

                # Create a mapping of stay_id to whether it has a positive label
                stay_id_positive_map = stay_id_has_positive.to_dict()

                # Add a column to merged_full indicating if the stay_id has a positive label anywhere
                merged_full["stay_has_positive"] = merged_full["stay_id"].map(
                    stay_id_positive_map
                )

                # Calculate row counts for each category using new naming conventions
                rows_of_controls = len(merged_full[~merged_full["stay_has_positive"]])
                rows_of_cases = len(merged_full[merged_full["stay_has_positive"]])
                negative_labels = len(merged_full[merged_full["label"] == False])
                positive_labels = len(merged_full[merged_full["label"] == True])

                # Format the statistics in a dictionary for further use
                stats = {
                    "Task": task,
                    "Dataset": dataset_name,
                    "Set": set_type,
                    "Total Stays": total_stays,
                    "Cases": positive_stays,
                    "Cases %": f"{(positive_stays/total_stays)*100:.1f}%",
                    "Controls": negative_stays,
                    "Controls %": f"{(negative_stays/total_stays)*100:.1f}%",
                    "Total Rows": total_rows,
                    "Rows of Cases": rows_of_cases,
                    "Rows of Cases %": f"{(rows_of_cases/total_rows)*100:.1f}%",
                    "Rows of Controls": rows_of_controls,
                    "Rows of Controls %": f"{(rows_of_controls/total_rows)*100:.1f}%",
                    "Positive Labels": positive_labels,
                    "Positive Labels %": f"{(positive_labels/total_rows)*100:.1f}%",
                    "Negative Labels": negative_labels,
                    "Negative Labels %": f"{(negative_labels/total_rows)*100:.1f}%",
                }

                return stats

        finally:
            # Clean up intermediate objects
            if (
                self.task != "mortality"
            ):  # Skip cleanup for mortality as these objects won't exist
                if "X_train_indexed" in locals():
                    del X_train_indexed
                if "y_train_indexed" in locals():
                    del y_train_indexed
                if "merged_full" in locals():
                    del merged_full
                if "stay_id_has_positive" in locals():
                    del stay_id_has_positive
                if "stay_id_positive_map" in locals():
                    del stay_id_positive_map
                gc.collect()

    def save_preprocessed_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_val: pd.DataFrame,
        y_test: pd.DataFrame,
    ) -> None:
        """
        Save processed data to parquet files, both in scratch and permanent storage if applicable.

        Args:
            X_train (pd.DataFrame): Training features
            X_val (pd.DataFrame): Validation features
            X_test (pd.DataFrame): Testing features
            y_train (pd.DataFrame): Training labels
            y_val (pd.DataFrame): Validation labels
            y_test (pd.DataFrame): Testing labels
        """
        # Generate directory name based on preprocessing configuration
        config_dirname = self.generate_preprocessing_dirname()

        # Construct directory path with task subfolder and config-based directory name
        directory = os.path.join(
            self.base_path,
            f"datasets/preprocessed_splits/{self.task}/{self.dataset_name}/{config_dirname}",
        )

        os.makedirs(directory, exist_ok=True)

        # Save files
        X_train.to_parquet(os.path.join(directory, "X_train.parquet"))
        y_train.to_parquet(os.path.join(directory, "y_train.parquet"))
        X_val.to_parquet(os.path.join(directory, "X_val.parquet"))
        y_val.to_parquet(os.path.join(directory, "y_val.parquet"))
        X_test.to_parquet(os.path.join(directory, "X_test.parquet"))
        y_test.to_parquet(os.path.join(directory, "y_test.parquet"))

        logger.info("Data saved to %s", directory)

        # Check if we're running on scratch (if original_base_path attribute exists)
        # Save to permanent storage as well
        if hasattr(self, "original_base_path"):
            permanent_directory = os.path.join(
                self.original_base_path,
                f"datasets/preprocessed_splits/{self.task}/{self.dataset_name}/{config_dirname}",
            )

            # Create permanent directory and save
            os.makedirs(permanent_directory, exist_ok=True)
            X_train.to_parquet(os.path.join(permanent_directory, "X_train.parquet"))
            y_train.to_parquet(os.path.join(permanent_directory, "y_train.parquet"))
            X_val.to_parquet(os.path.join(permanent_directory, "X_val.parquet"))
            y_val.to_parquet(os.path.join(permanent_directory, "y_val.parquet"))
            X_test.to_parquet(os.path.join(permanent_directory, "X_test.parquet"))
            y_test.to_parquet(os.path.join(permanent_directory, "y_test.parquet"))

            logger.info("Data also saved to permanent storage: %s", permanent_directory)

    def preprocess(self, task: str, dataset_name: str, save_data: bool = None) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Run the full preprocessing pipeline for a given task and dataset.

        Args:
            task (str): Task name ('mortality', 'aki', 'sepsis')
            dataset_name (str): Dataset name ('hirid', 'miiv', 'eicu')
            save_data (bool, optional): Whether to save the preprocessed data. Overrides config if provided.

        Returns:
            Tuple containing:
                X_train (pd.DataFrame): Training features
                X_val (pd.DataFrame): Validation features
                X_test (pd.DataFrame): Testing features
                y_train (pd.DataFrame): Training labels
                y_val (pd.DataFrame): Validation labels
                y_test (pd.DataFrame): Testing labels

        Notes:
            The following preprocessing steps are applied conditionally based on config:
            - Outlier removal ('replace_outliers')
            - NaN flagging ('flag_na')
            - Standardization ('standardize')
            - Static imputation ('static_imputation')
            - Dynamic imputation ('dynamic_imputation')

            For mortality task, reshape_mortality_data is always applied.

        Raises:
            Exception: If there's an error during preprocessing
        """
        try:
            # Store task and dataset name as instance variables
            self.task = task
            self.dataset_name = dataset_name

            # Determine whether to save data - parameter overrides config
            if save_data is None:
                save_data = self.config["save_data"]

            logger.info("%s", "=" * 40)
            logger.info("Processing Task: %s, Dataset: %s", task, dataset_name)
            logger.info("Active preprocessing options: %s", self.config)
            logger.info("%s", "=" * 40)

            # Load parquet files
            logger.info("Loading parquet files (original harmonized datasets)...")
            dyn_df, outc_df, sta_df = self.load_data(task, dataset_name)

            # Merge sta/dyn/outc dataframes
            logger.info("Merging dataframes...")
            dyn_sta = pd.merge(sta_df, dyn_df, on="stay_id")

            if task == "mortality":
                # For mortality task, merge only on stay_id (since outc has one row per stay_id)
                merged_data = pd.merge(dyn_sta, outc_df, on=["stay_id"], how="left")
            else:
                # For continuous classification tasks like sepsis and AKI, merge on both stay_id and time
                merged_data = pd.merge(
                    dyn_sta, outc_df, on=["stay_id", "time"], how="left"
                )

            X = merged_data.drop(columns=["label"])
            y = merged_data[["stay_id", "label"]]

            if "time" in X.columns:
                X = X.drop(columns=["time"])

            del dyn_df, sta_df, outc_df, dyn_sta, merged_data
            gc.collect()

            # Remove outliers - conditional
            if self.config["replace_outliers"]:
                logger.info("Removing outliers...")
                X_cleaned, outlier_counts = self.replace_outliers_with_nans(X)
            else:
                logger.info("Skipping outlier removal (disabled in config)")
                X_cleaned = X
            del X
            gc.collect()

            # Flag NaNs - conditional
            if self.config["flag_na"]:
                logger.info("Flagging NaN values...")
                X_flagged = self.flag_na_all(X_cleaned)
            else:
                logger.info("Skipping NaN flagging (disabled in config)")
                X_flagged = X_cleaned
            del X_cleaned
            gc.collect()

            # Split data with random seed
            logger.info(
                "Splitting data temporally into train, validation, and test sets..."
            )
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
                X_flagged, y
            )
            del X_flagged
            gc.collect()

            # Standardization - conditional
            if self.config["standardize"]:
                logger.info("Standardizing data...")
                X_train_standardized, X_val_standardized, X_test_standardized = (
                    self.standardize_all_sets(X_train, X_val, X_test)
                )
            else:
                logger.info("Skipping standardization (disabled in config)")
                X_train_standardized, X_val_standardized, X_test_standardized = (
                    X_train,
                    X_val,
                    X_test,
                )
            del X_train, X_val, X_test
            gc.collect()

            # Static imputation - conditional
            if self.config["static_imputation"]:
                logger.info("Performing static imputation...")
                (
                    X_train_sta_imputed,
                    X_val_sta_imputed,
                    X_test_sta_imputed,
                    global_means_sta_dict,
                ) = self.impute_static_all_sets(
                    X_train_standardized, X_val_standardized, X_test_standardized
                )
            else:
                logger.info("Skipping static imputation (disabled in config)")
                X_train_sta_imputed, X_val_sta_imputed, X_test_sta_imputed = (
                    X_train_standardized,
                    X_val_standardized,
                    X_test_standardized,
                )
            del X_train_standardized, X_val_standardized, X_test_standardized
            gc.collect()

            # Dynamic imputation - conditional
            if self.config["dynamic_imputation"]:
                logger.info("Performing dynamic imputation...")
                X_train_median = self.median_before_imputation(X_train_sta_imputed)

                X_train_imputed = self.dynamic_imputation(
                    X_train_sta_imputed, X_train_median
                )
                X_val_imputed = self.dynamic_imputation(
                    X_val_sta_imputed, X_train_median
                )
                X_test_imputed = self.dynamic_imputation(
                    X_test_sta_imputed, X_train_median
                )
            else:
                logger.info("Skipping dynamic imputation (disabled in config)")
                X_train_imputed, X_val_imputed, X_test_imputed = (
                    X_train_sta_imputed,
                    X_val_sta_imputed,
                    X_test_sta_imputed,
                )
            del X_train_sta_imputed, X_val_sta_imputed, X_test_sta_imputed
            gc.collect()

            # For mortality task, reshape the data
            if task == "mortality":
                logger.info("Applying mortality-specific reshaping into wide format...")
                X_train_imputed, y_train = self.reshape_mortality_data(
                    X_train_imputed, y_train, set_name="train"
                )
                X_val_imputed, y_val = self.reshape_mortality_data(
                    X_val_imputed, y_val, set_name="val"
                )
                X_test_imputed, y_test = self.reshape_mortality_data(
                    X_test_imputed, y_test, set_name="test"
                )

            # Add a warning about potential missing values if imputation was skipped
            if (
                not self.config["static_imputation"]
                or not self.config["dynamic_imputation"]
            ):
                na_counts = {
                    "X_train": X_train_imputed.isna().sum().sum(),
                    "X_val": X_val_imputed.isna().sum().sum(),
                    "X_test": X_test_imputed.isna().sum().sum(),
                }

                if any(na_counts.values()):
                    logger.warning(
                        "Datasets contain missing values after preprocessing: %s",
                        na_counts,
                    )
                    logger.warning(
                        "This may cause issues with models that don't handle NaN values."
                    )

            # Final shapes
            logger.info("Final dataset shapes:")
            logger.info("X_train: %s", X_train_imputed.shape)
            logger.info("y_train: %s", y_train.shape)
            logger.info("X_val: %s", X_val_imputed.shape)
            logger.info("y_val: %s", y_val.shape)
            logger.info("X_test: %s", X_test_imputed.shape)
            logger.info("y_test: %s", y_test.shape)

            # Save data if requested
            if save_data:
                self.save_preprocessed_data(
                    X_train_imputed,
                    X_val_imputed,
                    X_test_imputed,
                    y_train,
                    y_val,
                    y_test,
                )

            # # Calculate and print set statistics
            # train_stats = self.calculate_dataset_statistics(X_train_imputed, y_train, "train")
            # val_stats = self.calculate_dataset_statistics(X_val_imputed, y_val, "validation")
            # test_stats = self.calculate_dataset_statistics(X_test_imputed, y_test, "test")
            # self.print_statistics([train_stats, val_stats, test_stats])

            return (
                X_train_imputed,
                X_val_imputed,
                X_test_imputed,
                y_train,
                y_val,
                y_test,
            )

        except Exception as e:
            logger.error("Error in preprocessing pipeline: %s", e)
            raise

    def load_preprocessed_data(self, task: str, dataset_name: str) -> Optional[Tuple]:
        """
        Load preprocessed data for a specific task and dataset.

        Args:
            task (str): Task name ('mortality', 'aki', 'sepsis')
            dataset_name (str): Dataset name ('hirid', 'miiv', 'eicu')

        Returns:
            Optional[Tuple]: Tuple containing:
                X_train (pd.DataFrame): Training features
                X_val (pd.DataFrame): Validation features
                X_test (pd.DataFrame): Testing features
                y_train (pd.DataFrame): Training labels
                y_val (pd.DataFrame): Validation labels
                y_test (pd.DataFrame): Testing labels
        """
        # Store task and dataset name as instance variables
        self.task = task
        self.dataset_name = dataset_name

        # Generate directory name based on current configuration
        config_dirname = self.generate_preprocessing_dirname()

        # Construct directory path with task subfolder and config-based directory name
        directory = os.path.join(
            self.base_path,
            f"datasets/preprocessed_splits/{task}/{dataset_name}/{config_dirname}",
        )

        # Check if directory exists with current configuration
        if not os.path.exists(directory):
            return None
        else:
            # Load files
            X_train = pd.read_parquet(os.path.join(directory, "X_train.parquet"))
            y_train = pd.read_parquet(os.path.join(directory, "y_train.parquet"))
            X_val = pd.read_parquet(os.path.join(directory, "X_val.parquet"))
            y_val = pd.read_parquet(os.path.join(directory, "y_val.parquet"))
            X_test = pd.read_parquet(os.path.join(directory, "X_test.parquet"))
            y_test = pd.read_parquet(os.path.join(directory, "y_test.parquet"))

            logger.info("Loaded preprocessed data from %s", directory)

            return X_train, X_val, X_test, y_train, y_val, y_test

    def print_statistics(self, statistics_list: List[Dict[str, Any]]) -> None:
        """
        Print statistics in a formatted way for easy reading and Excel import.

        Args:
            statistics_list (List[Dict[str, Any]]): List of statistics dictionaries
        """

        if not statistics_list:
            logger.info("No statistics available to display")
            return

        logger.info("%s", "." * 40)
        logger.info(
            "DATASET STATISTICS - EXCEL FORMAT (Use Text to Columns with '/' as delimiter)"
        )

        # Print header row with slash delimiters
        logger.info(
            "Task/Dataset/Set/Total Stays/Cases/Controls/Total Rows/Rows of Cases/Rows of Controls/Positive Labels/Negative Labels"
        )

        # Print each row of statistics with slash delimiters
        for stat in statistics_list:
            # Format numbers with commas for thousands
            total_stays = f"{stat['Total Stays']:,}"
            cases = f"{stat['Cases']:,}"
            controls = f"{stat['Controls']:,}"
            total_rows = f"{stat['Total Rows']:,}"
            rows_cases = f"{stat['Rows of Cases']:,}"
            rows_controls = f"{stat['Rows of Controls']:,}"
            pos_labels = f"{stat['Positive Labels']:,}"
            neg_labels = f"{stat['Negative Labels']:,}"

            # For mortality task, simplify display (only show stays, cases, and controls)
            if stat["Task"] == "mortality":
                logger.info(
                    f"{stat['Task']}/"
                    f"{stat['Dataset']}/"
                    f"{stat['Set']}/"
                    f"{total_stays}/"
                    f"{cases} ({stat['Cases %']})/"
                    f"{controls} ({stat['Controls %']})/"
                    f"NA/NA/NA/NA/NA"
                )
            else:
                logger.info(
                    f"{stat['Task']}/"
                    f"{stat['Dataset']}/"
                    f"{stat['Set']}/"
                    f"{total_stays}/"
                    f"{cases} ({stat['Cases %']})/"
                    f"{controls} ({stat['Controls %']})/"
                    f"{total_rows}/"
                    f"{rows_cases} ({stat['Rows of Cases %']})/"
                    f"{rows_controls} ({stat['Rows of Controls %']})/"
                    f"{pos_labels} ({stat['Positive Labels %']})/"
                    f"{neg_labels} ({stat['Negative Labels %']})"
                )

        logger.info(f"{'.'*40}")

    def generate_preprocessing_dirname(self) -> str:
        """
        Generate a directory name based on the current preprocessing configuration.
        The name encodes which preprocessing steps were applied and with what parameters.
        Format: <preprocessing_steps>_split<train><val><test>
        Returns:
        str: Directory name encoding the preprocessing configuration
        """
        parts = []
        # Add part for outlier replacement
        if self.config["replace_outliers"]:
            parts.append("outliers")

        # Add part for NA flagging
        if self.config["flag_na"]:
            parts.append("flagna")

        # Get split ratios and format them
        train_pct = int(self.config["split_ratios"]["train"] * 100)
        val_pct = int(self.config["split_ratios"]["val"] * 100)
        test_pct = int(self.config["split_ratios"]["test"] * 100)
        split_part = f"split{train_pct}{val_pct}{test_pct}"

        # Always add split information
        parts.append(split_part)

        # Add part for standardization (after split)
        if self.config["standardize"]:
            parts.append("standardized")

        # Add part for static imputation
        if self.config["static_imputation"]:
            parts.append("staimputed")

        # Add part for dynamic imputation
        if self.config["dynamic_imputation"]:
            parts.append("dynimputed")

        # Create the directory name
        if parts:
            dirname = "_".join(parts)
        else:
            # If no preprocessing steps are enabled, use a default name with just split info
            dirname = f"raw_{split_part}"

        return dirname
