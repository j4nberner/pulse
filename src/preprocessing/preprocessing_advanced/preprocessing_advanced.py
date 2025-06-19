import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.util.data_util import (get_feature_name, get_feature_reference_range,
                                get_feature_uom)

# Set up logger
logger = logging.getLogger("PULSE_logger")


class PreprocessorAdvanced:
    """
    Advanced data preprocessing operations beyond baseline preprocessing and windowing.

    This class implements methods for more complex preprocessing tasks such as:
    - Matching absolute onset times between different data sources
    - Aggregating windowed data with various statistics
    - Selecting features based on importance or other criteria
    - Generating new features through transformations and combinations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the PreprocessorAdvanced class with configuration parameters.

        Args:
            config (Dict[str, Any], optional): Configuration options. Defaults to None.
        """
        self.config = config or {}
        logger.debug("Initialized PreprocessorAdvanced")

    def prepare_feature_descriptions(self, base_features, X_cols):
        """Prepare feature descriptions with name, unit of measurement, and reference range.

        Args:
            base_features: Set of base feature names
            X_cols: DataFrame columns to check for additional features

        Returns:
            Feature descriptions text as a formatted string
        """
        # Generate feature descriptions for the reference section
        feature_descriptions = []
        for feature in sorted(base_features):  # Sort for consistent order
            feature_name = get_feature_name(feature)
            uom = get_feature_uom(feature)
            range_values = get_feature_reference_range(feature)

            if range_values:  # Check if the range exists (not empty tuple)
                range_str = f"{range_values[0]} - {range_values[1]}"
                feature_descriptions.append(
                    f"- {feature_name}: Unit: {uom}. Reference range: {range_str}."
                )
            else:
                feature_descriptions.append(
                    f"- {feature_name}: Unit: {uom}. Reference range: /."
                )

        # Add weight and height to feature descriptions if they exist in the columns
        if "weight" in X_cols:
            weight_name = get_feature_name("weight")
            weight_uom = get_feature_uom("weight")
            feature_descriptions.append(
                f"- {weight_name}: Unit: {weight_uom}. Reference range: /."
            )

        if "height" in X_cols:
            height_name = get_feature_name("height")
            height_uom = get_feature_uom("height")
            feature_descriptions.append(
                f"- {height_name}: Unit: {height_uom}. Reference range: /."
            )

        # Join all feature descriptions into a single string
        return "\n".join(feature_descriptions)

    def format_patient_data(self, row, base_features, X_cols, data_window):
        """Format patient data for prompting.

        Args:
            row: Patient data row
            base_features: Set of base feature names
            X_cols: DataFrame columns to extract feature columns from

        Returns:
            Tuple of (patient_info, patient_features_text)
        """
        # Extract patient demographic info
        sex = row.get("sex", "unknown")
        age = row.get("age", "unknown")
        patient_info = f"The patient is a {sex}, aged {age} years."

        # Format feature values
        patient_features = []

        # Process dynamic features (those with time series)
        for feature in sorted(base_features):
            # Get columns for this feature (e.g., hr_1, hr_2, etc.)
            feature_cols = [col for col in X_cols if col.startswith(f"{feature}_")]

            # Filter to only include columns with numeric indices
            feature_cols = [col for col in feature_cols if col.split("_")[1].isdigit()]

            # Print warning if the number of feature columns doesn't match the data window
            if len(feature_cols) != data_window:
                logger.warning(
                    "Feature '%s' has %s columns, but expected %s columns.",
                    feature,
                    len(feature_cols),
                    data_window,
                )

            # Sort columns by time point
            if feature_cols:  # Only sort if there are valid columns
                feature_cols.sort(key=lambda x: int(x.split("_")[1]))

            # Extract values for this feature across all time points
            values = [f"{float(row[col]):.2f}" for col in feature_cols]
            values_str = f'"{", ".join(values)}"'

            # Use the proper feature name from dictionary
            feature_name = get_feature_name(feature)
            patient_features.append(f"- {feature_name}: {values_str}")

        # Get number of time points from dynamic features
        num_timepoints = len(feature_cols) if "feature_cols" in locals() else 6

        # Process static features (weight and height) - repeat value for all time points
        if "weight" in row.index and not pd.isna(row["weight"]):
            weight_value = f"{float(row['weight']):.2f}"
            weight_values = [weight_value] * num_timepoints
            weight_str = f'"{", ".join(weight_values)}"'
            weight_name = get_feature_name("weight")
            patient_features.append(f"- {weight_name}: {weight_str}")

        if "height" in row.index and not pd.isna(row["height"]):
            height_value = f"{float(row['height']):.2f}"
            height_values = [height_value] * num_timepoints
            height_str = f'"{", ".join(height_values)}"'
            height_name = get_feature_name("height")
            patient_features.append(f"- {height_name}: {height_str}")

        # Join patient features into a string
        patient_features_text = "\n".join(patient_features)

        return patient_info, patient_features_text

    @classmethod
    def aggregate_feature_windows(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate feature values in a DataFrame based on shared base names and return with static columns.

        Args:
            df: Input DataFrame with features named in a specific pattern (e.g., name_number)

        Returns:
            DataFrame with aggregated features (min, max, mean) for each base name, along with static columns.
        """
        grouped_values = {}
        dynamic_cols = set()

        # Group columns by base name (e.g., alb_0, alb_1 → alb)
        for col in df.columns:
            if "_" in col and col.split("_")[-1].isdigit():
                base_name = col.split("_")[0]
                grouped_values.setdefault(base_name, []).append(df[col])
                dynamic_cols.add(col)

        # Aggregate grouped values efficiently
        agg_data = {}
        for base_name, value_list in grouped_values.items():
            group_df = pd.concat(value_list, axis=1)
            agg_data[f"{base_name}_min"] = group_df.min(axis=1)
            agg_data[f"{base_name}_max"] = group_df.max(axis=1)
            agg_data[f"{base_name}_mean"] = group_df.mean(axis=1)

        # Create a single DataFrame from aggregated data
        agg_df = pd.DataFrame(agg_data, index=df.index)

        # Keep static columns (those not ending with _number)
        static_cols = [col for col in df.columns if col not in dynamic_cols]
        static_df = df[static_cols].copy()

        # Combine static and aggregated data
        result_df = pd.concat([static_df, agg_df], axis=1)

        return result_df

    def categorize_features(
        self, df, base_features=None, X_cols=None, num_categories=3, for_llm=False
    ):
        """
        Categorize features across an entire dataframe based on reference ranges.

        Args:
            df: Input DataFrame with patient data rows
            base_features: Set of base feature names (optional, will be extracted if not provided)
            X_cols: DataFrame columns to use (optional, will use df.columns if not provided)
            num_categories: Number of categories to use (3 or 5)
                - 3: -1 (too low), 0 (normal), 1 (too high)
                - 5: -1 (very low), -0.5 (slightly low), 0 (normal), 0.5 (slightly high), 1 (very high)
                  where "slightly" means within 50% of the reference range
            for_llm: If True, return descriptive strings instead of numeric categories

        Returns:
            DataFrame with the same index as input but with base features as columns and categorized values
            If for_llm=True, values are descriptive strings, otherwise numeric categories
        """
        # Use provided parameters or extract from dataframe
        if X_cols is None:
            X_cols = df.columns

        if base_features is None:
            # Extract base feature names
            base_features = set()
            for col in X_cols:
                if "_" in col and col.split("_")[-1].isdigit():
                    base_name = col.split("_")[0]
                    base_features.add(base_name)

        if num_categories not in [3, 5]:
            raise ValueError("num_categories must be either 3 or 5")

        # Initialize result dataframe with the same index as input
        result_df = pd.DataFrame(index=df.index)

        # Process each base feature
        for feature in sorted(base_features):
            # Get columns for this feature (e.g., hr_1, hr_2, etc.)
            feature_cols = [col for col in X_cols if col.startswith(f"{feature}_")]

            # Filter to only include columns with numeric indices
            feature_cols = [col for col in feature_cols if col.split("_")[1].isdigit()]

            # Skip if no valid columns for this feature
            if not feature_cols:
                continue

            # Get reference range
            reference_range = get_feature_reference_range(feature)

            # Skip if no reference range available
            if not reference_range:
                continue

            # Extract values for this feature
            feature_matrix = df[feature_cols].apply(pd.to_numeric, errors="coerce")

            # Calculate row-wise mean, ignoring NaN values
            row_means = feature_matrix.mean(axis=1, skipna=True)

            if num_categories == 3:
                # Categorize based on reference range with 3 categories
                feature_values = np.select(
                    [row_means < reference_range[0], row_means > reference_range[1]],
                    [-1, 1],  # Too low, Too high
                    default=0,  # Normal
                )
            else:  # num_categories == 5
                # Calculate range size for determining slightly low/high thresholds
                range_size = reference_range[1] - reference_range[0]

                # Calculate thresholds for 5-category classification
                very_low_threshold = reference_range[0] - 0.5 * range_size
                very_high_threshold = reference_range[1] + 0.5 * range_size

                # Categorize based on reference range with 5 categories
                feature_values = np.select(
                    [
                        row_means <= very_low_threshold,  # Very low
                        (row_means > very_low_threshold)
                        & (row_means < reference_range[0]),  # Slightly low
                        (row_means >= reference_range[0])
                        & (row_means <= reference_range[1]),  # Normal
                        (row_means > reference_range[1])
                        & (row_means < very_high_threshold),  # Slightly high
                        row_means >= very_high_threshold,  # Very high
                    ],
                    [-1, -0.5, 0, 0.5, 1],
                    default=0,  # Default to normal if there's any issue
                )

            # Add the feature to the result dataframe
            if for_llm:
                # Convert numeric categories to descriptive strings for LLM prompts
                if num_categories == 3:
                    category_mapping = {-1: "too low", 0: "normal", 1: "too high"}
                else:  # num_categories == 5
                    category_mapping = {
                        -1.0: "very low",
                        -0.5: "slightly low",
                        0.0: "normal",
                        0.5: "slightly high",
                        1.0: "very high",
                    }

                # Map numeric values to strings
                string_values = [
                    category_mapping.get(val, "unknown range") for val in feature_values
                ]
                result_df[feature] = string_values
            else:
                # Return numeric categories
                result_df[feature] = feature_values

        return result_df

    #######################################
    # PLACEHOLDER FOR FUTURE IMPLEMENTATION
    #######################################

    def match_absolute_onset(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Align temporal windows of cases and controls based on time since ICU admission

        """

        # WORK IN PROGRESS
        # Implementation logic here

        # Placeholder for X_aligned and y_aligned until implementation is complete
        X_aligned = X.copy()
        y_aligned = y.copy()
        return X_aligned, y_aligned

    def select_features(
        self,
        df: pd.DataFrame,
        method: str = "correlation",
        target_col: Optional[str] = None,
        threshold: float = 0.8,
        k_features: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Select relevant features from the dataset based on statistical criteria.

        This method implements various feature selection approaches:
        - correlation: Remove highly correlated features
        - variance: Remove low variance features
        - importance: Select top k features based on importance to target

        Args:
            df: Input DataFrame with features
            method: Feature selection method ('correlation', 'variance', 'importance')
            target_col: Target column name (required for 'importance' method)
            threshold: Threshold value for selection (correlation coefficient or variance)
            k_features: Number of top features to select (for 'importance' method)

        Returns:
            DataFrame with selected features
        """

        # WORK IN PROGRESS

        logger.info("Selecting features using method: %s", method)

        if method == "correlation":
            # Implementation for removing highly correlated features
            # [Code would compute correlation matrix and remove redundant features]
            pass

        elif method == "variance":
            # Implementation for removing low variance features
            # [Code would compute variances and filter based on threshold]
            pass

        elif method == "importance":
            if target_col is None:
                raise ValueError("target_col must be specified for 'importance' method")
            # Implementation for selecting features based on importance to target
            # [Code would compute feature importance and select top k features]

        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        # Placeholder
        selected_df = df.copy()
        return selected_df

    def generate_features(
        self, df: pd.DataFrame, operations: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Generate new features through transformations and combinations of existing features.

        This method creates new features based on specified operations:
        - blood_draws: Count frequency of blood draws from lab test features
        - hours_since_admission: Calculate time since admission for each patient

        Args:
            df: Input DataFrame
            operations: List of dictionaries specifying feature generation operations
                      Each dict should have keys:
                      - 'type': operation type (e.g., 'blood_draws')
                      - 'columns': columns to use
                      - 'params': additional parameters for the operation

        Returns:
            DataFrame with original and newly generated features
        """

        # WORK IN PROGRESS

        logger.info("Generating %d new feature sets", len(operations))

        result_df = df.copy()

        for op in operations:
            op_type = op.get("type")
            op.get("columns", [])
            op.get("params", {})

            if op_type == "blood_draws":
                # Count frequency of blood draws from lab test features
                # This would analyze when lab tests were performed and count draw frequency
                # Implementation would identify lab test timestamps and count frequency
                pass
            elif op_type == "hours_since_admission":
                # Calculate time since admission for each patient
                # if 'admission_time' in df.columns:
                # result_df['hours_since_admission'] = ...
                pass
            else:
                logger.warning(f"Unknown feature generation operation type: {op_type}")

        logger.info(
            "Generated features: original=%d, new=%d", df.shape[1], result_df.shape[1]
        )
        return result_df
