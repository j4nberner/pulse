import gc
import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.preprocessing.preprocessing_baseline.preprocessing_baseline import \
    PreprocessorBaseline

# Set up logger
logger = logging.getLogger("PULSE_logger")


class Windower:
    """
    Class for creating windowed data with specified data window, prediction window, and step size.

    This class transforms time-series ICU data into fixed-width windows suitable for
    machine learning models. It handles:
    - Taking preprocessed dataframes and applying windowing transformation
    - Creating windows with configurable data window, prediction window, and step size
    - Saving windowed data to parquet files if specified
    - Loading previously windowed data
    """

    def __init__(
        self,
        base_path,
        save_data=False,
        app_mode="benchmark",
        debug_data_length=100,
        original_base_path=None,
        preprocessor_config=None,
    ):
        """
        Initialize the Windower.

        Args:
            base_path (str): Base path for data directories
            save_data (bool): Whether to save windowed data
            app_mode (str): Identifier for the application mode (e.g., benchmark, debug, )
            original_base_path (str): Original base path for permanent storage (for Slurm jobs)
        """
        self.base_path = base_path
        self.save_data = save_data
        self.app_mode = app_mode
        self.debug_data_length = debug_data_length
        self.original_base_path = original_base_path
        self.preprocessor_config = preprocessor_config

    def create_windows(self, data_dict, data_window, prediction_window, step_size=1):
        """
        Create windowed data with specified data window, prediction window, and step size.

        Args:
            data_dict (dict): Dictionary containing train, val, test data with X and y keys
            data_window (int): Size of the data window (number of time steps to include)
            prediction_window (int): Size of the prediction window (time to prediction)
            step_size (int): Step size for window shifting

        Returns:
            dict: Dictionary containing windowed data
        """
        results = {}

        for set_type in ["train", "val", "test"]:
            X = data_dict[f"X_{set_type}"]
            y = data_dict[f"y_{set_type}"]

            X_np = X.values
            y_np = y["label"].values
            stay_id_np = X["stay_id"].values

            # Define static columns that will be preserved in order
            static_columns = ["stay_id", "age", "sex", "height", "weight"]
            columns_index = [
                X.columns.get_loc(col) for col in static_columns if col in X.columns
            ]
            non_static_columns = [
                i for i in range(X_np.shape[1]) if i not in columns_index
            ]

            result_rows = []
            result_labels = []
            unique_stay_ids = np.unique(stay_id_np)

            logger.info(
                "Processing %s set with %s stay_ids", set_type, len(unique_stay_ids)
            )

            # Initialize variables for batch processing
            batch_size = 10000  # Adjust based on memory constraints
            current_batch = 0
            X_window = None
            y_window = None

            for stay_id in tqdm(
                unique_stay_ids,
                mininterval=1.0,
                desc=f"{set_type} stay_ids",
            ):
                mask = stay_id_np == stay_id
                X_stay = X_np[mask]
                y_stay = y_np[mask]

                # Minimum length depends on prediction window
                min_length = data_window + prediction_window
                if prediction_window == 0:
                    min_length = data_window

                if len(X_stay) < min_length:
                    continue

                # Get static values for this stay
                static_row = {
                    X.columns[col_idx]: X_stay[0, col_idx] for col_idx in columns_index
                }

                # Adjust the range based on prediction window
                max_start = len(X_stay) - min_length + 1

                # Use step_size for window sliding
                for start in range(0, max_start, step_size):
                    X_window_data = X_stay[start : start + data_window]

                    # Skip if we don't have a full window (could happen with the last window)
                    if len(X_window_data) < data_window:
                        continue

                    row = {
                        f"{X.columns[col_idx]}_{hour}": X_window_data[hour, col_idx]
                        for col_idx in non_static_columns
                        for hour in range(data_window)
                    }

                    row.update(static_row)
                    result_rows.append(row)

                    # For prediction_window=0, get label from the last position of data window
                    # For positive prediction_window, get label from the position after data window + prediction_window - 1
                    label_position = start + data_window - 1
                    if prediction_window > 0:
                        label_position = start + data_window + prediction_window - 1

                    result_labels.append(y_stay[label_position])

                    # Process in batches to reduce memory usage
                    if len(result_rows) >= batch_size:
                        X_window, y_window = self._process_batch(
                            result_rows, result_labels, X, X_window, y_window
                        )

                        # Clear memory
                        result_rows = []
                        result_labels = []
                        current_batch += 1
                        gc.collect()  # Force garbage collection

            # Process any remaining data
            if result_rows:
                X_window, y_window = self._process_batch(
                    result_rows, result_labels, X, X_window, y_window
                )

            # Reorder columns to have stay_id, sex, age, height, weight first
            all_columns = list(X_window.columns)
            ordered_static_columns = [
                col for col in static_columns if col in all_columns
            ]
            other_columns = [
                col for col in all_columns if col not in ordered_static_columns
            ]
            X_window = X_window[ordered_static_columns + other_columns]

            # Log the shape of the windowed dataset
            logger.info(
                "Windowed %s set - X shape: %s, y shape: %s",
                set_type,
                X_window.shape,
                y_window.shape,
            )

            results[set_type] = {"X": X_window, "y": y_window}

        # Force garbage collection to free memory
        gc.collect()

        return results

    def _process_batch(self, result_rows, result_labels, X, X_window, y_window):
        """
        Process a batch of windowed data to create DataFrames with proper data types.

        Args:
            result_rows (list): List of dictionaries containing feature data
            result_labels (list): List of labels
            X (pd.DataFrame): Original feature DataFrame for reference
            X_window (pd.DataFrame): Existing X window DataFrame, can be None
            y_window (pd.DataFrame): Existing y window DataFrame, can be None

        Returns:
            tuple: Updated (X_window, y_window) DataFrames
        """
        batch_X = pd.DataFrame(result_rows)
        batch_y = pd.DataFrame(result_labels, columns=["label"])

        # Add stay_id to batch_y
        batch_y["stay_id"] = [row["stay_id"] for row in result_rows]

        # Process data types for the batch
        for col in batch_X.columns:
            if "_" in col and col.split("_")[-1].isdigit():
                # Extract the base column name (everything before the last underscore)
                base_col = "_".join(col.split("_")[:-1])
                if base_col in X.columns:
                    batch_X[col] = batch_X[col].astype(X[base_col].dtype)
            else:
                # For static columns
                if col in X.columns:
                    batch_X[col] = batch_X[col].astype(X[col].dtype)

        batch_y["label"] = batch_y["label"].astype(int)

        # Either initialize or append to existing dataframes
        if X_window is None:
            X_window = batch_X
            y_window = batch_y
        else:
            X_window = pd.concat([X_window, batch_X], ignore_index=True)
            y_window = pd.concat([y_window, batch_y], ignore_index=True)

        return X_window, y_window

    def save_windowed_data(
        self, results, task, dataset, data_window, prediction_window, step_size
    ):
        """
        Save windowed data to parquet files, both in scratch and permanent storage if applicable.

        Args:
            results (dict): Dictionary containing windowed data
            task (str): Task name
            dataset (str): Dataset name
            data_window (int): Size of the data window
            prediction_window (int): Size of the prediction window
            step_size (int): Step size for window shifting
        """

        # Generate dynamic directory name based on preprocessing configuration
        preprocessor = PreprocessorBaseline(
            base_path=self.base_path, config=self.preprocessor_config
        )  # Pass config to temporary instance
        config_dirname = preprocessor.generate_preprocessing_dirname()

        # Directory naming adjusted based on debug mode
        save_directory = f"datasets/preprocessed_splits/{task}/{dataset}/{config_dirname}/{data_window}_dw_{prediction_window}_pw_{step_size}_sz"
        os.makedirs(os.path.join(self.base_path, save_directory), exist_ok=True)

        # Save to current base_path (might be scratch)
        for set_type in ["train", "val", "test"]:
            # Save files
            x_path = os.path.join(
                self.base_path, save_directory, f"X_{set_type}.parquet"
            )
            y_path = os.path.join(
                self.base_path, save_directory, f"y_{set_type}.parquet"
            )
            if not os.path.exists(x_path):
                results[set_type]["X"].to_parquet(x_path)
            else:
                logger.warning(
                    "File %s already exists, skipping save for X_%s", x_path, set_type
                )

            if not os.path.exists(y_path):
                results[set_type]["y"].to_parquet(y_path)
            else:
                logger.warning(
                    "File %s already exists, skipping save for y_%s", y_path, set_type
                )

        logger.info(
            "Windowed data saved to %s", os.path.join(self.base_path, save_directory)
        )

        # If original_base_path is set, also save to permanent storage
        if self.original_base_path:
            permanent_directory = os.path.join(self.original_base_path, save_directory)
            os.makedirs(permanent_directory, exist_ok=True)

            for set_type in ["train", "val", "test"]:
                # Save files
                x_path = os.path.join(permanent_directory, f"X_{set_type}.parquet")
                y_path = os.path.join(permanent_directory, f"y_{set_type}.parquet")
                if not os.path.exists(x_path):
                    results[set_type]["X"].to_parquet(x_path)
                else:
                    logger.warning(
                        "File %s already exists, skipping save for X_%s",
                        x_path,
                        set_type,
                    )

                if not os.path.exists(y_path):
                    results[set_type]["y"].to_parquet(y_path)
                else:
                    logger.warning(
                        "File %s already exists, skipping save for y_%s",
                        y_path,
                        set_type,
                    )

            logger.info("Windowed data in permanent storage: %s", permanent_directory)

    def load_windowed_data(
        self, task, dataset, data_window, prediction_window, step_size
    ):
        """
        Load previously windowed data from parquet files.

        Args:
            task (str): Task name
            dataset (str): Dataset name
            data_window (int): Size of the data window
            prediction_window (int): Size of the prediction window
            step_size (int): Step size for window shifting

        Returns:
            dict: Dictionary containing windowed data or None if loading fails
        """
        # Generate dynamic directory name based on preprocessing configuration
        preprocessor = PreprocessorBaseline(
            base_path=self.base_path, config=self.preprocessor_config
        )  # Pass config to temporary instance
        config_dirname = preprocessor.generate_preprocessing_dirname()

        # Check for debug mode directory first
        load_directory = f"datasets/preprocessed_splits/{task}/{dataset}/{config_dirname}/{data_window}_dw_{prediction_window}_pw_{step_size}_sz"
        full_path = os.path.join(self.base_path, load_directory)

        # If not in debug mode and regular files don't exist
        if not os.path.exists(full_path):
            logger.info("Windowed data directory %s does not exist", full_path)
            return None

        results = {}

        try:
            for set_type in ["train", "val", "test"]:
                X_path = os.path.join(full_path, f"X_{set_type}.parquet")
                y_path = os.path.join(full_path, f"y_{set_type}.parquet")

                if not os.path.exists(X_path) or not os.path.exists(y_path):
                    logger.error("Missing windowed data files in %s", full_path)
                    return None

                X = pd.read_parquet(X_path)
                y = pd.read_parquet(y_path)

                results[set_type] = {"X": X, "y": y}

                logger.info(
                    "Loaded windowed %s set - X shape: %s, y shape: %s",
                    set_type,
                    X.shape,
                    y.shape,
                )

            logger.info("Loaded windowed data from %s", full_path)

            return results

        except Exception as e:
            logger.error("Error loading windowed data from %s: %s", full_path, e)
            return None

    def read_preprocessed_data(self, task, dataset):
        """
        Read preprocessed train, validation and test parquet files.

        Args:
            task (str): Task name
            dataset (str): Dataset name

        Returns:
            dict: Dictionary containing preprocessed data or None if loading fails
        """
        # Generate dynamic directory name based on preprocessing configuration
        preprocessor = PreprocessorBaseline(
            base_path=self.base_path, config=self.preprocessor_config
        )  # Pass config to temporary instance
        config_dirname = preprocessor.generate_preprocessing_dirname()

        read_directory = (
            f"datasets/preprocessed_splits/{task}/{dataset}/{config_dirname}"
        )
        full_path = os.path.join(self.base_path, read_directory)

        if not os.path.exists(full_path):
            return None

        try:
            data_sets = {
                "train": {
                    "X": pd.read_parquet(os.path.join(full_path, "X_train.parquet")),
                    "y": pd.read_parquet(os.path.join(full_path, "y_train.parquet")),
                },
                "val": {
                    "X": pd.read_parquet(os.path.join(full_path, "X_val.parquet")),
                    "y": pd.read_parquet(os.path.join(full_path, "y_val.parquet")),
                },
                "test": {
                    "X": pd.read_parquet(os.path.join(full_path, "X_test.parquet")),
                    "y": pd.read_parquet(os.path.join(full_path, "y_test.parquet")),
                },
            }

            # Log shapes of loaded datasets
            for set_type in data_sets:
                logger.info(
                    "Loaded %s set before windowing - X shape: %s, y shape: %s",
                    set_type,
                    data_sets[set_type]["X"].shape,
                    data_sets[set_type]["y"].shape,
                )

            return data_sets

        except Exception as e:
            logger.error("Error reading preprocessed data from %s: %s", full_path, e)
            return None

    def window_data(self, task, dataset, config, data_dict=None):
        """
        Apply windowing to a dataset, always trying to load presaved data first.

        Args:
            task (str): Task name
            dataset (str): Dataset name
            config (dict): Configuration for windowing
            data_dict (dict, optional): Dictionary containing data to window. If None, will load from preprocessed files.

        Returns:
            dict: Dictionary containing windowed data or original data if windowing is not applicable
        """

        # Extract windowing parameters from config
        data_window = config.get("data_window", 6)
        prediction_window = config.get("prediction_window", 0)
        step_size = config.get("step_size", 1)

        logger.info(
            "Applying windowing to %s_%s with data_window=%s, "
            "prediction_window=%s, step_size=%s",
            task,
            dataset,
            data_window,
            prediction_window,
            step_size,
        )

        # Create windows
        windowed_data = self.create_windows(
            data_dict, data_window, prediction_window, step_size
        )

        # Save windowed data only for full datasets
        if self.save_data and self.app_mode == "benchmark":
            logger.info("Saving windowed data for %s_%s", task, dataset)
            self.save_windowed_data(
                windowed_data, task, dataset, data_window, prediction_window, step_size
            )

        return windowed_data


class WindowedDataTo3D:
    """
    Class for converting windowed ICU data from flattened 2D pandas DataFrames to 3D numpy arrays
    suitable for deep learning models.

    This class treats static features (like demographics) as time series by repeating their values
    across the time dimension, resulting in a single 3D array output.
    """

    def __init__(self, architecture_type=None, config=None, task_name=None):
        """
        Initialize the WindowedDataTo3D converter.

        Args:
            logger: Logger instance for logging messages
            architecture_type: Base architecture of the convDL model ("CNN" or "RNN") to determine array format
            config (dict, optional): Configuration with windowing parameters
        """
        self.logger = logger or logging.getLogger("PULSE_logger")
        self.task_name = task_name
        self.architecture_type = architecture_type

        if self.architecture_type and self.architecture_type not in ["CNN", "RNN"]:
            self.logger.warning(
                f"Unknown architecture type: {architecture_type}. Using RNN format as default."
            )
            self.architecture_type = "RNN"

        # Extract window size from config if available
        self.window_size = 6  # Default
        if task_name is not None and task_name == "mortality":
            self.window_size = 25
            self.logger.info(
                "Mortality task detected - Setting window size to %s for mortality task",
                self.window_size,
            )
        elif config:
            if hasattr(config, "preprocessing_advanced"):
                preprocessing_advanced = config.preprocessing_advanced
                if hasattr(preprocessing_advanced, "windowing"):
                    windowing_config = preprocessing_advanced.windowing
                    windowing_enabled = getattr(windowing_config, "enabled", False)
                    if hasattr(windowing_config, "data_window"):
                        self.window_size = windowing_config.data_window
                        if windowing_enabled:
                            self.logger.info(
                                "Setting window size to %s from config",
                                self.window_size,
                            )

        # Add these properties for the enhanced functionality
        self.needs_conversion = True
        self.use_windowed_conversion = False
        self.input_shape = None

    def configure_conversion(self, windowing_enabled, input_shape):
        """
        Configure the conversion strategy based on windowing status.

        Args:
            windowing_enabled: Whether windowing was applied to data
            input_shape: Shape of input data
        """
        self.input_shape = input_shape
        self.needs_conversion = True
        self.use_windowed_conversion = windowing_enabled

        self.logger.info(
            "Converter configured: windowed=%s, input_shape=%s",
            windowing_enabled,
            input_shape,
        )

    def convert_batch_to_3d(
        self,
        batch_features,
        window_size=None,
        static_feature_count=4,
        id_column_index=None,
    ):
        """
        Convert a batch of features from 2D to 3D format suitable for temporal models.
        Works with tensors extracted directly from the dataloader.

        Args:
            batch_features (torch.Tensor): Batch of features in 2D format (batch_size, n_features)
            window_size (int, optional): Size of the time window. Defaults to self.window_size
            static_feature_count (int): Number of static features (excluding id_column)
            id_column_index (int): Index of the ID column to exclude (typically 0 for stay_id) -> in our case None because stay_id is dropped in dataloader

        Returns:
            torch.Tensor: 3D tensor ready for conventional DL model input
        """

        # If already 3D, return as is
        if len(batch_features.shape) == 3 or not self.needs_conversion:
            return batch_features

        # Override use_windowed_conversion for mortality task
        mortality_specific_task = (
            self.task_name is not None and self.task_name == "mortality"
        )
        if mortality_specific_task:
            self.use_windowed_conversion = True
            window_size = 25

        # Use windowed conversion if configured
        if self.use_windowed_conversion:

            batch_size, n_features = batch_features.shape

            # Use provided window_size or fall back to self.window_size
            if window_size is None:
                window_size = self.window_size

            # Determine model type (CNN or RNN)
            is_cnn = self.architecture_type == "CNN"

            # Skip the ID column
            if id_column_index is not None:
                # Create a mask to exclude the ID column
                keep_mask = torch.ones(n_features, dtype=torch.bool)
                keep_mask[id_column_index] = False

                # Apply the mask to get features without ID
                features_no_id = batch_features[:, keep_mask]
                n_features = features_no_id.shape[1]
            else:
                features_no_id = batch_features

            # Extract static features (typically columns 0-3 after removing ID)
            static_features = features_no_id[:, :static_feature_count]

            # Extract dynamic features (everything after static features)
            dynamic_features = features_no_id[:, static_feature_count:]
            n_dynamic_features = dynamic_features.shape[1]

            # Calculate number of actual features
            n_actual_dynamic_features = n_dynamic_features // window_size

            try:
                # Reshape dynamic features based on window size
                if is_cnn:
                    # For CNN: (batch, features, time)
                    dynamic_3d = dynamic_features.reshape(
                        batch_size, n_actual_dynamic_features, window_size
                    )

                    # Repeat static features for each time step
                    static_3d = static_features.unsqueeze(-1).repeat(1, 1, window_size)

                    # Combine on feature dimension (static first, then dynamic)
                    return torch.cat([static_3d, dynamic_3d], dim=1)
                else:
                    # For RNN: (batch, time, features)
                    dynamic_3d = dynamic_features.reshape(
                        batch_size, window_size, n_actual_dynamic_features
                    )

                    # Repeat static features for each time step
                    static_3d = static_features.unsqueeze(1).repeat(1, window_size, 1)

                    # Combine on feature dimension
                    return torch.cat([static_3d, dynamic_3d], dim=2)

            except Exception as e:
                self.logger.warning(
                    "Error reshaping batch to 3D: %s. Using simple approach.", e
                )

                # Fall back to simple reshape if the proper reshaping fails
                if is_cnn:
                    return features_no_id.unsqueeze(-1)  # (batch, features, 1)
                else:
                    return features_no_id.unsqueeze(1)  # (batch, 1, features)

        else:
            # Simple reshaping for non-windowed data
            if self.architecture_type == "CNN":
                return batch_features.unsqueeze(-1)  # For CNN: (batch, features, 1)
            else:
                return batch_features.unsqueeze(1)  # For RNN: (batch, 1, features)
