import gc
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.util.config_util import set_seeds
from src.preprocessing.preprocessing_advanced.windowing import Windower
from src.preprocessing.preprocessing_baseline.preprocessing_baseline import (
    PreprocessorBaseline,
)
from src.preprocessing.preprocessing_prompts import get_prompting_preprocessor

# Set up logger
logger = logging.getLogger("PULSE_logger")


class DatasetManager:
    """
    Manager class for handling datasets in the training framework.

    This class is responsible for:
    - Loading datasets
    - Preprocessing datasets
    - Providing access to preprocessed data

    Attributes:
        config (Dict): Configuration for datasets
        datasets (Dict): Dictionary of loaded datasets
        preprocessor (PreprocessorBaseline): Preprocessor instance
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize the DatasetManager.

        Args:
            config (dict): The full configuration
        """
        self.config = config
        self.datasets = {}
        self.preprocessor = None
        self.windower = None

        self.app_mode = config.general.app_mode
        self.debug_data_length = None

        match self.app_mode:
            case "debug":
                self.debug_data_length = config.general.debug_data_length
                logger.info(
                    "Running in debug mode. Limited data will be used for faster inference. (# of rows: %d)",
                    self.debug_data_length,
                )

            case "count_tokens":
                self.debug_data_length = config.general.debug_data_length
                logger.info(
                    "Running in count_tokens mode. Full test data and a small subset of train and val data will be used for token counting. (# of rows train/val: %d)",
                    self.debug_data_length,
                )
            case "benchmark":
                logger.info(
                    "Running in benchmark mode. Full data will be used for training and evaluation."
                )
            case _:
                logger.error(
                    "Invalid app_mode: %s. Must be one of ['debug', 'count_tokens', 'benchmark']",
                    self.app_mode,
                )
                raise ValueError(
                    f"Invalid app_mode: {self.app_mode}. Must be one of ['debug', 'count_tokens', 'benchmark']"
                )

        # Initialize preprocessing tools
        self._init_preprocessing_tools()

        # Initialize datasets based on config
        self._init_datasets()

        # Extract test_limited parameter from config
        self.test_limited = getattr(
            self.config.preprocessing_baseline.split_ratios, "test_limited", None
        )

    def _init_preprocessing_tools(self):
        """Initialize preprocessing tools based on configuration."""
        base_path = self.config.base_path
        random_seed = self.config.benchmark_settings.random_seed
        set_seeds(random_seed)

        # Get app_mode from config - using attribute style
        app_mode = self.config.general.app_mode

        # Get preprocessing_baseline configuration - using attribute style
        preprocessing_config = None
        if hasattr(self.config, "preprocessing_baseline"):
            preprocessing_config = self.config.preprocessing_baseline

        # Initialize preprocessor (add original_base_path attribute if run in HPC environment)
        if hasattr(self.config, "original_base_path"):
            self.preprocessor = PreprocessorBaseline(
                base_path=base_path,
                random_seed=random_seed,
                config=preprocessing_config,
                original_base_path=self.config.original_base_path,
            )
        else:
            self.preprocessor = PreprocessorBaseline(
                base_path=base_path,
                random_seed=random_seed,
                config=preprocessing_config,
            )

        # Initialize windower
        windowing_enabled = False
        save_windowed_data = False

        # Check if preprocessing_advanced exists in config - using attribute style
        if hasattr(self.config, "preprocessing_advanced"):
            if hasattr(self.config.preprocessing_advanced, "windowing"):
                windowing_config = self.config.preprocessing_advanced.windowing

                windowing_enabled = getattr(windowing_config, "enabled", False)
                save_windowed_data = getattr(windowing_config, "save_data", False)

        logger.info("Windowing enabled: %s, App mode: %s", windowing_enabled, app_mode)

        # Initialize windower with attribute style access
        if windowing_enabled:
            original_base_path = getattr(self.config, "original_base_path", None)
            self.windower = Windower(
                base_path=base_path,
                save_data=save_windowed_data,
                app_mode=app_mode,
                debug_data_length=self.debug_data_length,
                original_base_path=original_base_path,
                preprocessor_config=preprocessing_config,
            )

            logger.debug(
                "Windower initialized for advanced preprocessing with app mode: %s",
                app_mode,
            )

    def _init_datasets(self) -> None:
        """Initialize datasets based on configuration."""

        # Check if base_path exists
        if not os.path.exists(self.config.dataset_path):
            logger.error("Base path %s does not exist", self.config.base_path)
            sys.exit(1)

        # Process each task and dataset combination
        for task in self.config.tasks:
            for dataset_name in self.config.datasets:
                dataset_id = f"{task}_{dataset_name}"
                self.datasets[dataset_id] = {
                    "task": task,
                    "name": dataset_name,
                    "preprocessing_baseline": self.config.preprocessing_baseline,
                    "preprocessing_advanced": self.config.preprocessing_advanced,
                    "loaded": False,
                    "data": None,
                }
                logger.info("Initialized dataset: %s", dataset_id)

    def load_dataset(self, dataset_id: str) -> tuple[bool, Optional[Dict]]:
        """
        Load a specific dataset.
        Reversed loading logic.
        1. Loading saved preprocessing_advanced data if available
        2. Loading saved preprocessing_baseline data if available
        3. Loading raw data if no saved data is available

        Args:
            dataset_id (str): ID of the dataset to load

        Returns:
            tuple: (success: bool, dataset: Optional[Dict])
                - success (bool): True if the dataset was loaded successfully, False otherwise
                - dataset (Optional[Dict]): The loaded dataset or None if loading failed
        """
        if dataset_id not in self.datasets:
            logger.error("Dataset %s not found in configuration", dataset_id)
            return False

        dataset = self.datasets[dataset_id].copy()

        if dataset["loaded"]:
            logger.debug("Dataset %s already loaded", dataset_id)
            return True

        # Extract task and dataset name
        task = dataset["task"]
        name = dataset["name"]

        # 1. Load saved preprocessing_advanced data if available
        # Check if windowing is enabled and should be applied
        windowing_enabled = dataset["preprocessing_advanced"]["windowing"]["enabled"]
        windowing_config = dataset["preprocessing_advanced"]["windowing"]

        # Check if windowing is applicable for this task
        if task == "mortality":
            logger.warning(
                "Windowing is not applicable for the mortality task. Skipping windowing for task = 'mortality'."
            )
            windowing_enabled = False
            windowing_config = None

        # If windowing is enabled and applicable, try to load presaved windowed data
        if windowing_enabled:
            logger.debug("Attempting to load presaved windowed data for %s", dataset_id)
            windowed_data = self.windower.load_windowed_data(
                task=task,
                dataset=name,
                data_window=windowing_config.data_window,
                prediction_window=windowing_config.prediction_window,
                step_size=windowing_config.step_size,
            )

            if windowed_data is not None:
                # load presaved windowed data
                dataset["data"] = {
                    "X_train": windowed_data["train"]["X"],
                    "X_val": windowed_data["val"]["X"],
                    "X_test": windowed_data["test"]["X"],
                    "y_train": windowed_data["train"]["y"],
                    "y_val": windowed_data["val"]["y"],
                    "y_test": windowed_data["test"]["y"],
                }
                dataset["loaded"] = True
                dataset["preprocessing_advanced"]["windowing"]["loaded"] = True
                logger.info(
                    "Successfully loaded presaved windowed data for %s", dataset_id
                )
                return True, dataset

            logger.debug(
                "No presaved windowed data found for %s. Checking for baseline preprocessed data.",
                dataset_id,
            )

        # 2. Load saved preprocessing_baseline data if available
        data_sets = self.preprocessor.load_preprocessed_data(
            task=task, dataset_name=name
        )
        if data_sets is not None:
            X_train, X_val, X_test, y_train, y_val, y_test = data_sets
            logger.info("Loaded preprocessed data for %s", dataset_id)
            # Convert labels from boolean to int if necessary
            y_train["label"], y_val["label"], y_test["label"] = (
                y_train["label"].astype(int),
                y_val["label"].astype(int),
                y_test["label"].astype(int),
            )

            # Store the loaded data
            data_dict = {
                "train": {"X": X_train, "y": y_train},
                "val": {"X": X_val, "y": y_val},
                "test": {"X": X_test, "y": y_test},
            }

            # Store the processed data
            dataset["data"] = {
                "X_train": data_dict["train"]["X"],
                "X_val": data_dict["val"]["X"],
                "X_test": data_dict["test"]["X"],
                "y_train": data_dict["train"]["y"],
                "y_val": data_dict["val"]["y"],
                "y_test": data_dict["test"]["y"],
            }

            dataset["loaded"] = True
            dataset["preprocessing_advanced"]["windowing"]["loaded"] = False
            return True, dataset

        # 3. Baseline preprocessing and saving for future use
        logger.info(
            "Loading raw data for %s, applying baseline preprocessing and saving for future use.",
            dataset_id,
        )
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.preprocess(
            task=task,
            dataset_name=name,
            save_data=True,
        )
        logger.info("Loaded preprocessed data for %s", dataset_id)
        # Convert labels from boolean to int if necessary
        y_train["label"], y_val["label"], y_test["label"] = (
            y_train["label"].astype(int),
            y_val["label"].astype(int),
            y_test["label"].astype(int),
        )

        # Store the loaded data
        data_dict = {
            "train": {"X": X_train, "y": y_train},
            "val": {"X": X_val, "y": y_val},
            "test": {"X": X_test, "y": y_test},
        }

        # Store the processed data
        dataset["data"] = {
            "X_train": data_dict["train"]["X"],
            "X_val": data_dict["val"]["X"],
            "X_test": data_dict["test"]["X"],
            "y_train": data_dict["train"]["y"],
            "y_val": data_dict["val"]["y"],
            "y_test": data_dict["test"]["y"],
        }

        dataset["loaded"] = True
        dataset["preprocessing_advanced"]["windowing"]["loaded"] = False
        return True, dataset

    def get_preprocessed_data(
        self, dataset_id: str, model_name: str, mode: str = "train", **kwargs: Any
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
        """
        Get preprocessed data for a specific model.

        Args:
            dataset_id (str): ID of the dataset
            model_name (str): Name of the model
            mode (str): train, val, or test (default: train)
            **kwargs: Additional keyword arguments
            - prompting_id (str): ID of prompt preprocessing to apply
            - model_type (str): Type of model (e.g., "convML", "convDL", "LLM")
            - fine_tuning (bool): Whether to apply fine-tuning (default: False)


        Returns:
            Tuple[pd.DataFrame]: Features and labels

        Notes:
            - When mode is "test", if test_limited is set in config, only the first X stay_ids will be used
            - If test_limited is None, the full test set is used
        """
        if dataset_id not in self.datasets:
            logger.error("Dataset %s not found", dataset_id)
            return None, None, None, None, None, None

        dataset = self.datasets[dataset_id].copy()
        print_stats = self.datasets[dataset_id]["preprocessing_baseline"][
            "split_ratios"
        ]["print_stats"]

        if not dataset["loaded"]:
            success, dataset = self.load_dataset(dataset_id)
            if not success:
                return None, None, None, None, None, None

        data = dataset["data"]

        # Limit the loaded data to only load the necessary parts before processing
        # Take only n rows if in debug
        if self.app_mode == "debug":
            logger.info(
                "Debug mode: Taking only %d rows for %s",
                self.debug_data_length,
                dataset_id,
            )
            dataset["data"] = {
                "X_train": data["X_train"].iloc[: self.debug_data_length],
                "y_train": data["y_train"].iloc[: self.debug_data_length],
                "X_val": data["X_val"].iloc[: self.debug_data_length],
                "y_val": data["y_val"].iloc[: self.debug_data_length],
                "X_test": data["X_test"].iloc[: self.debug_data_length],
                "y_test": data["y_test"].iloc[: self.debug_data_length],
            }

        elif self.app_mode == "count_tokens":
            logger.debug(
                "Count tokens mode: Taking only %d rows for %s for train and val loader.",
                self.debug_data_length,
                dataset_id,
            )
            dataset["data"] = {
                "X_train": data["X_train"].iloc[: self.debug_data_length],
                "y_train": data["y_train"].iloc[: self.debug_data_length],
                "X_val": data["X_val"].iloc[: self.debug_data_length],
                "y_val": data["y_val"].iloc[: self.debug_data_length],
                "X_test": data["X_test"].iloc[:],
                "y_test": data["y_test"].iloc[:],
            }

        else:
            logger.debug("Running in benchmark mode.")

        if self.test_limited is not None:
            X = dataset["data"]["X_test"]
            y = dataset["data"]["y_test"]

            # Get unique stay_ids in ascending order
            unique_stay_ids = sorted(X["stay_id"].unique())
            # Take only the first x = test_limited stay_ids (or all if less than x = test_limited)
            selected_stay_ids = unique_stay_ids[: self.test_limited]
            # Filter X and y to include only the selected stay_ids
            X_limited = X[X["stay_id"].isin(selected_stay_ids)]
            y_limited = y[y["stay_id"].isin(selected_stay_ids)]

            # Replace X and y with the limited versions
            dataset["data"]["X_test"] = X_limited
            dataset["data"]["y_test"] = y_limited
            logger.info(
                "Limited test set to first %s stay_ids for %s",
                len(selected_stay_ids),
                dataset_id,
            )

        # Applying advanced preprocessing if and not already loaded from presaved files
        logger.debug(
            "Applying advanced preprocessing for %s.",
            dataset_id,
        )

        # Apply windowing if enabled and not already loaded from presaved files
        # newly windowed data will be saved only for app_mode = benchmark
        if (
            dataset["preprocessing_advanced"]["windowing"]["enabled"]
            and dataset["preprocessing_advanced"]["windowing"]["loaded"] is False
            and dataset["task"] != "mortality"
            and self.windower is not None
        ):
            logger.debug("Applying windowing to %s", dataset_id)

            data = self.windower.window_data(
                task=dataset["task"],
                dataset=dataset["name"],
                config=dataset["preprocessing_advanced"]["windowing"],
                data_dict=dataset["data"],
            )
            dataset["data"] = {
                "X_train": data["train"]["X"],
                "X_val": data["val"]["X"],
                "X_test": data["test"]["X"],
                "y_train": data["train"]["y"],
                "y_val": data["val"]["y"],
                "y_test": data["test"]["y"],
            }
            dataset["loaded"] = True
            dataset["preprocessing_advanced"]["windowing"]["loaded"] = True
            logger.info("Successfully windowed data for %s", dataset_id)

        del data  # Clear the data variable to free up memory

        # Model specific parameters from kwargs. Throws KeyError if not provided.
        model_type = kwargs["model_type"]

        # Convert categorical columns to numerical values for convML models
        if model_type in ["convML", "convDL"]:
            # Process gender column in X if it exists
            for data_set in ["X_train", "X_val", "X_test"]:
                X = dataset["data"][data_set]
                X["sex"] = X["sex"].map({"Male": 1, "Female": 0}).fillna(-1)
                dataset["data"][data_set] = X
            logger.debug("Converted gender column to numerical values")

        # Print statistics if requested (Print train, val and both original and limited test set statistics to compare distributions)
        if print_stats:
            train_stats = self.preprocessor.calculate_dataset_statistics(
                dataset["data"]["X_train"],
                dataset["data"]["y_train"],
                "train",
                task=dataset["task"],
                dataset_name=dataset["name"],
            )
            val_stats = self.preprocessor.calculate_dataset_statistics(
                dataset["data"]["X_val"],
                dataset["data"]["y_val"],
                "val",
                task=dataset["task"],
                dataset_name=dataset["name"],
            )
            test_stats = self.preprocessor.calculate_dataset_statistics(
                dataset["data"]["X_test"],
                dataset["data"]["y_test"],
                "test",
                task=dataset["task"],
                dataset_name=dataset["name"],
            )

            # Filter out None values before passing to print_statistics
            stats_to_print = [
                stat
                for stat in [train_stats, val_stats, test_stats]
                if stat is not None
            ]
            # Print statistics for all datasets
            self.preprocessor.print_statistics(stats_to_print)

        # Drop stay_id column after calculating statistics
        dataset["data"] = self._drop_stay_id_if_present(dataset["data"])
        logger.debug("Dropped stay_id column from all features and labels")

        # Apply advanced preprocessing if needed -> generate prompts
        if model_type == "LLM":
            prompting_id = kwargs["prompting_id"]
            fine_tuning = kwargs["fine_tuning"]

            prompting_preprocessor = get_prompting_preprocessor(
                prompting_id=prompting_id
            )
            num_shots = kwargs.get("num_shots", 0)
            data_window = self.config.preprocessing_advanced.windowing.data_window

            info_dict = {
                "dataset_name": dataset["name"],
                "task": dataset["task"],
                "model_name": model_name,
                "mode": mode,
                "num_shots": num_shots,
                "data_window": data_window,
            }

            # Add model instance to info_dict if provided in kwargs
            if "model_instance" in kwargs:
                info_dict["model_instance"] = kwargs["model_instance"]

            # Add output directory to info_dict
            info_dict["output_dir"] = getattr(self.config, "output_dir", None)

            logger.info(
                "Applying prompting preprocessor for prompting_id: %s, and number of shots: %s",
                prompting_id,
                num_shots,
            )

            # Apply advanced preprocessing
            if fine_tuning is True:
                # Training data
                X = [dataset["data"]["X_train"]]
                y = [dataset["data"]["y_train"]]
                info_dict["mode"] = "train"
                dataset["data"]["X_train"], dataset["data"]["y_train"] = (
                    prompting_preprocessor(X, y, info_dict)
                )

                # Validation data - Uses training data for few-shot learning
                X = [dataset["data"]["X_val"], dataset["data"]["X_train"]]
                y = [dataset["data"]["y_val"], dataset["data"]["y_train"]]
                info_dict["mode"] = "val"
                dataset["data"]["X_val"], dataset["data"]["y_val"] = (
                    prompting_preprocessor(X, y, info_dict)
                )

            # Test data
            X = [dataset["data"]["X_test"], dataset["data"]["X_train"]]
            y = [dataset["data"]["y_test"], dataset["data"]["y_train"]]
            info_dict["mode"] = "test"
            dataset["data"]["X_test"], dataset["data"]["y_test"] = (
                prompting_preprocessor(X, y, info_dict)
            )

            if fine_tuning is False:
                # Used only for few shot examples if no fine-tuning. Set to none.
                dataset["data"]["X_train"] = pd.DataFrame()
                dataset["data"]["y_train"] = pd.DataFrame()
                dataset["data"]["X_val"] = pd.DataFrame()
                dataset["data"]["y_val"] = pd.DataFrame()

            # Pass the loaded model back through the info_dict
            if "model_instance" in kwargs and "loaded_model" in info_dict:
                # Store the loaded model back to the benchmark.py flow
                kwargs["loaded_model"] = info_dict["loaded_model"]

        return (
            dataset["data"]["X_train"],
            dataset["data"]["y_train"],
            dataset["data"]["X_val"],
            dataset["data"]["y_val"],
            dataset["data"]["X_test"],
            dataset["data"]["y_test"],
        )

    def _drop_stay_id_if_present(self, data_dict: dict) -> dict:
        """
        Check if X and y sets have 'stay_id' column and drop it if present.

        Args:
            data_dict (dict): Dictionary containing X and y data splits

        Returns:
            dict: The modified data dictionary
        """
        # Loop through all dataframes in the data_dict
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and "stay_id" in df.columns:
                # Remove the stay_id column
                data_dict[key] = df.drop(columns=["stay_id"])
                logger.debug(f"Dropped 'stay_id' column from {key}")

                # Debug info for labels
                if key.startswith("y_"):
                    logger.debug(
                        f"{key} shape after dropping stay_id: {data_dict[key].shape}"
                    )
                    logger.debug(f"{key} columns: {data_dict[key].columns.tolist()}")

        return data_dict

    def release_dataset_cache(self, dataset_id=None):
        """Release cached data for a specific dataset or all datasets."""
        if dataset_id:
            if dataset_id in self.datasets and self.datasets[dataset_id]["loaded"]:
                self.datasets[dataset_id]["data"] = None
                self.datasets[dataset_id]["loaded"] = False
                logger.debug("Released cached data for %s", dataset_id)
                gc.collect()
        else:
            for ds_id, dataset in self.datasets.items():
                if dataset["loaded"]:
                    dataset["data"] = None
                    dataset["loaded"] = False
            logger.debug("Released all cached datasets")
            gc.collect()


class TorchDatasetWrapper(Dataset):
    """
    Memory-efficient wrapper class to convert pandas DataFrames to PyTorch Dataset.
    """

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Initialize the TorchDatasetWrapper with options for memory efficiency.

        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.DataFrame): Label dataframe
        """
        self.X = X
        self.y = y

        # Pre-compute indices for faster access
        self.indices = list(range(len(X)))

        # Optional: Convert DataFrames to NumPy arrays upfront if they fit in memory
        # This avoids repeated conversions during __getitem__ calls
        # Comment this out if data is too large
        # TODO: @sophiafe - Can we remove this? Very prone to memory issues. Or check available memory before converting.
        # self.X_array = X.values.astype(np.float32)
        # self.y_array = y.values.astype(np.float32)

        # Store column dtypes for efficient conversion
        self.dtypes = X.dtypes

        # Calculate pos/neg ratio, avoid division by zero
        neg = len(y) - y["label"].sum()
        pos = y["label"].sum()
        if pos == 0:
            logger.warning(
                "No positive samples found in the dataset. Setting pos_weight to 1."
            )
            self.pos_weight = 1.0
        elif neg == 0:
            logger.warning(
                "No negative samples found in the dataset. Setting pos_weight to 0."
            )
            self.pos_weight = 0.0
        else:
            self.pos_weight = neg / pos

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset with optimized memory usage.

        Args:
            idx (int or slice or list): Index/indices of the sample(s)

        Returns:
            tuple: (features, label) as torch.Tensor
        """
        # If we pre-computed arrays, use them
        if hasattr(self, "X_array") and hasattr(self, "y_array"):
            X_sample = self.X_array[idx]
            y_sample = self.y_array[idx]
        # For single integer index
        elif isinstance(idx, int):
            X_sample = self.X.iloc[idx].values.astype(np.float32)
            y_sample = self.y.iloc[idx].values.astype(np.float32)
        # For slices or lists of indices (batch access)
        else:
            X_sample = self.X.iloc[idx].values.astype(np.float32)
            y_sample = self.y.iloc[idx].values.astype(np.float32)

        # Convert to PyTorch tensors
        return torch.tensor(X_sample), torch.tensor(y_sample)

    # TODO: @sophiafe Is this still needed?
    def get_batch(self, indices):
        """
        Custom method to get a batch with explicit indices.
        More efficient than using DataLoader for large datasets.

        Args:
            indices (list): List of indices to include in batch

        Returns:
            tuple: (features, labels) for the specified indices
        """
        X_batch = self.X.iloc[indices].values.astype(np.float32)
        y_batch = self.y.iloc[indices].values.astype(np.float32)
        return X_batch, y_batch
