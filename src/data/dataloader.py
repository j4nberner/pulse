import gc
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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

    def __init__(self, config: dict):
        """
        Initialize the DatasetManager.

        Args:
            config (dict): The full configuration
        """
        self.config = config
        self.datasets = {}
        self.preprocessor = None
        self.windower = None

        # self.llm_model_list = ["Llama3Model"]

        self.debug_mode = getattr(self.config.general, "debug_mode", False)
        self.debug_data_length = getattr(self.config.general, "debug_data_length", 100)

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

        # Get debug_mode from config - using attribute style
        debug_mode = False
        if hasattr(self.config, "general"):
            debug_mode = getattr(self.config.general, "debug_mode", False)

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

        logger.info(
            "Windowing enabled: %s, Debug mode: %s", windowing_enabled, debug_mode
        )

        # Initialize windower with attribute style access
        if windowing_enabled:
            original_base_path = getattr(self.config, "original_base_path", None)
            self.windower = Windower(
                base_path=base_path,
                save_data=save_windowed_data,
                debug_mode=debug_mode,
                debug_data_length=self.debug_data_length,
                original_base_path=original_base_path,
                preprocessor_config=preprocessing_config,
            )

            logger.debug(
                "Windower initialized for advanced preprocessing with debug mode: %s",
                debug_mode,
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
                    "config": {"name": dataset_name, "task": task},
                    "loaded": False,
                    "data": None,
                }
                logger.info("Initialized dataset: %s", dataset_id)

    def load_dataset(self, dataset_id: str) -> bool:
        """
        Load a specific dataset.

        Args:
            dataset_id (str): ID of the dataset to load

        Returns:
            bool: True if loading was successful, False otherwise
        """
        if dataset_id not in self.datasets:
            logger.error("Dataset %s not found in configuration", dataset_id)
            return False

        dataset = self.datasets[dataset_id]

        if dataset["loaded"]:
            logger.debug("Dataset %s already loaded", dataset_id)
            return True

        try:
            # Extract task and dataset name
            task = dataset["task"]
            name = dataset["name"]

            # Check if windowing is enabled and should be applied
            windowing_enabled = False
            windowing_config = None

            if hasattr(self.config, "preprocessing_advanced"):
                if hasattr(self.config.preprocessing_advanced, "windowing"):
                    windowing_config = self.config.preprocessing_advanced.windowing
                    windowing_enabled = getattr(windowing_config, "enabled", False)

            # Check if windowing is applicable for this task
            if task == "mortality":
                logger.warning(
                    "Windowing is not applicable for the mortality task. Skipping windowing for task = 'mortality'."
                )

            # If windowing is enabled and not a mortality task, try to load presaved windowed data first
            if windowing_enabled and task != "mortality" and self.windower is not None:
                logger.debug(
                    "Attempting to load presaved windowed data for %s", dataset_id
                )
                windowed_data = self.windower.window_data(
                    task=task, dataset=name, config=windowing_config
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
                    logger.info(
                        "Successfully loaded presaved windowed data for %s", dataset_id
                    )
                    return True

                logger.info(
                    "No presaved windowed data found for %s, falling back to regular loading",
                    dataset_id,
                )

            # If not using presaved windowed data, proceed with regular loading/preprocessing
            try:
                # Try to load from preprocessed files
                X_train, X_val, X_test, y_train, y_val, y_test = (
                    self.preprocessor.load_preprocessed_data(
                        task=task, dataset_name=name
                    )
                )

                logger.info("Loaded preprocessed data for %s", dataset_id)

            except FileNotFoundError:
                # If not found, preprocess the data
                logger.info(
                    "Preprocessed data not found for %s, running preprocessing",
                    dataset_id,
                )

                X_train, X_val, X_test, y_train, y_val, y_test = (
                    self.preprocessor.preprocess(
                        task=task,
                        dataset_name=name,
                        save_data=getattr(
                            self.config.preprocessing_baseline, "save_data", True
                        ),
                    )
                )

                logger.info("Preprocessing Baseline completed for %s", dataset_id)

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

            # Apply windowing if enabled and not already loaded from presaved files
            if windowing_enabled and task != "mortality" and self.windower is not None:
                logger.debug("Applying windowing to %s", dataset_id)

                windowed_data = self.windower.window_data(
                    task=task,
                    dataset=name,
                    config=windowing_config,
                    data_dict=data_dict,
                )

                if windowed_data is not None:
                    data_dict = windowed_data
                    logger.info("Windowing applied to %s", dataset_id)

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
            return True

        except Exception as e:
            logger.error("Error loading dataset %s: %s", dataset_id, e)
            return False

    def get_preprocessed_data(
        self, dataset_id: str, model_name: str, mode: str = "train", **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get preprocessed data for a specific model.

        Args:
            dataset_id (str): ID of the dataset
            model_name (str): Name of the model
            mode (str): train, val, or test (default: train)
            **kwargs: Additional keyword arguments
            - debug (bool): If True, take only a specified number of rows
            - print_stats (bool): If True, print statistics for the datasets
            - prompting_id (str): ID of prompt preprocessing to apply

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels

        Notes:
            - When mode is "test", if test_limited is set in config, only the first X stay_ids will be used
            - If test_limited is None, the full test set is used
        """
        if dataset_id not in self.datasets:
            logger.error("Dataset %s not found", dataset_id)
            return None, None

        dataset = self.datasets[dataset_id]
        prompting_id = kwargs.get("prompting_id", None)

        if not dataset["loaded"]:
            success = self.load_dataset(dataset_id)
            if not success:
                return None, None

        data = dataset["data"]

        few_shot_list = [
            "liu_2023_few_shot_preprocessor",
            "zhu_2024a_one_shot_cot_preprocessor",
            "zhu_2024b_one_shot_preprocessor",
            "sarvari_2024_aggregation_preprocessor",
        ]

        # Take only n rows if in debug
        debug = kwargs.get("debug", False)
        if debug:
            logger.info(
                "Debug mode: Taking only %d rows for %s",
                self.debug_data_length,
                dataset_id,
            )
            data = {
                "X_train": data["X_train"].iloc[: self.debug_data_length],
                "y_train": data["y_train"].iloc[: self.debug_data_length],
                "X_val": data["X_val"].iloc[: self.debug_data_length],
                "y_val": data["y_val"].iloc[: self.debug_data_length],
                "X_test": data["X_test"].iloc[: self.debug_data_length],
                "y_test": data["y_test"].iloc[: self.debug_data_length],
            }

        # Initialize X_train and y_train to None for all modes
        X_train = None
        y_train = None

        # Get the appropriate split
        if mode == "train":
            X = data["X_train"]
            y = data["y_train"]

        elif mode == "val":
            if prompting_id in few_shot_list:
                # Some LLMs might need training data in validation set for few-shot learning
                X_train = data["X_train"]
                y_train = data["y_train"]
                X = data["X_val"]
                y = data["y_val"]
            else:
                # Normal case
                X = data["X_val"]
                y = data["y_val"]

        else:  # mode == "test"
            if prompting_id in few_shot_list:
                # Some LLMs might need training data in validation set for few-shot learning
                X_train = data["X_train"]
                y_train = data["y_train"]
                X = data["X_test"]
                y = data["y_test"]
            else:
                # Normal case
                X = data["X_test"]
                y = data["y_test"]

            # Handle limited test set if requested
            X_original = X.copy()
            y_original = y.copy()

            if self.test_limited is not None:
                logger.info(
                    "Limiting test set to first %s stay_ids for %s",
                    self.test_limited,
                    dataset_id,
                )
                # Get unique stay_ids in ascending order
                unique_stay_ids = sorted(X["stay_id"].unique())
                # Take only the first x = test_limited stay_ids (or all if less than x = test_limited)
                selected_stay_ids = unique_stay_ids[: self.test_limited]
                # Filter X and y to include only the selected stay_ids
                X_limited = X[X["stay_id"].isin(selected_stay_ids)]
                y_limited = y[y["stay_id"].isin(selected_stay_ids)]

                # Replace X and y with the limited versions
                X = X_limited
                y = y_limited

            # Print statistics if requested (Print train, val and both original and limited test set statistics to compare distributions)
            print_stats = kwargs.get("print_stats", False)  # set in train_models.py
            if print_stats:
                train_stats = self.preprocessor.calculate_dataset_statistics(
                    data["X_train"],
                    data["y_train"],
                    "train",
                    task=dataset["task"],
                    dataset_name=dataset["name"],
                )
                val_stats = self.preprocessor.calculate_dataset_statistics(
                    data["X_val"],
                    data["y_val"],
                    "val",
                    task=dataset["task"],
                    dataset_name=dataset["name"],
                )
                test_stats = self.preprocessor.calculate_dataset_statistics(
                    X_original,
                    y_original,
                    "test",
                    task=dataset["task"],
                    dataset_name=dataset["name"],
                )
                if self.test_limited is not None:
                    test_limited_stats = self.preprocessor.calculate_dataset_statistics(
                        X_limited,
                        y_limited,
                        f"test_limited{self.test_limited}",
                        task=dataset["task"],
                        dataset_name=dataset["name"],
                    )
                else:
                    test_limited_stats = None

                # Filter out None values before passing to print_statistics
                stats_to_print = [
                    stat
                    for stat in [train_stats, val_stats, test_stats, test_limited_stats]
                    if stat is not None
                ]
                # Print statistics for all datasets
                self.preprocessor.print_statistics(stats_to_print)

        # Drop stay_id columns BEFORE creating lists for few-shot learning
        if isinstance(X, pd.DataFrame) and "stay_id" in X.columns:
            X = X.drop(columns=["stay_id"])
        if isinstance(y, pd.DataFrame) and "stay_id" in y.columns:
            y = y.drop(columns=["stay_id"])

        # Also drop stay_id from training data used for few-shot examples
        if (
            X_train is not None
            and isinstance(X_train, pd.DataFrame)
            and "stay_id" in X_train.columns
        ):
            X_train = X_train.drop(columns=["stay_id"])
        if (
            y_train is not None
            and isinstance(y_train, pd.DataFrame)
            and "stay_id" in y_train.columns
        ):
            y_train = y_train.drop(columns=["stay_id"])

        logger.debug(
            "Dropped stay_id column from X and y (including for few-shot examples)"
        )

        # Convert categorical columns to numerical values for convML models
        convML_models = ["RandomForest", "XGBoost", "LightGBM"]
        if any(mdl in model_name for mdl in convML_models):
            # Process gender column in X if it exists
            if isinstance(X, pd.DataFrame) and "sex" in X.columns:
                X["sex"] = X["sex"].map({"Male": 1, "Female": 0}).fillna(-1)
                logger.debug("Converted gender column to numerical values")

        # Apply any model-specific preprocessing if needed.
        # For example, if you need to tokenize text data for LLMs
        if prompting_id is not None:
            prompting_preprocessor = get_prompting_preprocessor(
                prompting_id=prompting_id
            )
            num_shots = kwargs.get("num_shots", 0)
            data_window = self.config.preprocessing_advanced.windowing.data_window

            # Info dict needs to contain dataset name, task, and model name
            info_dict = {
                "dataset_name": dataset["name"],
                "task": dataset["task"],
                "model_name": model_name,
                "mode": mode,
                "num_shots": num_shots,
                "data_window": data_window,
            }
            if prompting_id in few_shot_list:
                # Add few-shot examples to info_dict if needed
                X = [X, X_train]
                y = [y, y_train]
            
            logger.info(
                "Applying prompting preprocessor for prompting_id: %s, and number of shots: %s",
                prompting_id,
                num_shots,
            )

            # Apply advanced preprocessing
            X, y = prompting_preprocessor(X, y, info_dict)

            # Log a sample prompt for debugging/verification
            if isinstance(X, pd.DataFrame) and mode == "test":
                prompt_column = (
                    "prompt"
                    if "prompt" in X.columns
                    else "text" if "text" in X.columns else None
                )
                if prompt_column and not X.empty:
                    sample_prompt = X[prompt_column].iloc[0]
                    logger.debug("Test loader length: %d", len(X))
                    logger.debug(
                        "Sample %s prompt with %s shots for %s:",
                        prompting_id,
                        num_shots,
                        dataset_id,
                    )
                    logger.debug("-" * 50)
                    logger.debug(sample_prompt)
                    logger.debug("-" * 50)

        return X, y

    def _drop_stay_id_if_present(self, data_dict: dict) -> dict:
        """
        Check if X and y sets have 'stay_id' column and drop it if present.

        Args:
            data_dict (dict): Dictionary containing X and y data splits

        Returns:
            dict: The modified data dictionary
        """
        # Log only once before processing all splits
        logger.debug("Dropping 'stay_id' column from all features and labels")

        for split in ["train", "val", "test"]:
            if split in data_dict:
                # Drop stay_id from X if present
                if "X" in data_dict[split]:
                    X_data = data_dict[split]["X"]
                    if isinstance(X_data, pd.DataFrame) and "stay_id" in X_data.columns:
                        data_dict[split]["X"] = X_data.drop(columns=["stay_id"])

                # Drop stay_id from y if present
                if "y" in data_dict[split]:
                    y_data = data_dict[split]["y"]
                    if isinstance(y_data, pd.DataFrame) and "stay_id" in y_data.columns:
                        data_dict[split]["y"] = y_data.drop(columns=["stay_id"])

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
        self.X_array = X.values.astype(np.float32)
        self.y_array = y.values.astype(np.float32)

        # Store column dtypes for efficient conversion
        self.dtypes = X.dtypes

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
