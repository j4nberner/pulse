import logging
import os
import sys
from typing import List, Any
from torch.utils.data import Dataset
import torch
from typing import Tuple, Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from src.preprocessing.preprocessing_baseline.preprocessing_baseline import (
    PreprocessorBaseline,
)
from src.preprocessing.preprocessing_advanced.windowing import Windower
from src.preprocessing.prompt_engineering import *

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

        # Initialize preprocessing tools
        self._init_preprocessing_tools()

        # Initialize datasets based on config
        self._init_datasets()

    def _init_preprocessing_tools(self):
        """Initialize preprocessing tools based on configuration."""
        base_path = self.config.base_path
        random_seed = self.config.random_seed

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

        logger.info(f"Windowing enabled: {windowing_enabled}, Debug mode: {debug_mode}")

        # Initialize windower with attribute style access
        if windowing_enabled:
            original_base_path = getattr(self.config, "original_base_path", None)
            self.windower = Windower(
                base_path=base_path,
                save_data=save_windowed_data,
                debug_mode=debug_mode,
                original_base_path=original_base_path,
                preprocessor_config=preprocessing_config,
            )

            logger.info(
                f"Windower initialized for advanced preprocessing with debug mode: {debug_mode}"
            )

    def _init_datasets(self) -> None:
        """Initialize datasets based on configuration."""

        # Check if base_path exists
        if not os.path.exists(self.config.dataset_path):
            logger.error(f"Base path {self.config.base_path} does not exist")
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
                logger.info(f"Initialized dataset: {dataset_id}")

    def load_dataset(self, dataset_id: str) -> bool:
        """
        Load a specific dataset.

        Args:
            dataset_id (str): ID of the dataset to load

        Returns:
            bool: True if loading was successful, False otherwise
        """
        if dataset_id not in self.datasets:
            logger.error(f"Dataset {dataset_id} not found in configuration")
            return False

        dataset = self.datasets[dataset_id]

        if dataset["loaded"]:
            logger.info(f"Dataset {dataset_id} already loaded")
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
                logger.info(
                    f"Attempting to load presaved windowed data for {dataset_id}"
                )
                windowed_data = self.windower.window_data(
                    task=task, dataset=name, config=windowing_config
                )

                if windowed_data is not None:
                    # Successfully loaded presaved windowed data
                    # Drop 'stay_id' column if present in y sets
                    windowed_data = self._drop_stay_id_if_present(windowed_data)

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
                        f"Successfully loaded presaved windowed data for {dataset_id}"
                    )
                    return True

                logger.info(
                    f"No presaved windowed data found for {dataset_id}, falling back to regular loading"
                )

            # If not using presaved windowed data, proceed with regular loading/preprocessing
            try:
                # Try to load from preprocessed files
                X_train, X_val, X_test, y_train, y_val, y_test = (
                    self.preprocessor.load_preprocessed_data(
                        task=task, dataset_name=name
                    )
                )

                logger.info(f"Loaded preprocessed data for {dataset_id}")

            except FileNotFoundError:
                # If not found, preprocess the data
                logger.info(
                    f"Preprocessed data not found for {dataset_id}, running preprocessing"
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

                logger.info(f"Preprocessing Baseline completed for {dataset_id}")

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
                logger.info(f"Applying windowing to {dataset_id}")

                windowed_data = self.windower.window_data(
                    task=task,
                    dataset=name,
                    config=windowing_config,
                    data_dict=data_dict,
                )

                if windowed_data is not None:
                    data_dict = windowed_data
                    logger.info(f"Windowing applied to {dataset_id}")

            # Drop 'stay_id' column if present in y sets (after windowing)
            data_dict = self._drop_stay_id_if_present(data_dict)

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
            logger.error(f"Error loading dataset {dataset_id}: {e}")
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

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
        """
        if dataset_id not in self.datasets:
            logger.error(f"Dataset {dataset_id} not found")
            return None, None

        dataset = self.datasets[dataset_id]

        if not dataset["loaded"]:
            success = self.load_dataset(dataset_id)
            if not success:
                return None, None

        data = dataset["data"]

        # Take only 100 rows if in debug
        debug = kwargs.get("debug", False)
        if debug:
            logger.info(f"Debug mode: Taking only 100 rows for {dataset_id}")
            data = {
                "X_train": data["X_train"].head(100),
                "y_train": data["y_train"].head(100),
                "X_val": data["X_val"].head(100),
                "y_val": data["y_val"].head(100),
                "X_test": data["X_test"].head(100),
                "y_test": data["y_test"].head(100),
            }

        # Get the appropriate split
        if mode == "test":
            X = data["X_test"]
            y = data["y_test"]

        elif mode == "val":
            X = data["X_val"]
            y = data["y_val"]
        else:
            X = data["X_train"]
            y = data["y_train"]

        # Apply any model-specific preprocessing if needed. Prompt engineering for LLMs, tokenization, etc.

        # For example, if you need to tokenize text data for LLMs
        preprocessing_id = kwargs.get("preprocessing_id", None)
        match preprocessing_id:
            case "Llama3Preprocessing":
                # Apply Llama3-specific preprocessing
                logger.info(f"Applying Llama3 preprocessing for {dataset_id}")
                dataset = kwargs.get("dataset", None)
                task = kwargs.get("task", None)
                info_dict = {"dataset": dataset, "task": task}
                X, y = apply_llama3_preprocessing(X, y, info_dict)
            case None:
                # No specific preprocessing needed
                logger.info(f"No specific preprocessing needed for {dataset_id}")

        return X, y

    def _drop_stay_id_if_present(self, data_dict: dict) -> dict:
        """
        Check if y sets have 'stay_id' column and drop it if present.

        Args:
            data_dict (dict): Dictionary containing X and y data splits

        Returns:
            dict: The modified data dictionary
        """
        # TODO: should X keep the stay_id?

        for split in ["train", "val", "test"]:
            if split in data_dict and "y" in data_dict[split]:
                y_data = data_dict[split]["y"]
                if isinstance(y_data, pd.DataFrame) and "stay_id" in y_data.columns:
                    logger.info(f"Dropping 'stay_id' column from train/val/test labels")
                    data_dict[split]["y"] = y_data.drop(columns=["stay_id"])

        return data_dict


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

        # TODO: je nach Modell Tensor anders stacken (if DL -> apply 3D stacking)

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
