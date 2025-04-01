import logging
import sys
from typing import List, Dict, Any
from torch.utils.data import Dataset
import os
import torch
import numpy as np

import yaml


# Set up logger
logger = logging.getLogger("PULSE_logger")


class HarmonizedIcu(Dataset):
    """A harmonized icu dataset class for loading and processing datasets."""

    def __init__(self, model_name: str, test: bool = False):
        """
        Initialize the dataset.

        Args:
            model_name: Name of the model for preprocessing.
            test: Whether to load the test set.
        """
        self.model_name = model_name
        # Placeholder for actual data loading logic
        if test:
            # Load test data
            self.data = [
                {"features": [i / 10, i / 5, i / 2], "label": i % 2} for i in range(100)
            ]
        else:
            # Load training data
            self.data = [
                {"features": [i / 10, i / 5, i / 2], "label": i % 2} for i in range(500)
            ]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        # Placeholder for actual data retrieval logic. Change preprocessing for each model.
        item = self.data[idx]
        features = item["features"]
        label = item["label"]

        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.model_name == "XGBoostModel":
            # XGBoost works with numpy arrays
            return features_tensor, label_tensor
        elif self.model_name == "SimpleDLModel":
            # For neural networks, return properly formatted tensors
            return features_tensor, label_tensor
        # Default case: return tensors for PyTorch models
        return features_tensor, label_tensor


class DatasetManager:
    """Handles ICU dataset operations."""

    def __init__(self, dataset_names: List[str]):
        """Initialize the DatasetManager with dataset names. Checks if datasets is available."""
        self.available_datasets = yaml.safe_load(
            open("datasets/datasets.yaml", "r", encoding="utf-8")
        )
        self.datasets = {}

        if not dataset_names:
            logger.error("No datasets specified.")
            sys.exit()

        for name in dataset_names:
            self.datasets[name] = self._load_dataset_attr(name)

    def __available_datasets__(self) -> List[str]:
        """Return the list of available datasets."""
        return list(self.available_datasets.keys())

    def get_preprocessed_data(
        self, dataset_name: str, model_name: str, test: bool = False
    ) -> Dataset:
        """
        Get preprocessed data for a specific dataset.

        Args:
            dataset_name: Name of the dataset to load.
            test: Whether to load the test set.

        Returns:
            Dataset: Preprocessed dataset.

        """
        if dataset_name not in self.datasets:
            logger.error("Dataset %s not supported.", dataset_name)
            return None

        logger.info("Loading preprocessed data for dataset: %s", dataset_name)

        match dataset_name:
            case "harmonized_icu":
                dataset = HarmonizedIcu(model_name, test=test)

        return dataset

    def _load_dataset_attr(self, dataset_name: str) -> Any:
        """
        Load attributes for specific dataset.

        Args:
            dataset_name: Name of the dataset to load.

        Returns:
            dict: Dictionary containing dataset attributes.

        """
        if dataset_name not in self.available_datasets:
            logger.error("Dataset %s not supported.", dataset_name)
            return None

        logger.info("Loading attributes for dataset: %s", dataset_name)

        # Placeholder - implement actual dataset attribute loading logic
        return {"name": dataset_name}
