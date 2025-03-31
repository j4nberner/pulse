import logging
import sys
from typing import List, Dict, Any
from torch.utils.data import Dataset
import os

import yaml


# Set up logger
logger = logging.getLogger(__name__)


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
            self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example data
        else:
            # Load training data
            self.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Example data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        # Placeholder for actual data retrieval logic. Change preprocessing for each model.
        if self.model_name == "XGBoost":
            # Apply XGBoost specific preprocessing
            pass
        elif self.model_name == "RandomForest":
            # Apply RandomForest specific preprocessing
            pass
        # Add more model-specific preprocessing as needed
        return self.data[idx]


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
