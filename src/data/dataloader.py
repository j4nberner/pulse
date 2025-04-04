import logging
import sys
from typing import List, Any
from torch.utils.data import Dataset, DataLoader
import torch
import yaml
from typing import Tuple, Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

from src.preprocessing.preprocessing_baseline.preprocessing_baseline import PreprocessorBaseline

# Set up logger
logger = logging.getLogger("PULSE_logger")


class HarmonizedIcu(Dataset):
    """A harmonized icu dataset class for loading and processing datasets."""

    def __init__(self, model_name: str, test: bool = False, **kwargs) -> None:
        """
        Initialize the dataset.

        Args:
            model_name: Name of the model for preprocessing.
            test: Whether to load the test set.
        """
        self.model_name = model_name
        # Placeholder for actual data loading logic
        # Do Baseline preprocessing here
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

        # General Preprocessing
        # preprocessor.preprocess(preprocessing_dict, data)

        # Do model-specific preprocessing here
        if self.model_name == "XGBoostModel":
            # Implement specific preprocessing for XGBoost here. Use src/preprocess/xgboost_preprocess.py import
            return features_tensor, label_tensor
        elif self.model_name == "SimpleDLModel":

            # For neural networks, return properly formatted tensors
            return features_tensor, label_tensor
        # Default case: return tensors for PyTorch models
        return features_tensor, label_tensor


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
        
        # Initialize datasets based on config
        self._init_datasets()
    
    def _init_datasets(self) -> None:
        """Initialize datasets based on configuration."""
        base_path = self.config.base_path
        random_seed = self.config.random_seed
        
        # Initialize preprocessor
        self.preprocessor = PreprocessorBaseline(
            base_path=base_path,
            random_seed=random_seed
        )
        
        # Process each task and dataset combination
        for task in self.config.tasks:
            for dataset_name in self.config.datasets:
                dataset_id = f"{task}_{dataset_name}"
                self.datasets[dataset_id] = {
                    'task': task,
                    'name': dataset_name,
                    'config': {'name': dataset_name, 'task': task},
                    'loaded': False,
                    'data': None
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
        
        if dataset['loaded']:
            logger.info(f"Dataset {dataset_id} already loaded")
            return True
        
        try:
            # Try to load preprocessed data first
            task = dataset['task']
            name = dataset['name']
            
            try:
                # Try to load from preprocessed files
                X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.load_preprocessed_data(
                    task=task,
                    dataset_name=name
                )
                
                logger.info(f"Loaded preprocessed data for {dataset_id}")
                
            except FileNotFoundError:
                # If not found, preprocess the data
                logger.info(f"Preprocessed data not found for {dataset_id}, running preprocessing")
                
                X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.preprocess(
                    task=task,
                    dataset_name=name,
                    save_data=True
                )
                
                logger.info(f"Preprocessing completed for {dataset_id}")
            
            # Store the loaded data
            dataset['data'] = {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
            dataset['loaded'] = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {e}")
            return False
    
    def get_preprocessed_data(
        self, 
        dataset_id: str, 
        model_name: str, 
        test: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get preprocessed data for a specific model.
        
        Args:
            dataset_id (str): ID of the dataset
            model_name (str): Name of the model
            test (bool): Whether to return test data
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
        """
        if dataset_id not in self.datasets:
            logger.error(f"Dataset {dataset_id} not found")
            return None, None
        
        dataset = self.datasets[dataset_id]
        
        if not dataset['loaded']:
            success = self.load_dataset(dataset_id)
            if not success:
                return None, None
        
        data = dataset['data']
        
        # Get the appropriate split
        if test:
            X = data['X_test']
            y = data['y_test']
        else:
            X = data['X_train']
            y = data['y_train']
        
        # Apply any model-specific preprocessing if needed
        # For now, we just return the data as is
        
        return X, y
    
    def get_validation_data(
        self, 
        dataset_id: str, 
        model_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get validation data for a specific model.
        
        Args:
            dataset_id (str): ID of the dataset
            model_name (str): Name of the model
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and labels
        """
        if dataset_id not in self.datasets:
            logger.error(f"Dataset {dataset_id} not found")
            return None, None
        
        dataset = self.datasets[dataset_id]
        
        if not dataset['loaded']:
            success = self.load_dataset(dataset_id)
            if not success:
                return None, None
        
        data = dataset['data']
        
        X_val = data['X_val']
        y_val = data['y_val']
        
        # Apply any model-specific preprocessing if needed
        # For now, we just return the data as is
        
        return X_val, y_val

class TorchDatasetWrapper(Dataset):
    """
    Wrapper class to convert pandas DataFrames to PyTorch Datasets.
    
    This class makes pandas DataFrames compatible with PyTorch DataLoader.
    
    Attributes:
        X (pd.DataFrame): Feature dataframe
        y (pd.DataFrame): Label dataframe
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        """
        Initialize the TorchDatasetWrapper.
        
        Args:
            X (pd.DataFrame): Feature dataframe
            y (pd.DataFrame): Label dataframe
        """
        self.X = X
        self.y = y
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (features, label) for the specified index
        """
        # Convert to numpy arrays
        X_sample = self.X.iloc[idx].values.astype(np.float32)
        y_sample = self.y.iloc[idx].values.astype(np.float32)
        
        return X_sample, y_sample
