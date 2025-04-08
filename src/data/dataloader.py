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
from src.preprocessing.preprocessing_advanced.windowing import Windower

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
        self.windower = None

        # Initialize preprocessing tools
        self._init_preprocessing_tools()
        
        # Initialize datasets based on config
        self._init_datasets()

    def _init_preprocessing_tools(self):
        """Initialize preprocessing tools based on configuration."""
        base_path = self.config.base_path
        random_seed = self.config.random_seed
        
        # Get debug_mode from config
        debug_mode = False
        if hasattr(self.config, 'general'):
            if isinstance(self.config.general, dict):
                debug_mode = self.config.general.get('debug_mode', False)
            else:
                debug_mode = getattr(self.config.general, 'debug_mode', False)
        
        # Get preprocessing_baseline configuration
        preprocessing_config = None
        if hasattr(self.config, 'preprocessing_baseline'):
            if isinstance(self.config.preprocessing_baseline, dict):
                preprocessing_config = self.config.preprocessing_baseline
            else:
                preprocessing_config = self.config.preprocessing_baseline.__dict__

        # Initialize preprocessor
        self.preprocessor = PreprocessorBaseline(
            base_path=base_path,
            random_seed=random_seed,
            config=preprocessing_config
        )
        
        # Initialize windower
        windowing_enabled = False
        save_windowed_data = False
        
        # Check if preprocessing_advanced exists in config
        if hasattr(self.config, 'preprocessing_advanced'):
            # Try different approaches to access the config
            if isinstance(self.config.preprocessing_advanced, dict):
                windowing_config = self.config.preprocessing_advanced.get('windowing', {})
            else:
                # If it's an object with attributes
                windowing_config = getattr(self.config.preprocessing_advanced, 'windowing', {})
                
            if windowing_config:
                if isinstance(windowing_config, dict):
                    windowing_enabled = windowing_config.get('enabled', False)
                    save_windowed_data = windowing_config.get('save_data', False)
                else:
                    windowing_enabled = getattr(windowing_config, 'enabled', False)
                    save_windowed_data = getattr(windowing_config, 'save_data', False)
        
        logger.info(f"Windowing enabled: {windowing_enabled}, Debug mode: {debug_mode}")
        
        if windowing_enabled:
            self.windower = Windower(
                base_path=base_path, 
                save_data=save_windowed_data,
                debug_mode=debug_mode
            )
            logger.info("Windower initialized for advanced preprocessing with debug mode: {debug_mode}")
    
    def _init_datasets(self) -> None:
        """Initialize datasets based on configuration."""

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
            # Extract task and dataset name
            task = dataset['task']
            name = dataset['name']
            
            # Check if windowing is enabled and should be applied
            windowing_enabled = False
            windowing_config = None
            
            if hasattr(self.config, 'preprocessing_advanced') and isinstance(self.config.preprocessing_advanced, dict):
                if 'windowing' in self.config.preprocessing_advanced:
                    windowing_config = self.config.preprocessing_advanced['windowing']
                    windowing_enabled = windowing_config.get('enabled', False)
            
            # Check if windowing is applicable for this task
            if task == "mortality":
                logger.warning(f"Windowing is not applicable for the mortality task. Skipping windowing for task = 'mortality'.")
            
            # If windowing is enabled and not a mortality task, try to load presaved windowed data first
            if windowing_enabled and task != "mortality" and self.windower is not None:
                logger.info(f"Attempting to load presaved windowed data for {dataset_id}")
                windowed_data = self.windower.window_data(
                    task=task,
                    dataset=name,
                    config=windowing_config
                )
                
                if windowed_data is not None:
                    # Successfully loaded presaved windowed data
                    # Drop 'stay_id' column if present in y sets
                    windowed_data = self._drop_stay_id_if_present(windowed_data)
                    
                    dataset['data'] = {
                        'X_train': windowed_data['train']['X'],
                        'X_val': windowed_data['val']['X'],
                        'X_test': windowed_data['test']['X'],
                        'y_train': windowed_data['train']['y'],
                        'y_val': windowed_data['val']['y'],
                        'y_test': windowed_data['test']['y']
                    }
                    dataset['loaded'] = True
                    logger.info(f"Successfully loaded presaved windowed data for {dataset_id}")
                    return True
                
                logger.info(f"No presaved windowed data found for {dataset_id}, falling back to regular loading")
            
            # If not using presaved windowed data, proceed with regular loading/preprocessing
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
                    save_data=self.config.preprocessing_baseline.get('save_data', True)
                )
                
                logger.info(f"Preprocessing completed for {dataset_id}")
            
            # Store the loaded data
            data_dict = {
                'train': {'X': X_train, 'y': y_train},
                'val': {'X': X_val, 'y': y_val},
                'test': {'X': X_test, 'y': y_test}
            }
            
            # Apply windowing if enabled and not already loaded from presaved files
            if windowing_enabled and task != "mortality" and self.windower is not None:
                logger.info(f"Applying windowing to {dataset_id}")
                
                windowed_data = self.windower.window_data(
                    task=task,
                    dataset=name,
                    config=windowing_config,
                    data_dict=data_dict
                )
                
                if windowed_data is not None:
                    data_dict = windowed_data
                    logger.info(f"Windowing applied to {dataset_id}")

            # Drop 'stay_id' column if present in y sets (after windowing)
            data_dict = self._drop_stay_id_if_present(data_dict)
            
            # Store the processed data
            dataset['data'] = {
                'X_train': data_dict['train']['X'],
                'X_val': data_dict['val']['X'],
                'X_test': data_dict['test']['X'],
                'y_train': data_dict['train']['y'],
                'y_val': data_dict['val']['y'],
                'y_test': data_dict['test']['y']
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

    def _drop_stay_id_if_present(self, data_dict: dict) -> dict:
            """
            Check if y sets have 'stay_id' column and drop it if present.
            
            Args:
                data_dict (dict): Dictionary containing X and y data splits
                
            Returns:
                dict: The modified data dictionary
            """
            for split in ['train', 'val', 'test']:
                if split in data_dict and 'y' in data_dict[split]:
                    y_data = data_dict[split]['y']
                    if isinstance(y_data, pd.DataFrame) and 'stay_id' in y_data.columns:
                        logger.info(f"Dropping 'stay_id' column from train/val/test labels")
                        data_dict[split]['y'] = y_data.drop(columns=['stay_id'])
            
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
        # If we pre-computed arrays, use them
        if hasattr(self, 'X_array') and hasattr(self, 'y_array'):
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