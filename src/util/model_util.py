import logging
from typing import Any, Dict, Optional
import os
import joblib
import torch
import torch.nn as nn

logger = logging.getLogger("PULSE_logger")


def save_torch_model(model_name: str, model: Any, save_dir: str) -> None:
    """Save the trained torch model to disk.

    Args:
        model_name: Name of the model to be saved
        model: The PyTorch model object to save
        save_dir: Parent directory where the model should be saved
    """

    try:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.pth")

        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), model_path)
        else:
            torch.save(model, model_path)

        logger.info(f"Model '{model_name}' saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model '{model_name}': {str(e)}")


# TODO: not tested
def save_sklearn_model(model_name: str, model: Any, save_dir: str) -> None:
    """Save the trained sklearn model to disk.

    Args:
        model_name: Name of the model to be saved
        model: The sklearn model object to save
        save_dir: Directory where the model should be saved
    """

    try:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.joblib")

        joblib.dump(model, model_path)

        logger.info(f"Model '{model_name}' saved to {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model '{model_name}': {str(e)}")


# TODO: not tested
def load_torch_model(model_name: str, save_dir: str) -> Any:
    """Load a PyTorch model from disk.

    Args:
        model_name: Name of the model to be loaded
        save_dir: Directory where the model is saved

    Returns:
        The loaded PyTorch model
    """

    try:
        model_path = os.path.join(save_dir, f"{model_name}.pth")
        model = torch.load(model_path)

        logger.info(f"Model '{model_name}' loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        return None


# TODO: not tested
def load_sklearn_model(model_name: str, save_dir: str) -> Any:
    """Load a sklearn model from disk.

    Args:
        model_name: Name of the model to be loaded
        save_dir: Directory where the model is saved

    Returns:
        The loaded sklearn model
    """

    try:
        model_path = os.path.join(save_dir, f"{model_name}.joblib")
        model = joblib.load(model_path)

        logger.info(f"Model '{model_name}' loaded from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {str(e)}")
        return None

def prepare_data_for_model_ml(
    train_dataloader,
    test_dataloader,
    logger_instance=None
) -> Dict[str, Any]:
    """
    Prepare data for machine learning models by converting PyTorch tensors
    from dataloaders to numpy arrays while preserving feature names.
    
    Args:
        train_dataloader: DataLoader containing the training data or list of data in debug mode
        test_dataloader: DataLoader containing the test data or list of data in debug mode
        logger_instance: Optional logger instance (uses default PULSE_logger if None)
    
    Returns:
        dict: Dictionary containing:
            - X_train: numpy array of training features
            - y_train: numpy array of training labels
            - X_test: numpy array of test features
            - y_test: numpy array of test labels
            - feature_names: list of feature names (if available)
    """
    import numpy as np
    import pandas as pd
    
    # Use provided logger or default
    log = logger_instance or logger
    
    # Extract data from dataloaders
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    # Check if we're in debug mode (dataloaders are lists)
    is_debug_mode = isinstance(train_dataloader, list)
    
    if is_debug_mode:
        # In debug mode, dataloaders are already list of batches
        for batch in train_dataloader:
            features, labels = batch
            X_train.extend(features.numpy())
            y_train.extend(labels.numpy().squeeze())
            
        for batch in test_dataloader:
            features, labels = batch
            X_test.extend(features.numpy())
            y_test.extend(labels.numpy().squeeze())
    else:
        # Process training data from DataLoader
        for batch in train_dataloader:
            features, labels = batch
            X_train.extend(features.numpy())
            y_train.extend(labels.numpy().squeeze())
        
        # Process test data from DataLoader
        for batch in test_dataloader:
            features, labels = batch
            X_test.extend(features.numpy())
            y_test.extend(labels.numpy().squeeze())
    
    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    # Log shapes
    log.info(f"Prepared data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
    log.info(f"Prepared data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Try to extract feature names from the original dataframe if available
    feature_names = None
    
    # Only try to extract feature names if not in debug mode
    if not is_debug_mode and hasattr(train_dataloader.dataset, 'X') and isinstance(train_dataloader.dataset.X, pd.DataFrame):
        feature_names = list(train_dataloader.dataset.X.columns)
        log.info(f"Extracted {len(feature_names)} feature names from original DataFrame")
    
    # Create fallback feature names if needed
    if feature_names is None or len(feature_names) != X_train.shape[1]:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        log.info(f"Using generated feature names (for debug mode)")
    
    # Return all processed data
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names
    }

def prepare_data_for_model_dl(
    data_loader,
    config: Dict,
    model_name: Optional[str] = None,
    logger_instance=None
) -> Dict:
    """
    Prepare data for deep learning models by determining appropriate conversions
    between 2D and 3D formats based on model requirements and input data shape.
    
    Args:
        data_loader: DataLoader containing the input data
        config: Configuration dictionary with preprocessing settings
        model_name: Name of the model to determine format requirements
        logger_instance: Optional logger instance (uses default PULSE_logger if None)
    
    Returns:
        dict: Configuration dictionary with:
            - reshape_needed (bool): Whether reshaping is needed for each batch
            - convert_method (str or None): Method to use for conversion ("windowed_to_3d" or None)
            - converter: Converter instance if needed (WindowedDataTo3D)
            - data_shape: Shape of the input data
    """
    # Use provided logger or default
    log = logger_instance or logger
    
    # Check if windowing is enabled in config
    windowing_enabled = False
    
    # Check for preprocessing_advanced config
    if "preprocessing_advanced" in config:
        preprocessing_advanced = config["preprocessing_advanced"]
        if "windowing" in preprocessing_advanced:
            windowing_config = preprocessing_advanced["windowing"]
            
            # Get the enabled value with proper type handling
            if "enabled" in windowing_config:
                enabled_value = windowing_config["enabled"]
                
                # Handle different boolean representations
                if isinstance(enabled_value, bool):
                    windowing_enabled = enabled_value
                else:
                    # Convert string/other types to boolean
                    enabled_str = str(enabled_value).lower()
                    windowing_enabled = (enabled_str == "true" or enabled_str == "1")
    
    log.info(f"Data preparation for model - Windowing enabled: {windowing_enabled}")
    
    result = {
        "reshape_needed": False,
        "convert_method": None,
        "converter": None,
        "data_shape": None
    }
    
    try:
        # Get a batch to inspect shape
        features, _ = next(iter(data_loader))
        result["data_shape"] = features.shape
        
        # Check data dimensionality and decide on conversion method
        if len(features.shape) == 3:
            log.info(f"Input data is already 3D with shape {features.shape}")
        else:
            log.info(f"Input data is 2D with shape {features.shape}, will convert to 3D")
            
            # Import only when needed to avoid circular imports
            from src.preprocessing.preprocessing_advanced.windowing import WindowedDataTo3D
            
            # Initialize the data converter
            converter = WindowedDataTo3D(
                logger=log,
                model_name=model_name,
                config=config
            )
            
            result["converter"] = converter
            
            # If windowing was already applied, use 3D conversion
            if windowing_enabled:
                result["convert_method"] = "windowed_to_3d"
                log.info("Will use WindowedDataTo3D converter for batches")
            else:
                # Simple reshaping needed (handled during training)
                result["reshape_needed"] = True
                log.info("Will use simple reshaping for batches")
        
    except Exception as e:
        log.error(f"Error preparing data: {e}")
    
    return result