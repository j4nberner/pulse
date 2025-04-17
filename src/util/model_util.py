import logging
from typing import Any, Dict, Optional
import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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


def save_sklearn_model(model_name: str, model: Any, save_dir: str) -> None:
    """
    Save the trained sklearn model to disk.

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


def prepare_data_for_model_ml(
    train_dataloader, test_dataloader, logger_instance=None
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

    # Use provided logger or default
    log = logger_instance or logger

    # Extract data from dataloaders
    X_train, y_train = [], []
    X_test, y_test = [], []
    feature_names = []

    if isinstance(train_dataloader[0], pd.DataFrame):
        # If DataLoader is a DataFrame, extract features and labels directly
        X_train = np.array(train_dataloader[0].values)
        y_train = np.array(train_dataloader[1].values).squeeze()
        X_test = np.array(test_dataloader[0].values)
        y_test = np.array(test_dataloader[1].values).squeeze()
        feature_names = list(train_dataloader[0].columns)

    else:
        # Convert lists to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

    # Log shapes
    log.info(
        f"Prepared data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}"
    )
    log.info(f"Prepared data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Try to extract feature names from the original dataframe if available
    # feature_names = None

    # # Only try to extract feature names if not in debug mode
    # if not is_debug_mode and hasattr(train_dataloader.dataset, 'X') and isinstance(train_dataloader.dataset.X, pd.DataFrame):
    #     feature_names = list(train_dataloader.dataset.X.columns)
    #     log.info(f"Extracted {len(feature_names)} feature names from original DataFrame")

    # # Create fallback feature names if needed
    # if feature_names is None or len(feature_names) != X_train.shape[1]:
    #     feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    #     log.info(f"Using generated feature names (for debug mode)")

    # Return all processed data
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


# implement conditional conversion for mortality (always, because it is never windowed (maybe first adapt ordering of columns during transformation in preprocessing))


def prepare_data_for_model_dl(
    data_loader,
    config: Dict,
    model_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Any:
    """
    Prepare data for deep learning models by returning a configured data converter.

    Args:
        data_loader: DataLoader containing the input data
        config: Configuration dictionary with preprocessing settings
        model_name: Name of the model to determine format requirements
        logger_instance: Optional logger instance
        task_name: Name of the current task (e.g., "mortality", "aki")

    Returns:
        WindowedDataTo3D: Configured converter instance ready to transform batches
    """

    # Import the converter
    from src.preprocessing.preprocessing_advanced.windowing import WindowedDataTo3D

    # Create converter with model name and config
    converter = WindowedDataTo3D(
        model_name=model_name, config=config, task_name=task_name
    )

    try:
        # Get a batch to inspect shape
        features, _ = next(iter(data_loader))

        # Configure converter based on data shape
        if len(features.shape) == 3:
            # Data is already 3D
            converter.needs_conversion = False
            logger.info("Input data is already 3D, no conversion needed")
        else:
            # Check if windowing is enabled in config
            windowing_enabled = False
            if task_name == "mortality":
                windowing_enabled = True
            elif "preprocessing_advanced" in config:
                preprocessing_advanced = config["preprocessing_advanced"]
                if "windowing" in preprocessing_advanced:
                    windowing_config = preprocessing_advanced["windowing"]
                    if "enabled" in windowing_config:
                        windowing_enabled = bool(windowing_config["enabled"])

            # Configure the converter based on windowing status
            converter.configure_conversion(windowing_enabled, features.shape)

            if windowing_enabled:
                logger.info("Will use 3D windowed conversion for batches")
            else:
                logger.info("Will use simple reshaping for batches")

    except Exception as e:
        logger.error(f"Error preparing data converter: {e}")

    return converter


def apply_model_prompt_format(model_id, prompt):
    """
    Apply model-specific prompt formatting.

    Args:
        model_id (str): The ID of the model.
        prompt (str): The prompt to format.
    """
    # Example formatting for Llama3
    if model_id == "Llama3Model":
        formatted_prompt = f"<|USER|>{prompt}<|ASSISTANT|>"
    else:
        formatted_prompt = prompt  # No formatting needed for other models

    return formatted_prompt
