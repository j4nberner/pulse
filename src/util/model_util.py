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



# TODO: add prepare_data logic (handling whether tensors need 3d-conversion for windowed DL models or conversion to numpy arrays for ML models)

def prepare_data_for_model(
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