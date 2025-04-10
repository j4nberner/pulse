import logging
from typing import Any
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
        save_dir: Directory where the model should be saved
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
