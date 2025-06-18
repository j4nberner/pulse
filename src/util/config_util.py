import logging
import os
import random
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

logger = logging.getLogger("PULSE_logger")


def load_config_with_models(base_config_path: str) -> OmegaConf:
    # Load the base YAML configuration file
    base_config = OmegaConf.load(base_config_path)

    # Get the list of model configuration file paths from the 'load_models' key in base_config
    model_files = base_config.get("load_models", [])

    # Create a dictionary to hold each model configuration
    models = {}
    for file_path in model_files:
        # Load the model configuration YAML file
        model_config = OmegaConf.load(file_path)

        # Extract the model name from the model configuration under the key 'name'
        model_name = model_config.get("name")
        if model_name is None:
            # If the 'name' key is missing, fall back to using the file name without extension.
            model_name = os.path.splitext(os.path.basename(file_path))[0]

        # Add global preprocessing configuration to each model config
        if "preprocessing_advanced" in base_config:
            model_config.params["preprocessing_advanced"] = (
                base_config.preprocessing_advanced
            )

        # Add ALL tasks to model config - this lets the training code select the current task
        if "tasks" in base_config:
            model_config.params["tasks"] = base_config.tasks

        # Add the loaded model config to the models dictionary under the extracted name
        models[model_name] = model_config

    # Add the models dictionary to the base config under the "models" key
    base_config.models = models

    return base_config


def save_config_file(config: OmegaConf, output_dir: str) -> None:
    """
    Copy the current configuration to the output directory.

    Args:
        config (OmegaConf): The configuration object to save.
        output_dir (str): The directory where the configuration file will be saved.
    """
    config_copy_path = os.path.join(output_dir, "config_copy.yaml")
    # Copy the configuration file to the output directory
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(config_copy_path), exist_ok=True)
    # Save the configuration to the output file
    OmegaConf.save(config, config_copy_path)
    logger.info("Configuration file copied to %s", config_copy_path)


def check_model_config_validity(model, config: OmegaConf) -> None:
    """
    This function checks for correct combination of keys and values in the configuration.

    Args:
        model: The model object to check.
        config (OmegaConf): The configuration object to validate.

    Raises:
        ValueError: If any required keys are missing or if the configuration is invalid.
    """
    if model.type == "convML":
        # Sanity check if data standardization was disabled
        if config.preprocessing_baseline.get("standardize"):
            logger.error(
                "Data standardization is enabled for convML models. Please disable it in the config."
            )
            sys.exit(1)

        # Check if debug_data_length is bigger than 999, to avoid having only one label in sets.
        if config.general.get("app_mode") == "debug":
            if config.general.get("debug_data_length", 0) < 1000:
                logger.error(
                    "debug_data_length for convML models should be at least 1000 to ensure proper label distribution. "
                    "Please set it to a higher value in the config."
                )
                sys.exit(1)

    if model.type == "convDL":
        # Sanity check if data standardization was enabled
        if not config.preprocessing_baseline.get("standardize"):
            logger.error(
                "Data standardization is not enabled for convDL models. Please enable it in the config."
            )
            sys.exit(1)

    if model.type == "LLM":
        # Sanity check if data standardization was disabled
        if config.preprocessing_baseline.get("standardize"):
            logger.error(
                "Data standardization is enabled for LLM models. Please disable it in the config."
            )
            sys.exit(1)


def get_pretrained_model_path(
    pretrained_model_list: list, task_name: str, dataset_name: str
) -> str:
    """
    Get the path to the pretrained model based on the provided configuration.

    Args:
        pretrained_model_list (dict): List containing pretrained model paths or parent folder.
        task_name (str): The name of the task.
        dataset_name (str): The name of the dataset.

    Returns:
        str: The path to the pretrained model.
    """
    if pretrained_model_list is None:
        return None

    pretrained_model_path = None
    # Check if the pretrained_model_list is a directory
    if os.path.isdir(pretrained_model_list):
        # If it's a directory, assume it contains multiple pretrained models
        pretrained_model_list = [
            os.path.join(pretrained_model_list, f)
            for f in os.listdir(pretrained_model_list)
            if os.path.isfile(os.path.join(pretrained_model_list, f))
        ]

    for path in pretrained_model_list:
        filename = os.path.basename(path)
        parts = filename.split("_")
        if len(parts) >= 4:
            task = parts[1]
            dataset = parts[2]
            if task == task_name and dataset == dataset_name:
                pretrained_model_path = path
                break

    return pretrained_model_path


# ------------------------------------
# Utilities for setting random seeds
# ------------------------------------


def set_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): The seed value to use
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For HuggingFace Transformers
    try:
        import transformers

        transformers.set_seed(seed)
    except (ImportError, AttributeError):
        pass


def get_deterministic_dataloader_args(seed):
    """
    Create a generator and worker_init_fn using the same seed for deterministic data loading.

    Args:
        seed (int): The base random seed

    Returns:
        dict: A dictionary containing 'generator' and 'worker_init_fn' for DataLoader
    """
    # Create and seed a generator for shuffle reproducibility
    g = torch.Generator()
    g.manual_seed(seed)

    # Create a worker_init_fn for worker process reproducibility
    def worker_init_fn(worker_id):
        # Each worker gets a different but deterministic seed derived from base seed
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Return both as a dictionary for easy unpacking
    return {"generator": g, "worker_init_fn": worker_init_fn}
