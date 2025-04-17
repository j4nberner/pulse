import os
import shutil
from omegaconf import OmegaConf
import logging

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
            model_config.params["preprocessing_advanced"] = base_config.preprocessing_advanced
        
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


# Example usage:
if __name__ == "__main__":
    config = load_config_with_models("path/to/base_config.yaml")
    print(OmegaConf.to_yaml(config))
