import logging
import os
from datetime import datetime

import yaml
from omegaconf import OmegaConf

import wandb

logger = logging.getLogger("PULSE_logger")


def setup_logger():
    """Creates a logger that logs to both a file and the console."""
    # Load the configuration file
    try:
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
        config_path = os.path.join(parent_dir, "configs", "config_benchmark.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
            config = OmegaConf.create(config_dict)
        else:
            raise FileNotFoundError(f"Config file not found at: {config_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("output", time_stamp)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"log_{time_stamp}.log")

    logger = logging.getLogger("PULSE_logger")

    # Set log level based on debug_logging config
    levelnamesmapping = logging.getLevelNamesMapping()
    logger_level = config.general.logging_level if "general" in config else "INFO"
    logger.setLevel(levelnamesmapping.get(logger_level, logging.INFO))

    # **Check if handlers already exist to prevent duplication**
    if not logger.hasHandlers():
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info("Logging to file: %s", log_file)

    return logger, log_dir


# Initialize wandb
def init_wandb(config: OmegaConf) -> bool:
    """
    Initialize Weights & Biases for experiment tracking

    Args:
        config (OmegaConf): Configuration object containing wandb settings.
    """
    if wandb.run is not None:
        wandb.finish()
    try:
        wandb.init(
            entity=config.wandb["entity"],  # needed for wandb
            name=config.get("run_name", None),  # optional run name
            group=config.get("group_name", None),  # optional group name
            config={k: v for k, v in vars(config).items() if not k.startswith("_")},
            reinit=True,
            settings=wandb.Settings(_disable_stats=True),
        )
        # Log model architecture if available
        # if hasattr(model, "get_config"):
        #     wandb.config.update(model.get_config())

        logger.info("Weights & Biases initialized successfully")
        return True
    except Exception as e:
        logger.warning("Failed to initialize Weights & Biases: %s", e)
        return False
