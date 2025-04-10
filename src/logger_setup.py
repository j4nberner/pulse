import os
import logging
from datetime import datetime
import wandb

logger = logging.getLogger("PULSE_logger")


def setup_logger():
    """Creates a logger that logs to both a file and the console."""
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("output", time_stamp)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"log_{time_stamp}.log")

    logger = logging.getLogger("PULSE_logger")
    logger.setLevel(logging.INFO)

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

        logger.info(f"Logging to file: {log_file}")

    return logger, log_dir


# Initialize wandb
def init_wandb(config):
    """Initialize Weights & Biases for experiment tracking"""
    try:
        wandb.init(
            entity=config.wandb["entity"],
            name=config.experiment_name,
            config={k: v for k, v in vars(config).items() if not k.startswith("_")},
        )
        logger.info("Weights & Biases initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Failed to initialize Weights & Biases: {e}")
        return False
