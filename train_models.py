import argparse
import os

import pandas as pd
import yaml
from torch.utils.data import DataLoader

from src.logger_setup import setup_logger, init_wandb
from src.data.dataloader import DatasetManager, TorchDatasetWrapper
from src.models.modelmanager import ModelManager
from src.util.slurm_util import copy_data_to_scratch, is_on_slurm, get_local_scratch_dir


# -------------------------Configure logging-------------------------

logger = setup_logger()


class TrainConfig:
    """Configuration settings for the training process."""

    def __init__(self, config_path: str):
        """Initialize training configuration."""
        self.config_path = config_path
        self.models = []
        self.tasks = []
        self.datasets = []
        self.metrics = []
        self.base_path = ""
        self.output_dir = r"output/results"
        self.experiment_name = (
            f"experiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.debug_mode = False
        self.general = {}  # Add container for general settings

    def load_from_file(self):
        """Load configuration from a YAML file."""
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            # if hasattr(self, key):
            setattr(self, key, value)

        # Set debug_mode from general section if it exists
        if hasattr(self, 'general') and isinstance(self.general, dict) and 'debug_mode' in self.general:
            self.debug_mode = self.general['debug_mode']
            logger.info(f"DEBUG MODE set to: {self.debug_mode}")

        logger.info("Loaded configuration from %s", self.config_path)


class ModelTrainer:
    """Core training functionality for ML/DL models and LLMs."""

    def __init__(self, config: TrainConfig):
        """
        Initialize the model trainer.

        Args:
            config (TrainConfig): Configuration object containing training settings.
        """
        self.config = config

        # Log debug mode status
        if self.config.debug_mode:
            logger.info("DEBUG MODE ACTIVE: Training will use limited dataset size")

        # -------------------- Copy data to local scratch (Slurm) --------------------
        if is_on_slurm() and self.config.general.get('use_scratch', False):
            logger.info("Running on Slurm, preparing to copy data to scratch space...")
            scratch_dir = get_local_scratch_dir()
            if scratch_dir:
                logger.info(f"Scratch directory available at: {scratch_dir}")
                # Update the config with scratch space paths
                self.config, data_copied = copy_data_to_scratch(self.config)
            else:
                logger.warning("No scratch directory found, using original data paths")

        self.dm = DatasetManager(config)
        self.mm = ModelManager(config.models)

        # Create output directory
        self.output_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {self.output_dir}")

    def run(self):
        """Run the training process for all configured models and datasets."""
        logger.info("Starting training process...")

        # Check if debug mode is enabled
        if self.config.debug_mode:
            logger.info("DEBUG MODE ACTIVE: Training will use only one batch")

        results = {}

        # Train and evaluate each model on each dataset
        for dataset_name, _ in self.dm.datasets.items():
            logger.info("Processing dataset: %s", dataset_name)
            results[dataset_name] = {}

            for model in self.mm.models:
                model_name = model.__class__.__name__
                trainer_name = model.trainer_name
                logger.info("Training model: %s on %s", model_name, dataset_name)

                try:
                    # Preprocess data for corresponding model. Returns X and y as pandas DataFrames
                    X_train, y_train = self.dm.get_preprocessed_data(
                        dataset_name, model_name, test=False
                    )
                    X_test, y_test = self.dm.get_preprocessed_data(
                        dataset_name, model_name, test=True
                    )

                    # Wrap with TorchDatasetWrapper
                    # TODO: This should not be applied to all models but rather only to traditional ML models that do not need tensors as input?
                    train_dataset = TorchDatasetWrapper(X_train, y_train)
                    test_dataset = TorchDatasetWrapper(X_test, y_test)

                    # Get batch size with fallback using getattr
                    if isinstance(self.config.benchmark_settings, dict):
                        # If benchmark_settings is a dictionary
                        batch_size = self.config.benchmark_settings.get('batch_size', 100)
                    else:
                        # If benchmark_settings is an object with attributes
                        batch_size = getattr(self.config.benchmark_settings, 'batch_size', 100)
                    
                    logger.info(f"Using batch size: {batch_size} for {model_name} on {dataset_name}")

                    # Now create the DataLoaders with the wrapped datasets
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

                    # If in debug mode, limit to a single batch for both training and testing
                    if self.config.debug_mode:
                        try:
                            # Get just the first batch
                            first_train_batch = next(iter(train_loader))
                            first_test_batch = next(iter(test_loader))
                            
                            # Convert to single-batch iterables
                            train_loader = [first_train_batch]
                            test_loader = [first_test_batch]
                            
                            logger.info(f"DEBUG MODE: Limited to single batch for {model_name}")
                        except StopIteration:
                            logger.warning(f"Dataset {dataset_name} is empty, cannot extract batch")

                    model.set_trainer(
                        trainer_name, train_loader, test_loader
                    )  # Set trainer for the model

                    # Train and evaluate the model -> model specific
                    model.trainer.train()

                except Exception as e:
                    logger.error(
                        "Error training %s on %s: %s", model_name, dataset_name, str(e)
                    )

        logger.info("Training process completed.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM ICU Prediction Benchmark")

    parser.add_argument(
        "--config",
        type=str,
        default="config_train.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize configuration
    config = TrainConfig(args.config)
    config.load_from_file()

    # Log if running on Slurm
    if is_on_slurm():
        logger.info(f"Running on Slurm cluster (Job ID: {os.getenv('SLURM_JOB_ID')})")

    if config.wandb["enabled"]:
        init_wandb(config)  # Initialize Weights & Biases

    # Run training
    trainer = ModelTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
