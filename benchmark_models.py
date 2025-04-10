import argparse
import os
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import shutil

from src.logger_setup import setup_logger, init_wandb
from src.data.dataloader import DatasetManager, TorchDatasetWrapper
from src.models.modelmanager import ModelManager
from src.util.slurm_util import copy_data_to_scratch, is_on_slurm, get_local_scratch_dir
from src.util.config_util import load_config_with_models, save_config_file


logger, output_dir = setup_logger()


class ModelTrainer:
    """Core training functionality for ML/DL models and LLMs."""

    def __init__(self, config: OmegaConf):
        """
        Initialize the model trainer.

        Args:
            config (TrainConfig): Configuration object containing training settings.
        """
        self.config = config

        # Log debug mode status
        if self.config.general.debug_mode:
            logger.info("DEBUG MODE ACTIVE: Training will use limited dataset size")

        # -------------------- Copy data to local scratch (Slurm) --------------------
        if is_on_slurm() and self.config.general.get("use_scratch", False):
            logger.info("Running on Slurm, preparing to copy data to scratch space...")
            scratch_dir = get_local_scratch_dir()
            if scratch_dir:
                logger.info(f"Scratch directory available at: {scratch_dir}")
                # Update the config with scratch space paths
                self.config, data_copied = copy_data_to_scratch(self.config)
            else:
                logger.warning("No scratch directory found, using original data paths")

        logger.info("---------------Initializing Dataset Manager---------------")
        self.dm = DatasetManager(self.config)
        logger.info("---------------Initializing Model Manager---------------")
        self.mm = ModelManager(
            self.config.models, wandb=config.wandb, output_dir=config.output_dir
        )

    def run(self):
        """Run the training process for all configured models and datasets."""
        logger.info("Starting training process...")

        # Check if debug mode is enabled
        if self.config.general.debug_mode:
            logger.info("DEBUG MODE ACTIVE: Training will use only one batch")

        results = {}

        # Train and evaluate each model on each dataset
        for dataset_name, _ in self.dm.datasets.items():
            logger.info("Processing dataset: %s", dataset_name)
            results[dataset_name] = {}

            for model in self.mm.models:
                model_name = model.__class__.__name__
                trainer_name = model.trainer_name
                logger.info("--" * 30)
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
                    train_dataset = TorchDatasetWrapper(X_train, y_train)
                    test_dataset = TorchDatasetWrapper(X_test, y_test)

                    # Get batch size with fallback using getattr
                    if isinstance(self.config.benchmark_settings, dict):
                        # If benchmark_settings is a dictionary
                        batch_size = self.config.benchmark_settings.get(
                            "batch_size", 100
                        )
                    else:
                        # If benchmark_settings is an object with attributes
                        batch_size = getattr(
                            self.config.benchmark_settings, "batch_size", 100
                        )

                    logger.info(
                        f"Using batch size: {batch_size} for {model_name} on {dataset_name}"
                    )

                    # Now create the DataLoaders with the wrapped datasets
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True,
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        drop_last=True,
                    )

                    # If in debug mode, limit to a single batch for both training and testing
                    if self.config.general.debug_mode:
                        try:
                            # Get just the first batch
                            first_train_batch = next(iter(train_loader))
                            first_test_batch = next(iter(test_loader))

                            # Convert to single-batch iterables
                            train_loader = [first_train_batch]
                            test_loader = [first_test_batch]

                            logger.info(
                                f"DEBUG MODE: Limited to single batch for {model_name}"
                            )
                        except StopIteration:
                            logger.warning(
                                f"Dataset {dataset_name} is empty, cannot extract batch"
                            )

                    # Set trainer for the model
                    model.set_trainer(trainer_name, train_loader, test_loader)
                    # Train and evaluate the model -> model specific
                    model.trainer.train()

                except Exception as e:
                    logger.error(
                        "Error training %s on %s: %s",
                        model_name,
                        dataset_name,
                        str(e),
                        exc_info=True,
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
    config = load_config_with_models(args.config)
    config.output_dir = output_dir
    config.experiment_name = (
        f"experiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    )
    save_config_file(config, output_dir)  # Save the configuration file

    # Log if running on Slurm
    if is_on_slurm():
        logger.info(f"Running on Slurm cluster (Job ID: {os.getenv('SLURM_JOB_ID')})")

    if config.wandb["enabled"]:
        init_wandb(config)  # Initialize Weights & Biases

    # Run Benchmarking
    trainer = ModelTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
