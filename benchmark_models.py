import argparse
import os
import shutil

import pandas as pd
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataloader import DatasetManager, TorchDatasetWrapper
from src.logger_setup import init_wandb, setup_logger
from src.models.modelmanager import ModelManager
from src.util.config_util import load_config_with_models, save_config_file
from src.util.slurm_util import (copy_data_to_scratch, get_local_scratch_dir,
                                 is_on_slurm)

logger, output_dir = setup_logger()


class ModelBenchmark:
    """Core benchmark functionality for ML/DL models and LLMs."""

    def __init__(self, config: OmegaConf):
        """
        Initialize the model benchmark.

        Args:
            config (TrainConfig): Configuration object containing training settings.
        """
        raise NotImplementedError("ModelBenchmark is not implemented yet.")

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
        """Run the benchmark process for all configured models and datasets."""
        logger.info("Starting benchmarking process...")

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
                    X_test, y_test = self.dm.get_preprocessed_data(
                        dataset_name,
                        model_name,
                        mode="test",
                        dataset=self.config.datasets[0],
                        task=self.config.tasks[0],
                        debug=self.config.general.debug_mode,
                    )

                    # Check the model type
                    if model.type == "llm":
                        test_loader = zip(X_test["text"], y_test["label"])
                    elif model.type == "ml":
                        # Tradional ML model
                        test_loader = (X_test, y_test)
                    else:
                        # DL model
                        # Wrap with TorchDatasetWrapper
                        test_dataset = TorchDatasetWrapper(X_test, y_test)
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=self.config.general.batch_size,
                            shuffle=True,
                            drop_last=True,
                        )

                    # Set trainer for the model
                    model.eval(test_loader)

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

    # Run training
    bench = ModelBenchmark(config)
    bench.run()


if __name__ == "__main__":
    main()
