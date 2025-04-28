import argparse
import os
import sys

import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataloader import DatasetManager, TorchDatasetWrapper
from src.logger_setup import init_wandb, setup_logger
from src.models.modelmanager import ModelManager
from src.util.config_util import load_config_with_models, save_config_file
from src.util.slurm_util import copy_data_to_scratch, get_local_scratch_dir, is_on_slurm

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
        self.mm = ModelManager(self.config)

    def run(self):
        """Run the training process for all configured models and datasets."""
        logger.info("Starting training process...")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Check if debug mode is enabled
        if self.config.general.debug_mode:
            logger.info("DEBUG MODE ACTIVE: Training will use only one batch")

        # Train and evaluate each model on each dataset
        for task_dataset_name, _ in self.dm.datasets.items():
            logger.info("#" * 60)
            logger.info(f"Processing dataset: {task_dataset_name}")
            # Create a group name for wandb using task_dataset_name and timestamp
            wand_group_name = f"{task_dataset_name}_{timestamp}"

            # Extract task from dataset_name (format: task_dataset)
            task_name = task_dataset_name.split("_")[0]
            dataset_name = task_dataset_name.split("_")[-1]

            # Get fresh models for this dataset/task combination
            fresh_models = self.mm.get_models_for_task(task_dataset_name)

            # Each fresh model is used only for this dataset
            for model in fresh_models:
                model_name = model.__class__.__name__
                model.task_name = task_name
                model.dataset_name = dataset_name
                trainer_name = model.trainer_name
                logger.info("--" * 30)
                logger.info(f"Training model: {model_name} on {task_dataset_name}")

                # Initialize wandb tracing for this model/dataset/task combination
                if self.config.wandb.get("enabled", False):
                    # Create a unique run name for this model-dataset combination
                    run_name = f"{model_name}_{task_dataset_name}"
                    # Create wandb config as OmegaConf object
                    wandb_config = OmegaConf.create(
                        {
                            "task_dataset_name": wand_group_name,
                            "model_name": model_name,
                            "run_name": run_name,
                        }
                    )
                    # Merge the configurations using OmegaConf
                    wandb_config = OmegaConf.merge(wandb_config, self.config)
                    init_wandb(wandb_config)

                dm_kwargs = {
                    "dataset": self.config.datasets[0],
                    "task": self.config.tasks[0],
                    "debug": self.config.general.debug_mode,
                }

                try:
                    # Initialize variables
                    X_train, y_train = None, None
                    X_val, y_val = None, None

                    if model.type == "LLM":
                        dm_kwargs.append(
                            {
                                "preprocessing_id": model.preprocessing_id,
                                "num_shots": model.params.get("shots", None),
                            }
                        )
                    # Preprocess data for corresponding model
                    X_train, y_train = self.dm.get_preprocessed_data(
                        task_dataset_name, model_name, mode="train", **dm_kwargs
                    )
                    X_val, y_val = self.dm.get_preprocessed_data(
                        task_dataset_name, model_name, mode="val", **dm_kwargs
                    )
                    X_test, y_test = self.dm.get_preprocessed_data(
                        task_dataset_name,
                        model_name,
                        mode="test",
                        **dm_kwargs,
                        limit_test_set=True,
                        print_stats=False,
                    )

                    # Choose the appropriate DataLoader based on model type
                    if model.type == "ML":
                        train_loader = (X_train, y_train)
                        val_loader = (X_val, y_val)
                        test_loader = (X_test, y_test)
                    elif model.type == "LLM":
                        # Passing the text and labels directly for LLMs
                        train_loader = (pd.DataFrame(), pd.DataFrame())
                        val_loader = (pd.DataFrame(), pd.DataFrame())
                        test_loader = (X_test, y_test)
                    elif model.type == "DL":
                        # Wrap with TorchDatasetWrapper
                        train_dataset = TorchDatasetWrapper(X_train, y_train)
                        val_dataset = TorchDatasetWrapper(X_val, y_val)
                        test_dataset = TorchDatasetWrapper(X_test, y_test)

                        # Get batch size with fallback using getattr
                        if isinstance(self.config.benchmark_settings, dict):
                            batch_size = self.config.benchmark_settings.get(
                                "batch_size", 100
                            )
                        else:
                            batch_size = getattr(
                                self.config.benchmark_settings, "batch_size", 100
                            )

                        logger.info(
                            f"Using batch size: {batch_size} for {model_name} on {task_dataset_name}"
                        )

                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                        )
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                        )
                    else:
                        logger.error(
                            "Please specify a model type (ML, DL, LLM) in the config"
                        )
                        sys.exit(1)

                    # Set trainer for the model and train
                    model.set_trainer(
                        trainer_name, train_loader, val_loader, test_loader
                    )
                    model.trainer.train()

                except Exception as e:
                    logger.error(
                        f"Error training {model_name} on {task_dataset_name}: {str(e)}",
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

    # Run training
    trainer = ModelTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
