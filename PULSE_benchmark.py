import argparse
import gc
import os
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataloader import DatasetManager, TorchDatasetWrapper
from src.logger_setup import init_wandb, setup_logger
from src.models.modelmanager import ModelManager
from src.util.config_util import (
    load_config_with_models,
    save_config_file,
    set_seeds,
    get_deterministic_dataloader_args,
)
from src.util.slurm_util import copy_data_to_scratch, get_local_scratch_dir, is_on_slurm
from src.util.env_util import load_environment

logger, output_dir = setup_logger()


class ModelTrainer:
    """Core training functionality for convML/convDL models and LLMs."""

    def __init__(self, config: OmegaConf):
        """
        Initialize the model trainer.

        Args:
            config (TrainConfig): Configuration object containing training settings.
        """
        self.config = config
        self.config.output_dir = output_dir

        # Log general information
        logger.info("Initializing ModelTrainer with configuration:")
        logger.info("App Name: %s", config.general.app_name)
        logger.info("App Version: %s", config.general.app_version)
        logger.info("App Mode: %s", config.general.app_mode)
        logger.info("Logging Level: %s", config.general.logging_level)

        # Set random seeds for reproducibility
        random_seed = self.config.benchmark_settings.get("random_seed", 42)
        set_seeds(random_seed)
        logger.info("Setting random seed to %s for reproducibility", random_seed)

        # -------------------- Copy data to local scratch (Slurm) --------------------
        if is_on_slurm() and self.config.general.get("use_scratch", False):
            logger.info("Running on Slurm, preparing to copy data to scratch space...")
            scratch_dir = get_local_scratch_dir()
            if scratch_dir:
                logger.info("Scratch directory available at: %s", scratch_dir)
                # Update the config with scratch space paths
                self.config, _ = copy_data_to_scratch(self.config)
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

        # Train and evaluate each model on each dataset
        for task_dataset_name, _ in self.dm.datasets.items():
            logger.info("#" * 60)
            logger.info("Processing dataset: %s", task_dataset_name)

            # Extract task from dataset_name (format: task_dataset)
            task_name = task_dataset_name.split("_")[0]
            dataset_name = task_dataset_name.split("_")[-1]

            # Get fresh models for this dataset/task combination
            updated_models = self.mm.get_models_for_task(task_dataset_name)

            # Each fresh model is used only for this dataset
            for model in updated_models:
                model_name = model.__class__.__name__
                model.task_name = task_name
                model.dataset_name = dataset_name
                model.save_metadata = self.config.general.get("save_metadata", False)
                trainer_name = model.trainer_name
                logger.info("--" * 30)
                logger.info("Training model: %s on %s", model_name, task_dataset_name)

                # Initialize wandb tracing for this model/dataset/task combination
                if self.config.wandb.get("enabled", False):
                    # Create a unique run name for this model-dataset combination
                    run_name = f"{model_name}_{task_dataset_name}"
                    group_name = f"{model_name}_{timestamp}"
                    # Create wandb config as OmegaConf object
                    wandb_config = OmegaConf.create(
                        {
                            "group_name": group_name,
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
                    "print_stats": self.config.preprocessing_baseline.split_ratios.print_stats,
                    "model_type": model.type,
                }

                try:
                    # Initialize variables
                    X_train, y_train = None, None
                    X_val, y_val = None, None

                    if model.type == "convML":
                        # Sanity check if data standardization was disabled
                        if self.config.preprocessing_baseline.get("standardize"):
                            logger.error(
                                "Data standardization is enabled for convML models. Please disable it in the config."
                            )
                            sys.exit(1)

                    if model.type == "convDL":
                        # Sanity check if data standardization was enabled
                        if not self.config.preprocessing_baseline.get("standardize"):
                            logger.error(
                                "Data standardization is not enabled for convDL models. Please enable it in the config."
                            )
                            sys.exit(1)

                    if model.type == "LLM":
                        # Sanity check if data standardization was disabled
                        if self.config.preprocessing_baseline.get("standardize"):
                            logger.error(
                                "Data standardization is enabled for LLM models. Please disable it in the config."
                            )
                            sys.exit(1)

                        dm_kwargs.update(
                            {
                                "prompting_id": model.prompting_id,
                                "num_shots": self.config.prompting.get("shots", 0),
                                "fine_tuning": model.params.get("tuning", False),
                            }
                        )

                    # Check if this model requires an agent-based preprocessor
                    if (
                        model.type == "LLM"
                        and hasattr(model, "prompting_id")
                        and "agent" in model.prompting_id
                    ):
                        # Create an inference-only model for the agent - use cached instance if possible
                        agent_model = self.mm.create_agent_model(
                            model.__class__.__name__
                        )
                        if agent_model:
                            # Store reference to the agent model
                            model.agent_model = agent_model

                            # Pass to data manager
                            dm_kwargs["model_instance"] = agent_model
                            logger.info(f"Created agent model for {model.prompting_id}")

                    # Preprocess data for corresponding model
                    X_train, y_train, X_val, y_val, X_test, y_test = (
                        self.dm.get_preprocessed_data(
                            task_dataset_name, model_name, **dm_kwargs
                        )
                    )

                    # Log the shapes of the datasets
                    logger.info(
                        "Shapes - Train: %s, Val: %s, Test: %s",
                        X_train.shape,
                        X_val.shape,
                        X_test.shape,
                    )

                    # Check if we can reuse an already loaded model from the agent pipeline
                    loaded_model = None
                    if "loaded_model" in dm_kwargs:
                        loaded_model = dm_kwargs["loaded_model"]
                        if (
                            loaded_model
                            and hasattr(loaded_model, "is_loaded")
                            and loaded_model.is_loaded
                        ):
                            logger.info(
                                "Reusing already loaded model instance for evaluation"
                            )
                            model = loaded_model
                            # Update necessary properties for evaluation
                            model.task_name = task_name
                            model.dataset_name = dataset_name

                    # Choose the appropriate DataLoader based on model type
                    if model.type == "convML":
                        train_loader = (X_train, y_train)
                        val_loader = (X_val, y_val)
                        test_loader = (X_test, y_test)
                    elif model.type == "LLM":
                        # Passing the text and labels directly for LLMs
                        train_loader = (X_train, y_train)
                        val_loader = (X_val, y_val)
                        test_loader = (X_test, y_test)
                    elif model.type == "convDL":
                        # Wrap with TorchDatasetWrapper
                        train_dataset = TorchDatasetWrapper(X_train, y_train)
                        val_dataset = TorchDatasetWrapper(X_val, y_val)
                        test_dataset = TorchDatasetWrapper(X_test, y_test)

                        batch_size = getattr(
                            self.config.benchmark_settings, "batch_size"
                        )

                        logger.info(
                            "Using batch size: %s for %s on %s",
                            batch_size,
                            model_name,
                            task_dataset_name,
                        )

                        # Get the deterministic DataLoader arguments
                        random_seed = self.config.benchmark_settings.get("random_seed")
                        dataloader_args = get_deterministic_dataloader_args(random_seed)

                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            num_workers=4,  # Matches the number of requested CPU cores
                            pin_memory=False,  # Speeds up CPU-to-GPU transfers
                            prefetch_factor=2,  # Default value, can increase if GPU is idle
                            persistent_workers=True,  # Keeps workers alive between epochs
                            **dataloader_args,
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            **dataloader_args,
                        )
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            **dataloader_args,
                        )

                    else:
                        logger.error(
                            "Please specify a model type (convML, convDL, LLM) in the config"
                        )
                        sys.exit(1)


                    if self.config.general.app_mode == "count_tokens":
                        # Estimate number of tokens for LLMs. Implemented only for Llama3.
                        if model.model_name == "Llama3Model":
                            model.set_trainer(
                                trainer_name, train_loader, val_loader, test_loader, disable_model_load = True
                            )
                            model.trainer.estimate_nr_tokens()
                        else:
                            logger.warning(
                                "Token estimation is only applicable for LLM models."
                            )
                    else:
                        # Set trainer for the model and train
                        model.set_trainer(
                            trainer_name, train_loader, val_loader, test_loader
                        )
                        model.trainer.train()

                except Exception as e:
                    logger.error(
                        "Error training %s on %s: %s",
                        model_name,
                        task_dataset_name,
                        str(e),
                        exc_info=True,
                    )
                finally:
                    # Memory cleanup after training each model
                    if hasattr(model, "trainer"):
                        del model.trainer

                    # Clear variables that might hold large data
                    train_loader = val_loader = test_loader = None
                    X_train = y_train = X_val = y_val = X_test = y_test = None

                    # If using PyTorch with CUDA, empty the cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Force garbage collection
                    gc.collect()
                    logger.info("Memory cleaned up after training %s", model_name)

            # Memory cleanup after processing each task-dataset combination
            del updated_models
            self.dm.release_dataset_cache(
                task_dataset_name
            )  # Release dataset from cache
            gc.collect()
            logger.info("Memory cleaned up after processing %s", task_dataset_name)

        logger.info("Training process completed.")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLM ICU Prediction Benchmark")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_benchmark.yaml",
        help="Path to configuration file",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    config = load_config_with_models(args.config)
    config.output_dir = output_dir
    config.experiment_name = f"PULSE_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    save_config_file(config, output_dir)  # Save the configuration file

    # Log if running on Slurm
    if is_on_slurm():
        logger.info("Running on Slurm cluster (Job ID: %s)", os.getenv("SLURM_JOB_ID"))
        # Load .env from home directory
        env_path = os.path.expanduser("~/.env")
        logger.debug(f"Loading environment variables from {env_path}")
        load_environment(env_path)
    else:
        load_environment("secrets/.env")  # Load from default secrets folder

    # Run training
    trainer = ModelTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
