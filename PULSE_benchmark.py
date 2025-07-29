import argparse
import gc
import os
import sys

import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.data.dataloader import DatasetManager, TorchDatasetWrapper
from src.logger_setup import init_wandb, setup_logger
from src.models.modelmanager import ModelManager
from src.util.config_util import (
    check_model_config_validity,
    get_deterministic_dataloader_args,
    get_pretrained_model_path,
    load_config_with_models,
    save_config_file,
    set_seeds,
)
from src.util.env_util import load_environment
from src.util.slurm_util import copy_data_to_scratch, get_local_scratch_dir, is_on_slurm

logger, output_dir = setup_logger()


class PulseBenchmark:
    """
    Core benchmark functionality for convML/convDL models and LLMs.
    This class initializes the benchmark with the provided configuration,
    sets up the dataset and model managers, and runs the training process if applicable.
    """

    def __init__(self, config: OmegaConf):
        """
        Initialize the Pulse Benchmark.

        Args:
            config (OmegaConf): Configuration object containing benchmark settings.
        """
        self.config = config
        self.config.output_dir = output_dir

        # Log general information
        logger.info("App Name: %s", config.general.app_name)
        logger.info("App Version: %s", config.general.app_version)
        logger.info("App Mode: %s", config.general.app_mode)
        logger.info("Logging Level: %s", config.general.logging_level)

        # Set random seeds for reproducibility
        self.random_seed = self.config.benchmark_settings.get("random_seed", 42)

        set_seeds(self.random_seed)
        logger.info("Setting random seed to %s for reproducibility", self.random_seed)

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
        """Run the benchmark process for all configured models and datasets."""
        logger.info("Starting benchmark process...")
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        # Train and/or evaluate each model on each dataset
        for task_dataset_name, _ in self.dm.datasets.items():
            logger.info("#" * 60)
            logger.info("Processing dataset: %s", task_dataset_name)

            # Extract task from dataset_name (format: task_dataset)
            task_name = self.dm.datasets[task_dataset_name]["task"]
            dataset_name = self.dm.datasets[task_dataset_name]["name"]

            # Get updated models for this dataset/task combination
            updated_models = self.mm.get_models_for_task(task_dataset_name)

            # Each updated model is used only for this dataset
            for model in updated_models:

                # Update model attributes for this task and dataset
                model.task_name = task_name
                model.dataset_name = dataset_name
                model.save_metadata = self.config.general.save_metadata
                pretrained_path_list = self.config.models[model.model_name].get(
                    "pretrained_model_paths", []
                )
                model.pretrained_model_path = get_pretrained_model_path(
                    pretrained_path_list,
                    model.task_name,
                    model.dataset_name,
                )

                logger.info("--" * 30)
                logger.info(
                    "Running model: %s on %s", model.model_name, task_dataset_name
                )

                # Initialize wandb tracing for this model/dataset/task combination
                if self.config.wandb.get("enabled", False):
                    # Create a unique run name for this model-dataset combination
                    run_name = f"{model.model_name}_{task_dataset_name}"
                    group_name = f"{model.model_name}_{timestamp}"
                    # Create wandb config as OmegaConf object
                    wandb_config = OmegaConf.create(
                        {
                            "group_name": group_name,
                            "model_name": model.model_name,
                            "run_name": run_name,
                        }
                    )
                    # Merge the configurations using OmegaConf
                    wandb_config = OmegaConf.merge(wandb_config, self.config)
                    init_wandb(wandb_config)

                try:
                    # Initialize variables
                    X_train, y_train = None, None
                    X_val, y_val = None, None

                    check_model_config_validity(model, self.config)

                    # Prepare model-specific arguments for the data manager
                    dm_kwargs = {
                        "model_type": model.type,
                    }
                    if model.type == "LLM":
                        dm_kwargs.update(
                            {
                                "fine_tuning": model.params.get("tuning", None),
                                "prompting_id": model.prompting_id,
                                "num_shots": self.config.prompting.get("shots", None),
                            }
                        )

                    # Preprocess data for corresponding model
                    X_train, y_train, X_val, y_val, X_test, y_test = (
                        self.dm.get_preprocessed_data(
                            task_dataset_name, model.model_name, model.mode, **dm_kwargs
                        )
                    )

                    # Set agent flag (specified in prompting preprocessor, agent initialization will happen lazily)
                    is_agent = self.dm.is_agent
                    model.is_agent = is_agent
                    if is_agent:
                        logger.debug("Agent flag set for %s", model.model_name)

                    #######################################################################

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
                        train_dataset = TorchDatasetWrapper(
                            X_train, y_train, self.dm.pos_weight
                        )
                        val_dataset = TorchDatasetWrapper(
                            X_val, y_val, self.dm.pos_weight
                        )
                        test_dataset = TorchDatasetWrapper(
                            X_test, y_test, self.dm.pos_weight
                        )

                        batch_size = getattr(
                            self.config.benchmark_settings, "batch_size"
                        )

                        logger.info(
                            "Using batch size: %s for %s on %s",
                            batch_size,
                            model.model_name,
                            task_dataset_name,
                        )

                        # Get the deterministic DataLoader arguments
                        dataloader_args = get_deterministic_dataloader_args(
                            self.random_seed
                        )

                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=False,
                            **dataloader_args,
                        )
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=0,
                            **dataloader_args,
                        )
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=0,
                            **dataloader_args,
                        )

                    else:
                        logger.error(
                            "Please specify a model type (convML, convDL, LLM) in the config"
                        )
                        sys.exit(1)

                    if (
                        self.config.general.app_mode == "count_tokens"
                        and model.type == "LLM"
                    ):
                        # Estimate number of tokens for LLMs.
                        model.estimate_nr_tokens(test_loader)

                    else:
                        # Train the model if specified in the config
                        if model.mode == "train":
                            # Set trainer for the model and train
                            model.set_trainer(
                                model.trainer_name,
                                model,
                                train_loader,
                                val_loader,
                            )
                            model.trainer.train()

                        if model.batch_processing:
                            # If batch processing is enabled, evaluate in batches
                            logger.info(
                                "Batch processing enabled for %s on %s",
                                model.model_name,
                                task_dataset_name,
                            )
                            model.evaluate_batched(test_loader, save_report=True)

                        else:
                            model.evaluate(test_loader, save_report=True)

                        ### Uncomment to use for system message test.
                        # model.evaluate_sys_msgs(test_loader, save_report=True)

                        ### Offline evaluation option for claude if batch processing was used and not evaluated ###
                        # -> pass batch_id to the model from here.
                        # model.evaluate_batched_offline(test_loader, save_report=True, batch_id=batch_id)

                except Exception as e:
                    logger.error(
                        "Unexpected error running %s on %s: %s",
                        model.model_name,
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

                    # Force garbage collection
                    gc.collect()
                    logger.info("Memory cleaned up after running %s", model.model_name)

            # Memory cleanup after processing each task-dataset combination
            del updated_models
            self.dm.release_dataset_cache(
                task_dataset_name
            )  # Release dataset from cache
            gc.collect()
            logger.info("Memory cleaned up after processing %s", task_dataset_name)

        logger.info("Benchmark process completed.")


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
        logger.debug("Loading environment variables from %s", env_path)
        load_environment(env_path)
    else:
        load_environment("secrets/.env")  # Load from default secrets folder

    # Run training
    pulse_bench = PulseBenchmark(config)
    pulse_bench.run()


if __name__ == "__main__":
    main()
