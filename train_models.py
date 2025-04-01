import argparse
import os

import pandas as pd
import yaml
from torch.utils.data import DataLoader

from src.logger_setup import setup_logger, init_wandb
from src.data.dataloader import DatasetManager
from src.models.modelmanager import ModelManager


# -------------------------Configure logging-------------------------

logger = setup_logger()


class TrainConfig:
    """Configuration settings for the training process."""

    def __init__(self, config_path: str):
        """Initialize training configuration."""
        self.config_path = config_path
        self.models = []
        self.datasets = []
        self.metrics = ["accuracy", "auroc", "auprc", "f1_score"]
        self.output_dir = r"output/results"
        self.experiment_name = (
            f"experiment_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )

    def load_from_file(self):
        """Load configuration from a YAML file."""
        with open(self.config_path, "r") as f:
            config_data = yaml.safe_load(f)

        for key, value in config_data.items():
            # if hasattr(self, key):
            setattr(self, key, value)

        logger.info("Loaded configuration from %s", self.config_path)


class ModelTrainer:
    """Core training functionality for ML/DL models and LLMs."""

    def __init__(self, config: TrainConfig):
        """Initialize the model trainer."""
        self.config = config
        self.dm = DatasetManager(config.datasets)
        self.mm = ModelManager(config.models)

        # Create output directory
        self.output_dir = os.path.join(config.output_dir, config.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Results will be saved to {self.output_dir}")

    def run(self):
        """Run the training process for all configured models and datasets."""
        logger.info("Starting training process...")
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
                    # Preprocess data for corresponding model. Returns torch dataset.
                    train_dataset = self.dm.get_preprocessed_data(
                        dataset_name, model_name, test=False
                    )
                    test_dataset = self.dm.get_preprocessed_data(
                        dataset_name, model_name, test=True
                    )

                    # Torch dataloaders -> might need custom Loaders for some models
                    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
    if config.wandb["enabled"]:
        init_wandb(config)  # Initialize Weights & Biases

    # Run training
    trainer = ModelTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
