import logging
import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseModel
from src.util.config_util import set_seeds
from src.util.model_util import (EarlyStopping, initialize_weights,
                                 prepare_data_for_model_convdl,
                                 save_torch_model)

logger = logging.getLogger("PULSE_logger")


class CNNModel(PulseModel, nn.Module):
    """
    A Convolutional Neural Network (CNN) model for time series data.
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the CNN model.

        Args:
            params (Dict[str, Any]): Configuration parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        model_name = kwargs.pop("model_name", "CNNModel")
        trainer_name = params["trainer_name"]
        super().__init__(
            model_name=model_name,
            params=params,
            trainer_name=trainer_name,
            **kwargs,
        )
        nn.Module.__init__(self)

        # Define all required parameters
        required_params = [
            "output_shape",
            "kernel_size",
            "pool_size",
            "dropout_rate",
            "learning_rate",
            "num_epochs",
            "early_stopping_rounds",
            "save_checkpoint",
            "verbose",
            "grad_clip_max_norm",
            "scheduler_factor",
            "scheduler_patience",
            "scheduler_cooldown",
            "min_lr",
        ]

        # Check if all required parameters exist in config
        self.check_required_params(params, required_params)

        # Log the parameters being used
        logger.info("Initializing CNN with parameters: %s", self.params)

        # Set the number of channels based on the input shape
        self.params["num_channels"] = (
            10  # overwritten in trainer. needs to be > 1 for normalization to work
        )
        if params["preprocessing_advanced"]["windowing"]["enabled"]:
            self.params["window_size"] = params["preprocessing_advanced"]["windowing"][
                "data_window"
            ]
        else:
            self.params["window_size"] = 1  # Default to 1

        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the CNN model.
        """
        set_seeds(self.params["random_seed"])
        logger.debug(
            "Setting seed %d before CNN model initialization",
            self.params["random_seed"],
        )

        # -------------------------Define layers-------------------------
        self.conv1 = nn.Conv1d(
            in_channels=self.params["num_channels"],
            out_channels=self.params["num_channels"] * 4,
            kernel_size=self.params["kernel_size"][0],
            padding="same",
            stride=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.params["num_channels"] * 4,
            out_channels=self.params["num_channels"] * 2,
            kernel_size=self.params["kernel_size"][1],
            padding="same",
        )
        self.conv3 = nn.Conv1d(
            in_channels=self.params["num_channels"] * 2,
            out_channels=16,
            kernel_size=self.params["kernel_size"][2],
            padding="same",
        )

        self.norm1 = nn.BatchNorm1d(num_features=self.params["num_channels"] * 4)
        self.norm2 = nn.BatchNorm1d(num_features=self.params["num_channels"] * 2)
        self.norm3 = nn.BatchNorm1d(num_features=16)

        self.leaky_relu = nn.LeakyReLU()

        self.pool = nn.MaxPool1d(kernel_size=self.params["pool_size"])
        self.dropout = nn.Dropout(self.params["dropout_rate"])
        self.flatten = nn.Flatten()

        # Dummy forward to calculate fc1 input size
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.params["num_channels"], self.params["window_size"]
            )
            dummy_output = self._forward_features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, flattened_size // 2)
        self.fc1_bn = nn.BatchNorm1d(flattened_size // 2)
        self.fc2 = nn.Linear(flattened_size // 2, self.params["output_shape"])

        # Initialize weights with Xavier initialization
        self.apply(initialize_weights)

    # -------------------------Define layers-------------------------

    def _forward_features(self, x):
        x = self.leaky_relu(self.norm1(self.conv1(x)))
        x = self.leaky_relu(self.norm2(self.conv2(x)))
        x = self.leaky_relu(self.norm3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.leaky_relu(x)
        return self.fc2(x)

    def evaluate(self, data_loader, save_report: bool = False) -> float:
        """Evaluates the model on the given dataset."""
        set_seeds(self.params["random_seed"])
        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        verbose = self.params.get("verbose", 1)
        val_loss = []

        # Get the configured data converter
        converter = prepare_data_for_model_convdl(
            data_loader,
            self.params,
            architecture_type=self.params.get("architecture_type", "CNN"),
            task_name=self.task_name,
        )
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([data_loader.dataset.pos_weight])
        )
        criterion.to(self.device)
        # To identify num_channels: Get a sample batch and transform using the converter
        features, _ = next(iter(data_loader))
        transformed_features = converter.convert_batch_to_3d(features)

        # Load model from pretrained state if available and not in training mode
        if self.pretrained_model_path and self.mode != "train":
            # Update the model input shape based on the data
            self.params["num_channels"] = transformed_features.shape[1]
            self.params["window_size"] = transformed_features.shape[2]
            self._init_model()
            logger.info(self)
            logger.info(
                "Input shape to model (after transformation): %s",
                transformed_features.shape,
            )
            self.load_model_weights(self.pretrained_model_path)
        # Move model to device
        self.to(self.device)

        # Move model to device
        self.to(self.device)

        self.eval()

        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(data_loader):
                inputs = converter.convert_batch_to_3d(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)

                loss = criterion(outputs, labels).item()
                val_loss.append(loss)

                if verbose == 2:  # Verbose level 2: log every batch
                    logger.info("Testing - Batch %d: Loss = %.4f", batch + 1, loss)
                    if self.wandb:
                        wandb.log({"Test loss": loss})
                if verbose == 1:  # Verbose level 1: log every 10 batches
                    if batch % 10 == 0:
                        logger.info("Testing - Batch %d: Loss = %.4f", batch + 1, loss)
                    if self.wandb:
                        wandb.log({"Test loss": loss})

                metadata_dict = {
                    "batch": batch,
                    "prediction": outputs.cpu().numpy(),
                    "label": labels.cpu().numpy(),
                    "loss": loss,
                    "age": inputs[:, 0, 0].cpu().numpy(),
                    "sex": inputs[:, 1, 0].cpu().numpy(),
                    "height": inputs[:, 2, 0].cpu().numpy(),
                    "weight": inputs[:, 3, 0].cpu().numpy(),
                }
                # Append results to metrics tracker
                metrics_tracker.add_results(outputs.cpu().numpy(), labels.cpu().numpy())
                metrics_tracker.add_metadata_item(metadata_dict)

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.log_metadata(True)
            metrics_tracker.save_report()

        # Log results to console
        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        if self.wandb:
            wandb.log(
                {
                    "Test metrics": wandb.Table(
                        data=[
                            [metric, value]
                            for metric, value in metrics_tracker.summary.items()
                        ],
                        columns=["Metric", "Value"],
                    )
                }
            )

        # Only saveing the model when save_report is True -> when in test mode.
        if save_report:
            model_save_name = f"{self.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Log the model architecture and parameters to wandb
            if self.wandb:
                wandb.log(
                    {
                        "model_architecture": str(self.model),
                        "model_parameters": self.state_dict(),
                    }
                )

            save_torch_model(
                model_save_name, self, os.path.join(self.save_dir, "Models")
            )  # Save the final model

        return np.mean(val_loss)


class CNNTrainer:
    """Trainer for the CNN model."""

    def __init__(self, cnn_model, train_loader, val_loader):
        self.model = cnn_model
        self.params = cnn_model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seeds(self.model.params["random_seed"])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pos_weight = self.train_loader.dataset.pos_weight
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight])
        )  # inbalanced dataset
        self.optimizer = optim.Adam(
            self.model.parameters()
        )  # Update after model initialization
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(cnn_model.save_dir, "Models")
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.early_stopping = EarlyStopping(
            patience=self.params["early_stopping_rounds"],
            verbose=True,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            cooldown=self.params["scheduler_cooldown"],
            min_lr=self.params["min_lr"],
        )

        # Log optimizer and criterion
        logger.info("Using optimizer: %s", self.optimizer.__class__.__name__)
        logger.info(
            "Using criterion: %s with class weight adjustment",
            self.criterion.__class__.__name__,
        )

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Get the configured data converter
        self.converter = prepare_data_for_model_convdl(
            self.train_loader,
            self.params,
            architecture_type=self.params.get("architecture_type", "CNN"),
            task_name=self.task_name,
        )
        # To identify num_channels: Get a sample batch and transform using the converter
        features, _ = next(iter(self.train_loader))
        transformed_features = self.converter.convert_batch_to_3d(features)

        # Update the model input shape based on the data
        set_seeds(self.model.params["random_seed"])
        self.model.params["num_channels"] = transformed_features.shape[1]
        self.model.params["window_size"] = transformed_features.shape[2]
        self.model._init_model()
        logger.info(self.model)
        logger.info(
            "Input shape to model (after transformation): %s",
            transformed_features.shape,
        )

        # Try to load the model weights if they exist
        if self.model.pretrained_model_path:
            try:
                self.model.load_model_weights(self.model.pretrained_model_path)
                logger.info(
                    "Pretrained model weights loaded successfully from %s",
                    self.model.pretrained_model_path,
                )
            except Exception as e:
                logger.warning(
                    "Failed to load pretrained model weights: %s. Defaulting to random initialization.",
                    str(e),
                )
        logger.debug(
            "Using architecture type: %s for model: %s",
            self.params.get("architecture_type", "Unknown"),
            self.model.model_name,
        )

    def train(self):
        """Training loop."""
        set_seeds(self.model.params["random_seed"])

        num_epochs = self.params["num_epochs"]
        verbose = self.params.get("verbose", 1)

        self.optimizer = optim.Adam(
            self.model.parameters()
        )  # Update optimizer after model initialization

        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.train_epoch(epoch, verbose)
            logger.info("Epoch %d finished", epoch + 1)

            val_loss = self.model.evaluate(
                self.val_loader
            )  # Evaluate on validation set

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info(
                    "Early stopping triggered at epoch %d. Stopping training.",
                    epoch + 1,
                )
                break

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)
            logger.debug(
                "Learning rate after epoch %d: %f",
                epoch + 1,
                self.optimizer.param_groups[0]["lr"],
            )

        logger.info("Training finished.")
        self.early_stopping.load_best_model(self.model)  # Load the best model

    def train_epoch(self, epoch: int, verbose: int = 1) -> None:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            verbose (int): Verbosity level (0, 1, or 2).
        """
        self.model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs = self.converter.convert_batch_to_3d(inputs)

            inputs, labels = (
                inputs.to(self.device),
                labels.to(self.device).float(),
            )

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass, gradient clipping and optimize
            loss.backward()
            max_norm = self.params["grad_clip_max_norm"]
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_norm
            )
            if total_norm > max_norm:
                logger.info("Gradient norm %.4f clipped to %s", total_norm, max_norm)
            self.optimizer.step()

            running_loss += loss.item()

            # Reporting based on verbosity
            if verbose == 2 or (verbose == 1 and i % 100 == 99):
                loss_value = running_loss / (100 if verbose == 1 else 1)
                logger.info(
                    "Epoch %d, Batch %d: Loss = %.4f", epoch + 1, i + 1, loss_value
                )

                if self.wandb:
                    wandb.log({"train_loss": loss_value})

                running_loss = 0.0
