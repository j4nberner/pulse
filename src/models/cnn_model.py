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
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    EarlyStopping,
    prepare_data_for_model_dl,
    save_torch_model,
)

logger = logging.getLogger("PULSE_logger")


class CNNModel(PulseTemplateModel, nn.Module):
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
        # For trainer_name we still require it to be explicitly in the params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")

        # Use the class name as model_name if not provided in params
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)
        nn.Module.__init__(self)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Define all required parameters
        required_params = [
            "output_shape",
            "kernel_size",
            "pool_size",
            "learning_rate",
            "num_epochs",
            "save_checkpoint",
            "verbose",
        ]

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        # Extract parameters from config
        self.params = params  # {param: params[param] for param in required_params}

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

        # -------------------------Define layers-------------------------
        self.conv1 = nn.Conv1d(
            in_channels=self.params["num_channels"],
            out_channels=self.params["num_channels"] * 4,
            kernel_size=self.params["kernel_size"],
            padding=self.params["kernel_size"] // 2,
            stride=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.params["num_channels"] * 4,
            out_channels=self.params["num_channels"] * 2,
            kernel_size=self.params["kernel_size"],
            padding=self.params["kernel_size"] // 2,
        )
        self.conv3 = nn.Conv1d(
            in_channels=self.params["num_channels"] * 2,
            out_channels=16,
            kernel_size=self.params["kernel_size"],
            padding=self.params["kernel_size"] // 2,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=1, num_channels=self.params["num_channels"] * 4
        )
        self.norm2 = nn.GroupNorm(
            num_groups=1, num_channels=self.params["num_channels"] * 2
        )
        self.norm3 = nn.GroupNorm(num_groups=1, num_channels=16)

        self.leaky_relu = nn.ReLU()

        self.pool = nn.MaxPool1d(kernel_size=self.params["pool_size"])
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()

        # Dummy forward to calculate fc1 input size
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.params["num_channels"], self.params["window_size"]
            )
            dummy_output = self._forward_features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(flattened_size, flattened_size // 2)
        self.fc2 = nn.Linear(flattened_size // 2, self.params["output_shape"])

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
        x = self.leaky_relu(self.fc1(x))
        return self.fc2(x)

    def set_trainer(
        self, trainer_name: str, train_loader, val_loader, test_loader
    ) -> None:
        """
        Sets the trainer for the CNN model.

        Args:
            trainer_name (str): The name of the trainer to be used.
            train_loader: The DataLoader object for the training dataset.
            test_loader: The DataLoader object for the testing dataset.

        Returns:
            None
        """
        self.trainer = CNNTrainer(self, train_loader, val_loader, test_loader)


class CNNTrainer:
    """Trainer for the CNN model."""

    def __init__(self, cnn_model, train_loader, val_loader, test_loader):
        self.model = cnn_model
        self.params = cnn_model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(1.0)
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
            delta=0.0,
        )

        # Log optimizer and criterion
        logger.info("Using optimizer: %s", self.optimizer.__class__.__name__)
        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Get the configured data converter
        self.converter = prepare_data_for_model_dl(
            self.train_loader,
            self.params,
            model_name=self.model.model_name,
            task_name=self.task_name,
        )
        # To identify num_channels: Get a sample batch and transform using the converter
        features, _ = next(iter(self.train_loader))
        transformed_features = self.converter.convert_batch_to_3d(features)

        # Update the model input shape based on the data
        self.model.params["num_channels"] = transformed_features.shape[1]
        self.model.params["window_size"] = transformed_features.shape[2]
        self.model._init_model()

        logger.info(
            "Input shape to model (after transformation): %s",
            transformed_features.shape,
        )

        # Try to load the model weights if they exist
        if self.model.pretrained_model_path:
            try:
                self.model.load_model_weights(self.model.pretrained_model_path)
            except Exception as e:
                logger.warning(
                    "Failed to load pretrained model weights: %s. Defaulting to random initialization.",
                    str(e),
                )

    def train(self):
        """Training loop."""
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
            val_loss = self.evaluate(self.val_loader)  # Evaluate on validation set
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                logger.info(
                    "Early stopping triggered at epoch %d. Stopping training.",
                    epoch + 1,
                )
                break

        self.early_stopping.load_best_model(self.model)  # Load the best model
        self.evaluate(
            self.test_loader, save_report=True
        )  # Evaluate on test set and save metrics
        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        # Log the model architecture and parameters to wandb
        if self.wandb:
            wandb.log(
                {
                    "model_architecture": str(self.model),
                    "model_parameters": self.model.state_dict(),
                }
            )
        save_torch_model(
            model_save_name, self.model, self.model.save_dir
        )  # Save the final model

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
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if verbose == 2:  # Verbose level 2: log every batch
                logger.info(
                    "Training - Epoch %d, Batch %d: Loss = %.4f",
                    epoch + 1,
                    i + 1,
                    loss.item(),
                )
                if self.wandb:
                    wandb.log({"train_loss": loss.item()})
            elif verbose == 1:
                if i % 10 == 9:
                    logger.info(
                        "Training - Epoch %d, Batch %d: Loss = %.4f",
                        epoch + 1,
                        i + 1,
                        running_loss / 10,
                    )
                    if self.wandb:
                        wandb.log({"train_loss": running_loss / 10})
                    running_loss = 0.0

    def evaluate(self, data_loader, save_report: bool = False) -> float:
        """Evaluates the model on the given dataset."""
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose = self.params.get("verbose", 1)
        self.model.eval()
        val_loss = []

        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(data_loader):
                inputs = self.converter.convert_batch_to_3d(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels).item()
                val_loss.append(loss)

                # Append results to metrics tracker
                metrics_tracker.add_results(outputs.cpu().numpy(), labels.cpu().numpy())

                if verbose == 2:  # Verbose level 2: log every batch
                    logger.info("Testing - Batch %d: Loss = %.4f", batch + 1, loss)
                    if self.wandb:
                        wandb.log({"Test loss": loss})
                if verbose == 1:  # Verbose level 1: log every 10 batches
                    if batch % 10 == 0:
                        logger.info("Testing - Batch %d: Loss = %.4f", batch + 1, loss)
                    if self.wandb:
                        wandb.log({"Test loss": loss})

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        # Log results to console
        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return np.mean(val_loss)
