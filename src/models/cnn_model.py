from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    save_torch_model,
    prepare_data_for_model_dl,
)
from src.eval.metrics import MetricsTracker
from src.eval.metrics import calculate_all_metrics, calc_metric_stats

logger = logging.getLogger("PULSE_logger")


class CNNModel(PulseTemplateModel, nn.Module):
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
        self.save_dir = kwargs.get(f"output_dir", f"{os.getcwd()}/output")

        # Define all required parameters
        required_params = [
            "num_features",
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
        logger.info(f"Initializing CNN with parameters: {self.params}")

        # Set the number of channels based on the input shape
        if params["preprocessing_advanced"]["windowing"]["enabled"]:
            self.params["num_channels"] = params["preprocessing_advanced"]["windowing"][
                "data_window"
            ]
        else:
            self.params["num_channels"] = 1  # Default to 1 channel for 1 timestamp

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
            stride=1,
            padding=self.params["kernel_size"] // 2,
        )
        self.bn1 = nn.BatchNorm1d(self.params["num_channels"] * 4)
        self.leaky_relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=self.params["pool_size"])

        self.conv2 = nn.Conv1d(
            in_channels=self.params["num_channels"] * 4,
            out_channels=self.params["num_channels"] * 2,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm1d(self.params["num_channels"] * 2)
        self.conv3 = nn.Conv1d(
            in_channels=self.params["num_channels"] * 2,
            out_channels=self.params["num_channels"] * 1,
            kernel_size=5,
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(self.params["num_channels"] * 1)
        self.pool = nn.MaxPool1d(kernel_size=self.params["pool_size"])
        self.dropout = nn.Dropout(0.5)

        # Fully connected layer
        pooled_size = (self.params["num_features"] // self.params["pool_size"]) // 2
        # TODO: use pooled_size when input is fixed. hardcoded for now...
        self.fc1 = nn.Linear(
            self.params["num_channels"] * 49,
            self.params["num_channels"] * 49 // 2,
        )
        self.fc2 = nn.Linear(
            self.params["num_channels"] * 1 * 49 // 2,
            self.params["output_shape"],
        )
        # -------------------------Define layers-------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_features, seq_length]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_shape]


        """
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def set_trainer(self, trainer_name: str, train_dataloader, test_dataloader) -> None:
        """
        Sets the trainer for the CNN model.

        Args:
            trainer_name (str): The name of the trainer to be used.
            train_dataloader: The DataLoader object for the training dataset.
            test_dataloader: The DataLoader object for the testing dataset.

        Returns:
            None
        """
        self.trainer = CNNTrainer(self, train_dataloader, test_dataloader)


class CNNTrainer:
    """Trainer for the CNN model."""

    def __init__(self, cnn_model, train_dataloader, test_dataloader):
        self.model = cnn_model
        self.params = cnn_model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.BCEWithLogitsLoss()
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(cnn_model.save_dir, "Models")

        # Log optimizer and criterion
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Using criterion: {self.criterion.__class__.__name__}")

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Data preparation
        self._prepare_data()

    def train(self):
        """Training loop."""
        save_checkpoint = self.params["save_checkpoint"]
        num_epochs = self.params["num_epochs"]
        verbose = self.params.get("verbose", 1)

        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.train_epoch(epoch, verbose)
            logger.info(f"Epoch {epoch + 1} finished")
            # Save checkpoint every epoch
            checkpoint_name = f"{self.model.model_name}_epoch_{epoch + 1}"
            checkpoint_path = os.path.join(self.model_save_dir, "Checkpoints")
            if save_checkpoint != 0 and epoch % save_checkpoint == 0:
                save_torch_model(checkpoint_name, self.model, checkpoint_path)

        logger.info("Training finished.")
        self.evaluate(self.test_dataloader)
        save_torch_model(
            self.model.model_name, self.model, self.model_save_dir
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
        for i, (inputs, labels) in enumerate(self.train_dataloader):
            inputs = self._transform_features(inputs)
            # Move tensors to the same device as the model
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if verbose == 2 or (verbose == 1 and i % 10 == 9):
                logger.info(
                    f"Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / (10 if self.params.get('verbose', 1) == 1 else 1):.4f}"
                )
                if self.wandb:
                    wandb.log(
                        {
                            "loss": running_loss
                            / (10 if self.params.get("verbose", 1) == 1 else 1)
                        }
                    )
                if verbose == 1:
                    running_loss = 0.0

    def evaluate(self, val_dataloader):
        """
        Evaluates the model on the validation set.

        Args:
            val_dataloader: The DataLoader object for the validation dataset.
        """
        metrics_tracker = MetricsTracker(self.model.model_name, self.model.save_dir)
        verbose = self.params.get("verbose", 1)
        self.model.eval()

        with torch.no_grad():
            for batch, (inputs, labels) in enumerate(val_dataloader):
                inputs = self._transform_features(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)

                accuracy = (predicted == labels).sum().item() / labels.size(0)

                # Append results to metrics tracker
                metrics_tracker.add_results(
                    predicted.cpu().numpy(), labels.cpu().numpy()
                )
                if verbose == 2 or (verbose == 1 and batch % 10 == 9):
                    logger.info(
                        f"Evaluating batch {batch + 1}: " f"Accuracy = {accuracy}"
                    )

                    if self.wandb:
                        wandb.log({"accuracy": accuracy})

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()

    # TODO: Move this to a separate function in model_util.py
    def _prepare_data(self):
        """
        Prepare data for InceptionTime by ensuring it's in 3D format.
        """
        # Use the utility function from model_util.py
        data_prep_result = prepare_data_for_model_dl(
            self.train_dataloader, self.params, model_name=self.model.model_name
        )

        # Extract results
        self.reshape_needed = data_prep_result["reshape_needed"]
        self.convert_method = data_prep_result["convert_method"]
        self.converter = data_prep_result["converter"]

        # Log input data shape
        logger.info(f"Input data shape: {data_prep_result['data_shape']}")

    # TODO: Move this to a separate function in model_util.py
    def _transform_features(self, features):
        """Transform features to the correct format for the model."""
        # Apply the appropriate conversions
        if hasattr(self, "convert_method") and self.convert_method == "windowed_to_3d":
            features = self.converter.convert_batch_to_3d(features)
            features = features.permute(0, 2, 1)

        elif hasattr(self, "reshape_needed") and self.reshape_needed:
            # InceptionTime always uses CNN format (batch, channels, time_steps)
            features = features.unsqueeze(1)  # Add time dimension

        return features
