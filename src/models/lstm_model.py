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
    EarlyStopping,
    save_torch_model,
    prepare_data_for_model_dl,
)
from src.eval.metrics import MetricsTracker


logger = logging.getLogger("PULSE_logger")


class LSTMModel(PulseTemplateModel, nn.Module):
    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the LSTM model.

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
            "learning_rate",
            "num_epochs",
            "save_checkpoint",
            "verbose",
            "num_layers",  # Number of LSTM layers
            "output_shape",  # Size of output
            "hidden_size",  # Number of features in hidden state
            "dropout",  # Dropout rate
        ]

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        # Extract parameters from config
        self.params = params.copy()

        # Log the parameters being used
        logger.info(f"Initializing LSTM with parameters: {self.params}")

        # Set the number of channels based on the input shape
        self.input_size = 1  # overwritten in trainer
        if params["preprocessing_advanced"]["windowing"]["enabled"]:
            self.params["window_size"] = params["preprocessing_advanced"]["windowing"][
                "data_window"
            ]
        else:
            self.params["window_size"] = 1  # Default to 1

        self.hidden_size = self.params[
            "hidden_size"
        ]  # Number of features in hidden state. 32-256 is standard
        self.num_layers = self.params[
            "num_layers"
        ]  # Number of LSTM layers 1-4 is standard
        self.output_size = self.params["output_shape"]
        self.dropout = self.params["dropout"]

        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the LSTM model.
        """

        # -------------------------Define layers-------------------------
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

        # Fully connected layer for output
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        # -------------------------Define layers-------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        # x shape: [batch_size, seq_length, input_size]

        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        # Pass through fully connected layer
        out = self.fc(out)

        return out

    def set_trainer(
        self, trainer_name: str, train_loader, val_loader, test_loader
    ) -> None:
        """
        Sets the trainer for the LSTM model.

        Args:
            trainer_name (str): The name of the trainer to be used.
            train_loader: The DataLoader object for the training dataset.
            val_loader: The DataLoader object for the validation dataset.
            test_loader: The DataLoader object for the testing dataset.

        Returns:
            None
        """
        self.trainer = LSTMTrainer(self, train_loader, val_loader, test_loader)


class LSTMTrainer:
    """Trainer for the LSTM model."""

    def __init__(self, lstm_model, train_loader, val_loader, test_loader):
        self.model = lstm_model
        self.params = lstm_model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.params["learning_rate"]
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(lstm_model.save_dir, "Models")
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.early_stopping = EarlyStopping(
            patience=self.params["early_stopping_rounds"],
            delta=0.0,
        )

        # Log optimizer and criterion
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Using criterion: {self.criterion.__class__.__name__}")

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Data preparation
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

        # Get the number of channels from the transformed features
        input_dim = transformed_features.shape[-1]

        # Update the model input shape based on the data
        self.model.input_size = input_dim
        self.model._init_model()

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

        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.train_epoch(epoch, verbose)
            logger.info(f"Epoch {epoch + 1} finished")
            val_loss = self.evaluate(self.val_loader)
            self.early_stopping(val_loss, self.model)

        logger.info("Training finished.")
        self.early_stopping.load_best_model(self.model)  # Load the best model
        self.evaluate(self.test_loader, save_report=True)
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

        for i, (inputs, labels) in enumerate(self.train_loader):
            inputs = self.converter.convert_batch_to_3d(inputs)
            # Move tensors to the same device as the model
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Handle sequence dimension based on the input data shape
            if len(inputs.shape) == 2:  # [batch_size, features]
                inputs = inputs.unsqueeze(
                    1
                )  # Add sequence dimension [batch_size, seq_len=1, features]

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Reporting based on verbosity
            if verbose == 2 or (verbose == 1 and i % 10 == 9):
                loss_value = running_loss / (10 if verbose == 1 else 1)
                logger.info(
                    f"Epoch {epoch + 1}, Batch {i + 1}: Loss = {loss_value:.4f}"
                )

                if self.wandb:
                    wandb.log({"train_loss": loss_value})

                running_loss = 0.0

    def evaluate(self, test_loader, save_report: bool = False) -> float:
        """
        Evaluates the model on the test set.

        Args:
            test_loader: The DataLoader object for the testing dataset.
        """
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
            for batch, (inputs, labels) in enumerate(test_loader):
                inputs = self.converter.convert_batch_to_3d(inputs)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels).item()

                val_loss.append(loss)

                # Append results to metrics tracker
                metrics_tracker.add_results(outputs.cpu().numpy(), labels.cpu().numpy())
                if verbose == 2 or (verbose == 1 and batch % 10 == 9):
                    logger.info(f"Evaluating batch {batch + 1}: " f"Loss = {loss}")

                    if self.wandb:
                        wandb.log({"val_loss": loss})

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()
        return np.mean(val_loss)
