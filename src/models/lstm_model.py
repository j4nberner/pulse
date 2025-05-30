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
from src.util.config_util import set_seeds
from src.util.model_util import (
    EarlyStopping,
    calculate_pos_weight,
    prepare_data_for_model_convdl,
    save_torch_model,
    initialize_weights,
)

logger = logging.getLogger("PULSE_logger")


class LSTMModel(PulseTemplateModel, nn.Module):
    """
    LSTM model for time series data.
    """

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

        # Extract model_name from kwargs if it exists (passed from ModelManager)
        if "model_name" in kwargs:
            self.model_name = kwargs.pop("model_name")  # Remove to avoid duplication
        else:
            # Fallback to class name if model_name not in kwargs
            self.model_name = self.__class__.__name__.replace("Model", "")

        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params, **kwargs)
        nn.Module.__init__(self)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Define all required parameters
        required_params = [
            "learning_rate",
            "num_epochs",
            "save_checkpoint",
            "verbose",
            "num_layers",
            "output_shape",
            "hidden_size",
            "lstm_units",
            "dense_units",
            "dropout",
            "grad_clip_max_norm",
            "scheduler_factor",
            "scheduler_patience",
            "scheduler_cooldown",
            "min_lr",
            "early_stopping_rounds",
            "num_epochs",
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
        logger.info("Initializing LSTM with parameters: %s", self.params)

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
        set_seeds(self.params["random_seed"])

        # Get parameters for the architecture
        self.lstm_units = self.params.get(
            "lstm_units", [self.hidden_size] * self.num_layers
        )
        self.dense_units = self.params.get("dense_units", 64)

        # Create ModuleLists for the layers
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # Create separate dropout rates for different layers or use the same rate
        if hasattr(self.dropout, "__len__"):  # Check if it's sequence-like
            dropout_rates = list(self.dropout)
        else:
            # Create increasing dropout rates if single value provided
            base_rate = float(self.dropout)  # Convert to float
            dropout_rates = [
                min(base_rate * (i + 1), 0.5) for i in range(len(self.lstm_units))
            ]
        logger.info("Using dropout rates for LSTM layers: %s", dropout_rates)

        # Create the LSTM layers with respective dropout and batch norm
        input_size = self.input_size
        for i, units in enumerate(self.lstm_units):
            lstm_layer = nn.LSTM(input_size, units, batch_first=True)
            self.lstm_layers.append(lstm_layer)
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]))
            self.batch_norm_layers.append(nn.BatchNorm1d(units))
            input_size = units

        # Define the dense layers for output with LeakyReLU activations
        self.dense1 = nn.Linear(self.lstm_units[-1], self.dense_units)
        self.dense2 = nn.Linear(self.dense_units, 32)
        self.dense3 = nn.Linear(32, self.output_size)
        self.dropout_final = nn.Dropout(
            dropout_rates[-1]
        )  # Final dropout after first dense layer

        # Initialize weights with Xavier initialization
        self.apply(initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_length, input_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        # Process through each LSTM layer with dropout and batch norm
        for lstm, dropout, batch_norm in zip(
            self.lstm_layers, self.dropout_layers, self.batch_norm_layers
        ):
            x, _ = lstm(x)
            x = dropout(x)
            # Apply batch normalization on the feature dimension
            # Need to transpose as batch norm expects [batch, features, seq_len]
            x = batch_norm(x.transpose(1, 2)).transpose(1, 2)

        # Get the output from the last time step
        x = x[:, -1, :]

        # Process through dense layers with activation functions
        x = torch.nn.functional.leaky_relu(self.dense1(x))
        x = self.dropout_final(x)
        x = torch.nn.functional.leaky_relu(self.dense2(x))
        x = self.dense3(x)

        return x

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
        set_seeds(self.model.params["random_seed"])

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.params["learning_rate"]
        )
        self.pos_weight = calculate_pos_weight(self.train_loader)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight])
        )
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(lstm_model.save_dir, "Models")
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

        # Data preparation
        # Get the configured data converter
        self.converter = prepare_data_for_model_convdl(
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

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.params["learning_rate"]
        )

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

    def train(self):
        """Training loop."""
        set_seeds(self.model.params["random_seed"])
        num_epochs = self.params["num_epochs"]
        verbose = self.params.get("verbose", 1)

        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.train_epoch(epoch, verbose)
            logger.info("Epoch %d finished", epoch + 1)

            val_loss = self.evaluate(self.val_loader)

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
        self.evaluate(self.test_loader, save_report=True)

        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_torch_model(
            model_save_name, self.model, os.path.join(self.model.save_dir, "Models")
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
                    logger.info("Evaluating batch %d: Loss = %.4f", batch + 1, loss)

                    if self.wandb:
                        wandb.log({"val_loss": loss})

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        # Log results to console
        logger.info("Test evaluation completed for %s", self.model.model_name)
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

        return np.mean(val_loss)
