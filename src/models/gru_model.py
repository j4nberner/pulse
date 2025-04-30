import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    prepare_data_for_model_convdl,
    save_torch_model,
    calculate_pos_weight,
)

# Set up logger
logger = logging.getLogger("PULSE_logger")


class GRUModel(PulseTemplateModel, nn.Module):
    """
    Implementation of Gated Recurrent Unit (GRU) model for time series classification.

    The model uses GRU layers to process sequential data followed by
    fully connected layers for classification.
    """

    class EarlyStopping:
        def __init__(self, patience=5, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.early_stop = False
            self.best_val_loss = float("inf")
            self.delta = delta
            self.best_state_dict = None

        def __call__(self, val_loss, model):
            if val_loss > self.best_val_loss - self.delta:
                self.counter += 1
                if self.verbose:
                    logger.info(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                logger.info(
                    f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model state..."
                )
            self.best_state_dict = model.state_dict().copy()
            self.best_val_loss = val_loss

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the GRU model.

        Args:
            params: Dictionary of parameters from the config file.
            **kwargs: Additional keyword arguments.

        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        # Validate trainer_name in params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")

        # Use the class name as model_name if not provided in params
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)
        nn.Module.__init__(self)

        # Define required parameters based on GRUModel.yaml
        required_params = [
            "save_checkpoint_freq",
            "verbose",
            "num_epochs",
            "earlystopping_patience",
            "hidden_dim",
            "layer_dim",
            "dropout_rate",
            "optimizer_name",
            "learning_rate",
            "weight_decay",
            "grad_clip_max_norm",
            "scheduler_factor",
            "scheduler_patience",
            "min_lr",
        ]

        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        # Extract parameters from config
        self.params = params

        # Log configuration details
        logger.info(
            f"Initializing {self.model_name} model with parameters: {self.params}"
        )

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Check if wandb is enabled
        self.wandb = kwargs.get("wandb", False)

        # Extracting architecture parameters
        self.hidden_dim = self.params["hidden_dim"]
        self.layer_dim = self.params["layer_dim"]
        self.dropout_rate = self.params["dropout_rate"]

        # These will be set when data shape is known
        self.input_dim = None

        # Initialize early stopping
        self.early_stopping = self.EarlyStopping(
            patience=self.params["earlystopping_patience"], verbose=True
        )

        # Initialize the model architecture placeholder
        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the GRU model architecture with placeholder values.
        The actual input shape will be determined when data is prepared.
        """
        # The actual network will be built in create_network_with_input_shape
        # when we know the actual input shape
        pass

    def create_network_with_input_shape(self, input_dim: int) -> None:
        """
        Update the model architecture based on the actual input shape.

        Args:
            input_dim: Input dimension from the data
        """
        self.input_dim = input_dim

        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.layer_dim,
            batch_first=True,
            dropout=self.dropout_rate if self.layer_dim > 1 else 0,
        )

        # Fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

        logger.info(
            f"GRU network initialized with input_dim {input_dim}, "
            + f"hidden_dim {self.hidden_dim}, layer_dim {self.layer_dim}, "
            + f"{sum(p.numel() for p in self.parameters())} parameters"
        )

    def forward(self, x):
        """
        Forward pass through the GRU network.

        Args:
            x: Input tensor of shape [batch_size, seq_length, input_dim]

        Returns:
            Output tensor after passing through the GRU network
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)

        # Forward propagate the GRU
        out, _ = self.gru(x, h0)

        # Extract the output of the last time step
        out = out[:, -1, :]

        # Feed to fully connected layers
        out = self.fc(out)

        return out

    def set_trainer(self, trainer_name, train_loader, val_loader, test_loader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for testing data.

        Returns:
            Trainer instance
        """
        self.trainer = GRUTrainer(self, train_loader, val_loader, test_loader)


class GRUTrainer:
    """
    Trainer class for GRU models.

    This class handles the training workflow for GRU models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_loader, val_loader, test_loader):
        """
        Initialize the GRU trainer.

        Args:
            model: The GRU model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for testing data.
        """
        self.model = model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.wandb = self.model.wandb
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name

        # Create model save directory and checkpoint subdirectory if it doesn't exist
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_save_dir, "Checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.save_checkpoint_freq = self.params["save_checkpoint_freq"]

        # Log which task is being processed
        if self.task_name:
            logger.info(f"Preparing GRU model for task: {self.task_name}")

        # Data preparation
        self._prepare_data()

        # Set criterion after calculating class weights for imbalanced datasets
        self.pos_weight = calculate_pos_weight(self.train_loader)
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight]).to(self.device)
        )
        logger.info(
            f"Using criterion: {self.criterion.__class__.__name__} with class weight adjustment"
        )

        # Initialize optimizer based on config
        self.optimizer_name = self.params["optimizer_name"]
        lr = self.params["learning_rate"]
        weight_decay = self.params["weight_decay"]

        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer_name == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        # Initialize scheduler
        scheduler_factor = self.params["scheduler_factor"]
        scheduler_patience = self.params["scheduler_patience"]
        min_lr = self.params["min_lr"]
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=min_lr,
        )

    def _prepare_data(self):
        """Prepare data for GRU by getting a configured converter."""

        # Get the configured converter
        self.converter = prepare_data_for_model_convdl(
            self.train_loader,
            self.params,
            model_name=self.model.model_name,
            task_name=self.task_name,
        )

        # To identify input_dim: Get a sample batch and transform
        features, _ = next(iter(self.train_loader))
        transformed_features = self.converter.convert_batch_to_3d(features)

        # Get the input dimension from the transformed features
        # For RNN models: (batch_size, time_steps, num_features)
        num_channels = transformed_features.shape[-1]

        # Update model architecture with correct shape
        self.model.create_network_with_input_shape(num_channels)

        logger.info(
            f"Input shape to model (after transformation): {transformed_features.shape}"
        )
        logger.info(
            f"Model architecture initialized with {num_channels} input channels"
        )

    def train(self):
        """Train the GRU model using the provided data loaders."""
        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        # Initialize metrics tracking dictionary
        self.metrics = {"train_loss": [], "val_loss": []}

        logger.info(f"Starting training on device: {self.device}")

        for epoch in range(self.params["num_epochs"]):
            early_stopped = self.train_epoch(epoch, verbose=self.params["verbose"])

            # Check if early stopping was triggered
            if early_stopped:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Save checkpoint periodically
            if (
                self.save_checkpoint_freq > 0
                and (epoch + 1) % self.save_checkpoint_freq == 0
            ):
                checkpoint_name = f"{self.model.model_name}_epoch_{epoch + 1}"
                save_torch_model(checkpoint_name, self.model, self.checkpoint_path)

        logger.info("Training completed.")

        # After training loop, load best model weights and save final model
        if self.model.early_stopping.best_state_dict:
            self.model.load_state_dict(self.model.early_stopping.best_state_dict)

        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        save_torch_model(model_save_name, self.model, self.model_save_dir)
        # Evaluate the model on the testing set
        self.evaluate()

    def train_epoch(self, epoch: int, verbose: int = 1) -> None:
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            verbose (int): Verbosity level (0, 1, or 2).

        Returns:
            Boolean indicating if early stopping was triggered
        """
        self.model.train()
        train_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = self.converter.convert_batch_to_3d(features)
            features, labels = features.to(self.device), labels.to(self.device).float()

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())

            # Backward pass, gradient clipping and optimize
            loss.backward()
            max_norm = self.params["grad_clip_max_norm"]
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=max_norm
            )
            if total_norm > max_norm:
                logger.info(f"Gradient norm {total_norm:.4f} clipped to {max_norm}")
            self.optimizer.step()

            train_loss += loss.item()

            # Log progress for each batch if verbose=2, or every 100 batches if verbose=1
            if verbose == 2 or verbose == 1 and batch_idx % 100 == 0:
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}"
                )

        # Calculate average loss for the epoch
        avg_train_loss = train_loss / len(self.train_loader)
        self.metrics["train_loss"].append(avg_train_loss)

        # Validation phase
        val_loss = self._validate()
        self.metrics["val_loss"].append(val_loss)

        # Update learning rate based on validation loss
        self.scheduler.step(val_loss)

        # Log epoch summary
        logger.info(
            f"Epoch {epoch+1}/{self.params['num_epochs']} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
        )

        # Log to WandB if enabled
        if self.wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                }
            )

        # Check early stopping
        self.model.early_stopping(val_loss, self.model)
        if self.model.early_stopping.early_stop:
            return True  # Early stopping triggered
        return False  # No early stopping

    def _validate(self):
        """Validate the model and return validation loss."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = self.converter.convert_batch_to_3d(features)
                features, labels = (
                    features.to(self.device),
                    labels.to(self.device).float(),
                )

                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def evaluate(self):
        """
        Evaluate the model on the test set with comprehensive metrics.
        Logs all metrics to wandb and returns the results.
        """
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        self.model.eval()

        # Track both batches and per-batch metrics for logging
        batch_metrics = []

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(self.test_loader):
                # Convert features for the model
                features = self.converter.convert_batch_to_3d(features)
                features, labels = (
                    features.to(self.device),
                    labels.to(self.device).float(),
                )

                # Forward pass
                outputs = self.model(features)

                # Get predictions (sigmoid for binary classification)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).int()

                # Calculate batch accuracy for logging
                batch_accuracy = (preds == labels).sum().item() / labels.size(0)
                batch_metrics.append(batch_accuracy)

                # Add results to metrics tracker
                metrics_tracker.add_results(preds.cpu().numpy(), labels.cpu().numpy())

                # Log batch progress if verbose
                if self.params["verbose"] == 2 or (
                    self.params["verbose"] == 1 and batch_idx % 100 == 0
                ):
                    logger.info(
                        f"Evaluating batch {batch_idx+1}/{len(self.test_loader)}: Accuracy = {batch_accuracy:.4f}"
                    )

        # Calculate comprehensive metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()

        # Log results to console
        logger.info(f"Test evaluation completed for {self.model.model_name}")
        logger.info(f"Test metrics: {metrics_tracker.summary}")
        logger.info(
            f"Average batch accuracy: {sum(batch_metrics)/len(batch_metrics):.4f}"
        )

        # Log all metrics to wandb if enabled
        if self.wandb:
            # Create a dictionary with all metrics
            wandb_metrics = {f"test_{k}": v for k, v in metrics_tracker.summary.items()}
            # Add average batch accuracy
            wandb_metrics["test_avg_batch_accuracy"] = sum(batch_metrics) / len(
                batch_metrics
            )
            # Log all metrics at once
            wandb.log(wandb_metrics)
