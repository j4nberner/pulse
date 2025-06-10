import logging
import os
from datetime import datetime
from typing import Any, Dict

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

# Set up logger
logger = logging.getLogger("PULSE_logger")


class GRUModel(PulseModel, nn.Module):
    """
    Implementation of Gated Recurrent Unit (GRU) model for time series classification.

    The model uses GRU layers to process sequential data followed by
    fully connected layers for classification.
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the GRU model.

        Args:
            params: Dictionary of parameters from the config file.
            **kwargs: Additional keyword arguments.

        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        model_name = kwargs.pop("model_name", "GRUModel")
        trainer_name = params["trainer_name"]
        super().__init__(
            model_name=model_name,
            params=params,
            trainer_name=trainer_name,
            **kwargs,
        )
        nn.Module.__init__(self)

        # Define required parameters based on GRUModel.yaml
        required_params = [
            "save_checkpoint_freq",
            "verbose",
            "num_epochs",
            "earlystopping_patience",
            "hidden_size",
            "num_layers",
            "dropout_rate",
            "fc_layers",
            "activation",
            "optimizer_name",
            "learning_rate",
            "weight_decay",
            "grad_clip_max_norm",
            "scheduler_factor",
            "scheduler_patience",
            "scheduler_cooldown",
            "min_lr",
        ]

        # Check if all required parameters exist in config
        self.check_required_params(params, required_params)

        # Log configuration details
        logger.info(
            "Initializing %s model with parameters: %s", self.model_name, self.params
        )

        # Extracting architecture parameters
        self.hidden_size = self.params["hidden_size"]
        self.num_layers = self.params["num_layers"]
        self.dropout_rate = self.params["dropout_rate"]
        # Create ModuleLists for GRU layers with separate dropout and batch normalization
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        # These will be set when data shape is known
        self.input_dim = None

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.params["earlystopping_patience"], verbose=True
        )

    def create_network_with_input_shape(self, input_dim: int) -> None:
        """
        Update the model architecture based on the actual input shape.

        Args:
            input_dim: Input dimension from the data
        """
        set_seeds(self.params["random_seed"])
        self.gru_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()

        self.input_dim = input_dim

        # Create separate dropout rates for different layers or use the same rate
        if hasattr(self.dropout_rate, "__len__"):  # Check if it's sequence-like
            dropout_rates = list(self.dropout_rate)
        else:
            # Create increasing dropout rates if single value provided
            base_rate = float(self.dropout_rate)  # Convert to float, not list
            dropout_rates = [
                min(base_rate * (i + 1), 0.5)
                for i in range(
                    self.num_layers + len(self.params.get("fc_layers", [64, 16]))
                )
            ]

        # Create GRU layers
        input_size = input_dim
        for i in range(self.num_layers):
            self.gru_layers.append(
                nn.GRU(
                    input_size=input_size,
                    hidden_size=self.hidden_size,
                    num_layers=1,  # Each layer is separate now
                    batch_first=True,
                )
            )
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]))
            self.batch_norm_layers.append(nn.BatchNorm1d(self.hidden_size))
            input_size = self.hidden_size  # Output of previous layer is input to next

        # Get FC layer dimensions from config
        fc_dims = self.params.get("fc_layers", [64, 16])
        activation_type = self.params.get("activation", "leaky_relu").lower()

        # Determine activation function based on config
        if activation_type == "relu":
            activation = nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = nn.LeakyReLU()
        elif activation_type == "gelu":
            activation = nn.GELU()
        else:
            # Default to LeakyReLU
            activation = nn.LeakyReLU()

        # Build fully connected layers with dynamic dropout rates
        layers = []
        input_size = self.hidden_size

        # Add FC layers with separate dropout rates
        for i, dim in enumerate(fc_dims):
            dropout_idx = self.num_layers + i
            if dropout_idx < len(dropout_rates):
                dropout_layer = nn.Dropout(dropout_rates[dropout_idx])
            else:
                dropout_layer = nn.Dropout(dropout_rates[-1])

            layers.extend([dropout_layer, nn.Linear(input_size, dim), activation])
            input_size = dim

        # Add final output layer
        layers.append(nn.Linear(input_size, 1))

        # Create sequential model with all layers
        self.fc = nn.Sequential(*layers)

        # Initialize weights with Xavier initialization
        self.apply(initialize_weights)

        logger.info(
            "GRU network initialized with input_dim %d, hidden_size %d, num_layers %d, "
            "activation=%s, fc_layers=%s, dropout_rates=%s, %d parameters",
            input_dim,
            self.hidden_size,
            self.num_layers,
            activation_type,
            fc_dims,
            dropout_rates,
            sum(p.numel() for p in self.parameters()),
        )

    def forward(self, x):
        """
        Forward pass through the GRU network.
        """
        batch_size = x.size(0)

        # Process through GRU layers with dropout and batch norm
        for gru, dropout, batch_norm in zip(
            self.gru_layers, self.dropout_layers, self.batch_norm_layers
        ):
            # Initialize hidden state for this layer
            h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

            # Forward through GRU
            x, _ = gru(x, h0)

            # Apply dropout
            x = dropout(x)

            # Apply batch normalization on the feature dimension
            # Batch norm expects [batch, features, seq_len]
            x = batch_norm(x.transpose(1, 2)).transpose(1, 2)

        # Extract the output of the last time step
        out = x[:, -1, :]

        # Feed to fully connected layers
        out = self.fc(out)

        return out

    def evaluate(self, data_loader, save_report: bool = False):
        """
        Evaluate the model on a dataset with comprehensive metrics.
        Logs all metrics to wandb and returns the results.
        """
        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )

        # Get the configured data converter
        converter = prepare_data_for_model_convdl(
            data_loader,
            self.params,
            architecture_type=self.params.get("architecture_type", "RNN"),
            task_name=self.task_name,
        )

        # Load model from pretrained state if available and not in training mode
        if self.pretrained_model_path and self.mode != "train":
            # To identify input_dim: Get a sample batch and transform
            features, _ = next(iter(data_loader))
            transformed_features = converter.convert_batch_to_3d(features)
            num_channels = transformed_features.shape[-1]
            self.create_network_with_input_shape(num_channels)
            logger.info(self)
            self.load_model_weights(self.pretrained_model_path)

        self.to(self.device)

        # Track both batches and per-batch metrics for logging
        batch_metrics = []

        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(data_loader):
                # Convert features for the model
                features = converter.convert_batch_to_3d(features)
                features, labels = (
                    features.to(self.device),
                    labels.to(self.device).float(),
                )

                # Forward pass
                outputs = self(features)

                # Get predictions (sigmoid for binary classification)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).int()

                # Calculate batch accuracy for logging
                batch_accuracy = (preds == labels).sum().item() / labels.size(0)
                batch_metrics.append(batch_accuracy)

                # Log batch progress if verbose
                if self.params["verbose"] == 2 or (
                    self.params["verbose"] == 1 and batch_idx % 100 == 0
                ):
                    logger.info(
                        "Evaluating batch %d/%d: Accuracy = %.4f",
                        batch_idx + 1,
                        len(data_loader),
                        batch_accuracy,
                    )

                metadata_dict = {
                    "batch": batch_idx,
                    "prediction": outputs.cpu().numpy(),
                    "label": labels.cpu().numpy(),
                    "age": features[:, 0, 0].cpu().numpy(),
                    "sex": features[:, 0, 1].cpu().numpy(),
                    "height": features[:, 0, 2].cpu().numpy(),
                    "weight": features[:, 0, 3].cpu().numpy(),
                }
                # Append results to metrics tracker
                metrics_tracker.add_results(outputs.cpu().numpy(), labels.cpu().numpy())
                metrics_tracker.add_metadata_item(metadata_dict)

        # Calculate comprehensive metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            # Calculate and log metrics
            metrics_tracker.log_metadata(True)
            metrics_tracker.save_report()

        # Log results to console
        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)
        logger.info(
            "Average batch accuracy: %.4f", sum(batch_metrics) / len(batch_metrics)
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


class GRUTrainer:
    """
    Trainer class for GRU models.

    This class handles the training workflow for GRU models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Initialize the GRU trainer.

        Args:
            model: The GRU model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
        """
        self.model = model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb = self.model.wandb
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        set_seeds(self.model.params["random_seed"])

        # Create model save directory and checkpoint subdirectory if it doesn't exist
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_save_dir, "Checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        self.save_checkpoint_freq = self.params["save_checkpoint_freq"]

        # Log which task is being processed
        if self.task_name:
            logger.info("Preparing GRU model for task: %s", self.task_name)

        # Data preparation
        self._prepare_data()

        # Set criterion after calculating class weights for imbalanced datasets
        self.pos_weight = self.train_loader.dataset.pos_weight
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight]).to(self.device)
        )
        logger.info(
            "Using criterion: %s with class weight adjustment",
            self.criterion.__class__.__name__,
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
        logger.info("Using optimizer: %s", self.optimizer.__class__.__name__)

        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.params["scheduler_factor"],
            patience=self.params["scheduler_patience"],
            cooldown=self.params["scheduler_cooldown"],
            min_lr=self.params["min_lr"],
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

    def _prepare_data(self):
        """Prepare data for GRU by getting a configured converter."""

        # Get the configured converter
        self.converter = prepare_data_for_model_convdl(
            self.train_loader,
            self.params,
            architecture_type=self.params.get("architecture_type", "RNN"),
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
        logger.info(self.model)

        logger.info(
            "Input shape to model (after transformation): %s",
            transformed_features.shape,
        )
        logger.info(
            "Model architecture initialized with %d input channels", num_channels
        )

    def train(self):
        """Train the GRU model using the provided data loaders."""
        set_seeds(self.model.params["random_seed"])

        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info("Starting training on device: %s", self.device)

        for epoch in range(self.params["num_epochs"]):
            early_stopped = self.train_epoch(epoch, verbose=self.params["verbose"])

            # Check if early stopping was triggered
            if early_stopped:
                logger.info("Early stopping triggered after %d epochs", epoch + 1)
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
        self.model.early_stopping.load_best_model(self.model)
        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_torch_model(model_save_name, self.model, self.model_save_dir)

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
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = self.converter.convert_batch_to_3d(features)
            features, labels = features.to(self.device), labels.to(self.device).float()
            # Log device information for the first batch
            if batch_idx == 0:
                logger.debug("Training batch on device: %s", features.device)

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
                logger.info("Gradient norm %.4f clipped to %f", total_norm, max_norm)
            self.optimizer.step()

            train_loss += loss.item()
            running_loss += loss.item()  # Add to running loss

            # Reporting based on verbosity
            if verbose == 2 or (verbose == 1 and batch_idx % 100 == 99):
                loss_value = running_loss / (100 if verbose == 1 else 1)
                logger.info(
                    "Epoch %d, Batch %d/%d: Loss = %.4f",
                    epoch + 1,
                    batch_idx + 1,
                    len(self.train_loader),
                    loss_value,
                )

                if self.wandb:
                    wandb.log({"train_loss": loss_value})

                running_loss = 0.0  # Reset running loss after logging

        # Calculate average loss for the epoch
        avg_train_loss = train_loss / len(self.train_loader)

        # Validation phase
        val_loss = self._validate()

        # Update learning rate based on validation loss
        self.scheduler.step(val_loss)

        # Log epoch summary
        logger.info(
            "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f, LR: %.6f",
            epoch + 1,
            self.params["num_epochs"],
            avg_train_loss,
            val_loss,
            self.optimizer.param_groups[0]["lr"],
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
