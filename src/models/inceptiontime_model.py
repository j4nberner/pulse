import logging
import os
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class InceptionTimeModel(PulseModel, nn.Module):
    """
    Implementation of InceptionTime deep learning model for time series classification.
    """

    class Inception(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_sizes=[1, 3, 5]):
            super(InceptionTimeModel.Inception, self).__init__()
            self.bottleneck = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            # Create convolutional layers based on provided kernel sizes
            self.conv_layers = nn.ModuleList()
            for k_size in kernel_sizes:
                self.conv_layers.append(
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=k_size, padding="same"
                    )
                )

            self.conv_pool = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

            # BatchNorm layer needs to account for all kernel paths plus the pooling path
            self.batch_norm = nn.BatchNorm1d(out_channels * (len(kernel_sizes) + 1))

        def forward(self, x):
            x0 = self.bottleneck(x)

            # Apply each conv layer to the bottleneck output
            conv_outputs = [conv(x0) for conv in self.conv_layers]

            # Add the pooling path
            pool_output = self.conv_pool(self.pool(x))

            # Concatenate all outputs
            out = torch.cat(conv_outputs + [pool_output], dim=1)

            out = self.batch_norm(out)
            out = F.leaky_relu(out)

            return out

    class Residual(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(InceptionTimeModel.Residual, self).__init__()
            self.bottleneck = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            self.batch_norm = nn.BatchNorm1d(out_channels)

        def forward(self, x, y):
            y = y + self.batch_norm(self.bottleneck(x))
            y = F.leaky_relu(y)
            return y

    class Lambda(nn.Module):
        def __init__(self, f):
            super(InceptionTimeModel.Lambda, self).__init__()
            self.f = f

        def forward(self, x):
            return self.f(x)

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the InceptionTime model.

        Args:
            params: Dictionary of parameters from the config file.
            **kwargs: Additional keyword arguments.

        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        model_name = kwargs.pop("model_name", "InceptiontimeModel")
        trainer_name = params["trainer_name"]
        super().__init__(
            model_name=model_name,
            params=params,
            trainer_name=trainer_name,
            **kwargs,
        )
        nn.Module.__init__(self)

        # Define required parameters based on InceptionTimeModel.yaml
        required_params = [
            "save_checkpoint_freq",
            "verbose",
            "num_epochs",
            "earlystopping_patience",
            "depth",
            "kernel_sizes",
            "dropout_rate",
            "optimizer_name",
            "learning_rate",
            "weight_decay",
            "grad_clip_max_norm",
            "scheduler_factor",
            "scheduler_patience",
            "scheduler_cooldown",
            "min_lr",
        ]

        self.check_required_params(params, required_params)

        # Log configuration details
        logger.info(
            "Initializing %s model with parameters: %s", self.model_name, self.params
        )

        # Network architecture parameters directly from params
        self.depth = self.params["depth"]
        self.kernel_sizes = self.params["kernel_sizes"]
        logger.debug("Using kernel sizes: %s for inception modules", self.kernel_sizes)
        self.dropout_rate = self.params["dropout_rate"]

        # These will be set in _init_model
        self.in_channels = None
        self.out_channels = None
        self.network = None
        # Store inception and residual modules separately
        self.inception_modules = nn.ModuleList()
        self.residual_connections = nn.ModuleDict()

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=self.params["earlystopping_patience"], verbose=True
        )

        # Initialize the model architecture
        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the InceptionTime network architecture with placeholder values.
        The actual input shape will be determined when data is prepared.
        """
        set_seeds(self.params["random_seed"])

        # Just set up placeholder values (num_channels will be determined after data preparation)
        self._configure_channels(num_channels=1)

        # The network will be built in create_network_with_input_shape when we know the actual shape
        self.network = None

    def _configure_channels(self, num_channels: int) -> None:
        """
        Configure the channel dimensions for the InceptionTime network.

        Args:
            num_channels: Number of input channels
        """
        # Calculate number of paths in each inception block
        num_paths_per_block = len(self.kernel_sizes) + 1  # kernels + pooling path

        # Reset channel configurations
        self.in_channels = [num_channels]  # First layer input is data channels
        self.out_channels = [min(256, num_channels)]  # First layer output channels

        # Configure channel dimensions for each layer
        for i in range(1, self.depth):
            # Input to this layer is the concatenated output from previous layer
            prev_out = self.out_channels[i - 1]
            prev_in = prev_out * num_paths_per_block

            self.in_channels.append(prev_in)

            # Determine output channels based on depth position
            if i < self.depth // 3:
                self.out_channels.append(prev_out)  # Same as previous
            elif i < 2 * self.depth // 3:
                self.out_channels.append(max(prev_out // 2, 32))  # Half, min 32
            else:
                self.out_channels.append(max(prev_out // 2, 16))  # Half, min 16

    def create_network_with_input_shape(self, num_channels: int) -> None:
        """
        Update the model architecture based on the actual input shape.

        Args:
            num_channels: Number of input channels from the data
        """
        set_seeds(self.params["random_seed"])
        # Reset inception and residual modules
        self.inception_modules = nn.ModuleList()
        self.residual_connections = nn.ModuleDict()

        # Reconfigure channel dimensions using the helper method
        self._configure_channels(num_channels)

        # Calculate output channels for each inception block based on kernel sizes
        # Each kernel size contributes one path, plus there's one pooling path
        num_paths_per_block = len(self.kernel_sizes) + 1

        # Build inception modules with residual connections
        for d in range(self.depth):
            self.inception_modules.append(
                self.Inception(
                    in_channels=self.in_channels[d],
                    out_channels=self.out_channels[d],
                    kernel_sizes=self.kernel_sizes,
                )
            )
            if d % 3 == 2 and d >= 2:  # Add residual connection every 3rd block
                self.residual_connections[str(d)] = self.Residual(
                    in_channels=self.out_channels[d - 2] * num_paths_per_block,
                    out_channels=self.out_channels[d] * num_paths_per_block,
                )

        # Add global average pooling and fully connected layers
        self.global_avg_pool = self.Lambda(lambda x: torch.mean(x, dim=-1))
        self.dropout1 = nn.Dropout(self.dropout_rate)
        # Update FC layer input size to account for variable number of paths
        self.fc1 = nn.Linear(num_paths_per_block * self.out_channels[-1], 64)
        self.relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(64, 16)
        self.relu2 = nn.LeakyReLU()
        self.output = nn.Linear(16, 1)

        # Clear the network attribute
        self.network = None

        # Initialize weights with Xavier initialization
        self.apply(initialize_weights)

    def forward(self, x):
        """
        Forward pass through the network with optimized residual connection handling.
        """
        # Use a fixed-size list to track only the outputs needed for residual connections
        # We only need to remember the last 3 outputs (for the residual connections)
        recent_outputs = [None, None, None]

        # Process through inception modules with residual connections
        for i, inception_module in enumerate(self.inception_modules):
            # Apply inception module
            x = inception_module(x)

            # Apply residual connection if needed
            if str(i) in self.residual_connections and recent_outputs[0] is not None:
                # Only use residual connection if we have a valid tensor
                residual_input = recent_outputs[0]  # first element is oldest
                residual_module = self.residual_connections[str(i)]
                x = residual_module(residual_input, x)

            # Shift outputs window and store current output (circular buffer style)
            recent_outputs.pop(0)  # Remove oldest output
            recent_outputs.append(x.clone())  # Add current output

        # Apply final layers
        x = self.global_avg_pool(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.output(x)

        return x

    def evaluate(self, data_loader, save_report=False):
        """
        Evaluate the model on the specified data loader.
        Args:
            data_loader: DataLoader for evaluation data.
            save_report: Whether to save the evaluation report.
        """
        set_seeds(self.params["random_seed"])
        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        self.eval()

        # Track both batches and per-batch metrics for logging
        batch_metrics = []

        # Get the configured converter
        converter = prepare_data_for_model_convdl(
            data_loader,
            self.params,
            architecture_type=self.params.get("architecture_type", "CNN"),
            task_name=self.task_name,
        )

        # Load model from pretrained state if available and not in training mode
        if self.pretrained_model_path:
            # Configure the model input size based on the data
            features, _ = next(iter(data_loader))
            transformed_features = converter.convert_batch_to_3d(features)
            input_dim = transformed_features.shape[1]  # num_channels
            self.input_size = input_dim
            self.create_network_with_input_shape(input_dim)
            self.load_model_weights(self.pretrained_model_path)

        # Move model to device
        self.to(self.device)

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
                    "sex": features[:, 1, 0].cpu().numpy(),
                    "height": features[:, 2, 0].cpu().numpy(),
                    "weight": features[:, 3, 0].cpu().numpy(),
                }
                # Append results to metrics tracker
                metrics_tracker.add_results(outputs.cpu().numpy(), labels.cpu().numpy())
                metrics_tracker.add_metadata_item(metadata_dict)

        # Calculate and log metrics
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


class InceptionTimeTrainer:
    """
    Trainer class for InceptionTime models.

    This class handles the training workflow for InceptionTime models
    including data preparation, training, evaluation and saving.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Initialize the InceptionTime trainer.

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
        os.makedirs(os.path.join(self.model_save_dir, "Checkpoints"), exist_ok=True)
        self.save_checkpoint_freq = self.params["save_checkpoint_freq"]

        # Log which task is being processed
        if self.task_name:
            logger.info("Preparing InceptionTime model for task: %s", self.task_name)

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
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
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
        """Prepare data for InceptionTime by getting a configured converter."""
        set_seeds(self.model.params["random_seed"])

        # Get the configured converter
        self.converter = prepare_data_for_model_convdl(
            self.train_loader,
            self.params,
            architecture_type=self.params.get("architecture_type", "CNN"),
            task_name=self.task_name,
        )

        # To identify num_channels: Get a sample batch and transform using the converter
        features, _ = next(iter(self.train_loader))
        transformed_features = self.converter.convert_batch_to_3d(features)

        # Get the number of channels from the transformed features
        num_channels = transformed_features.shape[
            1
        ]  # CNN models (batch_size, num_channels, time_steps), RNN models (batch_size, time_steps, num_features)

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
        """Training loop."""
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
                break  # Exit the training loop

            # Save checkpoint periodically
            if (
                self.save_checkpoint_freq > 0
                and (epoch + 1) % self.save_checkpoint_freq == 0
            ):
                checkpoint_name = f"{self.model.model_name}_epoch_{epoch + 1}"
                save_torch_model(checkpoint_name, self.model, self.checkpoint_path)

        logger.info("Training completed.")

        # After training loop, load best model weights and save final model
        self.model = self.model.early_stopping.load_best_model(self.model)
        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model.pretrained_model_path = save_torch_model(
            model_save_name, self.model, self.model_save_dir
        )

    def train_epoch(self, epoch: int, verbose: int = 1) -> None:
        """
        Trains the model for one epoch.

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
                logger.info("Gradient norm %.4f clipped to %.4f", total_norm, max_norm)
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

        # Update learning rate
        self.scheduler.step(val_loss)

        # Log progress
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
            return True  # Return True to indicate early stopping was triggered
        return False  # Return False if early stopping was not triggered

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
