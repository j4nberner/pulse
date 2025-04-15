import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import logging
from collections import OrderedDict
from typing import Dict, Any, List, Optional
import wandb
import os

from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    save_torch_model,
    prepare_data_for_model_dl,
)
from src.eval.metrics import calculate_all_metrics, calc_metric_stats, MetricsTracker

# TODO: Why are calculate_all_metrics and cal_metric_stats imported but not used?

# Set up logger
logger = logging.getLogger("PULSE_logger")

class InceptionTimeModel(PulseTemplateModel, nn.Module):
    """
    Implementation of InceptionTime deep learning model for time series classification.
    """

    class Inception(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(InceptionTimeModel.Inception, self).__init__()
            self.bottleneck = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            self.conv1 = nn.Conv1d(
                out_channels, out_channels, kernel_size=1, padding="same"
            )
            self.conv2 = nn.Conv1d(
                out_channels, out_channels, kernel_size=3, padding="same"
            )
            self.conv3 = nn.Conv1d(
                out_channels, out_channels, kernel_size=5, padding="same"
            )

            self.conv_pool = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

            self.batch_norm = nn.BatchNorm1d(out_channels * 4)

        def forward(self, x):
            x0 = self.bottleneck(x)
            x1 = self.conv1(x0)
            x2 = self.conv2(x0)
            x3 = self.conv3(x0)
            x4 = self.conv_pool(self.pool(x))

            out = torch.cat([x1, x2, x3, x4], dim=1)
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
                logger.info(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model state...')
            self.best_state_dict = model.state_dict().copy()
            self.best_val_loss = val_loss

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the InceptionTime model.

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
        
        # Define required parameters based on InceptionTimeModel.yaml
        required_params = [
            'save_checkpoint_freq',
            'verbose',
            'num_epochs',
            'patience',
            'use_cuda',
            'depth',
            'dropout_rate',
            'optimizer_name',
            'learning_rate',
            'weight_decay',
            'factor',
            'min_lr'   
        ]
        
        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")
        
        # Extract parameters from config
        self.params = params

        # Log configuration details
        logger.info(f"Initializing {self.model_name} model with parameters: {self.params}")
        
        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Check if wandb is enabled
        self.wandb = kwargs.get("wandb", False)

        # Network architecture parameters directly from params
        self.depth = self.params["depth"]  
        self.dropout_rate = self.params["dropout_rate"]
        
        # These will be set in _init_model
        self.in_channels = None
        self.out_channels = None
        self.network = None
        
        # Initialize early stopping
        self.early_stopping = self.EarlyStopping(patience=self.params["patience"], verbose=True)

        # Initialize the model architecture
        self._init_model()

    def _init_model(self) -> None:
        """
        Initialize the InceptionTime network architecture with placeholder values.
        The actual input shape will be determined when data is prepared.
        """
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
        # Reset channel configurations
        self.in_channels = [num_channels]
        self.out_channels = [min(256, num_channels)]
        
        # Configure channel dimensions for each layer
        for i in range(1, self.depth):
            prev_out = self.out_channels[i - 1]
            if i < self.depth // 3:
                self.in_channels.append(prev_out * 4)
                self.out_channels.append(prev_out)
            elif i < 2 * self.depth // 3:
                self.in_channels.append(prev_out * 4)
                self.out_channels.append(max(prev_out // 2, 32))
            else:
                self.in_channels.append(prev_out * 4)
                self.out_channels.append(max(prev_out // 2, 16))

    def create_network_with_input_shape(self, num_channels: int) -> None:
        """
        Update the model architecture based on the actual input shape.
        
        Args:
            num_channels: Number of input channels from the data
        """

        # Reconfigure channel dimensions using the helper method
        self._configure_channels(num_channels)
                
        # Store inception and residual modules separately
        self.inception_modules = nn.ModuleList()
        self.residual_connections = {}
        
        # Build inception modules with residual connections
        for d in range(self.depth):
            self.inception_modules.append(
                self.Inception(
                    in_channels=self.in_channels[d], 
                    out_channels=self.out_channels[d]
                )
            )
            if d % 3 == 2 and d >= 2:  # Add residual connection every 3rd block
                self.residual_connections[d] = self.Residual(
                    in_channels=self.out_channels[d - 2] * 4, 
                    out_channels=self.out_channels[d] * 4
                )

        # Add global average pooling and fully connected layers
        self.global_avg_pool = self.Lambda(lambda x: torch.mean(x, dim=-1))
        self.dropout1 = nn.Dropout(self.dropout_rate)
        self.fc1 = nn.Linear(4 * self.out_channels[-1], 64)
        self.relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(64, 16)
        self.relu2 = nn.LeakyReLU()
        self.output = nn.Linear(16, 1)
        
        # Clear the network attribute
        self.network = None

    def forward(self, x):
        """
        Forward pass through the network with manual residual connection handling.
        """
        # Store outputs for residual connections
        layer_outputs = []
        
        # Process through inception modules with residual connections
        for i in range(len(self.inception_modules)):
            # Store input for residual connection
            if i > 0 and i % 3 == 0:
                residual_input = layer_outputs[i-3]
            
            # Apply inception module
            x = self.inception_modules[i](x)
            layer_outputs.append(x)
            
            # Apply residual connection if needed
            if i in self.residual_connections:
                residual_module = self.residual_connections[i]
                x = residual_module(layer_outputs[i-2], x)
                layer_outputs[i] = x  # Update current output
        
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

    def set_trainer(self, trainer_name, train_loader, val_loader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
                
        Returns:
            None
        """
        self.trainer = InceptionTimeTrainer(self, train_loader, val_loader)

    # def eval(self, test_loader):
    #     """
    #     Evaluate the model on the test set.
    #     Used by benchmark_models.py.
        
    #     Args:
    #         test_loader: DataLoader with test data
            
    #     Returns:
    #         Dictionary with evaluation metrics
    #     """
    #     # Initialize a temporary trainer if one doesn't exist
    #     if not hasattr(self, 'trainer'):
    #         logger.info("Creating temporary trainer for evaluation")
    #         self.trainer = InceptionTimeTrainer(self, None, test_loader)
        
    #     # Run evaluation
    #     return self.trainer.evaluate(test_loader)
    
class InceptionTimeTrainer:
    """
    Trainer class for InceptionTime models.

    This class handles the training workflow for InceptionTime models
    including data preparation, training, evaluation and saving.
    """

    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb = self.model.wandb

        # Create model save directory and checkpoint subdirectory if it doesn't exist
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.model_save_dir, "Checkpoints")
        os.makedirs(os.path.join(self.model_save_dir, "Checkpoints"), exist_ok=True)
        self.save_checkpoint_freq = self.params["save_checkpoint_freq"]

        # Data preparation
        self._prepare_data()

        # Initialize optimizer based on config
        self.optimizer_name = self.params["optimizer_name"]
        lr = self.params["learning_rate"]
        weight_decay = self.params["weight_decay"]

        if self.optimizer_name == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_name == "adamw":
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif self.optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.criterion = nn.BCEWithLogitsLoss()

        logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")
        logger.info(f"Using criterion: {self.criterion.__class__.__name__}")

        # Initialize scheduler
        factor = self.params["factor"]
        patience = self.params["patience"]
        min_lr = self.params["min_lr"]
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr
        )

    def _prepare_data(self):
        """Prepare data for InceptionTime by getting a configured converter."""

        # Get the configured converter
        self.converter = prepare_data_for_model_dl(
            self.train_loader, self.params, model_name=self.model.model_name
        )

        # To identify num_channels: Get a sample batch and transform using the converter
        features, _ = next(iter(self.train_loader))
        transformed_features = self.converter.convert_batch_to_3d(features)
        
        # Get the number of channels from the transformed features
        num_channels = transformed_features.shape[1] # CNN models (batch_size, num_channels, time_steps), RNN models (batch_size, time_steps, num_features)
        
        # Update model architecture with correct shape
        self.model.create_network_with_input_shape(num_channels)
    
        logger.info(f"Input shape to model (after transformation): {transformed_features.shape}")
        logger.info(f"Model architecture initialized with {num_channels} input channels")

    def train(self):
        """Training loop."""
            
        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        # Initialize metrics tracking dictionary (not used for earlystopping, logging or wandb)
        self.metrics = {
            "train_loss": [],
            "val_loss": []
        }

        logger.info("Starting training...")
        for epoch in range(self.params["num_epochs"]):
            self.train_epoch(epoch, verbose=self.params["verbose"])
            logger.info(f"Epoch {epoch + 1}/{self.params['num_epochs']} completed.")

            # Save checkpoint periodically
            if self.save_checkpoint_freq > 0 and (epoch + 1) % self.save_checkpoint_freq == 0:
                checkpoint_name = f"{self.model.model_name}_epoch_{epoch + 1}"
                save_torch_model(checkpoint_name, self.model, self.checkpoint_path)
        
        logger.info("Training completed.")    

        # After training loop, load best model weights and save final model
        if self.model.early_stopping.best_state_dict:
            self.model.load_state_dict(self.model.early_stopping.best_state_dict)
        save_torch_model(f"{self.model.model_name}", self.model, self.model_save_dir)
        
        # Evaluate the model on the testing set
        # TODO: How does this fit into the benchmark_models.py?
        self.evaluate()

    def train_epoch(self, epoch: int, verbose: int = 1) -> None:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            verbose (int): Verbosity level (0, 1, or 2).
        """
        self.model.train()
        train_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = self.converter.convert_batch_to_3d(features)

            features, labels = (
                features.to(self.device),
                labels.to(self.device).float(),
            )

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs.squeeze(), labels.squeeze())

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # Calculate average loss
            avg_train_loss = train_loss / len(self.train_loader)
            self.metrics["train_loss"].append(avg_train_loss)

            # Validation phase
            val_loss = self._validate()
            self.metrics["val_loss"].append(val_loss)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.params["num_epochs"]} - "
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
                break

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

    def evaluate(self, dataloader = None):
        """
        Evaluate the model on the specified data loader.
        
        Args:
            dataloader: DataLoader to use for evaluation. If None, uses val_loader.
        """
        metrics_tracker = MetricsTracker(self.model.model_name, self.model.save_dir)
        self.model.eval()

        # Use provided dataloader or fall back to validation loader
        eval_loader = dataloader if dataloader is not None else self.val_loader
        logger.info(f"Evaluating model on {'test' if dataloader is not None else 'validation'} data")

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(eval_loader):
                features = self.converter.convert_batch_to_3d(features)
                features, labels = (
                    features.to(self.device),
                    labels.to(self.device).float(),
                )
                outputs = self.model(features)
                _, predicted = torch.max(outputs.data, 1)

                accuracy = (predicted == labels).sum().item() / labels.size(0)

                # Append results to metrics tracker
                metrics_tracker.add_results(predicted.cpu().numpy(), labels.cpu().numpy())
                if self.params["verbose"] == 2 or (self.params["verbose"] == 1 and batch_idx % 10 == 9):
                    logger.info(
                        f"Evaluating batch {batch_idx + 1}: " f"Accuracy = {accuracy}"
                    )

                    if self.wandb:
                        wandb.log({"accuracy": accuracy})

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()