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
from src.preprocessing.preprocessing_advanced.windowing import WindowedDataTo3D
from src.util.model_util import (
    save_torch_model,
    prepare_data_for_model_dl,
)
from src.eval.metrics import calculate_all_metrics, calc_metric_stats, MetricsTracker

# TODO: Why are calculate_all_metrics and cal_metric_stats imported but not used?
# TODO: add new dataloader logic with mode=train/val/test to ML models & Inception Time (-> is val already used?)
# TODO: Model save via early stopping and at the end (use only save_torch_model util)

# Set up logger
logger = logging.getLogger("PULSE_logger")


class InceptionTimeModel(PulseTemplateModel):
    """
    Implementation of InceptionTime deep learning model for time series classification.

    The model follows the architecture described in the InceptionTime paper
    with inception blocks and residual connections.
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
            # The bottleneck should match the incoming data to the target output dimension
            self.bottleneck = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, padding="same"
            )
            self.batch_norm = nn.BatchNorm1d(out_channels)

        def forward(self, x, y):
            # x is the input from 3 layers back, y is the inception output
            # Make sure x is transformed to match y's dimensions
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
            self.model_path = None

        def __call__(self, val_loss, model, model_path):
            if val_loss > self.best_val_loss - self.delta:
                self.counter += 1
                if self.verbose:
                    logger.info(
                        f"EarlyStopping counter: {self.counter} out of {self.patience}"
                    )
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(val_loss, model, model_path)
                self.counter = 0

        def save_checkpoint(self, val_loss, model, model_path):
            if self.verbose:
                logger.info(
                    f"Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model..."
                )
            torch.save(model.state_dict(), model_path)
            self.best_val_loss = val_loss
            self.model_path = model_path

    class Network(nn.Module):
        def __init__(self, in_channels, out_channels, depth=12, dropout_rate=0.3):
            super().__init__()
            self.depth = depth

            # Create modules for each depth level
            for d in range(depth):
                # Add inception module
                self.add_module(
                    f"inception_{d}",
                    InceptionTimeModel.Inception(
                        in_channels=in_channels[d], out_channels=out_channels[d]
                    ),
                )

                # Add residual connection every 3 blocks
                if d % 3 == 2:
                    # The residual needs to connect from the input to the output of this inception block
                    res_in = in_channels[d - 2]  # Input channels from 3 layers back
                    res_out = (
                        out_channels[d] * 4
                    )  # Output channels × 4 (because inception outputs 4x channels)

                    self.add_module(
                        f"residual_{d}",
                        InceptionTimeModel.Residual(
                            in_channels=res_in, out_channels=res_out
                        ),
                    )

            self.model = nn.Sequential(
                InceptionTimeModel.Lambda(f=lambda x: torch.mean(x, dim=-1)),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=4 * out_channels[-1], out_features=64),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(in_features=64, out_features=16),
                nn.LeakyReLU(),
                nn.Linear(in_features=16, out_features=1),
            )

        def forward(self, input_tensor):
            residual_output = None
            for d in range(self.depth):
                # Access modules directly from self, not from self.model
                inception_output = self.get_submodule(f"inception_{d}")(
                    input_tensor if d == 0 else residual_output
                )
                if d % 3 == 2:
                    residual_output = self.get_submodule(f"residual_{d}")(
                        input_tensor, inception_output
                    )
                    input_tensor = residual_output
                else:
                    residual_output = inception_output

            # Use model for the output layers that are in Sequential
            y = self.model(residual_output)

            return y

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the InceptionTime model.

        Args:
            params: Dictionary of parameters from the config file.

        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        # Validate trainer_name in params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")

        # Use the class name as model_name if not provided in params
        model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        trainer_name = params["trainer_name"]

        # Call parent class initializer
        super().__init__(model_name, trainer_name, params=params)

        # Store configuration
        self.config = params

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Log configuration details including preprocessing settings
        logger.info(f"Initializing {model_name} model with parameters")
        if "preprocessing_advanced" in self.config:
            logger.info(
                f"Model initialized with preprocessing_advanced config: {self.config['preprocessing_advanced']}"
            )

        self.model = None

    def set_trainer(
        self, trainer_name, train_dataloader, val_dataloader, test_dataloader
    ):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            test_dataloader: DataLoader for testing data.
        """
        if trainer_name == "InceptionTimeTrainer":
            self.trainer = InceptionTimeTrainer(
                self, train_dataloader, val_dataloader, test_dataloader, self.config
            )
        else:
            raise ValueError(
                f"Trainer {trainer_name} not supported for {self.model_name}."
            )

        return self.trainer


class InceptionTimeTrainer:
    """
    Trainer class for InceptionTime models.

    This class handles the training workflow for InceptionTime models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(
        self, model, train_dataloader, val_dataloader, test_dataloader, config
    ):
        """
        Initialize the InceptionTime trainer.

        Args:
            model: The InceptionTime model to train.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            test_dataloader: DataLoader for testing data.
            config: Configuration dictionary for training parameters.
        """
        self.model_wrapper = model
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.test_loader = test_dataloader
        self.config = config

        # Set device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available() and config.get("use_cuda", True)
            else "cpu"
        )
        logger.info(f"Training on device: {self.device}")

        # Extract training parameters
        self.num_epochs = config.get("num_epochs", 60)
        self.patience = config.get("patience", 5)

        # Use the model's save_dir
        self.save_dir = model.save_dir
        self.model_save_dir = os.path.join(self.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_save_dir, "Checkpoints"), exist_ok=True)

        # Set save_path within the model_save_dir instead of using a separate path
        best_model_filename = config.get(
            "best_model_filename", f"{self.model_wrapper.model_name}_best.pt"
        )
        self.save_path = os.path.join(self.model_save_dir, best_model_filename)

        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            self.model_wrapper.model_name, self.save_dir
        )

        # Check if wandb is enabled
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb and not "wandb" in locals():
            try:
                import wandb

                logger.info("WandB imported successfully")
            except ImportError:
                logger.warning("WandB not found, disabling WandB logging")
                self.use_wandb = False

        # Data preparation
        self._prepare_data()

        # Initialize model architecture
        self._init_model()

        # Track metrics
        self.metrics = {"train_loss": [], "val_loss": [], "accuracy": 0, "rmse": 0}

    def _prepare_data(self):
        """
        Prepare data for InceptionTime by ensuring it's in 3D format.
        """
        # Use the utility function from model_util.py
        data_prep_result = prepare_data_for_model_dl(
            self.train_loader, self.config, model_name=self.model_wrapper.model_name
        )

        # Extract results
        self.reshape_needed = data_prep_result["reshape_needed"]
        self.convert_method = data_prep_result["convert_method"]
        self.converter = data_prep_result["converter"]

        # Log input data shape
        logger.info(f"Input data shape: {data_prep_result['data_shape']}")

    def _init_model(self):
        """Initialize the InceptionTime model architecture with configuration parameters."""
        try:
            # Get sample to determine input shape
            features, _ = next(iter(self.train_loader))

            # Apply the appropriate conversions to get actual input shape
            if (
                hasattr(self, "convert_method")
                and self.convert_method == "windowed_to_3d"
            ):
                features = self.converter.convert_batch_to_3d(features)
                logger.info(f"After conversion to 3D, features shape: {features.shape}")
            elif hasattr(self, "reshape_needed") and self.reshape_needed:
                features = features.unsqueeze(
                    -1
                )  # InceptionTime always uses CNN format
                logger.info(f"After reshaping, features shape: {features.shape}")

            # Get actual input dimensions
            num_channels = features.shape[1]  # Middle dimension for CNN is channels
            logger.info(f"Using {num_channels} input channels for InceptionTime")

            # Get model architecture parameters
            arch_config = self.config.get("model_architecture", {})
            depth = arch_config.get("depth", 12)
            dropout_rate = arch_config.get("dropout_rate", 0.3)

            # Define dynamic channel configurations based on input
            in_channels = [num_channels]
            out_channels = [min(256, num_channels)]

            # Build channel architecture dynamically
            for i in range(1, depth):
                # Reduce channel count as we go deeper
                prev_out = out_channels[i - 1]
                if i < depth // 3:
                    in_channels.append(
                        prev_out * 4
                    )  # 4× from concatenation in Inception block
                    out_channels.append(prev_out)
                elif i < 2 * depth // 3:
                    in_channels.append(prev_out * 4)
                    out_channels.append(max(prev_out // 2, 32))
                else:
                    in_channels.append(prev_out * 4)
                    out_channels.append(max(prev_out // 2, 16))

            # Log the channel architecture
            logger.info(f"Dynamic channel configuration - in_channels: {in_channels}")
            logger.info(f"Dynamic channel configuration - out_channels: {out_channels}")

            # Initialize the network with dynamic channels
            self.model = InceptionTimeModel.Network(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=depth,
                dropout_rate=dropout_rate,
            ).to(self.device)

            # Log model architecture
            logger.info(
                f"InceptionTime network initialized with depth {depth}, "
                + f"{sum(p.numel() for p in self.model.parameters())} parameters"
            )

            # Make model available to the wrapper
            self.model_wrapper.model = self.model

            # Calculate class weights for imbalanced datasets
            pos_weight = self._calculate_pos_weight()

            # Get optimizer settings
            opt_config = self.config.get("optimizer", {})
            optimizer_name = opt_config.get("name", "adamw").lower()
            learning_rate = opt_config.get("learning_rate", 0.01)
            weight_decay = opt_config.get("weight_decay", 0.01)

            # Initialize loss and optimizer based on config
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight).to(self.device)
            )

            if optimizer_name == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer_name == "sgd":
                self.optimizer = optim.SGD(
                    self.model.parameters(), lr=learning_rate, momentum=0.9
                )
            else:  # Default to AdamW
                self.optimizer = optim.AdamW(
                    self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
                )

            logger.info(
                f"Using optimizer: {optimizer_name} with learning rate: {learning_rate}"
            )

            # Get scheduler configuration
            scheduler_config = self.config.get("scheduler", {})
            factor = scheduler_config.get("factor", 0.9)
            scheduler_patience = scheduler_config.get("patience", 5)
            min_lr = scheduler_config.get("min_lr", 0.001)

            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=factor,
                patience=scheduler_patience,
                min_lr=min_lr,
            )

            # Initialize early stopping
            self.early_stopping = InceptionTimeModel.EarlyStopping(
                patience=self.patience, verbose=True
            )

        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise

    def _calculate_pos_weight(self):
        """Calculate positive class weight for imbalanced data."""
        try:
            all_labels = []
            for _, labels in self.train_loader:
                all_labels.extend(labels.cpu().numpy().flatten())

            all_labels = np.array(all_labels)
            neg_count = np.sum(all_labels == 0)
            pos_count = np.sum(all_labels == 1)

            if pos_count == 0:
                logger.warning("No positive samples found, using pos_weight=1.0")
                return 1.0

            weight = neg_count / pos_count
            logger.info(
                f"Class imbalance - Neg: {neg_count}, Pos: {pos_count}, Weight: {weight}"
            )
            return weight

        except Exception as e:
            logger.error(f"Error calculating class weights: {e}")
            return 1.0

    def _transform_features(self, features):
        """Transform features to the correct format for the model."""
        # Apply the appropriate conversions
        if hasattr(self, "convert_method") and self.convert_method == "windowed_to_3d":
            features = self.converter.convert_batch_to_3d(features)
        elif hasattr(self, "reshape_needed") and self.reshape_needed:
            # InceptionTime always uses CNN format (batch, channels, time_steps)
            features = features.unsqueeze(-1)  # Add time dimension

        return features

    def train(self):
        """Train the InceptionTime model using the provided data loaders."""
        logger.info("Starting InceptionTime training")

        # Setup checkpoint saving
        save_checkpoint_freq = self.config.get("save_checkpoint", 5)
        checkpoint_path = os.path.join(self.model_save_dir, "Checkpoints")
        os.makedirs(checkpoint_path, exist_ok=True)

        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            all_labels = []
            all_outputs = []

            for batch_idx, (features, labels) in enumerate(self.train_loader):
                features = self._transform_features(features)

                # Log the shape on first batch of first epoch
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"Input shape to model: {features.shape}")

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

                # Store predictions for metrics
                all_labels.append(labels.cpu().detach().numpy())
                all_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())

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
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save checkpoint periodically
            if save_checkpoint_freq > 0 and (epoch + 1) % save_checkpoint_freq == 0:
                checkpoint_name = f"{self.model_wrapper.model_name}_epoch_{epoch + 1}"
                save_torch_model(checkpoint_name, self.model, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_name}")

            # Log to WandB if enabled
            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Check early stopping
            self.early_stopping(val_loss, self.model, self.save_path)
            if self.early_stopping.early_stop:
                logger.info("Early stopping triggered")
                break

        # Load best model from early stopping
        if self.early_stopping.model_path:
            self.model.load_state_dict(torch.load(self.early_stopping.model_path))
            logger.info(f"Loaded best model from {self.early_stopping.model_path}")

        # Save final model using the utility function
        final_model_path = os.path.join(
            self.model_save_dir, f"{self.model_wrapper.model_name}_final.pt"
        )
        save_torch_model(
            f"{self.model_wrapper.model_name}_final", self.model, self.model_save_dir
        )
        logger.info(f"Saved final model to {final_model_path}")

        # Evaluate final model
        eval_results = self._evaluate()
        self.metrics.update(eval_results)

        return self.metrics

    def _validate(self):
        """Validate the model and return validation loss."""
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = self._transform_features(features)

                features, labels = (
                    features.to(self.device),
                    labels.to(self.device).float(),
                )
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                val_loss += loss.item()

        return val_loss / len(self.test_loader)

    def _evaluate(self):
        """Evaluate the model on the test set with comprehensive metrics."""
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for features, labels in self.test_loader:
                features = self._transform_features(features)

                features = features.to(self.device)
                outputs = self.model(features)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs >= 0.5).int()

                # Store results for metrics calculation
                all_labels.extend(labels.numpy().flatten())
                all_preds.extend(preds.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

                # Add results to metrics tracker
                self.metrics_tracker.add_results(preds.cpu().numpy(), labels.numpy())

        # Calculate comprehensive metrics
        metrics_summary = self.metrics_tracker.compute_overall_metrics()
        self.metrics_tracker.save_report()

        logger.info(f"Test metrics: {metrics_summary}")

        # For backwards compatibility with existing code
        accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
        rmse_score = np.sqrt(np.mean((np.array(all_labels) - np.array(all_preds)) ** 2))

        # Log results
        logger.info(f"Test accuracy: {accuracy:.4f}, RMSE: {rmse_score:.4f}")

        # Update WandB if enabled
        if self.use_wandb:
            wandb.log(
                {
                    "test_accuracy": accuracy,
                    "test_rmse": rmse_score,
                    **{f"test_{k}": v for k, v in metrics_summary.items()},
                }
            )

        return {"accuracy": accuracy, "rmse": rmse_score, **metrics_summary}

    def predict(self, features):
        """Make predictions with the trained model."""
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor if needed
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)

            # Convert 2D data to 3D based on model type
            if len(features.shape) == 2:
                features = self._transform_features(features)

            features = features.to(self.device)
            outputs = self.model(features)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).int()

        return {
            "probabilities": probs.cpu().numpy(),
            "predictions": preds.cpu().numpy(),
        }
