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
from src.util.model_util import save_torch_model, load_torch_model
from src.eval.metrics import calculate_all_metrics, calc_metric_stats, MetricsTracker

# TODO: Why are calculate_all_metrics and cal_metric_stats imported but not used?

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
            self.bottleneck = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")    
            self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding="same")
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same")
            self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding="same")

            self.conv_pool = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")       
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
            self.bottleneck = nn.Conv1d(in_channels, out_channels * 4, kernel_size=1, padding="same")
            self.batch_norm = nn.BatchNorm1d(out_channels * 4)

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
            self.best_val_loss = float('inf')
            self.delta = delta
            self.model_path = None

        def __call__(self, val_loss, model, model_path):
            if val_loss > self.best_val_loss - self.delta:
                self.counter += 1
                if self.verbose:
                    logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(val_loss, model, model_path)
                self.counter = 0

        def save_checkpoint(self, val_loss, model, model_path):
            if self.verbose:
                logger.info(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model...')
            torch.save(model.state_dict(), model_path)
            self.best_val_loss = val_loss
            self.model_path = model_path
    
    class Network(nn.Module):
        def __init__(self, in_channels, out_channels, depth, dropout_rate=0.3):
            super(InceptionTimeModel.Network, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.depth = depth
            
            modules = OrderedDict()
            
            for d in range(depth):
                modules[f'inception_{d}'] = InceptionTimeModel.Inception(
                    in_channels=in_channels[d], 
                    out_channels=out_channels[d]
                )

                if d % 3 == 2:
                    modules[f'residual_{d}'] = InceptionTimeModel.Residual(
                        in_channels=in_channels[0] if d == 2 else 4 * out_channels[d-1],
                        out_channels=out_channels[d],
                    )
            
            modules['avg_pool'] = InceptionTimeModel.Lambda(f=lambda x: torch.mean(x, dim=-1))
            modules['dropout'] = nn.Dropout(dropout_rate) 
            modules['linear1'] = nn.Linear(in_features=4 * out_channels[-1], out_features=64)
            modules['linear2'] = nn.Linear(in_features=64, out_features=16)
            modules['linear3'] = nn.Linear(in_features=16, out_features=1)

            self.model = nn.Sequential(modules)

        def forward(self, input_tensor):
            residual_output = None
            for d in range(self.depth):
                inception_output = self.model.get_submodule(f'inception_{d}')(input_tensor if d == 0 else residual_output)
                if d % 3 == 2:
                    residual_output = self.model.get_submodule(f'residual_{d}')(input_tensor, inception_output)
                    input_tensor = residual_output
                else:
                    residual_output = inception_output

            y = self.model.get_submodule('avg_pool')(residual_output)
            y = self.model.get_submodule('dropout')(y)
            y = self.model.get_submodule('linear1')(y)
            y = F.leaky_relu(y)
            y = self.model.get_submodule('dropout')(y)
            y = self.model.get_submodule('linear2')(y)
            y = F.leaky_relu(y)
            y = self.model.get_submodule('linear3')(y)
            
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
        model_name = params.get("model_name", self.__class__.__name__.replace("Model", ""))
        trainer_name = params["trainer_name"]
        
        # Call parent class initializer
        super().__init__(model_name, trainer_name)
        
        # Store configuration
        self.config = params
        
        # Log configuration details including preprocessing settings
        logger.info(f"Initializing {model_name} model with parameters")
        if "preprocessing_advanced" in self.config:
            logger.info(f"Model initialized with preprocessing_advanced config: {self.config['preprocessing_advanced']}")
        
        self.model = None
        
    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        if trainer_name == "InceptionTimeTrainer":
            self.trainer = InceptionTimeTrainer(self, train_dataloader, test_dataloader, self.config)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for {self.model_name}.")
        
        return self.trainer


class InceptionTimeTrainer:
    """
    Trainer class for InceptionTime models.
    
    This class handles the training workflow for InceptionTime models
    including data preparation, model training, evaluation and saving.
    """
    
    def __init__(self, model, train_dataloader, test_dataloader, config):
        """
        Initialize the InceptionTime trainer.
        
        Args:
            model: The InceptionTime model to train.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
            config: Configuration dictionary for training parameters.
        """
        self.model_wrapper = model
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader
        self.config = config
  
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  config.get("use_cuda", True) else "cpu")
        logger.info(f"Training on device: {self.device}")
        
        # Extract training parameters
        self.num_epochs = config.get("num_epochs", 60)
        self.patience = config.get("patience", 5)
        self.save_path = config.get("save_path", "output/models/inceptiontime_best.pt")
        
        # Set up model save directory
        self.save_dir = config.get("save_dir", os.path.join(os.getcwd(), "output"))
        self.model_save_dir = os.path.join(self.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_save_dir, "Checkpoints"), exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            self.model_wrapper.model_name,
            self.save_dir
        )
        
        # Check if wandb is enabled
        self.use_wandb = config.get("use_wandb", False)
        if self.use_wandb and not 'wandb' in locals():
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
        self.metrics = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": 0,
            "rmse": 0
        }
    
    def _prepare_data(self):
        """
        Prepare data for InceptionTime by ensuring it's in 3D format.
        """
        # Check if windowing is enabled in config
        windowing_enabled = False
        
        # Check for preprocessing_advanced config
        if "preprocessing_advanced" in self.config:
            preprocessing_advanced = self.config["preprocessing_advanced"]
            if "windowing" in preprocessing_advanced:
                windowing_config = preprocessing_advanced["windowing"]
                
                # Get the enabled value with proper type handling
                if "enabled" in windowing_config:
                    enabled_value = windowing_config["enabled"]
                    
                    # Handle different boolean representations
                    if isinstance(enabled_value, bool):
                        windowing_enabled = enabled_value
                    else:
                        # Convert string/other types to boolean
                        enabled_str = str(enabled_value).lower()
                        windowing_enabled = (enabled_str == "true" or enabled_str == "1")
        
        logger.info(f"Data preparation for InceptionTime - Windowing enabled: {windowing_enabled}")
        
        try:
            # Flag to track if we need to reshape during batching
            self.reshape_needed = False
            self.convert_method = None
            
            # Get a batch to inspect shape
            features, labels = next(iter(self.train_loader))
            
            # Check data dimensionality and decide on conversion method
            if len(features.shape) == 3:
                logger.info(f"Input data is already 3D with shape {features.shape}")
            else:
                logger.info(f"Input data is 2D with shape {features.shape}, will convert to 3D")
                
                # Initialize the data converter
                self.converter = WindowedDataTo3D(
                    logger=logger,
                    model_name=self.model_wrapper.model_name,
                    config=self.config
                )
                
                # If windowing was already applied, use 3D conversion
                if windowing_enabled:
                    self.convert_method = "windowed_to_3d"
                    logger.info("Will use WindowedDataTo3D converter for batches")
                    
                    # Test conversion on a sample batch
                    converted_batch = self.converter.convert_batch_to_3d(features)
                    logger.info(f"Converted shape: {converted_batch.shape}")
                else:
                    # Simple reshaping needed (handled during training)
                    self.reshape_needed = True
                    logger.info("Will use simple reshaping for batches")
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
    
    def _init_model(self):
        """Initialize the InceptionTime model architecture with configuration parameters."""
        try:
            # Get sample to determine input shape
            features, _ = next(iter(self.train_loader))
            num_features = features.shape[1]  # Assuming batch_size, features, sequence_length
            
            # Get model architecture parameters
            arch_config = self.config.get("model_architecture", {})
            depth = arch_config.get("depth", 12)
            dropout_rate = arch_config.get("dropout_rate", 0.3)
            
            # Define channel configurations
            in_channels = [num_features, 1024, 1024, 512, 512, 512, 512, 512, 512, 256, 256, 256]
            out_channels = [256, 256, 128, 128, 128, 128, 128, 128, 64, 64, 64, 32]
            
            # Initialize the network
            self.model = InceptionTimeModel.Network(
                in_channels=in_channels,
                out_channels=out_channels,
                depth=depth,
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # Log model architecture
            logger.info(f"InceptionTime network initialized with depth {depth}, " +
                       f"{sum(p.numel() for p in self.model.parameters())} parameters")
            
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
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
            
            if optimizer_name == "adam":
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer_name == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            else:  # Default to AdamW
                self.optimizer = optim.AdamW(self.model.parameters(), 
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
            
            logger.info(f"Using optimizer: {optimizer_name} with learning rate: {learning_rate}")
            
            # Get scheduler configuration
            scheduler_config = self.config.get("scheduler", {})
            factor = scheduler_config.get("factor", 0.9)
            scheduler_patience = scheduler_config.get("patience", 5)
            min_lr = scheduler_config.get("min_lr", 0.001)
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, 
                patience=scheduler_patience, min_lr=min_lr
            )
            
            # Initialize early stopping
            self.early_stopping = InceptionTimeModel.EarlyStopping(patience=self.patience, verbose=True)
            
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
            logger.info(f"Class imbalance - Neg: {neg_count}, Pos: {pos_count}, Weight: {weight}")
            return weight
            
        except Exception as e:
            logger.error(f"Error calculating class weights: {e}")
            return 1.0
    
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
                # Convert 2D data to 3D based on model type
                if hasattr(self, 'convert_method') and self.convert_method == "windowed_to_3d":
                    features = self.converter.convert_batch_to_3d(features)
                elif hasattr(self, 'reshape_needed') and self.reshape_needed:
                    if self.converter.model_type == "CNN":
                        features = features.unsqueeze(-1)  # (batch, features, 1)
                    else:  # RNN format
                        features = features.unsqueeze(1)   # (batch, 1, features)
                
                # Log the shape on first batch of first epoch
                if batch_idx == 0 and epoch == 0:
                    logger.info(f"Input shape to model: {features.shape}")
                
                features, labels = features.to(self.device), labels.to(self.device).float()
                
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
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint periodically
            if save_checkpoint_freq > 0 and (epoch + 1) % save_checkpoint_freq == 0:
                checkpoint_name = f"{self.model_wrapper.model_name}_epoch_{epoch + 1}"
                save_torch_model(checkpoint_name, self.model, checkpoint_path)
                logger.info(f"Saved checkpoint: {checkpoint_name}")
            
            # Log to WandB if enabled
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
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
        final_model_path = os.path.join(self.model_save_dir, f"{self.model_wrapper.model_name}_final.pt")
        save_torch_model(f"{self.model_wrapper.model_name}_final", self.model, self.model_save_dir)
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
                # Convert 2D data to 3D based on model type
                if hasattr(self, 'convert_method') and self.convert_method == "windowed_to_3d":
                    features = self.converter.convert_batch_to_3d(features)
                elif hasattr(self, 'reshape_needed') and self.reshape_needed:
                    if self.converter.model_type == "CNN":
                        features = features.unsqueeze(-1)  # (batch, features, 1)
                    else:  # RNN format
                        features = features.unsqueeze(1)   # (batch, 1, features)
                
                features, labels = features.to(self.device), labels.to(self.device).float()
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
                # Convert 2D data to 3D based on model type
                if hasattr(self, 'convert_method') and self.convert_method == "windowed_to_3d":
                    features = self.converter.convert_batch_to_3d(features)
                elif hasattr(self, 'reshape_needed') and self.reshape_needed:
                    if self.converter.model_type == "CNN":
                        features = features.unsqueeze(-1)  # (batch, features, 1)
                    else:  # RNN format
                        features = features.unsqueeze(1)   # (batch, 1, features)
                
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
            wandb.log({
                "test_accuracy": accuracy,
                "test_rmse": rmse_score,
                **{f"test_{k}": v for k, v in metrics_summary.items()}
            })
        
        return {
            "accuracy": accuracy,
            "rmse": rmse_score,
            **metrics_summary
        }
    
    def predict(self, features):
        """Make predictions with the trained model."""
        self.model.eval()
        with torch.no_grad():
            # Convert input to tensor if needed
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features, dtype=torch.float32)
            
            # Convert 2D data to 3D based on model type
            if len(features.shape) == 2:
                if hasattr(self, 'convert_method') and self.convert_method == "windowed_to_3d":
                    features = self.converter.convert_batch_to_3d(features)
                elif hasattr(self, 'reshape_needed') and self.reshape_needed:
                    if self.converter.model_type == "CNN":
                        features = features.unsqueeze(-1)  # (batch, features, 1)
                    else:  # RNN format
                        features = features.unsqueeze(1)   # (batch, 1, features)
            
            features = features.to(self.device)
            outputs = self.model(features)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).int()
        
        return {
            "probabilities": probs.cpu().numpy(),
            "predictions": preds.cpu().numpy()
        }