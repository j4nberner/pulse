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
from src.util.model_util import save_torch_model, load_torch_model, prepare_data_for_model_dl
from src.eval.metrics import calculate_all_metrics, calc_metric_stats, MetricsTracker

# Set up logger
logger = logging.getLogger("PULSE_logger")

class GRUModel(PulseTemplateModel):
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
            self.best_val_loss = float('inf')
            self.delta = delta
            self.best_state_dict = None

        def __call__(self, val_loss, model):
            if val_loss > self.best_val_loss - self.delta:
                self.counter += 1
                if self.verbose:
                    logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                logger.info(f'Validation loss decreased ({self.best_val_loss:.6f} --> {val_loss:.6f}). Saving model state...')
            # Store state dict in memory
            self.best_state_dict = model.state_dict().copy()
            self.best_val_loss = val_loss
    
    class Network(nn.Module):
        def __init__(self, input_dim, hidden_dim=256, layer_dim=2, output_dim=1, dropout_rate=0.3):
            super().__init__()
            
            self.hidden_dim = hidden_dim
            self.layer_dim = layer_dim
            self.dropout_rate = dropout_rate
            
            # GRU layers
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=layer_dim,
                batch_first=True,
                dropout=dropout_rate if layer_dim > 1 else 0
            )
            
            # Fully connected layers for classification
            self.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 64),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(64, 16),
                nn.LeakyReLU(),
                nn.Linear(16, output_dim)
            )
            
        def forward(self, x):
            # Initialize hidden state with zeros
            batch_size = x.size(0)
            h0 = torch.zeros(self.layer_dim, batch_size, self.hidden_dim).to(x.device)
            
            # Forward propagate the GRU
            out, _ = self.gru(x, h0)
            
            # Extract the output of the last time step
            out = out[:, -1, :]
            
            # Feed to fully connected layers
            out = self.fc(out)
            
            return out
    
    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the GRU model.
        
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

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")
        
        # Log configuration details including preprocessing settings
        logger.info(f"Initializing {model_name} model with parameters")
        if "preprocessing_advanced" in self.config:
            logger.info(f"Model initialized with preprocessing_advanced config: {self.config['preprocessing_advanced']}")
        
        self.model = None
        
    def set_trainer(self, trainer_name, train_dataloader, val_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            test_dataloader: DataLoader for testing data.
        """
        if trainer_name == "GRUTrainer":
            self.trainer = GRUTrainer(self, train_dataloader, val_dataloader, test_dataloader, self.config)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for {self.model_name}.")
        
        return self.trainer

class GRUTrainer:
    """
    Trainer class for GRU models.
    
    This class handles the training workflow for GRU models
    including data preparation, model training, evaluation and saving.
    """
    
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, config):
        """
        Initialize the GRU trainer.
        
        Args:
            model: The GRU model to train.
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
        self.device = torch.device("cuda" if torch.cuda.is_available() and 
                                  config.get("use_cuda", True) else "cpu")
        logger.info(f"Training on device: {self.device}")
        
        # Extract training parameters
        self.num_epochs = config.get("num_epochs", 60)
        self.patience = config.get("patience", 5)
        
        # Use the model's save_dir
        self.save_dir = model.save_dir
        self.model_save_dir = os.path.join(self.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.model_save_dir, "Checkpoints"), exist_ok=True)
        
        # Set save_path within the model_save_dir
        best_model_filename = config.get("best_model_filename", f"{self.model_wrapper.model_name}_best.pt")
        self.save_path = os.path.join(self.model_save_dir, best_model_filename)
        
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
        Prepare data for GRU by ensuring it's in the correct format.
        """
        # Use the utility function from model_util.py
        data_prep_result = prepare_data_for_model_dl(
            self.train_loader,
            self.config,
            model_name=self.model_wrapper.model_name
        )
        
        # Extract results
        self.reshape_needed = data_prep_result["reshape_needed"]
        self.convert_method = data_prep_result["convert_method"]
        self.converter = data_prep_result["converter"]
        
        # Log input data shape
        logger.info(f"Input data shape: {data_prep_result['data_shape']}")
    
    def _init_model(self):
        """Initialize the GRU model architecture with configuration parameters."""
        try:
            # Get sample to determine input shape
            features, _ = next(iter(self.train_loader))
            
            # Apply the appropriate conversions to get actual input shape
            if hasattr(self, 'convert_method') and self.convert_method == "windowed_to_3d":
                features = self.converter.convert_batch_to_3d(features)
                logger.info(f"After conversion to 3D, features shape: {features.shape}")
            elif hasattr(self, 'reshape_needed') and self.reshape_needed:
                if len(features.shape) == 2:
                    features = features.unsqueeze(1)  # Add feature dimension for GRU
                logger.info(f"After reshaping, features shape: {features.shape}")
            
            # Get model architecture parameters
            arch_config = self.config.get("model_architecture", {})
            
            # Extract input dimensions - for GRU, we need the feature dimension
            input_dim = features.shape[2]
                
            logger.info(f"Input dimension for GRU: {input_dim}")
            
            # Get GRU parameters from config
            hidden_dim = arch_config.get("hidden_dim", 256)
            layer_dim = arch_config.get("layer_dim", 2)
            dropout_rate = arch_config.get("dropout_rate", 0.3)
            
            # Initialize the network
            self.model = GRUModel.Network(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                layer_dim=layer_dim,
                output_dim=1,  # Binary classification
                dropout_rate=dropout_rate
            ).to(self.device)
            
            # Log model architecture
            logger.info(f"GRU network initialized with hidden_dim {hidden_dim}, " +
                       f"layer_dim {layer_dim}, " +
                       f"{sum(p.numel() for p in self.model.parameters())} parameters")
            
            # Make model available to the wrapper
            self.model_wrapper.model = self.model
            
            # Calculate class weights for imbalanced datasets
            pos_weight = self._calculate_pos_weight()
            
            # Get optimizer settings
            opt_config = self.config.get("optimizer", {})
            optimizer_name = opt_config.get("name", "adam").lower()
            learning_rate = opt_config.get("learning_rate", 0.001)
            weight_decay = opt_config.get("weight_decay", 1e-6)
            
            # Initialize loss and optimizer based on config
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(self.device))
            
            if optimizer_name == "adamw":
                self.optimizer = optim.AdamW(self.model.parameters(), 
                                          lr=learning_rate,
                                          weight_decay=weight_decay)
            elif optimizer_name == "sgd":
                self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
            else:  # Default to Adam
                self.optimizer = optim.Adam(self.model.parameters(), 
                                         lr=learning_rate,
                                         weight_decay=weight_decay)
            
            logger.info(f"Using optimizer: {optimizer_name} with learning rate: {learning_rate}")
            
            # Get scheduler configuration
            scheduler_config = self.config.get("scheduler", {})
            factor = scheduler_config.get("factor", 0.5)
            scheduler_patience = scheduler_config.get("patience", 3)
            min_lr = scheduler_config.get("min_lr", 1e-6)
            
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=factor, 
                patience=scheduler_patience, min_lr=min_lr
            )
            
            # Initialize early stopping
            self.early_stopping = GRUModel.EarlyStopping(patience=self.patience, verbose=True)
            
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
        
    def _transform_features(self, features):
        """Transform features to the correct format for the GRU model."""
        # Apply the appropriate conversions
        if hasattr(self, 'convert_method') and self.convert_method == "windowed_to_3d":
            features = self.converter.convert_batch_to_3d(features)
        elif hasattr(self, 'reshape_needed') and self.reshape_needed:
            if len(features.shape) == 2:  # (batch, features)
                features = features.unsqueeze(1)  # Add seq-length pseudo-dimension, for GRU, we need (batch, seq_len, features)
        
        return features
    
    def train(self):
        """Train the GRU model using the provided data loaders."""
        logger.info("Starting GRU training")
        
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
            
            # Log to WandB if enabled
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Check early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                break
        
            # After training loop, load best model weights and save final model
            if self.early_stopping.best_state_dict:
                self.model.load_state_dict(self.early_stopping.best_state_dict)
            save_torch_model(f"{self.model_wrapper.model_name}", self.model, self.model_save_dir)

        # Evaluate final model
        eval_results = self._evaluate()
        self.metrics.update(eval_results)
        
        return self.metrics
    
    def _validate(self):
        """Validate the model using the validation set and return validation loss."""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = self._transform_features(features)
                
                features, labels = features.to(self.device), labels.to(self.device).float()
                outputs = self.model(features)
                loss = self.criterion(outputs.squeeze(), labels.squeeze())
                val_loss += loss.item()
        
        return val_loss / len(self.val_loader)
    
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
            
            # Apply necessary transformations
            features = self._transform_features(features)
            
            features = features.to(self.device)
            outputs = self.model(features)
            probs = torch.sigmoid(outputs).squeeze()
            preds = (probs >= 0.5).int()
        
        return {
            "probabilities": probs.cpu().numpy(),
            "predictions": preds.cpu().numpy()
        }