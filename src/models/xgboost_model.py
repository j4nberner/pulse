from typing import Dict, Any, Optional
import logging
import numpy as np
from xgboost import XGBClassifier

from src.models.pulsetemplate_model import PulseTemplateModel
from src.eval.metrics import rmse

logger = logging.getLogger("PULSE_logger")

# TODO: add saving and loading functionality

class XGBoostModel(PulseTemplateModel):
    """
    Implementation of XGBoost model for classification and regression tasks.

    Attributes:
        model: The trained XGBoost model.
        model_type: Type of the model ('classifier' or 'regressor').
        params: Parameters used for the model.
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the XGBoost model.
        
        Args:
            params: Dictionary of parameters from the config file.
            
        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        # For trainer_name we still require it to be explicitly in the params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")
            
        # Use the class name as model_name if not provided in params
        model_name = params.get("model_name", self.__class__.__name__.replace("Model", ""))
        trainer_name = params["trainer_name"]
        super().__init__(model_name, trainer_name)
        
        # Define all required XGBoost parameters
        required_xgb_params = [
            "objective", 
            "n_estimators", 
            "learning_rate", 
            "random_state", 
            "verbosity",
            "max_depth", 
            "gamma", 
            "min_child_weight", 
            "subsample", 
            "colsample_bytree",
            "reg_alpha", 
            "reg_lambda", 
            "scale_pos_weight", 
            "n_jobs", "tree_method",
            "eval_metric", 
            "early_stopping_rounds"
        ]
        
        # Check if all required XGBoost parameters exist in config
        missing_params = [param for param in required_xgb_params if param not in params]
        if missing_params:
            raise KeyError(f"Required XGBoost parameters missing from config: {missing_params}")
        
        # For XGBoost 2.0.3: include early_stopping_rounds in model initialization
        model_params = {param: params[param] for param in required_xgb_params}
        
        # Store early_stopping_rounds for training
        self.early_stopping_rounds = params["early_stopping_rounds"]
        
        # Log the parameters being used
        logger.info(f"Initializing XGBoost with parameters: {model_params}")
        
        # Initialize the XGBoost model with parameters from config
        self.model = XGBClassifier(**model_params)

    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        if trainer_name == "XGBoostTrainer":
            self.trainer = XGBoostTrainer(self, train_dataloader, test_dataloader)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for XGBoost.")


class XGBoostTrainer:
    """
    Trainer class for XGBoost models.

    This class handles the training workflow for XGBoost models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_dataloader, test_dataloader) -> None:
        """
        Initialize the XGBoost trainer.
        
        Args:
            model: The XGBoost model to train.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        """Train the XGBoost model using the provided data loaders."""
        logger.info(f"Starting training process for XGBoost model...")

        # Extract data from dataloader
        X_train, y_train = [], []
        X_test, y_test = [], []

        for batch in self.train_dataloader:
            features, labels = batch
            # Convert PyTorch tensors to NumPy arrays
            X_train.extend(features.numpy())
            y_train.extend(labels.numpy().squeeze())

        for batch in self.test_dataloader:
            features, labels = batch
            X_test.extend(features.numpy())
            y_test.extend(labels.numpy().squeeze())

        # Convert lists to numpy arrays (necessary after extend)
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Log shapes before model training (after conversion)
        logger.info(f"Before XGBoost training - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"Before XGBoost training - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Log dataset sizes
        logger.info(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train model
        self.model.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        logger.info("XGBoost model trained successfully.")

        # Evaluate the model
        y_pred = self.model.model.predict(X_test)
        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)