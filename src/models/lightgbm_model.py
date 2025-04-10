from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from src.models.pulsetemplate_model import PulseTemplateModel
from src.eval.metrics import rmse

logger = logging.getLogger("PULSE_logger")

class LightGBMModel(PulseTemplateModel):
    """
    Implementation of LightGBM model for classification and regression tasks.

    Attributes:
        model: The trained LightGBM model.
        model_type: Type of the model ('classifier' or 'regressor').
        params: Parameters used for the model.
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the LightGBM model.
        
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
        
        # Define all required LightGBM parameters
        required_lgb_params = [
            "objective", 
            "n_estimators", 
            "learning_rate", 
            "random_state", 
            "verbose",
            "max_depth", 
            "num_leaves", 
            "min_child_samples", 
            "subsample", 
            "colsample_bytree",
            "reg_alpha", 
            "reg_lambda", 
            "n_jobs", 
            "boosting_type",
            "metric", 
            "early_stopping_rounds"
        ]
        
        # Check if all required LightGBM parameters exist in config
        missing_params = [param for param in required_lgb_params if param not in params]
        if missing_params:
            raise KeyError(f"Required LightGBM parameters missing from config: {missing_params}")
        
        # Store early_stopping_rounds for training
        self.early_stopping_rounds = params["early_stopping_rounds"]
        
        # Extract LightGBM parameters from config
        model_params = {param: params[param] for param in required_lgb_params if param != "early_stopping_rounds"}
        
        # Log the parameters being used
        logger.info(f"Initializing LightGBM with parameters: {model_params}")
        
        # Initialize the LightGBM model with parameters from config
        self.model = LGBMClassifier(**model_params)

    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        if trainer_name == "LightGBMTrainer":
            self.trainer = LightGBMTrainer(self, train_dataloader, test_dataloader)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for LightGBM.")


class LightGBMTrainer:
    """
    Trainer class for LightGBM models.

    This class handles the training workflow for LightGBM models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_dataloader, test_dataloader) -> None:
        """
        Initialize the LightGBM trainer.
        
        Args:
            model: The LightGBM model to train.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        """Train the LightGBM model using the provided data loaders."""
        logger.info(f"Starting training process for LightGBM model...")

        # Extract data from dataloader and preserve feature names
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        # Rest of your data extraction code
        for batch in self.train_dataloader:
            features, labels = batch
            X_train.extend(features.numpy())
            y_train.extend(labels.numpy().squeeze())

        for batch in self.test_dataloader:
            features, labels = batch
            X_test.extend(features.numpy())
            y_test.extend(labels.numpy().squeeze())

        # Convert lists to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Log shapes before model training
        logger.info(f"Before LightGBM training - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"Before LightGBM training - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Access the original x-dataframe from the TorchDatasetWrapper to get column names (e.g. for feature importance analysis)
        feature_names = None
        if hasattr(self.train_dataloader.dataset, 'X') and isinstance(self.train_dataloader.dataset.X, pd.DataFrame):
            feature_names = list(self.train_dataloader.dataset.X.columns)
            logger.info(f"Extracted {len(feature_names)} feature names from original DataFrame")

        # Create fallback feature names if needed
        if feature_names is None or len(feature_names) != X_train.shape[1]:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
            logger.info(f"Using generated feature names (couldn't access original names)")
        
        # Create early stopping callback with verbose setting based on model configuration
        early_stopping_callback = early_stopping(
            stopping_rounds=self.model.early_stopping_rounds,
            first_metric_only=True,
            verbose=(self.model.model.verbose > 0)  # Use model's verbose setting
        )
        
        # Train model with explicit feature names
        self.model.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping_callback],
            feature_name=feature_names  # Pass the extracted column names
        )
        logger.info("LightGBM model trained successfully.")
        
        # Create DataFrame with feature names for prediction to avoid warnings
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Evaluate the model
        y_pred = self.model.model.predict(X_test_df)
        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)