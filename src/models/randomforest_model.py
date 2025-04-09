from typing import Dict, Any, Optional
import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.models.pulsetemplate_model import PulseTemplateModel
from src.eval.metrics import rmse
import psutil
import os
import sys

logger = logging.getLogger("PULSE_logger")


class RandomForestModel(PulseTemplateModel):
    """
    Implementation of RandomForest model for classification and regression tasks.

    Attributes:
        model: The trained RandomForest model.
        model_type: Type of the model ('classifier' or 'regressor').
        params: Parameters used for the model.
    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the RandomForest model.

        Args:
            params: Dictionary of parameters from the config file.

        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        # For trainer_name we still require it to be explicitly in the params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")

        # Use the class name as model_name if not provided in params
        model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        trainer_name = params["trainer_name"]
        super().__init__(model_name, trainer_name)

        # Define all required scikit-learn RandomForest parameters
        required_rf_params = [
            "n_estimators", "n_jobs", "max_depth", 
            "min_samples_split", "min_samples_leaf", "max_features",
            "bootstrap", "oob_score", "random_state", "verbose",
            "criterion", "max_leaf_nodes", "min_impurity_decrease",
            "max_samples", "class_weight", "ccp_alpha"
        ]

        # Check if all required RandomForest parameters exist in config
        missing_params = [param for param in required_rf_params if param not in params]
        if missing_params:
            raise KeyError(
                f"Required RandomForest parameters missing from config: {missing_params}"
            )

        # Extract RandomForest parameters from config
        rf_params = {param: params[param] for param in required_rf_params}

        # Log the parameters being used
        logger.info(f"Initializing RandomForest with parameters: {rf_params}")

        # Initialize the RandomForest model with parameters from config
        self.model = RandomForestClassifier(**rf_params)

    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        if trainer_name == "RandomForestTrainer":
            self.trainer = RandomForestTrainer(self, train_dataloader, test_dataloader)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for RandomForest.")


class RandomForestTrainer:
    """
    Trainer class for RandomForest models.

    This class handles the training workflow for RandomForest models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_dataloader, test_dataloader) -> None:
        """
        Initialize the RandomForest trainer.

        Args:
            model: The RandomForest model to train.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        """Train the RandomForest model using the provided data loaders."""
        logger.info("Starting training process for RandomForest model...")

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
        logger.info(f"Before RandomForest training - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        logger.info(f"Before RandomForest training - X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        # Log dataset sizes
        logger.info(
            f"Training on {len(X_train)} samples, testing on {len(X_test)} samples"
        )

        # dummy training loop
        self.model.model.fit(X_train, y_train)
        logger.info("RandomForest model trained successfully.")

        # Evaluate the model
        y_pred = self.model.model.predict(X_test)
        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)
