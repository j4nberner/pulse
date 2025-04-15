from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import psutil
import os
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier
import wandb

from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    save_sklearn_model,
    prepare_data_for_model_ml,
)
from src.eval.metrics import MetricsTracker, rmse
from src.eval.metrics import calculate_all_metrics, calc_metric_stats

# TODO: fix evaluation metrics (report is empty) and wandb evaluation

# Filter the specific warning about feature names
# (This is because training is done with np arrays and prediction with pd dataframe to preserve feature names for feature importance etc.)
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but RandomForestClassifier was fitted without feature names",
)

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
        super().__init__(model_name, trainer_name, params=params)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

        # Define all required scikit-learn RandomForest parameters
        required_rf_params = [
            "n_estimators",
            "n_jobs",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
            "bootstrap",
            "oob_score",
            "random_state",
            "verbose",
            "criterion",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "max_samples",
            "class_weight",
            "ccp_alpha",
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
        self.wandb = model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        """Train the RandomForest model using the provided data loaders."""
        logger.info("Starting training process for RandomForest model...")

        # Use the utility function to prepare data
        prepared_data = prepare_data_for_model_ml(
            self.train_dataloader, self.test_dataloader, logger_instance=logger
        )

        # Extract all data from the prepared_data dictionary
        X_train = prepared_data["X_train"]
        y_train = prepared_data["y_train"]
        X_test = prepared_data["X_test"]
        y_test = prepared_data["y_test"]
        feature_names = prepared_data["feature_names"]

        # Log training start to wandb
        if self.wandb:
            wandb.log({"train_samples": len(X_train), "test_samples": len(X_test)})

        # Train the model
        self.model.model.fit(X_train, y_train)
        logger.info("RandomForest model trained successfully.")

        # Create DataFrame with feature names for prediction to avoid warnings
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Evaluate the model
        metrics_tracker = MetricsTracker(self.model.model_name, self.model.save_dir)

        y_pred = self.model.model.predict(X_test_df)
        y_pred_proba = self.model.model.predict_proba(X_test_df)
        metrics_tracker.add_results(y_pred, y_test)
        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)

        # Log metrics to wandb
        if self.wandb:
            # TODO: update wandb to use the new metrics tracker
            wandb.log({"rmse": rmse_score})

            # Calculate and log other metrics
            metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
            for metric_name, value in metrics.items():
                wandb.log({metric_name: value})

            # Log feature importance
            if hasattr(self.model.model, "feature_importances_"):
                feature_importance = {
                    f"importance_{feature_names[i]}": imp
                    for i, imp in enumerate(self.model.model.feature_importances_)
                }
                wandb.log(feature_importance)

                # Create a bar chart of feature importances
                importance_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": self.model.model.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)

                wandb.log(
                    {
                        "feature_importance": wandb.plot.bar(
                            wandb.Table(dataframe=importance_df),
                            "feature",
                            "importance",
                            title="Feature Importance",
                        )
                    }
                )

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()

        # Save the model
        save_sklearn_model(self.model.model_name, self.model.model, self.model_save_dir)
