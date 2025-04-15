from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import os
from xgboost import XGBClassifier
import wandb

from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    save_sklearn_model,
    prepare_data_for_model_ml,
)
from src.eval.metrics import MetricsTracker, rmse
from src.eval.metrics import calculate_all_metrics, calc_metric_stats

logger = logging.getLogger("PULSE_logger")


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
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

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
            "n_jobs",
            "tree_method",
            "eval_metric",
            "early_stopping_rounds",
        ]

        # Check if all required XGBoost parameters exist in config
        missing_params = [param for param in required_xgb_params if param not in params]
        if missing_params:
            raise KeyError(
                f"Required XGBoost parameters missing from config: {missing_params}"
            )

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
        self.wandb = model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        """Train the XGBoost model using the provided data loaders."""
        logger.info("Starting training process for XGBoost model...")

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
        self.model.model.fit(
            X_train, y_train, eval_set=[(X_test, y_test)], verbose=False
        )
        logger.info("XGBoost model trained successfully.")

        # Create DataFrame with feature names for prediction to avoid warnings
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Evaluate the model
        y_pred = self.model.model.predict(X_test_df)
        y_pred_proba = self.model.model.predict_proba(X_test_df)
        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)

        # Log metrics to wandb
        if self.wandb:
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

        # Save the model
        save_sklearn_model(self.model.model_name, self.model.model, self.model_save_dir)
