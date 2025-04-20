from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import os
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import confusion_matrix
import wandb

from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    save_sklearn_model,
    prepare_data_for_model_ml,
)
from src.eval.metrics import MetricsTracker, rmse

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
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

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
            "early_stopping_rounds",
        ]

        # Check if all required LightGBM parameters exist in config
        missing_params = [param for param in required_lgb_params if param not in params]
        if missing_params:
            raise KeyError(
                f"Required LightGBM parameters missing from config: {missing_params}"
            )

        # Store early_stopping_rounds for training
        self.early_stopping_rounds = params["early_stopping_rounds"]

        # Extract LightGBM parameters from config
        model_params = {
            param: params[param]
            for param in required_lgb_params
            if param != "early_stopping_rounds"
        }

        # Log the parameters being used
        logger.info(f"Initializing LightGBM with parameters: {model_params}")

        # Initialize the LightGBM model with parameters from config
        self.model = LGBMClassifier(**model_params)

    def set_trainer(self, trainer_name, train_loader, val_loader, test_loader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. (not used)
            test_loader: DataLoader for testing data.
        """
        if trainer_name == "LightGBMTrainer":
            self.trainer = LightGBMTrainer(self, train_loader, val_loader, test_loader)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for LightGBM.")


class LightGBMTrainer:
    """
    Trainer class for LightGBM models.

    This class handles the training workflow for LightGBM models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_loader, val_loader, test_loader) -> None:
        """
        Initialize the LightGBM trainer.

        Args:
            model: The LightGBM model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. (not used)
            test_loader: DataLoader for testing data.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.wandb = model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        """Train the LightGBM model using the provided data loaders."""
        logger.info("Starting training process for LightGBM model...")

        # Use the utility function to prepare data
        prepared_data = prepare_data_for_model_ml(
            self.train_loader, self.val_loader, logger_instance=logger
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

        # Create early stopping callback with verbose setting based on model configuration
        early_stopping_callback = early_stopping(
            stopping_rounds=self.model.early_stopping_rounds,
            first_metric_only=True,
            verbose=(self.model.model.verbose > 0),  # Use model's verbose setting
        )

        # Train model with explicit feature names
        self.model.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[early_stopping_callback],
            feature_name=feature_names,  # Pass the extracted column names
        )
        logger.info("LightGBM model trained successfully.")

        # Create DataFrame with feature names for prediction to avoid warnings
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Evaluate the model
        metrics_tracker = MetricsTracker(
            self.model.model_name, self.model.task_name, self.model.dataset_name
        )
        y_pred = self.model.model.predict(X_test_df)
        y_pred_proba = self.model.model.predict_proba(X_test_df)
        metrics_tracker.add_results(y_pred, y_test)

        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()

        # Save the model
        save_sklearn_model(self.model.model_name, self.model.model, self.model_save_dir)

        # Log metrics to wandb
        if self.wandb:
            # Get metrics from the metrics tracker
            metrics = metrics_tracker.compute_overall_metrics()

            # Log all metrics from the overall summary
            if "overall" in metrics:
                wandb.log(metrics["overall"])

            # Create and log confusion matrix
            y_pred_binary = (y_pred >= 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred_binary)
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        preds=y_pred_binary,
                        y_true=y_test,
                        class_names=["Negative", "Positive"],
                    )
                }
            )

            # Create and log ROC curve
            wandb.log(
                {
                    "roc_curve": wandb.plot.roc_curve(
                        y_true=y_test,
                        y_probas=y_pred_proba,
                        labels=["Negative", "Positive"],
                    )
                }
            )

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
