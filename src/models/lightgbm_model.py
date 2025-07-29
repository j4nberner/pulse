import logging
import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseModel
from src.util.model_util import (prepare_data_for_model_convml,
                                 save_sklearn_model)

logger = logging.getLogger("PULSE_logger")


class LightGBMModel(PulseModel):
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
        model_name = kwargs.pop("model_name", "LightGBM")
        trainer_name = params["trainer_name"]
        super().__init__(
            model_name=model_name, params=params, trainer_name=trainer_name, **kwargs
        )

        # Define all required LightGBM parameters
        required_lgb_params = [
            "objective",
            "n_estimators",
            "learning_rate",
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
        self.check_required_params(params, required_lgb_params)

        # Store early_stopping_rounds for training
        self.early_stopping_rounds = params["early_stopping_rounds"]

        # Extract LightGBM parameters from config
        model_params = {
            param: params[param]
            for param in required_lgb_params
            if param != "early_stopping_rounds"
        }
        model_params["random_state"] = params.get("random_seed")

        # Log the parameters being used
        logger.info("Initializing LightGBM with parameters: %s", model_params)

        # Initialize the LightGBM model with parameters from config
        self.model = LGBMClassifier(**model_params)

    def evaluate(self, data_loader, save_report=False):
        """
        Evaluate the LightGBM model on the provided data loader.

        Args:
            data_loader: DataLoader containing the data to evaluate.
            save_report: Whether to save the evaluation report.

        Returns:
            A dictionary containing evaluation metrics.
        """
        logger.info("Evaluating LightGBM model...")

        # Load model from pretrained state if available and not in training mode
        if self.pretrained_model_path and self.mode != "train":
            self.load_model_weights(self.pretrained_model_path)

        # Use the utility function to prepare data
        X_test, y_test, feature_names = prepare_data_for_model_convml(data_loader)

        # Create DataFrame with feature names for prediction to avoid warnings
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Evaluate the model
        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        y_pred = self.model.predict(X_test_df)
        y_pred_proba = self.model.predict_proba(X_test_df)

        metadata_dict = {
            "prediction": y_pred_proba[:, 1],
            "label": y_test,
            "age": X_test_df["age"].values,
            "sex": X_test_df["sex"].values,
            "height": X_test_df["height"].values,
            "weight": X_test_df["weight"].values,
        }

        metrics_tracker.add_results(y_pred_proba[:, 1], y_test)
        metrics_tracker.add_metadata_item(metadata_dict)

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()
            metrics_tracker.log_metadata()

        # Log results to console
        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        # Save the model
        model_save_name = f"{self.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_save_dir = os.path.join(self.save_dir, "Models")
        os.makedirs(model_save_dir, exist_ok=True)

        # Store feature names as an attribute before saving
        self.model._pulse_feature_names = feature_names

        save_sklearn_model(model_save_name, self.model, model_save_dir)

        # Log metrics to wandb
        if self.wandb:
            # Get metrics from the metrics tracker
            metrics = metrics_tracker.compute_overall_metrics()

            # Log all metrics from the overall summary
            if "overall" in metrics:
                wandb.log(metrics["overall"])

            # Create and log confusion matrix
            y_pred_binary = (y_pred >= 0.5).astype(int)
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
            if hasattr(self.model, "feature_importances_"):
                feature_importance = {
                    f"importance_{feature_names[i]}": imp
                    for i, imp in enumerate(self.model.feature_importances_)
                }
                wandb.log(feature_importance)

                # Create a bar chart of feature importances
                importance_df = pd.DataFrame(
                    {
                        "feature": feature_names,
                        "importance": self.model.feature_importances_,
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


class LightGBMTrainer:
    """
    Trainer class for LightGBM models.

    This class handles the training workflow for LightGBM models
    including data preparation, model training and saving.
    """

    def __init__(self, model, train_loader, val_loader) -> None:
        """
        Initialize the LightGBM trainer.

        Args:
            model: The LightGBM model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. (not used)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
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
        X_train, y_train, _ = prepare_data_for_model_convml(self.train_loader)
        X_val, y_val, _ = prepare_data_for_model_convml(self.val_loader)

        # Log training start to wandb
        if self.wandb:
            wandb.log(
                {
                    "train_samples": len(X_train),
                    "val_sample": len(X_val),
                }
            )

        # Create early stopping callback with verbose setting based on model configuration
        early_stopping_callback = early_stopping(
            stopping_rounds=self.model.early_stopping_rounds,
            first_metric_only=True,
            verbose=self.model.params["verbose"],
        )

        # Train model with explicit feature names
        self.model.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping_callback],
        )

        logger.info("LightGBM model trained successfully.")
