from typing import Dict, Any, Optional
import logging
import numpy as np
import pandas as pd
import psutil
import os
import sys
import warnings
from datetime import datetime
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
import wandb

from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (
    save_sklearn_model,
    prepare_data_for_model_ml,
)
from src.eval.metrics import MetricsTracker, rmse


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

        self.tune_hyperparameters = params.get("tune_hyperparameters", False)

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

    def set_trainer(self, trainer_name, train_loader, val_loader, test_loader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. (not used)
            test_loader: DataLoader for testing data.
        """
        if trainer_name == "RandomForestTrainer":
            self.trainer = RandomForestTrainer(
                self, train_loader, val_loader, test_loader
            )
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for RandomForest.")


class RandomForestTrainer:
    def __init__(
        self,
        model: PulseTemplateModel,
        train_loader: Any,
        val_loader: Optional[Any],
        test_loader: Any,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.wandb = model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)

    def _tune_hyperparameters(
        self, model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray
    ) -> BaseEstimator:
        param_grid: Dict[str, Any] = {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        logger.info("Starting GridSearchCV for hyperparameter tuning...")
        grid_search.fit(X_train, y_train)
        logger.info("Best params found: %s", grid_search.best_params_)

        if self.wandb:
            wandb.log(
                {
                    "best_params": grid_search.best_params_,
                    "best_score": grid_search.best_score_,
                }
            )

        return grid_search.best_estimator_

    def train(self) -> None:
        logger.info("Starting training process for RandomForest model...")

        # Use the utility function to prepare data
        prepared_data = prepare_data_for_model_ml(
            self.train_loader,
            self.val_loader,
            self.test_loader,
        )

        X_train = prepared_data["X_train"]
        y_train = prepared_data["y_train"]
        X_test = prepared_data["X_test"]
        y_test = prepared_data["y_test"]
        feature_names = prepared_data["feature_names"]

        if self.wandb:
            wandb.log({"train_samples": len(X_train), "test_samples": len(X_test)})

        # Optional: tune hyperparameters
        if self.model.tune_hyperparameters:
            self.model.model = self._tune_hyperparameters(
                self.model.model, X_train, y_train
            )

        # Train the model
        self.model.model.fit(X_train, y_train)
        logger.info("RandomForest model trained successfully.")

        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )

        y_pred = self.model.model.predict(X_test_df)
        y_pred_proba = self.model.model.predict_proba(X_test_df)
        metrics_tracker.add_results(y_pred_proba[:, 1], y_test)

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()

        # Log results to console
        logger.info(f"Test evaluation completed for {self.model.model_name}")
        logger.info(f"Test metrics: {metrics_tracker.summary}")

        # Save the model
        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        save_sklearn_model(model_save_name, self.model.model, self.model_save_dir)

        if self.wandb:
            metrics = metrics_tracker.compute_overall_metrics()
            if "overall" in metrics:
                wandb.log(metrics["overall"])

            y_pred_binary = (y_pred >= 0.5).astype(int)
            wandb.log(
                {
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        preds=y_pred_binary,
                        y_true=y_test,
                        class_names=["Negative", "Positive"],
                    ),
                    "roc_curve": wandb.plot.roc_curve(
                        y_true=y_test,
                        y_probas=y_pred_proba,
                        labels=["Negative", "Positive"],
                    ),
                }
            )

            if hasattr(self.model.model, "feature_importances_"):
                feature_importance = {
                    f"importance_{feature_names[i]}": imp
                    for i, imp in enumerate(self.model.model.feature_importances_)
                }
                wandb.log(feature_importance)

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


# class RandomForestTrainer:
#     """
#     Trainer class for RandomForest models.

#     This class handles the training workflow for RandomForest models
#     including data preparation, model training, evaluation and saving.
#     """

#     def __init__(self, model, train_loader, val_loader, test_loader) -> None:
#         """
#         Initialize the RandomForest trainer.

#         Args:
#             model: The RandomForest model to train.
#             train_loader: DataLoader for training data.
#             val_loader: DataLoader for validation data. (not used)
#             test_loader: DataLoader for testing data.
#         """
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.task_name = self.model.task_name
#         self.dataset_name = self.model.dataset_name
#         self.wandb = model.wandb
#         self.model_save_dir = os.path.join(model.save_dir, "Models")
#         # Create model save directory if it doesn't exist
#         os.makedirs(self.model_save_dir, exist_ok=True)

#     def train(self):
#         """Train the RandomForest model using the provided data loaders."""
#         logger.info("Starting training process for RandomForest model...")

#         # Use the utility function to prepare data
#         prepared_data = prepare_data_for_model_ml(
#             self.train_loader, self.test_loader, logger_instance=logger
#         )

#         # Extract all data from the prepared_data dictionary
#         X_train = prepared_data["X_train"]
#         y_train = prepared_data["y_train"]
#         X_test = prepared_data["X_test"]
#         y_test = prepared_data["y_test"]
#         feature_names = prepared_data["feature_names"]

#         # Log training start to wandb
#         if self.wandb:
#             wandb.log({"train_samples": len(X_train), "test_samples": len(X_test)})

#         # Train the model
#         self.model.model.fit(X_train, y_train)
#         logger.info("RandomForest model trained successfully.")

#         # Create DataFrame with feature names for prediction to avoid warnings
#         X_test_df = pd.DataFrame(X_test, columns=feature_names)

#         # Evaluate the model
#         metrics_tracker = MetricsTracker(
#             self.model.model_name,
#             self.model.task_name,
#             self.model.dataset_name,
#             self.model.save_dir,
#         )

#         y_pred = self.model.model.predict(X_test_df)
#         y_pred_proba = self.model.model.predict_proba(X_test_df)
#         metrics_tracker.add_results(y_pred, y_test)

#         rmse_score = rmse(y_test, y_pred)
#         logger.info("RMSE: %f", rmse_score)

#         # Calculate and log metrics
#         metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
#         metrics_tracker.save_report()

#         # Save the model
#         model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
#         save_sklearn_model(model_save_name, self.model.model, self.model_save_dir)

#         # Log metrics to wandb
#         if self.wandb:
#             # Get metrics from the metrics tracker
#             metrics = metrics_tracker.compute_overall_metrics()

#             # Log all metrics from the overall summary
#             if "overall" in metrics:
#                 wandb.log(metrics["overall"])

#             # Create and log confusion matrix
#             y_pred_binary = (y_pred >= 0.5).astype(int)
#             cm = confusion_matrix(y_test, y_pred_binary)
#             wandb.log(
#                 {
#                     "confusion_matrix": wandb.plot.confusion_matrix(
#                         preds=y_pred_binary,
#                         y_true=y_test,
#                         class_names=["Negative", "Positive"],
#                     )
#                 }
#             )

#             # Create and log ROC curve
#             wandb.log(
#                 {
#                     "roc_curve": wandb.plot.roc_curve(
#                         y_true=y_test,
#                         y_probas=y_pred_proba,
#                         labels=["Negative", "Positive"],
#                     )
#                 }
#             )

#             # Log feature importance
#             if hasattr(self.model.model, "feature_importances_"):
#                 feature_importance = {
#                     f"importance_{feature_names[i]}": imp
#                     for i, imp in enumerate(self.model.model.feature_importances_)
#                 }
#                 wandb.log(feature_importance)

#                 # Create a bar chart of feature importances
#                 importance_df = pd.DataFrame(
#                     {
#                         "feature": feature_names,
#                         "importance": self.model.model.feature_importances_,
#                     }
#                 ).sort_values("importance", ascending=False)

#                 wandb.log(
#                     {
#                         "feature_importance": wandb.plot.bar(
#                             wandb.Table(dataframe=importance_df),
#                             "feature",
#                             "importance",
#                             title="Feature Importance",
#                         )
#                     }
#                 )
