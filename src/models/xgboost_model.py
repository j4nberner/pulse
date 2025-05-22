import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import wandb.sklearn
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import (prepare_data_for_model_convml,
                                 save_sklearn_model)

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
        self.params = params

        # For trainer_name we still require it to be explicitly in the params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")

        # Use the class name as model_name if not provided in params
        self.model_name = self.params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

        self.tune_hyperparameters = params.get("tune_hyperparameters", False)

        # Define all required XGBoost parameters
        required_xgb_params = [
            "objective",
            "n_estimators",
            "learning_rate",
            "verbosity",
            "max_depth",
            "gamma",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
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
        model_params["random_state"] = params.get("random_seed")

        # Store early_stopping_rounds for training
        self.early_stopping_rounds = params["early_stopping_rounds"]

        # Log the parameters being used
        logger.info("Initializing XGBoost with parameters: %s", model_params)

        # Initialize the XGBoost model with parameters from config
        self.model = XGBClassifier(
            **model_params,
        )

    def set_trainer(self, trainer_name, train_loader, val_loader, test_loader):
        """
        Set the trainer for the model.

        Args:
            trainer_name: Name of the trainer.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data. (not used)
            test_loader: DataLoader for testing data.
        """
        if trainer_name == "XGBoostTrainer":
            self.trainer = XGBoostTrainer(self, train_loader, val_loader, test_loader)
        else:
            raise ValueError(f"Trainer {trainer_name} not supported for XGBoost.")


class XGBoostTrainer:
    """
    Trainer class for XGBoost models.

    This class handles the training workflow for XGBoost models
    including data preparation, model training, evaluation and saving.
    """

    def __init__(self, model, train_loader, val_loader, test_loader) -> None:
        """
        Initialize the XGBoost trainer.

        Args:
            model: The XGBoost model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
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

    def _tune_hyperparameters(
        self,
        model: XGBClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> XGBClassifier:
        param_dist = {
            "max_depth": np.arange(3, 10),
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": np.arange(1, 10),
            "gamma": [0, 0.1, 0.2, 0.5],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [1, 1.5, 2.0],
        }

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=50,
            scoring="roc_auc",
            n_jobs=-1,
            cv=3,
            verbose=1,
            random_state=42,
        )
        # Disable early stopping for hyperparameter tuning
        model.set_params(early_stopping_rounds=None)

        logger.info("Starting RandomizedSearchCV for hyperparameter tuning...")
        random_search.fit(X_train, y_train, verbose=True)
        logger.info("Best params found: %s", random_search.best_params_)

        if self.wandb:
            wandb.log(
                {
                    "best_params": random_search.best_params_,
                    "best_score": random_search.best_score_,
                }
            )

        return random_search.best_estimator_

    def train(self):
        """Train the XGBoost model using the provided data loaders."""
        logger.info("Starting training process for XGBoost model...")

        # Use the utility function to prepare data
        prepared_data = prepare_data_for_model_convml(
            self.train_loader,
            self.val_loader,
            self.test_loader,
        )

        # Extract all data from the prepared_data dictionary
        X_train = prepared_data["X_train"]
        y_train = prepared_data["y_train"]
        X_val = prepared_data["X_val"]
        y_val = prepared_data["y_val"]
        X_test = prepared_data["X_test"]
        y_test = prepared_data["y_test"]
        feature_names = prepared_data["feature_names"]

        # Log training start to wandb
        if self.wandb:
            wandb.log(
                {
                    "train_samples": len(X_train),
                    "val_sample": len(X_val),
                    "test_samples": len(X_test),
                }
            )

        # Optional: tune hyperparameters
        if self.model.tune_hyperparameters:
            self.model.model = self._tune_hyperparameters(
                self.model.model, X_train, y_train, X_val, y_val
            )

        # Train the model
        self.model.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )
        logger.info("XGBoost model trained successfully.")

        results = self.model.model.evals_result()

        if self.wandb:
            for i in range(len(results["validation_0"]["auc"])):
                wandb.log({"val_loss": results["validation_0"]["auc"][i], "step": i})

        # Load the best model if early stopping was used
        if hasattr(self.model.model, "best_iteration"):
            self.model.model.n_estimators = self.model.model.best_iteration
            logger.info(
                "Loading best iteration with n_estimators: %d",
                self.model.model.n_estimators,
            )

        # Create DataFrame with feature names for prediction to avoid warnings
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        # Evaluate the model
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )

        y_pred = self.model.model.predict(X_test_df)
        y_pred_proba = self.model.model.predict_proba(X_test_df)
        metrics_tracker.add_results(y_pred_proba[:, 1], y_test)

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        metrics_tracker.save_report()

        # Log results to console
        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        # Save the model
        model_save_name = f"{self.model.model_name}_{self.task_name}_{self.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_sklearn_model(model_save_name, self.model.model, self.model_save_dir)

        # Log metrics to wandb
        if self.wandb:
            if "overall" in metrics_tracker.summary:
                wandb.log(metrics_tracker.summary["overall"])

            y_pred_binary = (y_pred >= 0.5).astype(int)
            wandb.log(
                {
                    "confusion_matrix": wandb.sklearn.plot_confusion_matrix(
                        y_pred=y_pred_binary,
                        y_true=y_test,
                        labels=["Negative", "Positive"],
                    ),
                    "roc_curve": wandb.sklearn.plot_roc(
                        y_true=y_test,
                        y_probas=y_pred_proba,
                        labels=["Negative", "Positive"],
                    ),
                }
            )

            if hasattr(self.model.model, "feature_importances_"):
                # Feature importances
                importances = self.model.model.feature_importances_
                wandb.log(
                    {
                        "feature_importances": wandb.plot.bar(
                            wandb.Table(
                                data=[
                                    [f, i] for f, i in zip(feature_names, importances)
                                ],
                                columns=["Feature", "Importance"],
                            ),
                            "Feature",
                            "Importance",
                        )
                    }
                )
