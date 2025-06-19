import logging
import os
import warnings
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import wandb.sklearn
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseModel
from src.util.model_util import (prepare_data_for_model_convml,
                                 save_sklearn_model)

# Filter the specific warning about feature names
# (This is because training is done with np arrays and prediction with pd dataframe to preserve feature names for feature importance etc.)
warnings.filterwarnings(
    "ignore",
    message="X has feature names, but RandomForestClassifier was fitted without feature names",
)

logger = logging.getLogger("PULSE_logger")


class RandomForestModel(PulseModel):
    """
    Implementation of RandomForest model for classification and regression tasks.

    Attributes:
        params: Parameters used for the model.

    """

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the RandomForest model.

        Args:
            params: Dictionary of parameters from the config file.
            **kwargs: Additional keyword arguments:
            - output_dir (str, optional): Directory where model outputs will be saved.
              Defaults to `os.getcwd()/output`.
            - wandb (bool, optional): Whether to enable Weights & Biases (wandb) logging.
              Defaults to False.

        Raises:
            KeyError: If any required parameters are missing from the config.
        """
        model_name = kwargs.pop("model_name", "RandomForest")
        trainer_name = params["trainer_name"]
        super().__init__(model_name, params=params, trainer_name=trainer_name, **kwargs)

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
            "verbose",
            "criterion",
            "max_leaf_nodes",
            "min_impurity_decrease",
            "max_samples",
            "class_weight",
            "ccp_alpha",
        ]

        # Check if all required RandomForest parameters exist in config
        self.check_required_params(params, required_rf_params)

        # Extract RandomForest parameters from config
        rf_params = {param: params[param] for param in required_rf_params}
        rf_params["random_state"] = params.get("random_seed")

        # Log the parameters being used
        logger.info("Initializing RandomForest with parameters: %s", rf_params)

        # Initialize the RandomForest model with parameters from config
        self.model = RandomForestClassifier(**rf_params)

    def evaluate(self, data_loader: Any, save_report: bool = False) -> Dict[str, Any]:
        """
        Evaluate the model on the provided data loader.

        Args:
            data_loader: DataLoader containing the data to evaluate.
            save_report: Whether to save the evaluation report. Defaults to False.

        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info("Evaluating RandomForest model...")

        # Load model from pretrained state if available and not in training mode
        if self.pretrained_model_path and self.mode != "train":
            self.load_model_weights(self.pretrained_model_path)

        X_test, y_test, feature_names = prepare_data_for_model_convml(data_loader)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )

        y_pred = self.model.predict(X_test_df)
        y_pred_proba = self.model.predict_proba(X_test_df)

        metadata_dict = {
            "prediction": y_pred_proba[:, -1],
            "label": y_test,
            "age": X_test_df["age"].values,
            "sex": X_test_df["sex"].values,
            "height": X_test_df["height"].values,
            "weight": X_test_df["weight"].values,
        }

        metrics_tracker.add_results(y_pred_proba[:, -1], y_test)
        metrics_tracker.add_metadata_item(metadata_dict)

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()
            metrics_tracker.log_metadata(True)

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

            if hasattr(self.model, "feature_importances_"):
                # Feature importances
                importances = self.model.feature_importances_
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


class RandomForestTrainer:
    def __init__(
        self,
        model,
        train_loader: Any,
        val_loader: Optional[Any],
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.wandb = model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        os.makedirs(self.model_save_dir, exist_ok=True)

    def _tune_hyperparameters(
        self, model: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray
    ) -> BaseEstimator:
        param_grid = {
            "n_estimators": [100, 200, 500],  # Number of trees in the forest
            "max_depth": [None, 10, 20, 50],  # Max depth of each tree
            "min_samples_split": [2, 5, 10],  # Minimum samples to split a node
            "min_samples_leaf": [1, 2, 4],  # Minimum samples required at a leaf node
            "max_features": [
                "sqrt",
                "log2",
            ],  # Number of features to consider at each split
            "class_weight": [
                None,
                "balanced",
                "balanced_subsample",
            ],  # Handling class imbalance
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
        """Train and evaluate the RandomForest model using the provided data loaders."""

        logger.info("Starting training process for RandomForest model...")

        # Use the utility function to prepare data
        X_train, y_train, feature_names = prepare_data_for_model_convml(
            self.train_loader
        )
        X_val, y_val, _ = prepare_data_for_model_convml(self.val_loader)

        # Optional: tune hyperparameters
        if self.model.tune_hyperparameters:
            self.model.model = self._tune_hyperparameters(
                self.model.model, X_train, y_train
            )

        # Train the model
        self.model.model.fit(X_train, y_train)

        # Store feature names as an attribute for later retrieval
        self.model.model._pulse_feature_names = feature_names

        logger.info("RandomForest model trained successfully.")
