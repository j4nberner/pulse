import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple
import xgboost as xgb
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import logging
from .pulsetemplate_model import PulseTemplateModel

#!!! INGORE


class XGBoostModel(PulseTemplateModel):
    """XGBoost model implementation that inherits from PulseTemplateModel."""

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize XGBoost model with parameters.

        Args:
            params: Dictionary of XGBoost parameters
        """
        super().__init__(model_name="XGBoost", trainer_name="XGBoostTrainer")
        self.params = params
        self.model = None
        self.is_classifier = False

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        **kwargs,
    ) -> "XGBoostModel":
        """
        Train the XGBoost model.

        Args:
            X: Features for training
            y: Target values
            **kwargs: Additional arguments to pass to XGBoost's train method

        Returns:
            self: Trained model instance
        """
        # Determine if this is a classification task
        unique_values = np.unique(y)
        self.is_classifier = len(unique_values) <= 10  # Arbitrary threshold

        # Adjust objective based on task type if not explicitly set
        if "objective" not in self.params:
            if self.is_classifier:
                if len(unique_values) == 2:
                    self.params["objective"] = "binary:logistic"
                else:
                    self.params["objective"] = "multi:softprob"
                    self.params["num_class"] = len(unique_values)

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y)

        # Train model
        evaluation_results = {}
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params.get("n_estimators", 100),
            evals=[(dtrain, "train")],
            evals_result=evaluation_results,
            **kwargs,
        )

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Features for prediction

        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before predict().")

        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)

        # For classification tasks, convert probabilities to class labels
        if self.is_classifier:
            if self.params.get("objective") == "binary:logistic":
                return (predictions > 0.5).astype(int)
            else:  # multi-class
                return np.argmax(predictions, axis=1)

        return predictions

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for classification tasks.

        Args:
            X: Features for prediction

        Returns:
            np.ndarray: Class probabilities
        """
        if not self.is_classifier:
            raise ValueError(
                "predict_proba() is only available for classification tasks."
            )

        if self.model is None:
            raise ValueError(
                "Model has not been trained. Call fit() before predict_proba()."
            )

        dtest = xgb.DMatrix(X)
        probabilities = self.model.predict(dtest)

        # For binary classification, reshape to proper probability format
        if self.params.get("objective") == "binary:logistic":
            return np.vstack((1 - probabilities, probabilities)).T

        return probabilities

    def save(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        self.model.save_model(path)

    def load(self, path: str) -> "XGBoostModel":
        """
        Load the model from a file.

        Args:
            path: Path to load the model from

        Returns:
            self: Model instance with loaded model
        """
        self.model = xgb.Booster()
        self.model.load_model(path)
        return self

    def feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the model.

        Returns:
            Dict[str, float]: Feature importance scores
        """
        if self.model is None:
            raise ValueError(
                "Model has not been trained. Call fit() before feature_importance()."
            )

        importance = self.model.get_score(importance_type="gain")
        return importance

    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        if trainer_name == "XGBoostTrainer":
            self.trainer = XGBoostTrainer(self, train_dataloader, test_dataloader)
        else:
            raise ValueError(f"Trainer {trainer_name} not found.")


class XGBoostTrainer:
    """Trainer class for XGBoost models with cross-validation support."""

    def __init__(self, model, train_dataloader, test_dataloader):
        """
        Initialize XGBoost trainer.

        Args:
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for test data
            params: Dictionary of XGBoost parameters
        """
        self.cv_results = None
        self.best_params = None
        self.best_model = None

        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        """
        Train the XGBoost model using data from the DataLoader.

        Args:
            model: XGBoost model instance to train
            epochs: Number of epochs (overrides n_estimators in model params if provided)
            early_stopping_rounds: Number of rounds for early stopping
            verbose: Whether to print training progress

        Returns:
            Trained model
        """
        # Extract data from DataLoader
        X_train, y_train = [], []
        for batch in self.train_dataloader:
            features, targets = batch["features"], batch["label"]
            X_train.append(np.array(features))
            y_train.append(np.array(targets))

        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)

        # # Get validation data if available
        # if self.test_dataloader:
        #     X_val, y_val = [], []
        #     for batch in self.test_dataloader:
        #         features, targets = batch
        #         X_val.append(features.numpy())
        #         y_val.append(targets.numpy())

        #     X_val = np.vstack(X_val)
        #     y_val = np.concatenate(y_val)

        #     # Create DMatrix objects
        #     dtrain = xgb.DMatrix(X_train, label=y_train)
        #     dval = xgb.DMatrix(X_val, label=y_val)
        #     watchlist = [(dtrain, "train"), (dval, "val")]
        # else:
        #     dtrain = xgb.DMatrix(X_train, label=y_train)
        #     watchlist = [(dtrain, "train")]

        # # Set parameters
        # num_boost_round = (
        #     epochs if epochs is not None else model.params.get("n_estimators", 100)
        # )

        # # Train model
        # evaluation_results = {}
        # trained_model = xgb.train(
        #     params=model.params,
        #     dtrain=dtrain,
        #     num_boost_round=num_boost_round,
        #     evals=watchlist,
        #     evals_result=evaluation_results,
        #     early_stopping_rounds=early_stopping_rounds,
        #     verbose_eval=verbose,
        # )

        # model.model = trained_model
        # model.is_classifier = model.params.get("objective", "").startswith(
        #     ("binary", "multi")
        # )

        # return model

    def test(self, model):
        """
        Evaluate the XGBoost model on test data.

        Args:
            model: Trained XGBoost model instance

        Returns:
            dict: Dictionary with evaluation metrics
        """
        # Extract test data
        X_test, y_test = [], []
        for batch in self.test_dataloader:
            features, targets = batch
            X_test.append(features.numpy())
            y_test.append(targets.numpy())

        X_test = np.vstack(X_test)
        y_test = np.concatenate(y_test)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {}

        if model.is_classifier:
            metrics["accuracy"] = accuracy_score(y_test, y_pred)

            try:
                # For binary classification
                metrics["precision"] = precision_score(y_test, y_pred)
                metrics["recall"] = recall_score(y_test, y_pred)
                metrics["f1"] = f1_score(y_test, y_pred)
            except:
                # For multi-class, use average='macro'
                metrics["precision"] = precision_score(y_test, y_pred, average="macro")
                metrics["recall"] = recall_score(y_test, y_pred, average="macro")
                metrics["f1"] = f1_score(y_test, y_pred, average="macro")
        else:
            # Regression metrics
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])

        return metrics
