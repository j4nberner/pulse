from typing import Dict, Any, Optional
import logging
from sklearn.ensemble import RandomForestClassifier

from src.models.pulsetemplate_model import PulseTemplateModel
from src.eval.metrics import rmse

logger = logging.getLogger("PULSE_logger")


class RandomForestModel(PulseTemplateModel):
    """
    Implementation of RandomForest model for classification and regression tasks.

    Attributes:
        model: The trained RandomForest model.
        model_type: Type of the model ('classifier' or 'regressor').
        params: Parameters used for the model.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initialize the RandomForest model.

        Args:
            model_type: Type of model ('classifier' or 'regressor').
            params: Dictionary of parameters for the RandomForest model.
        """
        model_name = params.get("model_name")
        trainer_name = params.get("trainer_name")
        super().__init__(model_name, trainer_name)

        # Check input parameters
        rf_params = {
            "n_estimators": params.get("n_estimators", 100),
        }
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
            model_type: Type of model ('classifier' or 'regressor').
            params: Parameters for the RandomForest model.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        """
        Train the RandomForest model using the provided data loaders.
        """
        # training loop
        logger.info("Training RandomForest model...")
        # Extract data from dataloader
        X_train, y_train = [], []
        X_test, y_test = [], []

        for batch in self.train_dataloader:
            features, labels = batch
            X_train.extend(features.numpy())
            y_train.extend(labels.numpy())

        for batch in self.test_dataloader:
            features, labels = batch
            X_test.extend(features.numpy())
            y_test.extend(labels.numpy())

        # dummy training loop
        self.model.model.fit(X_train, y_train)
        logger.info("RandomForest model trained successfully.")

        # Evaluate the model
        y_pred = self.model.model.predict(X_test)
        rmse_score = rmse(y_test, y_pred)
        logger.info("RMSE: %f", rmse_score)
