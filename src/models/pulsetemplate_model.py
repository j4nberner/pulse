import logging
from typing import Any, Optional

import joblib
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("PULSE_logger")


class PulseTemplateModel:
    """
    Base model template that all other models will inherit from.

    This class provides the common attributes and methods that all models
    in the Pulse framework should implement.
    """

    def __init__(
        self, model_name: str, trainer_name: Optional[str] = None, **kwargs
    ) -> None:
        """Initialize a new Pulse model.

        Args:
            model_name: Name of the model
            trainer_name: Optional name of the trainer
        """
        params = kwargs.get("params", {})
        self.params = params
        self.model_name = model_name
        self.trainer_name = trainer_name
        self.trainer = None
        self.model = None
        self.dataset_name = None
        self.task_name = None
        self.save_metadata = None

        self.prompting_id = params.get("prompting_id", None)
        self.pretrained_model_path = kwargs.get("pretrained_model_path")
        self.type = params.get("type", None)

    def set_trainer(
        self,
        trainer_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        """
        Set the trainer for this model. This method should be overridden by subclasses.
        A trainer is responsible for training and evaluating the model.

        Args:
            trainer_name: Name of the trainer to use
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
        """
        self.trainer_name = trainer_name
        self.trainer = None

    def load_model_weights(self, model_path: str) -> None:
        """Load model weights from a specified path.

        Args:
            model_path: Path to the model weights file
        """
        if self.type == "convML":
            # Load the sklearn model using joblib
            self.model = joblib.load(model_path)

        elif self.type == "convDL":
            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))

            # Check if the loaded file is a full model or just weights
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()

            # Load the weights into the model
            if hasattr(self, "load_state_dict"):
                self.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            else:
                logger.warning(
                    "Model does not have load_state_dict method. Cannot load weights."
                )

        elif self.type == "LLM":
            # Load LLM model weights
            pass
        else:
            logger.warning("Model type not recognized. Cannot load model weights.")
