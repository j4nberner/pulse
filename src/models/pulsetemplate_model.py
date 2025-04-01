from typing import Any, Optional
from torch.utils.data import DataLoader


class PulseTemplateModel:
    """Base model template that all other models will inherit from.

    This class provides the common attributes and methods that all models
    in the Pulse framework should implement.
    """

    def __init__(self, model_name: str, trainer_name: Optional[str] = None):
        """Initialize a new Pulse model.

        Args:
            model_name: Name of the model
            trainer_name: Optional name of the trainer
        """
        self.model_name = model_name
        self.trainer_name = trainer_name
        self.trainer = None

    def set_trainer(
        self,
        trainer_name: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        """
        Set the trainer for this model. This method should be overridden by subclasses.
        A trainer is responsible for training and evaluating the model.

        Args:
            trainer_name: Name of the trainer to use
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for test data
        """
        self.trainer_name = trainer_name
        self.trainer = None
        # TODO: Implement dynamic loading of trainer class based on trainer_name
