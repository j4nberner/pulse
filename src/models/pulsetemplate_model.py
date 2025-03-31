from typing import Any, Optional


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
        self, trainer_name: str, train_dataloader: Any, test_dataloader: Any
    ) -> None:
        """Set the trainer for this model.

        Args:
            trainer_name: Name of the trainer to use
            train_dataloader: DataLoader for training data
            test_dataloader: DataLoader for test data
        """
        self.trainer_name = trainer_name
        # TODO: Implement dynamic loading of trainer class based on trainer_name

        # # Import the trainer dynamically
        # try:
        #     trainer_module = __import__(f"trainers.{trainer_name}", fromlist=[""])
        #     trainer_class = getattr(trainer_module, trainer_name)
        #     self.trainer = trainer_class(self, train_dataloader, test_dataloader)
        # except (ImportError, AttributeError) as e:
        #     raise ValueError(f"Failed to load trainer '{trainer_name}': {e}")

        # # Verify that trainer has required methods
        # if not hasattr(self.trainer, "train") or not hasattr(self.trainer, "test"):
        #     raise ValueError(
        #         f"Trainer '{trainer_name}' missing required methods (train or test)"
        #     )
