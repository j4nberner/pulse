from .example_trainer import ExampleTrainer
from .example_model import ExampleModel  # Placeholder for actual model class

trainer_cls_name_dict = {
    "ExampleTrainer": ExampleTrainer,
}
model_cls_name_dict = {
    "ExampleModel": ExampleModel,  # Placeholder for actual model class
}


def get_trainer_class(trainer_name: str):
    """
    Get the trainer class based on the trainer name.

    Args:
        trainer_name (str): The name of the trainer.

    Returns:
        class: The corresponding trainer class.
    """
    if trainer_name in trainer_cls_name_dict:
        return trainer_cls_name_dict[trainer_name]
    else:
        raise ValueError(f"Trainer {trainer_name} not found.")


def get_model_class(model_name: str):
    """
    Get the model class based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        class: The corresponding model class.
    """
    if model_name in model_cls_name_dict:
        return model_cls_name_dict[model_name]
    else:
        raise ValueError(f"Model {model_name} not found.")
