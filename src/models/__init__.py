from .simpledl_model import SimpleDLModel  # Placeholder for actual model class
from .xgboost import XGBoostModel
from .randomforest_model import RandomForestModel
from .cnn_model import CNNModel


model_cls_name_dict = {
    "RandomForest": RandomForestModel,
    "CNNModel": CNNModel,
}


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
