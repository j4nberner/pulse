from .cnn_model import CNNModel
from .gru_model import GRUModel
from .inceptiontime_model import InceptionTimeModel
from .lightgbm_model import LightGBMModel
from .llama3_model import Llama3Model
from .lstm_model import LSTMModel
from .randomforest_model import RandomForestModel
from .xgboost_model import XGBoostModel

model_cls_name_dict = {
    "RandomForest": RandomForestModel,
    "CNNModel": CNNModel,
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
    "LSTMModel": LSTMModel,
    "Llama3Model": Llama3Model,
    "InceptionTimeModel": InceptionTimeModel,
    "GRUModel": GRUModel,
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
