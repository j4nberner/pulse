from .cnn_model import CNNModel
from .deepseekr1_model import DeepseekR1Model
from .gemma3_model import Gemma3Model
from .gru_model import GRUModel
from .inceptiontime_model import InceptionTimeModel
from .lightgbm_model import LightGBMModel
from .llama3_model import Llama3Model
from .llama4_model import Llama4Model
from .lstm_model import LSTMModel
from .randomforest_model import RandomForestModel
from .xgboost_model import XGBoostModel

model_cls_name_dict = {
    "RandomForest": RandomForestModel,
    "CNNModel": CNNModel,
    "XGBoost": XGBoostModel,
    "LightGBM": LightGBMModel,
    "LSTMModel": LSTMModel,
    "InceptionTimeModel": InceptionTimeModel,
    "GRUModel": GRUModel,
    "Llama3Model": Llama3Model,
    "Llama4Model": Llama4Model,
    "Gemma3Model": Gemma3Model,
    "DeepseekR1Model": DeepseekR1Model,
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
