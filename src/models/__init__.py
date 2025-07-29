from .claude_sonnet4_model import ClaudeSonnet4Model
from .cnn_model import CNNModel, CNNTrainer
from .deepseekr1_model import DeepseekR1Model
from .gemini2p5flash_model import Gemini2p5Model
from .gemma3_model import Gemma3Model
from .gpt_model import GPTModel
from .grok4_model import Grok4Model
from .gru_model import GRUModel, GRUTrainer
from .inceptiontime_model import InceptionTimeModel, InceptionTimeTrainer
from .lightgbm_model import LightGBMModel, LightGBMTrainer
from .llama3_model import Llama3Model
from .llama4_model import Llama4Model
from .lstm_model import LSTMModel, LSTMTrainer
from .mistral_model import MistralModel
from .o3_model import OpenAIo3Model
from .randomforest_model import RandomForestModel, RandomForestTrainer
from .xgboost_model import XGBoostModel, XGBoostTrainer

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
    "DeepseekR1Qwen7bModel": DeepseekR1Model,
    "DeepseekR1Llama8bModel": DeepseekR1Model,
    "MistralModel": MistralModel,
    "Gemini2p5flashModel": Gemini2p5Model,
    "Gemini2p5proModel": Gemini2p5Model,
    "GPT4oModel": GPTModel,
    "ClaudeSonnet4Model": ClaudeSonnet4Model,
    "OpenAIo3Model": OpenAIo3Model,
    "Grok4Model": Grok4Model,
}

trainer_cls_name_dict = {
    "CNNTrainer": CNNTrainer,
    "LSTMTrainer": LSTMTrainer,
    "GRUTrainer": GRUTrainer,
    "InceptionTimeTrainer": InceptionTimeTrainer,
    "LightGBMTrainer": LightGBMTrainer,
    "XGBoostTrainer": XGBoostTrainer,
    "RandomForestTrainer": RandomForestTrainer,
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
