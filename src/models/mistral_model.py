import logging
import warnings
from typing import Any, Dict

from transformers import BitsAndBytesConfig

from src.models.pulse_model import PulseLLMModel

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class MistralModel(PulseLLMModel):
    """Mistral 7b model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the MistralModel with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "MistralModel")
        super().__init__(model_name, params, **kwargs)

        required_params = [
            "max_new_tokens",
            "verbose",
            "tuning",
            "num_epochs",
            "max_new_tokens",
            "max_length",
            "do_sample",
            "temperature",
        ]
        self.check_required_params(params, required_params)

        self.max_length: int = self.params.get("max_length", 5120)
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )
