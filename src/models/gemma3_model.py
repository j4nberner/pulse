import logging
import warnings
from typing import Any, Dict

import torch
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import (AutoProcessor, BitsAndBytesConfig, Gemma3ForConditionalGeneration)

from src.models.pulse_model import PulseLLMModel

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class Gemma3Model(PulseLLMModel):
    """Gemma 3 model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the Gemma3Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "Gemma3Model")
        kwargs.get("trainer_name", "Llama3Trainer")
        super().__init__(model_name, params, **kwargs)

        required_params = [
            "max_new_tokens",
            "verbose",
            "tuning",
            "quantization",
            "num_epochs",
            "max_new_tokens",
            "max_length",
            "do_sample",
            "temperature",
        ]
        self.check_required_params(params, required_params)

        self.max_length: int = self.params.get("max_length", 5120)
        self.quantization = params["quantization"]

        if self.quantization:
            logger.info("Using quantization for Gemma3 model")
            self.quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    def load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        try:
            # Skip loading if already loaded
            if self.is_loaded:
                logger.info("Model already loaded, reusing existing instance")
                return

            logger.debug(f"Loading model %s", self.model_id)
            self.tokenizer = AutoProcessor.from_pretrained(
                self.model_id,
                padding_side="left",
            )
            if self.quantization:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    quantization_config=self.quantization_config,
                    attn_implementation="eager",
                )
            else:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                )

            if self.params.get("tuning", False):
                logger.info("Applying Prompt Tuning")
                tuning_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    tokenizer_name_or_path=self.model_id,
                    num_virtual_tokens=20,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Classify the diagnosis of following ICU data:",
                )
                self.model = get_peft_model(self.model, tuning_config)
                logger.debug(self.model.print_trainable_parameters())

            logger.info("Successfully loaded Gemma3 model: %s", self.model_id)

            logger.debug("GPU memory allocated: %s",
                         torch.cuda.memory_allocated() / (1024 ** 3))

            # Mark model as loaded after successful loading
            self.is_loaded = True

        except Exception as e:
            logger.error("Failed to load Gemma3 model: %s", e)
            raise
