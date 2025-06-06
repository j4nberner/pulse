import json
import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Gemma3ForConditionalGeneration)

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseLLMModel
from src.util.config_util import set_seeds
from src.util.model_util import extract_dict, prompt_template_hf

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
        trainer_name = kwargs.get("trainer_name", "Llama3Trainer")
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

    def load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        # TODO: @j4nberner Check if AutoModel also works here -> if yes, inherit directly form PulseLLMModel
        try:
            # Skip loading if already loaded
            if self.is_loaded:
                logger.info("Model already loaded, reusing existing instance")
                return

            logger.debug(f"Loading model %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, padding_side="left"
            )
            self.model = Gemma3ForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
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

            # Mark model as loaded after successful loading
            self.is_loaded = True

        except Exception as e:
            logger.error("Failed to load Gemma3 model: %s", e)
            raise
