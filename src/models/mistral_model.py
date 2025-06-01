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
                          BitsAndBytesConfig)

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


class MistralModel(PulseLLMModel):
    """Mistral 7b model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the MistralModel with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.get("model_name", "MistralModel")
        super().__init__(model_name, params=params)

        self.inference_only = kwargs.get("inference_only", False)
        if self.inference_only:
            # For inference-only mode (agentic workflow)
            self.trainer_name = params.get("trainer_name", "MistralTrainer")
            # Skip parent initialization for agentic workflow
            self.random_seed = self.params.get("random_seed", 42)
            logger.debug("Using random seed: %d", self.random_seed)

            # Set necessary parameters for inference
            self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")
            self.wandb = kwargs.get("wandb", False)
            self.task_name = kwargs.get("task_name")
            self.dataset_name = kwargs.get("dataset_name")

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
