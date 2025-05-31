import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import AzureOpenAI

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseTemplateModel
from src.util.model_util import extract_dict, prompt_template_hf

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class GPTModel(PulseTemplateModel):
    """GPT model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the GPTModel with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "GPTModel")
        super().__init__(model_name, params, **kwargs)

        required_params = [
            "max_new_tokens",
            "api_version",
            "api_key_name",
            "api_uri_name",
        ]
        self.check_required_params(params, required_params)

        api_key = os.environ.get(params["api_key_name"])
        endpoint_uri = os.environ.get(params["api_uri_name"])

        self.client = AzureOpenAI(
            api_version=params["api_version"],
            azure_endpoint=endpoint_uri,
            api_key=api_key,
        )
        self.deployment = params["deployment"]
        self.prompting_id = params.get("prompting_id", None)

    def generate(
        self,
        input_text: str,
        custom_system_message: str = None,
        force_raw_text: bool = False,
    ) -> Dict[str, Any]:
        """Runs the model on the input and extracts diagnosis, explanation, and probability.

        Args:
            input_text: The text to analyze
            custom_system_message: Optional custom system message
            force_raw_text: If True, returns raw text output without JSON parsing
        """
        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text = prompt_template_hf(
            input_text, custom_system_message, model="GPTModel"
        )

        # Generate output with scores
        infer_start = time.perf_counter()
        outputs = self.client.chat.completions.create(
            messages=input_text,
            max_tokens=10000,
            temperature=0.4,
            model=self.deployment,
        )
        infer_time = time.perf_counter() - infer_start

        # For Azure OpenAI, usage is in outputs.usage
        num_input_tokens = (
            outputs.usage.prompt_tokens if hasattr(outputs, "usage") else None
        )
        num_output_tokens = (
            outputs.usage.completion_tokens if hasattr(outputs, "usage") else None
        )

        decoded_output = outputs.choices[0].message.content
        logger.debug("Decoded output:\n %s", decoded_output)

        # Extract dict from the decoded output (e.g., via regex or JSON parsing)
        parsed = extract_dict(decoded_output)

        # Check if probability is a number or string, try to convert, else default to 0.5
        prob = parsed.get("probability", 0.5)
        try:
            prob = float(prob)
        except (ValueError, TypeError):
            logger.warning("Failed to convert probability to float. Defaulting to 0.5")
            prob = 0.5
        parsed["probability"] = prob

        logger.info(
            f"Inference time: {infer_time:.4f}s | Tokens: {num_input_tokens + num_output_tokens}"
        )

        return {
            "generated_text": parsed,
            "infer_time": infer_time,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
        }
