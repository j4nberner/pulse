import logging
import os
import random
import time
import warnings
from typing import Any, Dict

import numpy as np

# import openai
import anthropic

from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseModel, PulseLLMModel
from src.util.config_util import set_seeds
from src.util.model_util import (
    parse_llm_output,
    prompt_template_hf,
)


warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class ClaudeSonnet4Model(PulseLLMModel):
    """Claude Sonnet 4 model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the ClaudeSonnet4Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "ClaudeSonnet4Model")
        super().__init__(model_name, params, **kwargs)

        required_params = [
            "max_new_tokens",
            "model_id",
            "thinking_budget",
            "temperature",
            "api_key_name",
        ]
        self.check_required_params(params, required_params)

        self.client = anthropic.Anthropic(api_key=os.getenv(params["api_key_name"]))
        self.model_id = params["model_id"]
        self.prompting_id = params.get("prompting_id", None)
        self.max_new_tokens = params["max_new_tokens"]
        self.thinking_budget = params["thinking_budget"]
        self.temperature = params["temperature"]

        # self.model = GenerativeModel(self.model_id)
        self.is_agent = False
        self.agent_instance = None
        self.is_loaded = True  # Gemini models are loaded by default

    def _generate_standard(
        self,
        input_text: str,
        custom_system_message: str = None,
        parse_json: bool = True,
        generate_raw_text: bool = False,
    ) -> Dict[str, Any]:
        """Standard generation logic for non-agent models."""
        # Set seed for deterministic generation
        set_seeds(self.random_seed)

        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text, sys_msg = prompt_template_hf(
            input_text, custom_system_message, self.model_name, task=self.task_name
        )

        infer_start = time.perf_counter()
        # Retry logic for rate limiting
        num_retries = 10
        delay = 1
        exponential_base = 2.0

        for attempt in range(num_retries + 1):
            try:
                infer_start = time.perf_counter()
                response = self.client.messages.create(
                    model=self.model_id,
                    system=sys_msg,
                    max_tokens=self.max_new_tokens,
                    messages=input_text,
                    temperature=self.temperature,
                    thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
                )
                infer_time = time.perf_counter() - infer_start
                break  # Success, exit retry loop

            except Exception as e:
                error_message = str(e)
                logger.info("Error during inference: %s", error_message)

                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > num_retries:
                    raise Exception(
                        f"Maximum number of retries ({num_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + random.random())

                # Sleep for the delay
                time.sleep(delay)

        infer_time = time.perf_counter() - infer_start

        num_input_tokens = response.usage.input_tokens
        num_output_tokens = response.usage.output_tokens
        num_thinking_tokens = (
            0  # Model provides a summary and not the actual reasoning tokens
        )

        thinking_output = ""
        answer_output = ""
        for block in response.content:
            if block.type == "thinking":
                thinking_output = block.thinking
            elif block.type == "text":
                answer_output = block.text

        logger.debug("Decoded output:\n %s", answer_output)

        # Parse the output if parse_json is True
        if parse_json:
            generated_text = parse_llm_output(answer_output)
        else:
            generated_text = response

        logger.info(
            "Inference time: %.4fs | Input Tokens: %d | Output Tokens: %d | Thinking Budget: %d",
            infer_time,
            num_input_tokens,
            num_output_tokens,
            num_thinking_tokens,
        )

        # Return consistent result structure
        return {
            "generated_text": generated_text,
            "thinking_output": thinking_output,
            "token_time": 0.0,
            "infer_time": infer_time,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": num_thinking_tokens,
        }
