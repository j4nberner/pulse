import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
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


class DeepseekR1Model(PulseLLMModel):
    """DeepseekR1 model wrapper using LangChain for prompt templating and inference."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the DeepseekR1Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "DeepseekR1Model")
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

    def generate(
        self,
        input_text: str,
        custom_system_message: str = None,
        force_raw_text: bool = False,
    ) -> Dict[str, Any]:
        """Runs the HF model on the input and extracts diagnosis, explanation, and probability.

        Args:
            input_text: The text to analyze
            custom_system_message: Optional custom system message
            force_raw_text: If True, returns raw text output without JSON parsing
        """
        # Set seed for deterministic generation
        set_seeds(self.random_seed)

        # Ensure model is loaded before trying to use it
        if not self.is_loaded:
            logger.debug("Model not loaded yet for inference, loading now...")
            self.load_model()

        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text = prompt_template_hf(
            input_text, custom_system_message, self.model_name
        )

        chat_prompt = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )
        token_start = time.perf_counter()
        tokenized_inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
        )
        token_time = time.perf_counter() - token_start
        num_input_tokens = tokenized_inputs["input_ids"].shape[-1]

        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)

        infer_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.params["max_new_tokens"],
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=self.params["do_sample"],
                temperature=self.params["temperature"],
            )

        infer_time = time.perf_counter() - infer_start

        # Get generated token ids (excluding prompt)
        gen_ids = outputs.sequences[0][num_input_tokens:]
        num_output_tokens = gen_ids.shape[-1]

        # Decode the full generated string
        decoded_output = self.tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        # Extract the thinking part
        thinking_output = decoded_output.split("</think>")[0]
        answer_output = decoded_output.split("</think>")[1]

        # logger.debug("Thinking output:\n %s", thinking_output)
        logger.debug("Answer output:\n %s", answer_output)

        # Check if we should return raw text or parsed JSON (important for multi-turn conversations)
        if force_raw_text:
            # For text-only outputs like summaries
            return {
                "generated_text": answer_output,  # Return raw text
                "token_time": token_time,
                "infer_time": infer_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "thinking_output": thinking_output,  # Not used
            }

        # Extract dict from the decoded output
        try:
            parsed = extract_dict(answer_output)
            # logger.debug("Parsed output: %s", parsed)
        except Exception as e:
            logger.warning(f"Failed to parse output: {decoded_output}")
            parsed = {
                "diagnosis": None,
                "probability": 0.5,
                "explanation": decoded_output,
            }

        logger.info(
            "Tokenization time: %.4fs | Inference time: %.4fs | Tokens: %d",
            token_time,
            infer_time,
            num_input_tokens + num_output_tokens,
        )

        return {
            "generated_text": parsed,
            "token_time": token_time,
            "infer_time": infer_time,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "thinking_output": thinking_output,
        }