import logging
from typing import Any, Dict, List, Optional
import traceback
from types import SimpleNamespace

import torch

from src.models.llama3_model import Llama3Model
from src.util.model_util import prompt_template_hf

logger = logging.getLogger("PULSE_logger")


class Llama3ModelAdapter:
    """Adapter for using Llama3Model with the agent framework."""

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", **kwargs):
        """Initialize the model adapter.

        Args:
            model_id: The model ID to use
            **kwargs: Additional arguments passed to Llama3Model
        """
        self.model_id = model_id
        self.model = None
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        """Load the model on demand."""
        if self.model is None:
            try:
                # Create a params dictionary with all necessary parameters
                params = {
                    "trainer_name": "Llama3Trainer",
                    "model_id": self.model_id,
                    "type": "LLM",
                    "max_new_tokens": 1000,
                    "do_sample": False,
                    "temperature": 0.0,
                    **self.kwargs,
                }

                # Initialize the model
                self.model = Llama3Model(
                    model_name="Llama3Model", params=params
                )

                # Enhanced SimpleNamespace with get() method to maintain compatibility
                class EnhancedNamespace(SimpleNamespace):
                    def get(self, key, default=None):
                        return getattr(self, key, default)

                # Convert params dictionary to our enhanced namespace
                self.model.params = EnhancedNamespace(**params)

                self.model._load_model()
                logger.info(f"Loaded model {self.model_id}")
            except Exception as e:
                logger.error(f"Error loading model {self.model_id}: {e}")
                logger.debug(f"Detailed error: {traceback.format_exc()}")
                raise

    def __call__(
        self, messages: List[Dict[str, str]], parse_json: bool = True
    ) -> Dict[str, Any]:
        """Call the model with the provided messages."""
        try:
            # Ensure model is loaded
            self._load_model()

            # Debug message content
            logger.debug(f"Calling LLM with messages: {[m['role'] for m in messages]}")
            logger.debug(f"JSON parsing enabled: {parse_json}")

            # Extract system message if present
            system_message = None
            for message in messages:
                if message["role"] == "system":
                    system_message = message["content"]
                    logger.debug(f"System message: {system_message[:100]}...")
                    break

            # Format the messages for the model - construct a single string
            prompt_text = ""
            for message in messages:
                if message["role"] == "system":
                    prompt_text += f"System: {message['content']}\n\n"
                elif message["role"] == "user":
                    prompt_text += f"User: {message['content']}\n\n"
                elif message["role"] == "assistant":
                    prompt_text += f"Assistant: {message['content']}\n\n"

            # Call the model with the formatted prompt and parsing preference
            result = self.model.infer_llm(
                prompt_text, system_message, force_raw_text=not parse_json
            )

            # Get the generated text
            generated_text = (
                str(result["generated_text"])
                if isinstance(result, dict) and "generated_text" in result
                else str(result)
            )

            # Extract metrics from result
            token_time = round(result.get("token_time", 0.0), 4)
            infer_time = round(result.get("infer_time", 0.0), 4)
            num_input_tokens = result.get("num_tokens", 0)

            # Calculate output tokens
            num_output_tokens = 0
            if self.model and hasattr(self.model, "tokenizer"):
                # Use the model's tokenizer if available
                encoded_output = self.model.tokenizer.encode(
                    generated_text, add_special_tokens=False
                )
                num_output_tokens = len(encoded_output)
            else:
                logger.warning("num_output_tokens cannot be calculated")

            # Return both the generated text and system message plus metrics
            return {
                "generated_text": generated_text,
                "system_message": system_message,
                "num_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "token_time": token_time,
                "infer_time": infer_time,
            }

        except Exception as e:
            logger.error(f"Error calling LLM: {e}", exc_info=True)
            return {
                "generated_text": f"Error calling LLM: {str(e)}",
                "system_message": (
                    system_message if "system_message" in locals() else None
                ),
                "num_tokens": 0,
                "num_output_tokens": 0,
                "token_time": 0.0,
                "infer_time": 0.0,
            }

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "model") and self.model is not None:
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
