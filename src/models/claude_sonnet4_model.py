import logging
import os
import random
import time
import warnings
from typing import Any, Dict

# import openai
import anthropic
import numpy as np
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseLLMModel
from src.util.config_util import set_seeds
from src.util.model_util import parse_llm_output, prompt_template_hf

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
            "batch_processing",
        ]
        self.check_required_params(params, required_params)

        self.client = anthropic.Anthropic(api_key=os.getenv(params["api_key_name"]))
        self.model_id = params["model_id"]
        self.prompting_id = params.get("prompting_id", None)
        self.max_new_tokens = params["max_new_tokens"]
        self.thinking_budget = params["thinking_budget"]
        self.temperature = params["temperature"]
        self.batch_processing = params["batch_processing"]

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

    def evaluate_batched(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set using batch processing when available.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        logger.info("Starting Claude Batch Processing...")
        # Creating an array of json tasks
        tasks = []
        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            idx = X[0]
            if self.is_agent:
                X_input = X[1]  # Full pandas Series with all patient features
            else:
                X_input = X[1].iloc[0]  # Single text prompt for standard models
            y_true = y[1].iloc[0]

            # system_message
            input_text, sys_msg = prompt_template_hf(
                X_input, None, self.model_name, task=self.task_name
            )

            task = Request(
                custom_id=f"task-{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    system=sys_msg,
                    max_tokens=self.max_new_tokens,
                    messages=input_text,
                    temperature=1.0,
                    thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
                ),
            )
            tasks.append(task)

        batch = self.client.messages.batches.create(requests=tasks)

        # Wait for the batch job to complete
        message_batch = None
        while True:
            message_batch = self.client.messages.batches.retrieve(batch.id)
            if message_batch.processing_status == "ended":
                break

            logger.info(f"Batch {batch.id} is still processing...")
            time.sleep(60)

        results = []
        for r in self.client.messages.batches.results(batch.id):
            results.append(r)

        # Sort by custom_id to maintain order
        results.sort(key=lambda x: x.custom_id.split("-")[1])

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        for r in results:
            idx = int(r.custom_id.split("-")[1])
            if idx not in test_loader[1].index or idx not in test_loader[0].index:
                logger.warning(
                    f"Result index {idx} not found in test_loader, skipping."
                )
                continue
            y_true = test_loader[1].loc[idx].iloc[0]
            X_input = test_loader[0].loc[idx].iloc[0]

            model_output = r.result
            if model_output.type == "errored":
                logger.warning(
                    "Error in model output for task %s: %s",
                    r.custom_id,
                    model_output.error,
                )
                metrics_tracker.add_results(np.nan, y_true)
                metrics_tracker.add_metadata_item(
                    {
                        "Input Prompt": X_input,
                        "Target Label": y_true,
                        "Predicted Probability": np.nan,
                        "Predicted Diagnosis": "error",
                        "Predicted Explanation": "error",
                        "Tokenization Time": 0,
                        "Inference Time": 0,
                        "Input Tokens": np.nan,
                        "Output Tokens": np.nan,
                        "Thinking Tokens": np.nan,
                        "Thinking Output": "error",
                    }
                )
                continue

            num_input_tokens = r.result.message.usage.input_tokens
            num_output_tokens = r.result.message.usage.output_tokens
            num_thinking_tokens = (
                0  # Model provides a summary and not the actual reasoning tokens
            )

            thinking_output = ""
            answer_output = ""
            for block in r.result.message.content:
                if block.type == "thinking":
                    thinking_output = block.thinking.encode(
                        "ascii", errors="replace"
                    ).decode("ascii")

                elif block.type == "text":
                    answer_output = block.text

            answer_output = answer_output.encode("ascii", errors="replace").decode(
                "ascii"
            )
            logger.debug("Decoded output:\n %s", answer_output)
            result_dict = parse_llm_output(answer_output)

            logger.info(
                "Input Tokens: %d | Output Tokens: %d | Thinking Budget: %d",
                num_input_tokens,
                num_output_tokens,
                num_thinking_tokens,
            )

            # Handle case where generated_text is a string instead of dict (when parsing fails)
            if isinstance(result_dict, dict):
                predicted_probability = float(result_dict.get("probability", np.nan))
                predicted_diagnosis = result_dict.get("diagnosis", "error")
                generated_explanation = result_dict.get("explanation", "error")
            else:
                predicted_probability = np.nan
                predicted_diagnosis = "error"
                generated_explanation = "error"

            logger.info(
                "Predicted probability: %s | True label: %s",
                predicted_probability,
                y_true,
            )

            metrics_tracker.add_results(predicted_probability, y_true)
            metrics_tracker.add_metadata_item(
                {
                    "Input Prompt": X_input,
                    "Target Label": y_true,
                    "Predicted Probability": predicted_probability,
                    "Predicted Diagnosis": predicted_diagnosis,
                    "Predicted Explanation": generated_explanation,
                    "Tokenization Time": 0,
                    "Inference Time": 0,
                    "Input Tokens": num_input_tokens,
                    "Output Tokens": num_output_tokens,
                    "Thinking Tokens": num_thinking_tokens,
                    "Thinking Output": thinking_output,
                }
            )
            if len(metrics_tracker.items) > 100:
                # Log metadata periodically to avoid memory issues
                metrics_tracker.log_metadata()

        metrics_tracker.log_metadata(save_to_file=self.save_metadata)
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report(prompting_id=self.prompting_id)

        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

    def evaluate_batched_offline(
        self, test_loader: Any, save_report: bool = False, batch_id=None
    ) -> float:
        """Evaluates the model on a given test set using batch processing when available.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        logger.info("Starting Claude Batch Processing...")
        # Creating an array of json tasks
        tasks = []
        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            idx = X[0]
            if self.is_agent:
                X_input = X[1]  # Full pandas Series with all patient features
            else:
                X_input = X[1].iloc[0]  # Single text prompt for standard models
            y_true = y[1].iloc[0]

            # system_message
            input_text, sys_msg = prompt_template_hf(
                X_input, None, self.model_name, task=self.task_name
            )

            task = Request(
                custom_id=f"task-{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    system=sys_msg,
                    max_tokens=self.max_new_tokens,
                    messages=input_text,
                    temperature=1.0,
                    thinking={"type": "enabled", "budget_tokens": self.thinking_budget},
                ),
            )
            tasks.append(task)

        # Wait for the batch job to complete
        message_batch = None
        while True:
            message_batch = self.client.messages.batches.retrieve(batch_id)
            if message_batch.processing_status == "ended":
                break

            logger.info(f"Batch {batch_id} is still processing...")
            time.sleep(60)

        results = []
        for r in self.client.messages.batches.results(batch_id):
            results.append(r)

        # Sort by custom_id to maintain order
        results.sort(key=lambda x: x.custom_id.split("-")[1])
        logger.debug(f"Shape of results: {len(results)}")
        logger.debug(f"Shape of y_true: {test_loader[1].shape}")
        logger.debug(f"Shape of X_input: {test_loader[0].shape}")

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )

        X_df = test_loader[0]
        y_df = test_loader[1].reset_index(
            drop=True
        )  # was not sorted in prompt preprocessing

        for r in results:
            idx_str = r.custom_id.split("-", 1)[1]
            # Ensure index type matches test_loader indices
            idx = None
            # Try integer first
            try:
                idx_int = int(idx_str)
                if idx_int in y_df.index and idx_int in X_df.index:
                    idx = idx_int
            except Exception:
                pass
            # Fallback to string
            if idx is None and idx_str in y_df.index and idx_str in X_df.index:
                idx = idx_str
            if idx is None:
                logger.warning(
                    f"Result index {idx_str} not found in test_loader, skipping."
                )
                continue
            y_true = y_df.loc[idx].iloc[0]
            X_input = X_df.loc[idx].iloc[0]
            X_input = test_loader[0].loc[idx].iloc[0]

            model_output = r.result
            if model_output.type == "errored":
                logger.warning(
                    "Error in model output for task %s: %s",
                    r.custom_id,
                    model_output.error,
                )
                metrics_tracker.add_results(np.nan, y_true)
                metrics_tracker.add_metadata_item(
                    {
                        "Input Prompt": X_input,
                        "Target Label": y_true,
                        "Predicted Probability": np.nan,
                        "Predicted Diagnosis": "error",
                        "Predicted Explanation": "error",
                        "Tokenization Time": 0,
                        "Inference Time": 0,
                        "Input Tokens": np.nan,
                        "Output Tokens": np.nan,
                        "Thinking Tokens": np.nan,
                        "Thinking Output": "error",
                    }
                )
                continue

            num_input_tokens = r.result.message.usage.input_tokens
            num_output_tokens = r.result.message.usage.output_tokens
            num_thinking_tokens = (
                0  # Model provides a summary and not the actual reasoning tokens
            )

            thinking_output = ""
            answer_output = ""
            for block in r.result.message.content:
                if block.type == "thinking":
                    thinking_output = block.thinking.encode(
                        "ascii", errors="replace"
                    ).decode("ascii")

                elif block.type == "text":
                    answer_output = block.text

            answer_output = answer_output.encode("ascii", errors="replace").decode(
                "ascii"
            )
            logger.debug("Decoded output:\n %s", answer_output)
            result_dict = parse_llm_output(answer_output)

            logger.info(
                "Input Tokens: %d | Output Tokens: %d | Thinking Budget: %d",
                num_input_tokens,
                num_output_tokens,
                num_thinking_tokens,
            )

            # Handle case where generated_text is a string instead of dict (when parsing fails)
            if isinstance(result_dict, dict):
                predicted_probability = float(result_dict.get("probability", np.nan))
                predicted_diagnosis = result_dict.get("diagnosis", "error")
                generated_explanation = result_dict.get("explanation", "error")
            else:
                predicted_probability = np.nan
                predicted_diagnosis = "error"
                generated_explanation = "error"

            logger.info(
                "Predicted probability: %s | True label: %s",
                predicted_probability,
                y_true,
            )

            metrics_tracker.add_results(predicted_probability, y_true)
            metrics_tracker.add_metadata_item(
                {
                    "Input Prompt": X_input,
                    "Target Label": y_true,
                    "Predicted Probability": predicted_probability,
                    "Predicted Diagnosis": predicted_diagnosis,
                    "Predicted Explanation": generated_explanation,
                    "Tokenization Time": 0,
                    "Inference Time": 0,
                    "Input Tokens": num_input_tokens,
                    "Output Tokens": num_output_tokens,
                    "Thinking Tokens": num_thinking_tokens,
                    "Thinking Output": thinking_output,
                }
            )
            if len(metrics_tracker.items) > 100:
                # Log metadata periodically to avoid memory issues
                metrics_tracker.log_metadata()

        metrics_tracker.log_metadata(save_to_file=self.save_metadata)
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report(prompting_id=self.prompting_id)

        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)
