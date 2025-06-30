import logging
import os
import time
import warnings
from typing import Any, Dict

import numpy as np
import torch
from openai import AzureOpenAI

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseModel
from src.util.model_util import extract_dict, parse_llm_output, prompt_template_hf, system_message_samples

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class GPTModel(PulseModel):
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
            input_text, custom_system_message, model="GPTModel", task=self.task_name
        )

        # Generate output with scores
        infer_start = time.perf_counter()
        outputs = self.client.chat.completions.create(
            messages=input_text,
            max_tokens=self.params["max_new_tokens"],
            temperature=self.params.get("temperature", 0.4),
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

        generated_text = parse_llm_output(decoded_output)

        # Check if probability is a number or string, try to convert, else default to 0.5
        prob = generated_text.get("probability", 50)
        try:
            prob = float(prob)
        except (ValueError, TypeError):
            logger.warning("Failed to convert probability to float. Defaulting to 0.5")
            prob = 0.5
        generated_text["probability"] = prob

        logger.info(
            f"Inference time: {infer_time:.4f}s | Tokens: {num_input_tokens + num_output_tokens}"
        )

        return {
            "generated_text": generated_text,
            "infer_time": infer_time,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
        }

    def evaluate(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        logger.info("Starting test evaluation...")

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        criterion = torch.nn.BCELoss()
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        sys_msg = system_message_samples(task=self.task_name)[1]
        logger.info("System Message:\n\n %s", sys_msg)

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            y_true = y[1].iloc[0]

            result_dict = self.generate(X_input, custom_system_message=sys_msg)

            generated_text = result_dict["generated_text"]
            infer_time = result_dict["infer_time"]
            num_input_tokens = result_dict["num_input_tokens"]
            num_output_tokens = result_dict["num_output_tokens"]

            predicted_probability = float(generated_text.get("probability", 50))

            logger.info(
                "Predicted probability: %s | True label: %s",
                predicted_probability,
                y_true,
            )
            if verbose > 1:
                logger.info("Diagnosis for: %s", generated_text["diagnosis"])
                logger.info(
                    "Generated explanation: %s \n", generated_text["explanation"]
                )
            if verbose > 2:
                logger.info("Input prompt: %s \n", X_input)

            predicted_label = torch.tensor(
                predicted_probability, dtype=torch.float32
            ).unsqueeze(0)
            target = torch.tensor(float(y_true), dtype=torch.float32).unsqueeze(0)

            loss = criterion(predicted_label, target)
            val_loss.append(loss.item())

            if self.wandb:
                wandb.log(
                    {
                        "val_loss": loss.item(),
                        "infer_time": infer_time,
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                    }
                )

            metrics_tracker.add_results(predicted_probability, y_true)
            metrics_tracker.add_metadata_item(
                {
                    "Input Prompt": X_input,
                    "Target Label": y_true,
                    "Predicted Probability": predicted_probability,
                    "Predicted Diagnosis": generated_text.get("diagnosis", ""),
                    "Predicted Explanation": generated_text.get("explanation", ""),
                    "Inference Time": infer_time,
                    "Input Tokens": num_input_tokens,
                    "Output Tokens": num_output_tokens,
                }
            )

        metrics_tracker.log_metadata(save_to_file=self.save_metadata)
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report(prompting_id=self.prompting_id)

        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return float(np.mean(val_loss))
