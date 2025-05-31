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
        self.model_name = kwargs.get("model_name", "GPTModel")
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        self.save_dir: str = kwargs.get("output_dir", f"{os.getcwd()}/output")
        self.wandb: bool = kwargs.get("wandb", False)

        required_params = [
            "max_new_tokens",
            "api_version",
            "api_key_name",
            "api_uri_name",
        ]
        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        self.params: Dict[str, Any] = params

        api_key = os.environ.get(params["api_key_name"])
        endpoint_uri = os.environ.get(params["api_uri_name"])

        self.client = AzureOpenAI(
            api_version=params["api_version"],
            azure_endpoint=endpoint_uri,
            api_key=api_key,
        )
        self.deployment = params["deployment"]

    def infer_llm(
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

    def set_trainer(
        self,
        trainer_name: str,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        **kwargs: Any,
    ) -> None:
        """Sets the associated trainer instance.

        Args:
            trainer_name: Name of the trainer class.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for test data.
        """
        self.trainer = GPTTrainer(self, train_loader, val_loader, test_loader, **kwargs)


class GPTTrainer:
    """Trainer class for GPTModel."""

    def __init__(
        self, model: GPTModel, train_loader, val_loader, test_loader, **kwargs
    ) -> None:
        """
        Initialize the GPT trainer. Finetruning is not implemented yet.
        This is a wrapper for inference only.

        Args:
            model (GPTModel): The GPT model to be trained.
            train_loader: The DataLoader object for the training dataset.
            val_loader: The DataLoader object for the validation dataset.
            test_loader: The DataLoader object for the testing dataset.
        """
        self.model = model
        self.params = model.params
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_test_set = self.params.get("save_test_set", False)

        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name

        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Exit if test_loader is bigger than 100
        if len(self.test_loader) > 1000:
            logger.warning(
                "Test set is larger than 1000 samples. Exiting to avoid bankrupcy..."
            )
            sys.exit(1)

    def train(self):
        """Training loop."""
        verbose = self.params.get("verbose", 1)
        logger.info("System message: %s", prompt_template_hf("")[0])
        logger.info("Starting inference...")

        self.evaluate_single(self.test_loader, save_report=True)

    def evaluate_single(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        if self.save_test_set:
            # Save test set to CSV
            test_loader[0].to_csv(
                os.path.join(self.model.save_dir, "test_set.csv"), index=False
            )
            test_loader[1].to_csv(
                os.path.join(self.model.save_dir, "test_labels.csv"), index=False
            )
            logger.info("Test set saved to %s", self.model.save_dir)
        logger.info("Starting test evaluation...")

        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            y_true = y[1].iloc[0]

            result_dict = self.model.infer_llm(X_input)

            generated_text = result_dict["generated_text"]
            infer_time = result_dict["infer_time"]
            num_input_tokens = result_dict["num_input_tokens"]
            num_output_tokens = result_dict["num_output_tokens"]

            predicted_probability = float(generated_text.get("probability", 0.5))

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

            loss = self.criterion(predicted_label, target)
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

        metrics_tracker.log_metadata(save_to_file=self.model.save_metadata)
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return float(np.mean(val_loss))
