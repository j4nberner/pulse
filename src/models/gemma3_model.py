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
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
)

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseTemplateModel
from src.util.model_util import extract_dict, prompt_template_hf
from src.util.config_util import set_seeds

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class Gemma3Model(PulseTemplateModel):
    """Gemma 3 model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the Gemma3Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        # Add model loading flag
        self.is_loaded = False

        # Initialize essential properties first
        self.model_name = kwargs.get("model_name", "Gemma3Model")
        self.inference_only = kwargs.get("inference_only", False)
        self.params = params

        if self.inference_only:
            # For inference-only mode (agentic workflow)
            self.trainer_name = params.get("trainer_name", "Gemma3Trainer")
            # Skip parent initialization for agentic workflow
            self.random_seed = self.params.get("random_seed", 42)
            logger.debug("Using random seed: %d", self.random_seed)

            # Set necessary parameters for inference
            self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")
            self.wandb = kwargs.get("wandb", False)
            self.task_name = kwargs.get("task_name")
            self.dataset_name = kwargs.get("dataset_name")
        else:
            # Full model initialization for standard workflow
            self.trainer_name = params["trainer_name"]
            super().__init__(self.model_name, self.trainer_name, params=params)

            # Store random seed from params (added by ModelManager)
            self.random_seed = self.params.get("random_seed", 42)
            logger.debug("Using random seed: %d", self.random_seed)

            self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")
            self.wandb = kwargs.get("wandb", False)

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

        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        self.params: Dict[str, Any] = params
        self.params["save_test_set"] = kwargs.get("save_test_set", False)

        self.model_id: str = self.params.get("model_id", "google/gemma-3-12b-it")
        self.max_length: int = self.params.get("max_length", 5120)

        self.tokenizer: Optional[Any] = None
        self.llm_model: Optional[Any] = None

        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Number of GPUs: %d", torch.cuda.device_count())

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        try:
            # Skip loading if already loaded
            if (
                self.tokenizer is not None
                and self.llm_model is not None
                and self.is_loaded
            ):
                logger.info("Model already loaded, reusing existing instance")
                return

            logger.debug(f"Loading model %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, padding_side="left"
            )
            self.llm_model = Gemma3ForConditionalGeneration.from_pretrained(
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
                self.llm_model = get_peft_model(self.llm_model, tuning_config)
                logger.debug(self.llm_model.print_trainable_parameters())

            logger.info("Successfully loaded Gemma3 model: %s", self.model_id)

            # Only log pipeline initialization in full training mode
            if not self.inference_only:
                logger.info(
                    "Initializing Hugging Face pipeline with parameters: %s",
                    self.params,
                )

            # Mark model as loaded after successful loading
            self.is_loaded = True

        except Exception as e:
            logger.error("Failed to load Gemma3 model: %s", e)
            raise

    def infer_llm(
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
        if self.tokenizer is None or self.llm_model is None:
            logger.debug("Model not loaded yet for inference, loading now...")
            self._load_model()

        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text = prompt_template_hf(
            input_text, custom_system_message, self.model_name
        )

        # Tokenize with chat template
        chat_prompt = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )

        token_start = time.perf_counter()
        tokenized_inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
        )
        token_time = time.perf_counter() - token_start

        # num_input_tokens = tokenized_inputs["input_ids"].size(1)
        num_input_tokens = tokenized_inputs["input_ids"].shape[-1]

        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)

        # Generate output with scores
        infer_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.llm_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.params["max_new_tokens"],
                return_dict_in_generate=True,
                output_scores=False,
                output_hidden_states=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=self.params["do_sample"],
                temperature=self.params["temperature"],
            )
        infer_time = time.perf_counter() - infer_start

        # Get first sequence, decode to string
        generated_text = self.tokenizer.decode(
            outputs.sequences[0][num_input_tokens:], skip_special_tokens=True
        )
        num_output_tokens = outputs.sequences[0].size(0) - num_input_tokens

        # Trim after first <end_of_turn>
        generated_text = generated_text.split("<end_of_turn>")[0]
        logger.debug("Generated text: %s", generated_text)

        # Check if we should return raw text or parsed JSON (important for multi-turn conversations)
        if force_raw_text:
            # For text-only outputs like summaries
            return {
                "generated_text": generated_text,  # Return raw text
                "token_time": token_time,
                "infer_time": infer_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
            }

        # Extract dict from the decoded output (e.g., via regex or JSON parsing)
        parsed = extract_dict(generated_text)

        # Check if probability is a number or string, try to convert, else default to 0.5
        prob = parsed.get("probability", 0.5)
        try:
            prob = float(prob)
        except (ValueError, TypeError):
            logger.warning("Failed to convert probability to float. Defaulting to 0.5")
            prob = 0.5
        parsed["probability"] = prob

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
        }

    def calculate_tokens(self, input_text: str) -> Dict[str, Any]:
        """
        Runs the full inference without loading the model and calculates the number of input and output tokens.
        Assuming num_output_tokens = max_new_tokens.

        Args:
            input_text: The input text to be tokenized.
        Returns:
            A dictionary containing the number of input and output tokens.
        """

        # Format input using prompt template
        input_text = prompt_template_hf(input_text)

        # Tokenize with chat template
        chat_prompt = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )
        tokenized_inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
        )
        num_input_tokens = tokenized_inputs["input_ids"].size(1)

        return {
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": self.params.max_new_tokens,
        }

    def set_trainer(
        self,
        trainer_name: str,
        train_dataloader: Any,
        val_dataloader: Any,
        test_dataloader: Any,
        **kwargs: Any,
    ) -> None:
        """Sets the associated trainer instance.

        Args:
            trainer_name: Name of the trainer class.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            test_dataloader: DataLoader for test data.
        """
        self.trainer = Gemma3Trainer(
            self, train_dataloader, val_dataloader, test_dataloader, **kwargs
        )


class Gemma3Trainer:
    """Trainer class for Gemma3Model."""

    def __init__(
        self, model: Gemma3Model, train_loader, val_loader, test_loader, **kwargs
    ) -> None:
        """
        Initialize the Gemma3 trainer. Finetruning is not implemented yet.
        This is a wrapper for inference only.

        Args:
            model (Gemma3Model): The Gemma3 model to be trained.
            train_loader: The DataLoader object for the training dataset.
            val_loader: The DataLoader object for the validation dataset.
            test_loader: The DataLoader object for the testing dataset.
        """
        # Set seed for deterministic generation
        set_seeds(model.random_seed)

        # Load the model and tokenizer
        if kwargs.get("disable_model_load", False):
            logger.info("Skipping model loading for debugging purposes.")
        else:
            model._load_model()  #

        self.model = model
        self.llm_model = model.llm_model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_test_set = self.params.get("save_test_set", False)

        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.tuning = self.params.get("tuning", False)

        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        """Training loop."""
        # Set seed for deterministic generation
        set_seeds(self.model.random_seed)

        verbose = self.params.get("verbose", 1)
        logger.info("System message: %s", prompt_template_hf("")[0])
        logger.info("Starting training...")

        if self.tuning:
            logger.info(
                "Tuning model with prompt tuning. Model is saved in %s",
                self.model_save_dir,
            )
            optimizer = optim.AdamW(
                self.llm_model.parameters(), lr=self.params.get("lr", 1e-4)
            )
            num_epochs = self.params.get("num_epochs", 1)

            self.llm_model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                logger.info(f"Epoch {epoch + 1} started...")
                for i, (X, y) in enumerate(
                    zip(
                        self.train_loader[0].iterrows(), self.train_loader[1].iterrows()
                    )
                ):
                    # Input prompt
                    X_input = prompt_template_hf(X[1].iloc[0])
                    inputs = self.model.tokenizer.apply_chat_template(
                        X_input, tokenize=False, add_generation_prompt=True
                    )

                    # Build target output label
                    probability = y[1].iloc[0]  # float
                    diagnosis = (
                        "not-" if probability < 0.5 else ""
                    ) + self.model.task_name
                    target_output = (
                        "{\n"
                        f'  "diagnosis": "{diagnosis}",\n'
                        f'  "probability": {round(probability, 4)},\n'
                        '  "explanation": "N/A"\n'
                        "}\n\n"
                    )

                    encoded = self.encode_prompt_target(
                        inputs,
                        target_output,
                        max_len=self.model.tokenizer.model_max_length,
                    )

                    optimizer.zero_grad()
                    outputs = self.llm_model(
                        input_ids=encoded["input_ids"].to(self.device),
                        attention_mask=encoded["attention_mask"].to(self.device),
                        labels=encoded["labels"].to(self.device),
                    )

                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()
                    epoch_loss += loss.item()

                    logger.info(
                        "Step %d/%d, Loss: %.4f",
                        i + 1,
                        len(self.train_loader[0]),
                        loss.item(),
                    )

                    if self.wandb:
                        wandb.log({"train_loss": loss.item()})

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Avg Total Loss: {epoch_loss/len(self.train_loader[0]):.4f}"
                )
                if self.wandb:
                    wandb.log(
                        {f"avg_epoch_loss": epoch_loss / len(self.train_loader[0])}
                    )

                val_loss = self.evaluate_single(self.val_loader)
                logger.info("Validation loss: %s", val_loss)

                self.llm_model.save_pretrained(self.model_save_dir)
                self.model.tokenizer.save_pretrained(self.model_save_dir)
                logger.info("Model saved to %s", self.model_save_dir)

        self.evaluate_single(self.test_loader, save_report=True)

    def evaluate_single(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        # Set seed for deterministic generation
        set_seeds(self.model.random_seed)

        # Check if model is already loaded before attempting to load
        if not self.model.is_loaded:
            self.model._load_model()
        else:
            logger.info("Using already loaded model instance for evaluation")

        logger.info("Starting test evaluation...")

        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        self.llm_model.eval()

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            idx = X[0]  # The index of the current row
            X_input = X[1].iloc[0]  # The input text for standard pipeline
            y_true = y[1].iloc[0]  # The true label

            # Check if this row contains an agent prediction
            is_agent_prediction = False
            if "is_agent_prediction" in test_loader[0].columns:
                is_agent_prediction = bool(
                    test_loader[0].at[idx, "is_agent_prediction"]
                )
                logger.debug(
                    "Sample %s: is_agent_prediction = %s (type: %s)",
                    idx,
                    is_agent_prediction,
                    type(is_agent_prediction),
                )

            if is_agent_prediction:
                logger.info(
                    "Found agent prediction - using directly without additional inference"
                )

                try:
                    # Parse the agent's prediction JSON
                    agent_output = (
                        json.loads(X_input) if isinstance(X_input, str) else X_input
                    )

                    # Extract prediction fields
                    predicted_probability = float(agent_output.get("probability", 0.5))
                    diagnosis = agent_output.get("diagnosis", "")
                    explanation = agent_output.get("explanation", "")

                    # Get token metrics if available
                    token_time = 0.0  # Placeholder values
                    infer_time = 0.0
                    num_input_tokens = (
                        test_loader[0].at[idx, "num_input_tokens"]
                        if "num_input_tokens" in test_loader[0].columns
                        else 100
                    )
                    num_output_tokens = (
                        test_loader[0].at[idx, "num_output_tokens"]
                        if "num_output_tokens" in test_loader[0].columns
                        else 50
                    )

                    # Create result structure matching what infer_llm would return
                    result_dict = {
                        "generated_text": {
                            "diagnosis": diagnosis,
                            "probability": predicted_probability,
                            "explanation": explanation,
                        },
                        "token_time": token_time,
                        "infer_time": infer_time,
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                    }

                    logger.info(
                        "Using agent prediction: %s with probability %s",
                        diagnosis,
                        predicted_probability,
                    )

                except Exception as e:
                    logger.error(
                        f"Error parsing agent prediction: {e} - Falling back to standard inference"
                    )
                    # Run normal inference as fallback
                    result_dict = self.model.infer_llm(X_input)
            else:
                # Standard inference for non-agent predictions
                result_dict = self.model.infer_llm(X_input)

            generated_text = result_dict["generated_text"]
            token_time = result_dict["token_time"]
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
                        "token_time": token_time,
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
                    "Tokenization Time": token_time,
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

    def estimate_nr_tokens(self) -> int:
        """Estimates the number of tokens for a task-dataset combination.

        Returns:
            The estimated number of tokens.
        """
        logger.info("Estimating number of tokens for the dataset...")
        # Load the tokenizer
        self.model.tokenizer = AutoTokenizer.from_pretrained(
            self.model.model_id, use_fast=False, padding_side="left"
        )

        test_loader = self.test_loader
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        num_input_tokens = 0
        num_output_tokens = 0

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            token_dict = self.model.calculate_tokens(X_input)
            num_input_tokens = token_dict["num_input_tokens"]
            num_output_tokens = token_dict["num_output_tokens"]
            total_input_tokens += num_input_tokens
            total_output_tokens += num_output_tokens
            total_tokens += num_input_tokens + num_output_tokens
            logger.debug(
                "Input tokens: %s | Output tokens: %s",
                num_input_tokens,
                num_output_tokens,
            )

        logger.info(
            f"Total tokens for the task {self.model.task_name} dataset {self.model.dataset_name}: {total_tokens}"
        )
        logger.info("Total input tokens: %s", total_input_tokens)
        logger.info("Total output tokens: %s", total_output_tokens)
        logger.info(
            "Average input tokens: %s", total_input_tokens / len(test_loader[0])
        )
        logger.info(
            "Average output tokens: %s", total_output_tokens / len(test_loader[0])
        )
        return total_tokens

    def evaluate_batched(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set in batches.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        NotImplementedError(
            "Batch evaluation is not implemented for Gemma3Model. Use evaluate_single instead."
        )

    def encode_prompt_target(
        self,
        prompt: str,
        target: str,
        max_len: int = 5000,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Tokenize and encode prompt and target into input_ids and labels for causal LM training.

        Args:
            prompt (str): The input prompt string.
            target (str): The target output string.
            max_len (int): The maximum length of the final sequence.
            add_special_tokens (bool): Whether to add special tokens during tokenization.

        Returns:
            dict: Dictionary containing input_ids, labels, and attention_mask.
        """
        # Tokenize prompt and target
        prompt_ids = self.model.tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )
        target_ids = self.model.tokenizer.encode(
            target, add_special_tokens=add_special_tokens
        )

        # Truncate from the start if too long
        input_ids = prompt_ids + target_ids
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]

        # Recompute where the target starts (after possible truncation of prompt)
        prompt_len = len(prompt_ids)
        total_len = len(input_ids)
        target_start_idx = max(0, total_len - len(target_ids))

        # Create labels: -100 for prompt, real target IDs for target
        labels = [-100] * target_start_idx + input_ids[target_start_idx:]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(
            labels
        ), f"input_ids and labels length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "labels": torch.tensor(
                labels, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=self.device
            ).unsqueeze(0),
        }
