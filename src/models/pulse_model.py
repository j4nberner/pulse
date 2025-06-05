import gc
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import joblib
import numpy as np
import psutil
import torch
import torch.nn as nn
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb
from src.eval.metrics import MetricsTracker
from src.util.config_util import set_seeds
from src.util.model_util import extract_dict, prompt_template_hf, sys_msg_smpls

logger = logging.getLogger("PULSE_logger")


class PulseModel:
    """
    Base pulse model.

    This class provides the common attributes and methods that all models
    in the Pulse framework should implement.
    """

    def __init__(
        self, model_name: str, params, trainer_name: Optional[str] = None, **kwargs
    ) -> None:
        """Initialize a new Pulse model with the model name and configuration parameters.

        Args:
            model_name: Name of the model
            params: Dictionary of parameters for the model
            trainer_name: Optional name of the trainer
            **kwargs: Additional keyword arguments for model configuration
        """
        # Required parameters for all models
        self.model_name = model_name
        self.params = params
        self.model = None
        self.type = params.get("type", None)
        self.mode = params.get("mode", "inference")  # train, inference
        self.is_loaded = False
        self.dataset_name = None
        self.task_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.random_seed = self.params.get("random_seed", 42)
        set_seeds(self.params["random_seed"])
        logger.debug("Using random seed: %d", self.random_seed)

        self.trainer_name = trainer_name
        self.trainer = None
        self.criterion = None

        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")
        self.save_metadata = kwargs.get("save_metadata", True)
        self.wandb = kwargs.get("wandb", False)
        self.pretrained_model_path = kwargs.get("pretrained_model_path", None)

    def set_trainer(
        self,
        trainer_name: str,
        model: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        """
        Set the trainer for this model. This method should be overridden by subclasses.
        A trainer is responsible for training and evaluating the model.

        Args:
            trainer_name: Name of the trainer to use
            model: The model instance to train
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
        """
        from src.models import get_trainer_class

        self.trainer_name = trainer_name
        cls = get_trainer_class(trainer_name)
        self.trainer = cls(model, train_loader, val_loader)

    def check_required_params(self, params: dict, required_params: list) -> None:
        """Check if all required parameters are present in the params dictionary.

        Args:
            params: Dictionary of parameters
            required_params: List of required parameter names

        Raises:
            ValueError: If any required parameter is missing
        """
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(
                f"Missing required parameters: {', '.join(missing_params)}"
            )

    def load_model_weights(self, model_path: str) -> None:
        """Load model weights from a specified path.

        Args:
            model_path: Path to the model weights file
        """
        if self.type == "convML":
            # Load the sklearn model using joblib
            self.model = joblib.load(model_path)
            self.is_loaded = True
            logger.info("Sklearn model loaded successfully from %s", model_path)

        elif self.type == "convDL":
            logger.info("Loading model weights from %s", model_path)
            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))

            # Check if the loaded file is a full model or just weights
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()

            # Load the weights into the model
            if hasattr(self, "load_state_dict"):
                self.load_state_dict(state_dict, strict=False)
                self.is_loaded = True
            else:
                logger.warning(
                    "Model does not have load_state_dict method. Cannot load weights."
                )

        elif self.type == "LLM":
            # Load LLM model weights
            pass
        else:
            logger.warning("Model type not recognized. Cannot load model weights.")


class PulseLLMModel(PulseModel):
    """
    Base model for Huggingface-LLMs that inherits from PulseTemplateModel.
    This class provides additional attributes and methods specific to LLMs.
    """

    def __init__(
        self,
        model_name: str,
        params: Dict[str, Any],
        trainer_name: Optional[str] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        """Initialize a new Pulse LLM model.
        Args:
            model_name: Name of the model
            params: Dictionary of parameters for the model
            trainer_name: Optional name of the trainer
            **kwargs: Additional keyword arguments for model configuration
        """
        super().__init__(model_name, params, trainer_name, **kwargs)

        self.device = kwargs.get(
            "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        logger.debug("Number of GPUs: %d", torch.cuda.device_count())

        self.model_id = params.get("model_id", None)
        self.tokenizer = None

        self.inference_only = kwargs.get(
            "inference_only", False
        )  # TODO: @sophiafe needed?
        self.prompting_id = params.get("prompting_id", None)

    def load_model(self) -> None:
        """Loads the tokenizer and model weights."""
        try:
            # Skip loading if already loaded
            if self.is_loaded:
                logger.info("Model already loaded, reusing existing instance")
                return

            logger.debug("Loading model %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=False, padding_side="left"
            )

            # Check if model is already loaded on CPU and move to GPU if needed
            if self.model is not None:
                # Check if model is on CPU and CUDA is available
                if torch.cuda.is_available() and all(
                    p.device.type == "cpu" for p in self.model.parameters()
                ):
                    logger.info("Moving model from CPU to GPU")
                    self.model.to(self.device)
                else:
                    logger.info("Model already loaded")
            else:
                # Load model from pretrained
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    device_map="auto" if torch.cuda.is_available() else None,
                    torch_dtype=(
                        torch.bfloat16 if torch.cuda.is_available() else torch.float32
                    ),
                )

            # Apply tuning only in full training mode and if specified
            if not self.inference_only and self.params.get("tuning", False):
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

            logger.info("Successfully loaded %s model.", self.model_id)

            # Only log pipeline initialization in full training mode
            if not self.inference_only:
                logger.info(
                    "Initializing Hugging Face pipeline with parameters: %s",
                    self.params,
                )

            # Mark model as loaded after successful loading
            self.is_loaded = True

        except Exception as e:
            logger.error("Failed to load the %s model.", self.model_id)
            logger.exception(e)
            raise e

    def offload_model(self) -> None:
        """
        Offloads the model from GPU memory to CPU. Sets is_loaded to False.
        If CPU memory is insufficient, deletes the model and clears cache.
        """
        if self.is_loaded:
            logger.info("Offloading model %s to CPU", self.model_id)
            try:
                self.model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()
                self.is_loaded = False
                logger.info("Model offloaded successfully")
            except RuntimeError as e:
                logger.warning(
                    "Failed to offload model to CPU due to insufficient memory: %s. Deleting model from memory.",
                    e,
                )
                del self.model
                gc.collect()
                torch.cuda.empty_cache()
                self.is_loaded = False
        else:
            logger.warning("Model is not loaded, nothing to offload")

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
        num_input_tokens = tokenized_inputs["input_ids"].size(1)

        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)

        # Generate output with scores
        infer_start = time.perf_counter()

        with torch.no_grad():
            outputs = self.model.generate(
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

        # Get generated token ids (excluding prompt) and convert to a Python list
        generated_token_ids_list = outputs.sequences[0][num_input_tokens:].tolist()
        num_output_tokens = len(generated_token_ids_list)

        decoded_output = self.tokenizer.decode(
            generated_token_ids_list,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

        if self.model_name == "Gemma3Model":
            # Trim after first <end_of_turn>
            decoded_output = decoded_output.split("<end_of_turn>")[0]

        logger.debug("Decoded output:\n %s", decoded_output)

        # Check if we should return raw text or parsed JSON (important for multi-turn conversations)
        if force_raw_text:
            # For text-only outputs like summaries
            return {
                "generated_text": decoded_output,  # Return raw text
                "token_time": token_time,
                "infer_time": infer_time,
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
            }

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

    def evaluate(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss

        # Check if model is already loaded before attempting to load
        if not self.is_loaded:
            self.load_model()
        else:
            logger.info("Using already loaded model instance for evaluation")

        logger.info("Starting test evaluation...")

        # Set seed for deterministic generation
        set_seeds(self.random_seed)

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        self.model.eval()

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
                        "Error parsing agent prediction: %s - Falling back to standard inference",
                        e,
                    )
                    # Run normal inference as fallback
                    result_dict = self.generate(X_input)
            else:
                # Standard inference for non-agent predictions
                result_dict = self.generate(X_input)

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

            loss = criterion(predicted_label, target)
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


        metrics_tracker.log_metadata(save_to_file=self.save_metadata)
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        logger.info("Test evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        self.offload_model()

        return float(np.mean(val_loss))

    def evaluate_sys_msgs(self, test_loader: Any, save_report: bool = True) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        criterion = nn.BCELoss()  # Binary Cross Entropy Loss

        # Check if model is already loaded before attempting to load
        if not self.is_loaded:
            self.load_model()
        else:
            logger.info("Using already loaded model instance for evaluation")

        logger.info("Starting test evaluation...")

        # Set seed for deterministic generation
        set_seeds(self.random_seed)

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        sys_msgs = sys_msg_smpls(task=self.task_name)

        self.model.eval()

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]  # The input text for standard pipeline
            y_true = y[1].iloc[0]  # The true label

            for i, sys_msg in enumerate(sys_msgs):
                # Standard inference for non-agent predictions
                result_dict = self.generate(input_text=X_input, custom_system_message=sys_msg)

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

                loss = criterion(predicted_label, target)
                val_loss.append(loss.item())

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
                        "System Message": sys_msg,
                        "System Message Index": i,
                    }
                )

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.log_metadata(save_to_file=self.save_metadata)
            metrics_tracker.save_report()

        logger.info("System Message evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        self.offload_model()

        return float(np.mean(val_loss))

    def estimate_nr_tokens(self, data_loader) -> int:
        """Estimates the number of tokens for a task-dataset combination.

        Returns:
            The estimated number of tokens.
        """
        logger.info("Estimating number of tokens for the dataset...")
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, use_fast=False, padding_side="left"
        )
        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )

        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        num_input_tokens = 0
        num_output_tokens = 0

        for X, y in zip(data_loader[0].iterrows(), data_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            # Format input using prompt template
            input_text = prompt_template_hf(X_input)

            # Tokenize with chat template
            chat_prompt = self.tokenizer.apply_chat_template(
                input_text, tokenize=False, add_generation_prompt=True
            )
            tokenized_inputs = self.tokenizer(
                chat_prompt,
                return_tensors="pt",
            )
            num_input_tokens = tokenized_inputs["input_ids"].size(1)
            token_dict = {
                "Input Prompt": input_text,
                "Input Tokens": num_input_tokens,
                "Output Tokens": self.params.max_new_tokens,
            }

            metrics_tracker.add_metadata_item(token_dict)
            num_input_tokens = token_dict["Input Tokens"]
            num_output_tokens = token_dict["Output Tokens"]
            total_input_tokens += num_input_tokens
            total_output_tokens += num_output_tokens
            total_tokens += num_input_tokens + num_output_tokens
            logger.debug(
                "Input tokens: %s | Output tokens: %s",
                num_input_tokens,
                num_output_tokens,
            )

        metrics_tracker.log_metadata(save_to_file=self.save_metadata)
        return total_tokens
