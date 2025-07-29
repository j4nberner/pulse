import gc
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

import joblib
import numpy as np
import torch
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb

if TYPE_CHECKING:
    pass

from src.eval.metrics import MetricsTracker
from src.models.agents import create_agent_instance
from src.util.config_util import set_seeds
from src.util.model_util import (
    parse_llm_output,
    prompt_template_hf,
    system_message_samples,
)

logger = logging.getLogger("PULSE_logger")

os.environ["TORCHDYNAMO_DISABLE"] = "1"  # Disabled to not get Gemma errors


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
        self.inference_only = self.mode == "inference"

        self.prompting_id = params.get("prompting_id", None)
        self.is_agent = False
        self.agent_instance = None
        self.model_size_mb = None

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

            # Load model from pretrained
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=(
                    torch.bfloat16 if torch.cuda.is_available() else torch.float32
                ),
            )
            self.model_size_mb = sum(
                p.numel() * p.element_size() for p in self.model.parameters()
            ) / (1024 * 1024)
            logger.info("Model size: %.2f MB", self.model_size_mb)

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

    def delete_model(self) -> None:
        """
        Delete the model from GPU memory. Sets is_loaded to False.
        """
        if self.is_loaded:
            logger.info("Deleting the model %s", self.model_id)

            del self.model
            gc.collect()
            torch.cuda.empty_cache()
            self.is_loaded = False
            return

        else:
            logger.warning("Model is not loaded, nothing to delete.")

    def generate(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Unified generation method that automatically routes based on model configuration.

        Args:
            input_data: Input data (string for standard models, pd.Series for agents)
            **kwargs: Additional arguments passed to generation methods

        Returns:
            Dictionary with generated_text and metrics
        """
        # Ensure model is loaded
        if not self.is_loaded:
            logger.debug("Model not loaded yet for inference, loading now...")
            self.load_model()

        # Route based on agent flag
        if self.is_agent:
            return self._generate_with_agent(input_data, **kwargs)
        else:
            return self._generate_standard(input_data, **kwargs)

    def _generate_standard(
        self,
        input_text: str,
        custom_system_message: str = None,
        parse_json: bool = True,
        force_raw_text: bool = False,
    ) -> Dict[str, Any]:
        """Standard generation logic for non-agent models."""
        # Set seed for deterministic generation
        set_seeds(self.random_seed)

        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text = prompt_template_hf(
            input_text,
            custom_system_message=custom_system_message,
            model=self.model_name,
            task=self.task_name,
        )

        # Tokenize with chat template
        tokenized_inputs = self.tokenizer.apply_chat_template(
            input_text,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)

        num_input_tokens = tokenized_inputs["input_ids"].size(1)

        logger.debug(
            "GPU memory allocated: %s", torch.cuda.memory_allocated() / (1024**3)
        )

        # Generate output with scores
        infer_start = time.perf_counter()

        # Set model-specific generation parameters
        output_scores = True if self.model_name == "DeepseekR1Model" else False

        with torch.no_grad():
            outputs = self.model.generate(
                **tokenized_inputs,
                max_new_tokens=self.params["max_new_tokens"],
                return_dict_in_generate=True,
                output_scores=output_scores,
                output_hidden_states=False,
                # pad_token_id=self.tokenizer.pad_token_id,
                # eos_token_id=self.tokenizer.eos_token_id,
                do_sample=self.params["do_sample"],
                temperature=self.params["temperature"],
            )
        infer_time = time.perf_counter() - infer_start

        # Get generated token ids (excluding prompt) and convert to a Python list
        generated_token_ids_list = outputs.sequences[0][num_input_tokens:].tolist()
        num_output_tokens = len(generated_token_ids_list)

        decoded_output = self.tokenizer.decode(
            generated_token_ids_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # Initialize thinking_output for models that don't use it
        thinking_output = ""
        num_thinking_tokens = 0

        if self.model_name == "Gemma3Model":
            # Trim after first <end_of_turn>
            decoded_output = decoded_output.split("<end_of_turn>")[0]
        elif self.model_name == "DeepseekR1Model":
            # Handle DeepSeek thinking tokens
            if "</think>" in decoded_output:
                thinking_output = decoded_output.split("</think>")[0]
                decoded_output = decoded_output.split("</think>")[1]
                num_thinking_tokens = len(
                    self.tokenizer(
                        thinking_output,
                        return_tensors="pt",
                    )[
                        "input_ids"
                    ][0]
                )
            logger.debug("Answer output:\n %s", decoded_output)

        # Log the decoded output for debugging
        logger.debug("Decoded output:\n %s", decoded_output)

        # Parse the output if parse_json is True and force_raw_text is False
        if parse_json and not force_raw_text:
            generated_text = parse_llm_output(decoded_output)
        else:
            generated_text = decoded_output

        logger.info(
            "Inference time: %.4fs | Input Tokens: %d | Output Tokens: %d | Thinking Budget: %d",
            infer_time,
            num_input_tokens,
            num_output_tokens,
            num_thinking_tokens,
        )

        # Build result structure
        result = {
            "generated_text": generated_text,
            "thinking_output": thinking_output,
            "token_time": 0.0,
            "infer_time": infer_time,
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": num_output_tokens,
            "num_thinking_tokens": num_thinking_tokens,
        }

        return result

    def evaluate(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        # Check if model is already loaded before attempting to load
        if not self.is_loaded:
            self.load_model()
        else:
            logger.info("Using already loaded model instance for evaluation")

        logger.info("Starting test evaluation...")

        metrics_tracker = MetricsTracker(
            self.model_name,
            self.task_name,
            self.dataset_name,
            self.save_dir,
        )

        # For agents: Initialize agent and set the metrics_tracker in the agent's memory manager
        if self.is_agent:
            if not self.agent_instance:
                self._initialize_agent()
            if self.agent_instance:
                self.agent_instance.memory.metrics_tracker = metrics_tracker
                logger.debug(
                    "Set MetricsTracker in agent memory manager for %s", self.model_name
                )

        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        try:
            self.model.eval()
        except Exception as e:
            pass

        # Ensure sequential Sample IDs for agent models
        if self.is_agent:
            test_loader = (
                test_loader[0].reset_index(drop=True),
                test_loader[1].reset_index(drop=True),
            )

        skip_samples = 0

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            # Skip first n samples if specified in params
            if X[0] < skip_samples:
                continue

            idx = X[0]
            if self.is_agent:
                X_input = X[1]  # Full pandas Series with all patient features
            else:
                X_input = X[1].iloc[0]  # Single text prompt for standard models
            y_true = y[1].iloc[0]

            try:
                # Set target for agent models
                if self.is_agent and self.agent_instance:
                    self.agent_instance.memory.set_current_sample_target(y_true)

                # Get raw result from generation
                result_dict = self.generate(X_input)

            except Exception as e:
                logger.error(
                    "Error during inference for sample %s: %s", idx, e, exc_info=True
                )
                # Create fallback result
                result_dict = {
                    "generated_text": {
                        "diagnosis": "error",
                        "probability": np.nan,
                        "explanation": f"error: {str(e)}",
                    },
                    "token_time": 0.0,
                    "infer_time": 0.0,
                    "num_input_tokens": 0,
                    "num_output_tokens": 0,
                    "num_thinking_tokens": 0,
                    "thinking_output": "",
                }

            # Extract results
            generated_text = result_dict["generated_text"]
            token_time = result_dict["token_time"]
            infer_time = result_dict["infer_time"]
            num_input_tokens = result_dict["num_input_tokens"]
            num_output_tokens = result_dict["num_output_tokens"]
            num_thinking_tokens = result_dict["num_thinking_tokens"]
            thinking_output = result_dict["thinking_output"]

            # Handle case where generated_text is a string instead of dict (when parsing fails)
            if isinstance(generated_text, dict):
                predicted_probability = float(generated_text.get("probability", np.nan))
                predicted_diagnosis = generated_text.get("diagnosis", "error")
                generated_explanation = generated_text.get("explanation", "error")
            else:
                predicted_probability = np.nan
                predicted_diagnosis = "error"
                generated_explanation = "error"

            logger.info(
                "Predicted probability: %s | True label: %s",
                predicted_probability,
                y_true,
            )

            if verbose > 1:
                logger.info("Diagnosis for: %s", predicted_diagnosis)
                logger.info("Generated explanation: %s \n", generated_explanation)
            if verbose > 2:
                logger.info("Input prompt: %s \n", X_input)

            if self.wandb:
                wandb.log(
                    {
                        "token_time": token_time,
                        "infer_time": infer_time,
                        "num_input_tokens": num_input_tokens,
                        "num_output_tokens": num_output_tokens,
                    }
                )

            metrics_tracker.add_results(predicted_probability, y_true)

            if not self.is_agent:  # Agent models handle metadata logging in add_step()
                metrics_tracker.add_metadata_item(
                    {
                        "Input Prompt": X_input,
                        "Target Label": y_true,
                        "Predicted Probability": predicted_probability,
                        "Predicted Diagnosis": predicted_diagnosis,
                        "Predicted Explanation": generated_explanation,
                        "Tokenization Time": token_time,
                        "Inference Time": infer_time,
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

        # self.delete_model()

        return 0.0

    def evaluate_sys_msgs(self, test_loader: Any, save_report: bool = True) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        # Criterion not needed for LLM inference
        # criterion = nn.BCELoss()  # Binary Cross Entropy Loss

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

        sys_msgs = system_message_samples(task=self.task_name)

        # Skip first n samples if specified
        skip_samples = self.params.get("skip_samples", 0)
        if skip_samples > 0:
            logger.info(
                "Skipping first %d samples in the test set for system message evaluation",
                skip_samples,
            )
            X_test, y_test = test_loader
            X_test = X_test.iloc[skip_samples:]
            y_test = y_test.iloc[skip_samples:]
            test_loader = (X_test, y_test)

        self.model.eval()

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]  # The input text for standard pipeline
            y_true = y[1].iloc[0]  # The true label

            for i, sys_msg in enumerate(sys_msgs):
                # Standard inference for non-agent predictions
                result_dict = self.generate(
                    input_data=X_input, custom_system_message=sys_msg
                )

                generated_text = result_dict["generated_text"]
                token_time = result_dict["token_time"]
                infer_time = result_dict["infer_time"]
                num_input_tokens = result_dict["num_input_tokens"]
                num_output_tokens = result_dict["num_output_tokens"]

                # Handle case where generated_text is a string instead of dict (when parsing fails)
                if isinstance(generated_text, dict):
                    predicted_probability = float(
                        generated_text.get("probability", np.nan)
                    )
                else:
                    predicted_probability = np.nan

                logger.info(
                    "Predicted probability: %s | True label: %s",
                    predicted_probability,
                    y_true,
                )
                if verbose > 1:
                    if isinstance(generated_text, dict):
                        logger.info(
                            "Diagnosis for: %s", generated_text.get("diagnosis", "")
                        )
                        logger.info(
                            "Generated explanation: %s \n",
                            generated_text.get("explanation", ""),
                        )
                    else:
                        logger.info("Raw output: %s", generated_text)
                if verbose > 2:
                    logger.info("Input prompt: %s \n", X_input)

                predicted_label = torch.tensor(
                    predicted_probability, dtype=torch.float32
                ).unsqueeze(0)
                target = torch.tensor(float(y_true), dtype=torch.float32).unsqueeze(0)

                val_loss.append(np.nan)

                metrics_tracker.add_results(predicted_probability, y_true)
                # Handle case where generated_text is a string instead of dict
                if isinstance(generated_text, dict):
                    diagnosis = generated_text.get("diagnosis", "")
                    explanation = generated_text.get("explanation", "")
                else:
                    diagnosis = ""
                    explanation = str(generated_text) if generated_text else ""

                metrics_tracker.add_metadata_item(
                    {
                        "Input Prompt": X_input,
                        "Target Label": y_true,
                        "Predicted Probability": predicted_probability,
                        "Predicted Diagnosis": diagnosis,
                        "Predicted Explanation": explanation,
                        "Tokenization Time": token_time,
                        "Inference Time": infer_time,
                        "Input Tokens": num_input_tokens,
                        "Output Tokens": num_output_tokens,
                        "System Message": sys_msg,
                        "System Message Index": i,
                    }
                )
                if len(metrics_tracker.items) > 100:
                    # Log metadata periodically to avoid memory issues
                    metrics_tracker.log_metadata()

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.log_metadata()
            metrics_tracker.save_report()

        logger.info("System Message evaluation completed for %s", self.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

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

    # -------------------------------
    # Agent-specific methods
    # -------------------------------

    def _generate_with_agent(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Generate using agent-based multi-step reasoning."""
        try:
            # Initialize agent if not already done
            if not self.agent_instance:
                self._initialize_agent()

            # If agent initialization failed, fallback to standard
            if not self.agent_instance:
                logger.warning(
                    "Agent initialization failed for %s, using standard generation",
                    self.model_name,
                )
                return self._generate_standard(str(input_data), **kwargs)

            # Update agent context
            if self.agent_instance:
                self.agent_instance.update_task_context(
                    getattr(self, "task_name", None),
                    getattr(self, "dataset_name", None),
                )

            # Validate input data type
            if isinstance(input_data, str):
                logger.warning("Agent received string input instead of structured data")
                return self._generate_standard(input_data, **kwargs)

            # Set current sample in agent memory for tracking
            sample_id = getattr(input_data, "name", "default")
            self.agent_instance.memory.set_current_sample(sample_id)

            # Process through agent - returns raw unparsed output
            result = self.agent_instance.process_single(input_data)

            # Adding default Thinking Output for compatibility
            result["thinking_output"] = ""
            result["num_thinking_tokens"] = 0

            return result

        except Exception as e:
            logger.error("Error in agent processing: %s", e, exc_info=True)
            # Fallback to standard generation
            return self._generate_standard(str(input_data), **kwargs)

    def _initialize_agent(self) -> None:
        """Initialize the agent instance based on prompting_id."""
        self.agent_instance = create_agent_instance(
            prompting_id=self.prompting_id,
            model=self,
            task_name=getattr(self, "task_name", None),
            dataset_name=getattr(self, "dataset_name", None),
            output_dir=getattr(self, "save_dir", None),
            metrics_tracker=None,  # Will be set in evaluate method
        )

        if self.agent_instance:
            self.is_agent = True
        else:
            self.is_agent = False
