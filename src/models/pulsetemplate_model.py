import logging
from typing import Any, Dict, Optional

import joblib
import torch
from torch.utils.data import DataLoader
import gc
import psutil
import time
import os
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from src.util.config_util import set_seeds
from src.util.model_util import extract_dict, prompt_template_hf


logger = logging.getLogger("PULSE_logger")


class PulseTemplateModel:
    """
    Base model template that all other models will inherit from.

    This class provides the common attributes and methods that all models
    in the Pulse framework should implement.
    """

    def __init__(
        self, model_name: str, trainer_name: Optional[str] = None, **kwargs
    ) -> None:
        """Initialize a new Pulse model.

        Args:
            model_name: Name of the model
            trainer_name: Optional name of the trainer
        """
        params = kwargs.get("params", {})
        self.params = params
        self.model_name = model_name
        self.trainer_name = trainer_name
        self.trainer = None
        self.model = None
        self.dataset_name = None
        self.task_name = None
        self.save_metadata = None

        self.pretrained_model_path = kwargs.get("pretrained_model_path")
        self.type = params.get("type", None)

        if self.type == "LLM":
            self.prompting_id = params.get("prompting_id", None)
        else:
            self.prompting_id = None

    def set_trainer(
        self,
        trainer_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
    ) -> None:
        """
        Set the trainer for this model. This method should be overridden by subclasses.
        A trainer is responsible for training and evaluating the model.

        Args:
            trainer_name: Name of the trainer to use
            train_loader: DataLoader for training data
            test_loader: DataLoader for test data
        """
        self.trainer_name = trainer_name
        self.trainer = None

    def check_required_params(params: dict, required_params: list) -> None:
        """Check if all required parameters are present in the params dictionary.

        Args:
            params: Dictionary of parameters
            required_params: List of required parameter names

        Raises:
            ValueError: If any required parameter is missing
        """
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")


    def load_model_weights(self, model_path: str) -> None:
        """Load model weights from a specified path.

        Args:
            model_path: Path to the model weights file
        """
        if self.type == "convML":
            # Load the sklearn model using joblib
            self.model = joblib.load(model_path)

        elif self.type == "convDL":
            # Load the state dictionary
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))

            # Check if the loaded file is a full model or just weights
            if hasattr(state_dict, "state_dict"):
                state_dict = state_dict.state_dict()

            # Load the weights into the model
            if hasattr(self, "load_state_dict"):
                self.load_state_dict(state_dict)
                logger.info("Model weights loaded successfully")
            else:
                logger.warning(
                    "Model does not have load_state_dict method. Cannot load weights."
                )

        elif self.type == "LLM":
            # Load LLM model weights
            pass
        else:
            logger.warning("Model type not recognized. Cannot load model weights.")
    
    def offload_model_to_cpu(self) -> None:
        """Offloads the model from GPU memory (if applicable)."""
        if self.type in ["convDL", "LLM"]:
            # For PyTorch and LLM models
            if self.type == "LLM":
                free_mem = psutil.virtual_memory().available / (1024 ** 2)  # in MB
                model_size = sum(p.numel() for p in self.llm_model.parameters()) * 4 / (1024 ** 2)  # float32 assumed
                logger.debug("CPU free memory: %.2f MB | Model size: %.2f MB", free_mem, model_size)
                if free_mem < model_size * 1.2:  # safety margin
                    logger.warning("Not enough CPU memory to offload the model. Free: %.2f MB, Required: %.2f MB. Deleting from GPU only")
                    # Delete from GPU only
                    del self.llm_model
                    gc.collect()
                    torch.cuda.empty_cache()

                else:
                    # Check if only partially offloaded: if any parameter is still on CUDA, delete the model
                    if any(p.device.type == "cuda" for p in self.llm_model.parameters()):
                        logger.warning("Model is only partially offloaded. Deleting model from GPU.")
                        del self.llm_model
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        # Fully offload the model to CPU
                        self.llm_model.to("cpu")
                        logger.info("LLM model offloaded to CPU memory")
            else:
                # TODO: implement for convDL models
                # self.model.to("cpu")
                pass
            torch.cuda.empty_cache()
            logger.info("Model offloaded from GPU memory")
            self.is_loaded = False

        elif self.type == "convML":
            # Sklearn models are always on CPU
            logger.info("Sklearn model is always on CPU; nothing to offload")
        else:
            logger.warning("Unknown model type; cannot offload")

    def load_model_to_gpu(self) -> None:
        """Loads the model to GPU memory (if applicable)."""
        if self.type in ["convDL", "LLM"]:
            # For PyTorch and LLM models
            if torch.cuda.is_available():
                device = getattr(self, "device", torch.device("cuda"))
                if self.type == "LLM":
                    torch.cuda.empty_cache()
                    gc.collect()
                    free_mem = torch.cuda.mem_get_info(device)[0] / (1024 ** 2)  # in MB
                    model_size = sum(p.numel() for p in self.llm_model.parameters()) * 4 / (1024 ** 2)  # float32 assumed
                    logger.debug("GPU free memory: %.2f MB | Model size: %.2f MB", free_mem, model_size)
                    if free_mem < model_size * 1.2:  # add a safety margin
                        logger.warning("Not enough GPU memory to load the model. Free: %.2f MB, Required: %.2f MB", free_mem, model_size)
                    self.llm_model.to(device)
                else:
                    # For convDL models, load the model to the specified device
                    #TODO: implement for confDL models
                    pass
                    # self.model.to(device)
                logger.info("Model loaded to GPU memory")
                self.is_loaded = True
            else:
                logger.warning("CUDA not available.")
        elif self.type == "convML":
            # Sklearn models cannot be loaded to GPU
            pass
        else:
            logger.warning("Unknown model type; cannot load to GPU")


class PulseLLMModel(PulseTemplateModel):
    """
    Base model template for LLMs that inherits from PulseTemplateModel.
    This class provides additional attributes and methods specific to LLMs.
    """

    def __init__(self, model_name: str, **kwargs) -> None:
        super().__init__(model_name, **kwargs)
        self.llm_model = None
        self.is_loaded = False

        self.device = kwargs.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = kwargs.get("model_name", None)
        self.inference_only = kwargs.get("inference_only", True)
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")
        self.wandb = kwargs.get("wandb", False)
        self.task_name = kwargs.get("task_name")
        self.dataset_name = kwargs.get("dataset_name")

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights."""
        try:
            # Skip loading if already loaded
            if self.is_loaded:
                logger.info("Model already loaded, reusing existing instance")
                return

            logger.debug(f"Loading model %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=False, padding_side="left"
            )

            # Check if there is enough GPU memory for the model
            if torch.cuda.is_available():
                device = getattr(self, "device", torch.device("cuda"))
                torch.cuda.empty_cache()
                free_mem = torch.cuda.mem_get_info(device)[0] / (1024 ** 2)
                model_size = (
                    sum(p.numel() for p in self.llm_model.parameters()) * 4 / (1024 ** 2)
                )
                logger.debug(
                    "GPU free memory: %.2f MB | Model size: %.2f MB",
                    free_mem,
                    model_size,
                )
                if free_mem < model_size * 1.2:  # add a safety margin
                    logger.warning(
                        "Not enough GPU memory to load the model. Free: %.2f MB, Required: %.2f MB",
                        free_mem,
                        model_size,
                    )
            else:
                logger.error("CUDA not available.")

            # Common model loading configuration
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
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
                self.llm_model = get_peft_model(self.llm_model, tuning_config)
                logger.debug(self.llm_model.print_trainable_parameters())

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
        num_input_tokens = tokenized_inputs["input_ids"].size(1)

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

        # Get generated token ids (excluding prompt) and convert to a Python list
        generated_token_ids_list = outputs.sequences[0][num_input_tokens:].tolist()

        num_output_tokens = len(generated_token_ids_list)

        decoded_output = self.tokenizer.decode(
            generated_token_ids_list,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
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