import logging
from typing import Any, Optional

import joblib
import torch
from torch.utils.data import DataLoader
import gc
import psutil
import tempfile
import os

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

        self.prompting_id = params.get("prompting_id", None)
        self.pretrained_model_path = kwargs.get("pretrained_model_path")
        self.type = params.get("type", None)

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
                if free_mem < model_size * 1.2:  # add a safety margin
                    logger.warning("Not enough CPU memory to offload the model. Free: %.2f MB, Required: %.2f MB. Try to reduce the number of loaded models.")
                else:
                    self.llm_model.to("cpu")
                self.llm_model.to("cpu")
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

