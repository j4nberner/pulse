from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import save_torch_model
from src.eval.metrics import MetricsTracker
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer


logger = logging.getLogger("PULSE_logger")


class Llama3Model(PulseTemplateModel):
    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the Llama3 model.

        Args:
            params (Dict[str, Any]): Configuration parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        # Use the class name as model_name if not provided in params
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        # Set the model save directory
        self.save_dir = kwargs.get("output_dir", f"{os.getcwd()}/output")

        # Define all required parameters
        required_params = []

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        # Extract parameters from config
        self.params = {
            param: params[param] for param in required_params if param in params
        }

        # Model defaults
        self.model_id = self.params.get("model_id", "meta-llama/Llama-2-7b-hf")
        self.max_length = self.params.get("max_length", 512)

        # Initialize the model and tokenizer
        try:
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_id)
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto"
            )
            logger.info(f"Successfully loaded Llama3 model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load Llama3 model: {e}")
            raise

        # Log the parameters being used
        logger.info(f"Initializing Llama3 with parameters: {self.params}")

    def evaluate(self, test_dataloader) -> Dict[str, Any]:
        """
        Evaluate the Llama3 model on the provided test dataloader.

        Args:
            test_dataloader: The DataLoader object for the testing dataset.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        # Initialize metrics tracker
        metrics = MetricsTracker(self.model_name, self.task_name, self.dataset_name)

        # Iterate over the test dataloader
        for X, y in test_dataloader:
            # Tokenize the input data
            inputs = self.tokenizer(
                X,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
            )
            # Generate predictions
            with torch.no_grad():
                outputs = self.llama_model.generate(
                    inputs["input_ids"],
                    max_length=self.max_length,
                    num_return_sequences=1,
                )
            # Decode the generated outputs
            generated_text = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Extract the predicted labels from the generated text
            predicted_labels = [
                text.split("\n")[-1] for text in generated_text
            ]  # TODO: adjust
            metrics.add_results(y, predicted_labels)
            # Print the generated text
            logger.info(f"Generated text: {generated_text}")

        # Log the evaluation results
        metrics.compute_overall_metrics()
        metrics.save_report()

    def set_trainer(
        self, trainer_name: str, train_dataloader, val_dataloader, test_dataloader
    ) -> None:
        """
        Sets the trainer for the Llama3 model.

        Args:
            trainer_name (str): The name of the trainer to be used.
            train_dataloader: The DataLoader object for the training dataset.
            val_dataloader: The DataLoader object for the validation dataset.
            test_dataloader: The DataLoader object for the testing dataset.

        Returns:
            None
        """
        # This is a wrapper for inference only, so training is not implemented
        logger.warning("Training is not implemented for the Llama3 wrapper model.")
        pass
