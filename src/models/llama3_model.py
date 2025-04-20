from datetime import datetime
from typing import Dict, Any
import logging
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch
import wandb
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import prepare_data_for_model_dl, save_torch_model
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
        self.params = params

        # Model defaults
        self.model_id = self.params.get("model_id", "meta-llama/Llama-3.1-8B")
        self.max_length = self.params.get("max_length", 512)

        self.tokenizer = None
        self.llama_model = None

    def _load_model(self) -> None:
        """
        Load the Llama3 model and tokenizer. Done only at inference time to save resources.
        """
        try:
            # Load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

            # Use disk_offload to handle large models
            with init_empty_weights():
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                    self.model_id, torch_dtype=torch.float16
                )
            self.llama_model = load_checkpoint_and_dispatch(
                self.llama_model,
                self.model_id,
                device_map="auto",
                offload_folder="offload",
                offload_state_dict=True,
            )

            logger.info(f"Successfully loaded Llama3 model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to load Llama3 model: {e}")
            raise

        # Log the parameters being used
        logger.info(f"Initializing Llama3 with parameters: {self.params}")

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
        self.trainer = Llama3Trainer(
            self, train_dataloader, val_dataloader, test_dataloader
        )


class Llama3Trainer:

    def __init__(
        self, model: Llama3Model, train_loader, val_loader, test_loader
    ) -> None:
        """
        Initialize the Llama3 trainer. Finetruning is not implemented yet.
        This is a wrapper for inference only.

        Args:
            model (Llama3Model): The Llama3 model to be trained.
            train_loader: The DataLoader object for the training dataset.
            val_loader: The DataLoader object for the validation dataset.
            test_loader: The DataLoader object for the testing dataset.
        """
        self.model = model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = nn.BCEWithLogitsLoss()
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name

        logger.info(f"Using criterion: {self.criterion.__class__.__name__}")

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

        # Get the configured data converter
        # TODO: implement this for LLMs?
        # self.converter = prepare_data_for_model_dl(
        #     self.train_loader,
        #     self.params,
        #     model_name=self.model.model_name,
        #     task_name=self.task_name,
        # )

        # Load the model and tokenizer
        self.model._load_model()

    def train(self):
        """Training loop."""
        num_epochs = self.params["num_epochs"]
        verbose = self.params.get("verbose", 1)

        # Move to GPU if available
        self.model.to(self.device)
        self.criterion.to(self.device)

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            val_loss = self.evaluate(self.val_loader)  # Evaluate on validation set

        self.evaluate(
            self.test_loader, save_report=True
        )  # Evaluate on test set and save metrics

    def evaluate(self, test_loader, save_report: bool = False) -> float:
        """
        Evaluate the Llama3 model on the provided test dataloader.

        Args:
            test_dataloader: The DataLoader object for the testing or validation dataset.

        Returns:
            float: The average loss on the test set.
        """
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose = self.params.get("verbose", 1)
        val_loss = []

        # Set the model to evaluation mode
        self.model.llama_model.eval()
        self.model.tokenizer.eval()

        # Move the model and tokenizer to the appropriate device
        self.model.tokenizer.to(self.device)
        self.model.llama_model.to(self.device)

        # Iterate over the test dataloader
        # TODO: maybe need to force batch size to 1 for Llama3?
        for X, y in test_loader:
            # Tokenize the input data
            inputs = self.model.tokenizer(
                X,
                return_tensors="pt",
                truncation=True,
                max_length=self.model.max_length,
            )
            # Generate predictions
            with torch.no_grad():
                outputs = self.model.llama_model.generate(
                    inputs["input_ids"],
                    max_length=self.model.max_length,
                    num_return_sequences=1,
                )
            # Decode the generated outputs
            generated_text = self.model.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

            # Extract the predicted labels from the generated text
            predicted_labels = [
                text.split("\n")[-1] for text in generated_text
            ]  # TODO: adjust

            # Convert the predicted labels to tensors
            predicted_labels = torch.tensor(
                [int(label) for label in predicted_labels], dtype=torch.float32
            ).to(self.device)
            y = y.to(self.device)
            # Calculate the loss
            loss = self.criterion(predicted_labels, y)
            val_loss.append(loss.item())

            metrics_tracker.add_results(y, predicted_labels)
            # Print the generated text
            logger.info(f"Generated text: {generated_text}")

        # Calculate and log metrics
        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()
        return np.mean(val_loss)
