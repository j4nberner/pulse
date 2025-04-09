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
from src.eval.metrics import calculate_all_metrics


logger = logging.getLogger("PULSE_logger")


class CNNModel(PulseTemplateModel, nn.Module):
    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """
        Initialize the CNN model.

        Args:
            params (Dict[str, Any]): Configuration parameters for the model.
            **kwargs: Additional keyword arguments.
        """
        # For trainer_name we still require it to be explicitly in the params
        if "trainer_name" not in params:
            raise KeyError("Required parameter 'trainer_name' not found in config")

        # Use the class name as model_name if not provided in params
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        trainer_name = params["trainer_name"]
        super().__init__(self.model_name, trainer_name)
        nn.Module.__init__(self)

        # Define all required parameters
        required_params = [
            "num_features",
            "output_shape",
            "num_channels",
            "kernel_size",
            "pool_size",
            "learning_rate",
            "num_epochs",
            "save_checkpoint",
            "verbose",
        ]

        # Check if wandb is enabled and set up
        self.wandb = kwargs.get("wandb", False)

        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        # Extract parameters from config
        self.params = {param: params[param] for param in required_params}

        # Log the parameters being used
        logger.info(f"Initializing CNN with parameters: {self.params}")

        model_dir = self.params.get("model_dir", r"output/models")
        self.model_dir = os.path.join(
            model_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.model_dir, exist_ok=True)
        logger.info(f"Model directory created at: {self.model_dir}")

        # -------------------------Define layers-------------------------
        self.conv1 = nn.Conv1d(
            in_channels=self.params["num_channels"],
            out_channels=256,
            kernel_size=self.params["kernel_size"],
            stride=1,
            padding=self.params["kernel_size"] // 2,
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=self.params["pool_size"])

        # Fully connected layer
        pooled_size = self.params["num_features"] // self.params["pool_size"]
        self.fc = nn.Linear(
            256 * pooled_size,
            self.params["output_shape"],
        )
        # -------------------------Define layers-------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_features, seq_length]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_shape]
        """
        # Apply convolutional layer
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Flatten the output
        x = x.view(x.size(0), -1)

        # Fully connected layer
        x = self.fc(x)

        return x

    def set_trainer(self, trainer_name: str, train_dataloader, test_dataloader) -> None:
        """
        Sets the trainer for the CNN model.

        Args:
            trainer_name (str): The name of the trainer to be used.
            train_dataloader: The DataLoader object for the training dataset.
            test_dataloader: The DataLoader object for the testing dataset.

        Returns:
            None
        """
        self.trainer = CNNTrainer(self, train_dataloader, test_dataloader)

    def save_model(self, checkpoint=False, epoch=None):
        """
        Saves the model to a file.

        Args:
            checkpoint (bool): If True, saves a checkpoint of the model.
            epoch (int): The current epoch number. Used for naming the checkpoint file.
        """
        if checkpoint and epoch is not None:
            model_path = os.path.join(
                self.model_dir, f"model_checkpoint_epoch_{self.model_name}_{epoch}.pth"
            )
        else:
            model_path = os.path.join(self.model_dir, f"{self.model_name}.pth")
        torch.save(self.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    def load_model(self, model_path=None):
        """Loads the model from a file."""
        if model_path is None:
            model_path = os.path.join(self.model_dir, "model.pth")  # Default path

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded from {model_path}")


class CNNTrainer:
    """Trainer for the CNN model."""

    def __init__(self, cnn_model, train_dataloader, test_dataloader):
        self.model = cnn_model
        self.params = cnn_model.params
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.wandb = self.model.wandb

    def train(self):
        """Training loop."""
        save_checkpoint = self.params["save_checkpoint"]
        num_epochs = self.params["num_epochs"]
        verbose = self.params.get("verbose", 1)

        logger.info("Starting training...")
        for epoch in range(num_epochs):
            self.train_epoch(epoch, verbose)
            logger.info(f"Epoch {epoch + 1} finished")
            # Save checkpoint every epoch
            if save_checkpoint != 0 and epoch % save_checkpoint == 0:
                self.model.save_model(checkpoint=True, epoch=epoch)

        logger.info("Training finished.")
        self.evaluate(self.test_dataloader)
        self.model.save_model()  # Save the final model

    def train_epoch(self, epoch: int, verbose: int = 1) -> None:
        """
        Trains the model for one epoch.

        Args:
            epoch (int): Current epoch number.
            verbose (int): Verbosity level (0, 1, or 2).
        """
        self.model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(self.train_dataloader):
            # Reshape inputs to (batch_size, 1, input_size) for CNN
            inputs = inputs.unsqueeze(1)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            if verbose == 2 or (verbose == 1 and i % 10 == 9):
                logger.info(
                    f"Epoch {epoch + 1}, Batch {i + 1}: Loss = {running_loss / (10 if self.params.get('verbose', 1) == 1 else 1):.4f}"
                )
                if self.wandb:
                    wandb.log(
                        {
                            "loss": running_loss
                            / (10 if self.params.get("verbose", 1) == 1 else 1)
                        }
                    )
                if verbose == 1:
                    running_loss = 0.0

    def evaluate(self, test_dataloader):
        """
        Evaluates the model on the test set.

        Args:
            test_dataloader: The DataLoader object for the testing dataset.
        """
        self.model.eval()
        metrics_tracker = {
            "auroc": [],
            "auprc": [],
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "sensitivity": [],
            "specificity": [],
        }

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                # Reshape inputs to (batch_size, 1, input_size) for CNN
                inputs = inputs.unsqueeze(1)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                metrics = calculate_all_metrics(labels, predicted)
                for metric, value in metrics.items():
                    metrics_tracker[metric].append(value)

        # Average metrics over the dataset
        for metric in metrics_tracker:
            metrics_tracker[metric] = np.mean(metrics_tracker[metric])

        logger.info(f"Evaluation metrics: {metrics_tracker}")
        if self.wandb:
            wandb.log(metrics_tracker)
