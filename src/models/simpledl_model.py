from typing import Any, Dict
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from .pulsetemplate_model import PulseTemplateModel


logger = logging.getLogger("PULSE_logger")


class SimpleDLModel(nn.Module, PulseTemplateModel):
    """
    A Deeplearning example PyTorch model with a simple architecture.
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the model with configurable layer sizes.

        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden layer
            output_size (int): Number of output classes
        """
        # Set model parameters.
        model_name = "SimpleDLModel"  # Required for all models
        trainer_name = params.get("trainer_name")  # Required for all models
        input_size = params.get("input_size")
        hidden_size = params.get("hidden_size")
        output_size = params.get("output_size")

        # Initialize both parent classes
        nn.Module.__init__(self)
        PulseTemplateModel.__init__(self, model_name, trainer_name)

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_size]
        """
        # Flatten input if needed (assuming x might be [batch_size, channels, height, width])
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)

        # First layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Second layer with ReLU activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Output layer (no activation - will be applied in the loss function)
        x = self.fc3(x)

        return x

    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name (str): Name of the trainer to be used.
        """
        if trainer_name == "SimpleDLTrainer":
            self.trainer = SimpleDLTrainer(self, train_dataloader, test_dataloader)
        else:
            raise ValueError(f"Trainer {trainer_name} not found.")


class SimpleDLTrainer:
    """
    Simple Deep Learning Model trainer class for demonstration purposes.
    This class is a placeholder and should be replaced with actual training logic.
    """

    def __init__(self, model, train_dataloader, test_dataloader):
        """
        Initialize the ExampleTrainer.

        Args:
            model: The model to be trained.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for testing data.
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

    def train(self):
        """
        Train the model using the loaded dataset.
        This is a placeholder method and should be replaced with actual training logic.
        """
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training parameters
        num_epochs = 5
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                # Move data to device
                data, target = data.to(device), target.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, target)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item()

                if batch_idx % 100 == 99:  # Print every 100 batches
                    logger.info(
                        f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}"
                    )
                    # Log to wandb
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "batch": batch_idx + 1,
                            "training_loss": running_loss / 100,
                        }
                    )
                    running_loss = 0.0

            # Validate after each epoch
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in self.test_dataloader:
                    data, target = data.to(device), target.to(device)
                    outputs = self.model(data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += target.size(0)
                    correct += (predicted == target).sum().item()

            logger.info(f"Epoch {epoch+1} Accuracy: {100 * correct / total:.2f}%")
            # Log to wandb
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "accuracy": 100 * correct / total,
                }
            )
