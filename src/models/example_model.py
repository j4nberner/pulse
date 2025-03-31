import torch
import torch.nn as nn
import torch.nn.functional as F


class ExampleModel(nn.Module):
    """
    A dummy example PyTorch model with a simple architecture.
    """

    def __init__(self, trainer_name, input_size=784, hidden_size=128, output_size=10):
        """
        Initialize the model with configurable layer sizes.

        Args:
            input_size (int): Size of the input features
            hidden_size (int): Size of the hidden layer
            output_size (int): Number of output classes
        """
        super(ExampleModel, self).__init__()

        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(p=0.2)

        self.name = "ExampleModel"
        self.trainer_name = trainer_name
        self.trainer = None

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

    def predict(self, x):
        """
        Make a prediction (with softmax for probabilities).

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Probability distribution over output classes
        """
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def set_trainer(self, trainer_name, train_dataloader, test_dataloader):
        """
        Set the trainer for the model.

        Args:
            trainer_name (str): Name of the trainer to be used.
        """
        if trainer_name == "ExampleTrainer":
            self.trainer = ExampleTrainer(self, train_dataloader, test_dataloader)
        else:
            raise ValueError(f"Trainer {trainer_name} not found.")


class ExampleTrainer:
    """
    Example trainer class for demonstration purposes.
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

        # Placeholder for actual training logic
        print(f"Training {self.model.name}...")

    def test(self):
        """
        Test the model using the loaded dataset.
        This is a placeholder method and should be replaced with actual testing logic.
        """

        # Placeholder for actual testing logic
        print(f"Testing {self.model.name}...")
