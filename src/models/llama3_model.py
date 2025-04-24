from datetime import datetime
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from src.models.pulsetemplate_model import PulseTemplateModel
from src.eval.metrics import MetricsTracker
import logging
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import torch
import wandb

logger = logging.getLogger("PULSE_logger")


class Llama3Model(PulseTemplateModel):
    """Llama 3 model wrapper using LangChain for prompt templating and inference."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the Llama3Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        self.model_name = params.get(
            "model_name", self.__class__.__name__.replace("Model", "")
        )
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        self.save_dir: str = kwargs.get("output_dir", f"{os.getcwd()}/output")
        self.wandb: bool = kwargs.get("wandb", False)
        self.params: Dict[str, Any] = params
        self.model_id: str = self.params.get("model_id", "meta-llama/Llama-3.1-8B")
        self.max_length: int = self.params.get("max_length", 512)

        self.tokenizer: Optional[Any] = None
        self.llama_model: Optional[Any] = None
        self.lc_llm: Optional[Any] = None
        self.prompt_template: Optional[PromptTemplate] = None
        self.lc_chain: Optional[Runnable] = None

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes LangChain pipeline."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto"
            )
            logger.info("Successfully loaded Llama3 model: %s", self.model_id)
        except Exception as e:
            logger.error("Failed to load Llama3 model: %s", e)
            raise

        logger.info("Initializing Llama3 with parameters: %s", self.params)
        self._init_langchain()

    def _init_langchain(self) -> None:
        """Initializes the LangChain LLM and prompt pipeline."""
        pipe = pipeline(
            "text-generation",
            model=self.llama_model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_new_tokens=self.max_length,
        )
        self.lc_llm = pipe

        self.prompt_template = PromptTemplate(
            input_variables=["input"],
            template=self.params.get("prompt_template", "{input}"),
        )

        self.lc_chain = self.prompt_template | self.lc_llm

    def infer_llm(self, input_text: str) -> str:
        """Runs the LangChain pipeline with the given input.

        Args:
            input_text: A string input to feed into the prompt.

        Returns:
            The generated text response from the model.
        """
        return self.lc_chain.invoke({"input": input_text})

    def parse_output(self, generated_text: str) -> float:
        """Parses the generated text output into a float probability or label.

        Args:
            generated_text: Raw output from the LLM.

        Returns:
            A float probability or classification result.
        """
        try:
            if "yes" in generated_text.lower():
                return 1.0
            elif "no" in generated_text.lower():
                return 0.0
            return float(generated_text.strip())
        except Exception:
            return 0.5

    def set_trainer(
        self,
        trainer_name: str,
        train_dataloader: Any,
        val_dataloader: Any,
        test_dataloader: Any,
    ) -> None:
        """Sets the associated trainer instance.

        Args:
            trainer_name: Name of the trainer class.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            test_dataloader: DataLoader for test data.
        """
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
        # Load the model and tokenizer
        model._load_model()  # Comment out to only test preprocessing

        self.model = model
        self.llama_model = model.llama_model
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

        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

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

    def train(self):
        """Training loop."""
        verbose = self.params.get("verbose", 1)

        # Move to GPU if available
        # TODO: Managed by accelerate. Check if this is needed at all.
        # self.llama_model.to(self.device)
        # self.criterion.to(self.device)

        logger.info("Starting training...")
        val_loss = self.evaluate(self.val_loader)  # Evaluate on validation set
        logger.info("Validation loss: %s", val_loss)

        self.evaluate(
            self.test_loader, save_report=True
        )  # Evaluate on test set and save metrics

    def evaluate(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        self.llama_model.eval()

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            y_true = y[1].iloc[0]

            generated_text: str = self.model.infer_llm(X_input)
            predicted_probability: float = self.model.parse_output(generated_text)

            predicted_label = torch.tensor(
                predicted_probability, dtype=torch.float32
            ).unsqueeze(0)
            target = torch.tensor(float(y_true), dtype=torch.float32).unsqueeze(0)

            loss = self.criterion(predicted_label, target)
            val_loss.append(loss.item())

            logger.info(
                "Validation loss: %s | Prediction: %s",
                loss.item(),
                generated_text.strip(),
            )

            if self.wandb:
                wandb.log({"val_loss": loss.item()})

            metrics_tracker.add_results(y_true, predicted_label)

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return float(np.mean(val_loss))
