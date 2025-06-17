import logging
import os
import warnings
from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import BitsAndBytesConfig

from src.models.pulse_model import PulseLLMModel
from src.util.config_util import set_seeds
from src.util.model_util import prompt_template_hf

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class Llama3Model(PulseLLMModel):
    """Llama 3 model wrapper."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the Llama3Model with parameters and paths.

        Args:
            params: Configuration dictionary with model specific parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        model_name = kwargs.pop("model_name", "Llama3Model")
        kwargs.get("trainer_name", "Llama3Trainer")

        super().__init__(model_name, params, **kwargs)

        # Check if all required parameters exist in config
        required_params = [
            "max_new_tokens",
            "verbose",
            "tuning",
            "num_epochs",
            "max_length",
            "do_sample",
            "temperature",
        ]
        self.check_required_params(params, required_params)

        # Extract commonly used parameters
        self.max_length = self.params["max_length"]

        # Setup quantization config and device
        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )

    def set_trainer(
        self,
        trainer_name: str,
        train_loader: Any,
        val_loader: Any,
        **kwargs: Any,
    ) -> None:
        """Sets the associated trainer instance.

        Args:
            trainer_name: Name of the trainer class.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
        """
        self.trainer_name = trainer_name
        logger.info("Setting trainer: %s", self.trainer_name)
        self.trainer = Llama3Trainer(self, train_loader, val_loader, **kwargs)


class Llama3Trainer:
    """Trainer class for Llama3Model."""

    def __init__(self, model: Llama3Model, train_loader, val_loader, **kwargs) -> None:
        """
        Initialize the Llama3 trainer. Finetruning is not implemented yet.
        This is a wrapper for inference only.

        Args:
            model (Llama3Model): The Llama3 model to be trained.
            train_loader: The DataLoader object for the training dataset.
            val_loader: The DataLoader object for the validation dataset.
            test_loader: The DataLoader object for the testing dataset.
        """
        raise NotImplementedError("Finetuning is not implemented yet for Llama3Model.")
        # Set seed for deterministic generation
        set_seeds(model.random_seed)

        # Load the model and tokenizer
        if kwargs.get("disable_model_load", False):
            logger.info("Skipping model loading for debugging purposes.")
        else:
            model._load_model()  #

        self.model = model
        self.model = model.model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.BCELoss()  # Binary Cross Entropy Loss
        self.wandb = self.model.wandb
        self.model_save_dir = os.path.join(model.save_dir, "Models")
        self.task_name = self.model.task_name
        self.dataset_name = self.model.dataset_name
        self.tuning = self.params.get("tuning", False)

        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        """Training loop."""
        # Set seed for deterministic generation
        set_seeds(self.model.random_seed)

        logger.info("System message: %s", prompt_template_hf("")[0])
        logger.info("Starting training...")

        if self.tuning:
            raise NotImplementedError(
                "Prompt tuning is not implemented for Llama3Model yet. Set tuning parameter to false."
            )
            # logger.info(
            #     "Tuning model with prompt tuning. Model is saved in %s",
            #     self.model_save_dir,
            # )
            # optimizer = optim.AdamW(
            #     self.model.parameters(), lr=self.params.get("lr", 1e-4)
            # )
            # num_epochs = self.params.get("num_epochs", 1)

            # self.model.train()
            # for epoch in range(num_epochs):
            #     epoch_loss = 0.0
            #     logger.info(f"Epoch {epoch + 1} started...")
            #     for i, (X, y) in enumerate(
            #         zip(
            #             self.train_loader[0].iterrows(), self.train_loader[1].iterrows()
            #         )
            #     ):
            #         # Input prompt
            #         X_input = prompt_template_hf(X[1].iloc[0])
            #         inputs = self.model.tokenizer.apply_chat_template(
            #             X_input, tokenize=False, add_generation_prompt=True
            #         )

            #         # Build target output label
            #         probability = y[1].iloc[0]  # float
            #         diagnosis = (
            #             "not-" if probability < 0.5 else ""
            #         ) + self.model.task_name
            #         target_output = (
            #             "{\n"
            #             f'  "diagnosis": "{diagnosis}",\n'
            #             f'  "probability": {round(probability, 4)},\n'
            #             '  "explanation": "N/A"\n'
            #             "}\n\n"
            #         )

            #         encoded = self.encode_prompt_target(
            #             inputs,
            #             target_output,
            #             max_len=self.model.tokenizer.model_max_length,
            #         )

            #         optimizer.zero_grad()
            #         outputs = self.model(
            #             input_ids=encoded["input_ids"].to(self.device),
            #             attention_mask=encoded["attention_mask"].to(self.device),
            #             labels=encoded["labels"].to(self.device),
            #         )

            #         loss = outputs.loss
            #         loss.backward()

            #         optimizer.step()
            #         epoch_loss += loss.item()

            #         logger.info(
            #             "Step %d/%d, Loss: %.4f",
            #             i + 1,
            #             len(self.train_loader[0]),
            #             loss.item(),
            #         )

            #         if self.wandb:
            #             wandb.log({"train_loss": loss.item()})

            #     logger.info(
            #         "Epoch %d/%d, Avg Total Loss: %.4f",
            #         epoch + 1,
            #         num_epochs,
            #         epoch_loss / len(self.train_loader[0]),
            #     )
            #     if self.wandb:
            #         wandb.log(
            #             {f"avg_epoch_loss": epoch_loss / len(self.train_loader[0])}
            #         )

            #     val_loss = self.evaluate_single(self.val_loader)
            #     logger.info("Validation loss: %s", val_loss)

            #     self.model.save_pretrained(self.model_save_dir)
            #     self.model.tokenizer.save_pretrained(self.model_save_dir)
            #     logger.info("Model saved to %s", self.model_save_dir)

    def encode_prompt_target(
        self,
        prompt: str,
        target: str,
        max_len: int = 5000,
        add_special_tokens: bool = True,
    ) -> dict:
        """
        Tokenize and encode prompt and target into input_ids and labels for causal LM training.

        Args:
            prompt (str): The input prompt string.
            target (str): The target output string.
            max_len (int): The maximum length of the final sequence.
            add_special_tokens (bool): Whether to add special tokens during tokenization.

        Returns:
            dict: Dictionary containing input_ids, labels, and attention_mask.
        """
        # Tokenize prompt and target
        prompt_ids = self.model.tokenizer.encode(
            prompt, add_special_tokens=add_special_tokens
        )
        target_ids = self.model.tokenizer.encode(
            target, add_special_tokens=add_special_tokens
        )

        # Truncate from the start if too long
        input_ids = prompt_ids + target_ids
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]

        # Recompute where the target starts (after possible truncation of prompt)
        len(prompt_ids)
        total_len = len(input_ids)
        target_start_idx = max(0, total_len - len(target_ids))

        # Create labels: -100 for prompt, real target IDs for target
        labels = [-100] * target_start_idx + input_ids[target_start_idx:]

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)

        assert len(input_ids) == len(
            labels
        ), f"input_ids and labels length mismatch: {len(input_ids)} vs {len(labels)}"

        return {
            "input_ids": torch.tensor(
                input_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "labels": torch.tensor(
                labels, dtype=torch.long, device=self.device
            ).unsqueeze(0),
            "attention_mask": torch.tensor(
                attention_mask, dtype=torch.long, device=self.device
            ).unsqueeze(0),
        }
