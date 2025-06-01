import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulse_model import PulseTemplateModel
from src.util.model_util import extract_dict, prompt_template_hf

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class MeditronModel(PulseTemplateModel):
    """Meditron model wrapper using LangChain for prompt templating and inference."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the MeditronModel with parameters and paths.

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

        required_params = [
            "max_new_tokens",
        ]
        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        self.params: Dict[str, Any] = params
        self.params["save_test_set"] = kwargs.get("save_test_set", False)

        self.model_id: str = self.params.get("model_id", "epfl-llm/meditron-7b")
        self.max_length: int = self.params.get("max_length", 5120)

        self.tokenizer: Optional[Any] = None
        self.meditron_model: Optional[Any] = None
        self.lc_llm: Optional[Any] = None
        self.prompt_template: Optional[PromptTemplate] = None
        self.lc_chain: Optional[Runnable] = None

        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.meditron_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            if self.params.get("tuning", False):
                logger.info("Applying Prompt Tuning")
                tuning_config = PromptTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    tokenizer_name_or_path=self.model_id,
                    num_virtual_tokens=20,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    prompt_tuning_init_text="Classify the diagnosis of following ICU data:",
                )
                self.meditron_model = get_peft_model(self.meditron_model, tuning_config)
                logger.debug(self.meditron_model.print_trainable_parameters())

            logger.info("Successfully loaded Meditron model: %s", self.model_id)
        except Exception as e:
            logger.error("Failed to load Meditron model: %s", e)
            raise

        logger.info(
            "Initializing Hugging Face pipeline with parameters: %s", self.params
        )

    def infer_llm(self, input_text: str) -> Dict[str, Any]:
        """Runs the HF pipeline with the given input and logs timing/token info.

        Args:
            input_text: A string input to feed into the prompt.

        Returns:
            A dictionary with the generated text, timing information, and token count.
        """
        if not isinstance(input_text, str):
            input_text = str(input_text)

        input_text = prompt_template_hf(
            input_text, model="MeditronModel"
        )  # Apply prompt template to structure the input and guide output.

        token_start = time.perf_counter()
        # chat_prompt = self.tokenizer.apply_chat_template(
        #     input_text, tokenize=False, add_generation_prompt=True
        # )

        # logger.debug("-------------CHAT PROMPT-------------")
        # logger.debug(input_text)

        tokenized_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
        )
        token_time = time.perf_counter() - token_start
        num_tokens = tokenized_inputs["input_ids"].numel()
        logger.debug(f"NR Tokens: {num_tokens}")

        # logger.debug("-------------DECODED CHAT PROMPT-------------")
        # logger.debug(
        #     self.tokenizer.decode(
        #         tokenized_inputs["input_ids"][0],
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=True,
        #     )
        # )

        infer_start = time.perf_counter()
        self.meditron_model.to(self.device)

        with torch.no_grad():
            outputs = self.meditron_model.generate(
                input_ids=tokenized_inputs["input_ids"].to(self.device),
                attention_mask=tokenized_inputs["attention_mask"].to(self.device),
                max_new_tokens=self.params.max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # logger.debug("-------------GENERATED OUTPUTS-------------")
        # logger.debug(
        #     self.tokenizer.decode(
        #         outputs[0],
        #         skip_special_tokens=True,
        #         clean_up_tokenization_spaces=True,
        #     )
        # )
        # logger.debug("-------------GENERATED OUTPUTS END-------------")

        # 3) Slice off the prompt part:
        gen_ids = outputs[0, num_tokens:]

        # 4) Decode just the generated tokens:
        generated_text = self.tokenizer.decode(
            outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        infer_time = time.perf_counter() - infer_start
        logger.debug(
            "Decoded full outputs: %s",
            generated_text,
        )

        generated_text = extract_dict(
            generated_text
        )  # Extract dict from the generated text.

        generated_text["probability"] = round(
            (
                abs(generated_text["probability"] - 1.0)
                if "not-" in generated_text["diagnosis"]
                else abs(generated_text["probability"])
            ),
            3,
        )

        logger.info(
            f"Tokenization time: {token_time:.4f}s | Inference time: {infer_time:.4f}s | Tokens: {num_tokens}"
        )

        return {
            "generated_text": generated_text,
            "token_time": token_time,
            "infer_time": infer_time,
            "num_tokens": num_tokens,
        }

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
        self.trainer = MeditronTrainer(
            self, train_dataloader, val_dataloader, test_dataloader
        )

    def parse_output(self, output: str) -> float:
        """Parses the output string to extract the predicted probability.

        Args:
            output: The generated text from the model.

        Returns:
            A float representing the predicted probability.
        """
        try:
            # Extract the floating-point number from the output
            if "not-" in output:
                probability = np.abs(float(output.split(":")[-1].strip()) - 1.0)
            else:
                probability = float(output.split(":")[-1].strip())
            return probability
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse output. Defaulting to 0.5: %s", e)
            # Log the error and return a default value
            logger.info("Output: %s", output)
            return 0.5  # Default to 0.5 if parsing fails


class MeditronTrainer:
    """Trainer class for MeditronModel."""

    def __init__(
        self, model: MeditronModel, train_loader, val_loader, test_loader
    ) -> None:
        """
        Initialize the Meditron trainer. Finetruning is not implemented yet.
        This is a wrapper for inference only.

        Args:
            model (MeditronModel): The Meditron model to be trained.
            train_loader: The DataLoader object for the training dataset.
            val_loader: The DataLoader object for the validation dataset.
            test_loader: The DataLoader object for the testing dataset.
        """
        # Load the model and tokenizer
        model._load_model()  # Comment out to only test preprocessing

        self.model = model
        self.meditron_model = model.meditron_model
        self.params = model.params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.save_test_set = self.params.get("save_test_set", False)

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
        verbose = self.params.get("verbose", 1)
        logger.info("System message: %s", prompt_template_hf("")[0])
        logger.info("Starting training...")

        if self.tuning:
            logger.info(
                "Tuning model with prompt tuning. Model is saved in %s",
                self.model_save_dir,
            )
            optimizer = optim.AdamW(
                self.meditron_model.parameters(), lr=self.params.get("lr", 1e-4)
            )
            num_epochs = self.params.get("num_epochs", 1)

            self.meditron_model.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                logger.info(f"Epoch {epoch + 1} started...")
                for i, (X, y) in enumerate(
                    zip(
                        self.train_loader[0].iterrows(), self.train_loader[1].iterrows()
                    )
                ):
                    # Input prompt
                    X_input = prompt_template_hf(X[1].iloc[0])
                    inputs = self.model.tokenizer.apply_chat_template(
                        X_input, tokenize=False, add_generation_prompt=True
                    )

                    # Build target output label
                    probability = y[1].iloc[0]  # float
                    diagnosis = (
                        "not-" if probability < 0.5 else ""
                    ) + self.model.task_name
                    target_output = (
                        "{\n"
                        f'  "diagnosis": "{diagnosis}",\n'
                        f'  "probability": {round(probability, 3)},\n'
                        '  "explanation": "N/A"\n'
                        "}\n\n"
                    )

                    encoded = self.encode_prompt_target(
                        inputs,
                        target_output,
                        max_len=self.model.tokenizer.model_max_length,
                    )

                    optimizer.zero_grad()
                    outputs = self.meditron_model(
                        input_ids=encoded["input_ids"].to(self.device),
                        attention_mask=encoded["attention_mask"].to(self.device),
                        labels=encoded["labels"].to(self.device),
                    )

                    loss = outputs.loss
                    loss.backward()

                    optimizer.step()
                    epoch_loss += loss.item()

                    logger.info(
                        "Step %d/%d, Loss: %.4f",
                        i + 1,
                        len(self.train_loader[0]),
                        loss.item(),
                    )

                    if self.wandb:
                        wandb.log({"train_loss": loss.item()})

                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Avg Total Loss: {epoch_loss/len(self.train_loader[0]):.4f}"
                )
                if self.wandb:
                    wandb.log(
                        {f"avg_epoch_loss": epoch_loss / len(self.train_loader[0])}
                    )

                val_loss = self.evaluate_single(self.val_loader)
                logger.info("Validation loss: %s", val_loss)

                self.meditron_model.save_pretrained(self.model_save_dir)
                self.model.tokenizer.save_pretrained(self.model_save_dir)
                logger.info("Model saved to %s", self.model_save_dir)

        self.evaluate_single(self.test_loader, save_report=True)

    def evaluate_single(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        if self.save_test_set:
            # Save test set to CSV
            test_loader[0].to_csv(
                os.path.join(self.model.save_dir, "test_set.csv"), index=False
            )
            test_loader[1].to_csv(
                os.path.join(self.model.save_dir, "test_labels.csv"), index=False
            )
            logger.info("Test set saved to %s", self.model.save_dir)
        logger.info("Starting test evaluation...")

        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        self.meditron_model.eval()

        total_tokens = 0
        total_token_time = 0.0
        total_infer_time = 0.0

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            y_true = y[1].iloc[0]

            result_dict = self.model.infer_llm(X_input)

            generated_text = result_dict["generated_text"]
            token_time = result_dict["token_time"]
            infer_time = result_dict["infer_time"]
            num_tokens = result_dict["num_tokens"]

            total_token_time += token_time
            total_infer_time += infer_time
            total_tokens += num_tokens

            predicted_probability = float(generated_text.get("probability", 0.5))

            logger.info(
                "Predicted probability: %s | True label: %s",
                predicted_probability,
                y_true,
            )

            predicted_label = torch.tensor(
                predicted_probability, dtype=torch.float32
            ).unsqueeze(0)
            target = torch.tensor(float(y_true), dtype=torch.float32).unsqueeze(0)

            loss = self.criterion(predicted_label, target)
            val_loss.append(loss.item())

            if self.wandb:
                wandb.log(
                    {
                        "val_loss": loss.item(),
                        "token_time": token_time,
                        "infer_time": infer_time,
                        "num_tokens": num_tokens,
                    }
                )

            metrics_tracker.add_results(predicted_probability, y_true)

        # After evaluation loop
        logger.info("Total tokens: %s", total_tokens)
        logger.info(
            "Average tokenization time: %.4fs", total_token_time / len(test_loader[0])
        )
        logger.info(
            "Average inference time: %.4fs", total_infer_time / len(test_loader[0])
        )

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return float(np.mean(val_loss))

    def evaluate_batched(self, test_loader: Any, save_report: bool = False) -> float:
        """Evaluates the model on a given test set in batches.

        Args:
            test_loader: Tuple of (X, y) test data in DataFrame form.
            save_report: Whether to save the evaluation report.

        Returns:
            The average validation loss across the test dataset.
        """
        NotImplementedError(
            "Batch evaluation is not implemented for MeditronModel. Use evaluate_single instead."
        )

    def encode_prompt_target(
        self,
        prompt: str,
        target: str,
        max_len: int = 512,
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
        prompt_len = len(prompt_ids)
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
