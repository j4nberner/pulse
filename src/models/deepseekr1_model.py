import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import Runnable
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulsetemplate_model import PulseTemplateModel

logger = logging.getLogger("PULSE_logger")


class DeepSeekR1Model(PulseTemplateModel):
    """DeepSeek-R1 model wrapper using LangChain for prompt templating and inference."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        NotImplementedError("DeepSeek-R1 model is not implemented yet.")

        self.model_name = params.get("model_name", "DeepSeekR1")
        self.trainer_name = params["trainer_name"]
        super().__init__(self.model_name, self.trainer_name, params=params)

        self.save_dir: str = kwargs.get("output_dir", f"{os.getcwd()}/output")
        self.wandb: bool = kwargs.get("wandb", False)
        self.params: Dict[str, Any] = params
        self.model_id: str = self.params.get("model_id", "deepseek-ai/DeepSeek-R1")
        self.max_length: int = self.params.get("max_length", 512)

        self.tokenizer: Optional[Any] = None
        self.deepseek_model: Optional[Any] = None
        self.hf_pipeline: Optional[Any] = None

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
            self.deepseek_model = AutoModelForCausalLM.from_pretrained(
                self.model_id, torch_dtype=torch.float16, device_map="auto"
            )

            if self.params.get("prefix_tuning", False):
                logger.info("Applying Prefix Tuning")
                prefix_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    num_virtual_tokens=self.params.get("num_virtual_tokens", 30),
                    encoder_hidden_size=self.params.get("encoder_hidden_size", 4096),
                )
                self.deepseek_model = get_peft_model(self.deepseek_model, prefix_config)

            logger.info("Successfully loaded DeepSeek-R1 model: %s", self.model_id)
        except Exception as e:
            logger.error("Failed to load DeepSeek-R1 model: %s", e)
            raise

        logger.info(
            "Initializing Hugging Face pipeline with parameters: %s", self.params
        )

        self.hf_pipeline = pipeline(
            "text-generation",
            model=self.deepseek_model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_new_tokens=5,
            do_sample=False,
            temperature=0.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )

    def infer_llm(self, input_text: str) -> Dict[str, Any]:
        """Runs the HF pipeline with the given input and logs timing/token info."""
        if not isinstance(input_text, str):
            input_text = str(input_text)

        token_start = time.perf_counter()
        tokens = self.tokenizer(input_text, return_tensors="pt")
        token_time = time.perf_counter() - token_start
        num_tokens = len(tokens["input_ids"][0])

        infer_start = time.perf_counter()
        result = self.hf_pipeline(input_text)
        infer_time = time.perf_counter() - infer_start

        generated_text = result[0]["generated_text"].replace(input_text, "").strip()
        logger.info(f"Generated text: {generated_text}")

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
        """Sets the associated trainer instance."""
        self.trainer = DeepSeekR1Trainer(
            self, train_dataloader, val_dataloader, test_dataloader
        )

    def parse_output(self, output: str) -> float:
        """Parses the output string to extract the predicted probability."""
        try:
            probability = float(output.split(":")[-1].strip())
            return probability
        except (ValueError, IndexError) as e:
            logger.warning("Failed to parse output. Defaulting to 0.5: %s", e)
            logger.info("Output: %s", output)
            return 0.5  # Default to 0.5 if parsing fails


class DeepSeekR1Trainer:
    def __init__(
        self, model: DeepSeekR1Model, train_loader, val_loader, test_loader
    ) -> None:
        model._load_model()

        self.model = model
        self.model_instance = model.model
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
        self.prefix_tuning = self.params.get("prefix_tuning", False)

        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        verbose = self.params.get("verbose", 1)

        logger.info("Starting training...")

        if self.prefix_tuning:
            logger.info(
                "Tuning model with prefix tuning. Model is saved in %s",
                self.model_save_dir,
            )
            optimizer = optim.AdamW(
                self.model_instance.parameters(), lr=self.params.get("lr", 1e-4)
            )
            num_epochs = self.params.get("epochs", 3)

            self.model_instance.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                for X, y in zip(
                    self.train_loader[0].iterrows(), self.train_loader[1].iterrows()
                ):
                    X_input = X[1].iloc[0]
                    y_true = (
                        torch.tensor(float(y[1].iloc[0]), dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )

                    inputs = self.model.tokenizer(
                        X_input, return_tensors="pt", truncation=True, padding=True
                    ).to(self.device)
                    labels = y_true.unsqueeze(0).expand(
                        inputs["input_ids"].shape[0], -1
                    )

                    outputs = self.model_instance(**inputs, labels=labels)
                    loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    if self.wandb:
                        wandb.log({"train_loss": loss.item()})

                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

                val_loss = self.evaluate_single(self.val_loader)
                logger.info("Validation loss: %s", val_loss)

        self.evaluate_single(self.test_loader, save_report=True)

    def evaluate_single(self, test_loader: Any, save_report: bool = False) -> float:
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        self.model_instance.eval()

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

            predicted_probability = self.model.parse_output(generated_text)

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

        logger.info(f"Total tokens: {total_tokens}")
        logger.info(
            f"Average tokenization time: {total_token_time / len(test_loader[0]):.4f}s"
        )
        logger.info(
            f"Average inference time: {total_infer_time / len(test_loader[0]):.4f}s"
        )

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return float(np.mean(val_loss))
