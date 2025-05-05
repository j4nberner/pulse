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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import extract_dict, prompt_template_hf

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

        required_params = [
            "max_new_tokens",
            "do_sample",
            "temperature",
        ]
        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        self.params: Dict[str, Any] = params

        # self.quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        self.model_id: str = self.params.get(
            "model_id", "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.max_length: int = self.params.get("max_length", 512)

        self.tokenizer: Optional[Any] = None
        self.llama_model: Optional[Any] = None
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=False, padding_side="left"
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                # low_cpu_mem_usage=True,
                # device_map=None # need to load later for memory reasons
            )

            if self.params.get("prefix_tuning", False):
                logger.info("Applying Prefix Tuning")
                prefix_config = PrefixTuningConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    num_virtual_tokens=self.params.get("num_virtual_tokens", 30),
                    encoder_hidden_size=self.params.get("encoder_hidden_size", 4096),
                )
                self.llama_model = get_peft_model(self.llama_model, prefix_config)

            logger.info("Successfully loaded Llama3 model: %s", self.model_id)
        except Exception as e:
            logger.error("Failed to load Llama3 model: %s", e)
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
            input_text
        )  # Apply prompt template to structure the input and guide output.

        token_start = time.perf_counter()
        chat_prompt = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )

        tokenized_inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
        )
        token_time = time.perf_counter() - token_start
        num_tokens = tokenized_inputs["input_ids"].numel()

        infer_start = time.perf_counter()
        self.llama_model.to(self.device)

        with torch.no_grad():
            outputs = self.llama_model.generate(
                input_ids=tokenized_inputs["input_ids"].to(self.device),
                attention_mask=tokenized_inputs["attention_mask"].to(self.device),
                max_new_tokens=self.params.max_new_tokens,
                do_sample=self.params.do_sample,
                temperature=self.params.temperature,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 3) Slice off the prompt part:
        #    outputs is shape (1, prompt_len + gen_len)
        gen_ids = outputs[0, num_tokens:]

        # 4) Decode just the generated tokens:
        generated_text = self.tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        infer_time = time.perf_counter() - infer_start
        logger.debug(
            "Decoded full outputs: %s",
            generated_text,
        )

        generated_text = extract_dict(
            generated_text
        )  # Extract dict from the generated text.


        generated_text["probability"] = round(abs(generated_text["probability"] - 1.0) if "not-" in generated_text["diagnosis"] else abs(generated_text["probability"]), 3)

        logger.info(
            f"Tokenization time: {token_time:.4f}s | Inference time: {infer_time:.4f}s | Tokens: {num_tokens}"
        )

        # torch.cuda.empty_cache()

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
        self.trainer = Llama3Trainer(
            self, train_dataloader, val_dataloader, test_dataloader
        )

    def parse_output(self, output: str) -> float:
        """Parses the output string to extract the predicted probability.

        Args:
            output: The generated text from the model.

        Returns:
            A float representing the predicted probability.
        """
        # TODO: Implement a more robust parsing method
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


class Llama3Trainer:
    """Trainer class for Llama3Model."""

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
        self.prefix_tuning = self.params.get("prefix_tuning", False)

        logger.info("Using criterion: %s", self.criterion.__class__.__name__)

        # Create model save directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def train(self):
        """Training loop."""
        verbose = self.params.get("verbose", 1)

        logger.info("Starting training...")

        if self.prefix_tuning:
            logger.info(
                "Tuning model with prefix tuning. Model is saved in %s",
                self.model_save_dir,
            )
            optimizer = optim.AdamW(
                self.llama_model.parameters(), lr=self.params.get("lr", 1e-4)
            )
            num_epochs = self.params.get("epochs", 3)

            self.llama_model.train()
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

                    outputs = self.llama_model(**inputs, labels=labels)
                    loss = outputs.loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    if self.wandb:
                        wandb.log({"train_loss": loss.item()})

                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

                val_loss = self.evaluate_single(
                    self.val_loader
                )  # Evaluate on validation set
                logger.info("Validation loss: %s", val_loss)

        self.evaluate_single(
            self.test_loader, save_report=True
        )  # Evaluate on test set and save metrics

    def evaluate_single(self, test_loader: Any, save_report: bool = False) -> float:
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
            "Batch evaluation is not implemented for Llama3Model. Use evaluate_single instead."
        )
        metrics_tracker = MetricsTracker(
            self.model.model_name,
            self.model.task_name,
            self.model.dataset_name,
            self.model.save_dir,
        )
        verbose: int = self.params.get("verbose", 1)
        val_loss: list[float] = []

        self.llama_model.eval()

        df_X, df_y = test_loader  # X: prompts, y: true labels
        prompts = df_X.iloc[:, 0].tolist()
        true_labels = df_y.iloc[:, 0].tolist()

        # Batch inference with Hugging Face pipeline
        results = self.model.hf_pipeline(
            prompts,
        )

        logger.debug(f"Results from hf_pipeline: {results}")

        for i, result_dict in enumerate(results):
            generated_text = result_dict["generated_text"]
            y_true = true_labels[i]

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
                wandb.log({"val_loss": loss.item()})

            metrics_tracker.add_results(predicted_probability, y_true)

        metrics_tracker.summary = metrics_tracker.compute_overall_metrics()
        if save_report:
            metrics_tracker.save_report()

        logger.info("Test evaluation completed for %s", self.model.model_name)
        logger.info("Test metrics: %s", metrics_tracker.summary)

        return float(np.mean(val_loss))
