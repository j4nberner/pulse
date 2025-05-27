import logging
import os
import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from peft import PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch.nn import functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Llama4ForConditionalGeneration)

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import extract_dict, prompt_template_hf

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

logger = logging.getLogger("PULSE_logger")


class Llama4Model(PulseTemplateModel):
    """Llama 4 model wrapper using LangChain for prompt templating and inference."""

    def __init__(self, params: Dict[str, Any], **kwargs) -> None:
        """Initializes the Llama4Model with parameters and paths.

        Args:
            params: Configuration dictionary with model parameters.
            **kwargs: Additional optional parameters such as `output_dir` and `wandb`.
        """
        raise NotImplementedError(
            "Llama4Model is not implemented yet."
        )
        self.model_name = kwargs.get("model_name", "Llama4Model")
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

        self.model_id: str = self.params.get(
            "model_id", 
            "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        )
        self.max_length: int = self.params.get("max_length", 5120)

        self.tokenizer: Optional[Any] = None
        self.llama_model: Optional[Any] = None

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",  # Options: "nf4", "fp4"
            bnb_4bit_use_double_quant=True  # Optional: enables nested quantization
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, padding_side="left"
            )
            self.llama_model = Llama4ForConditionalGeneration.from_pretrained(
                self.model_id,
                attn_implementation="sdpa", # good for long context windows
                device_map="auto",
                torch_dtype=torch.bfloat16,
                load_in_4bit=True,
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
                self.llama_model = get_peft_model(self.llama_model, tuning_config)
                logger.debug(self.llama_model.print_trainable_parameters())

            logger.info("Successfully loaded Llama4 model: %s", self.model_id)
        except Exception as e:
            logger.error("Failed to load Llama4 model: %s", e)
            raise

        logger.info(
            "Initializing Hugging Face pipeline with parameters: %s", self.params
        )

    def infer_llm(self, input_text: str) -> Dict[str, Any]:
        """Runs the HF model on the input and extracts diagnosis, explanation, and probability."""
        logger.info("---------------------------------------------")

        if not isinstance(input_text, str):
            input_text = str(input_text)

        # Format input using prompt template
        input_text = prompt_template_hf(input_text)

        # Apply chat template
        chat_prompt = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )

        token_start = time.perf_counter()
        tokenized_inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
        )
        token_time = time.perf_counter() - token_start
        num_prompt_tokens = tokenized_inputs["input_ids"].size(1)

        yes_token_id = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        no_token_id = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
        # self.llama_model.to(self.device)
        input_ids = tokenized_inputs["input_ids"].to(self.device)
        attention_mask = tokenized_inputs["attention_mask"].to(self.device)

        # Generate output with scores
        infer_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.llama_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.params.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        infer_time = time.perf_counter() - infer_start

        # Get generated token ids (excluding prompt)
        gen_ids = outputs.sequences[0][num_prompt_tokens:]

        # Decode the full generated string
        decoded_output = self.tokenizer.decode(
            gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug("Decoded output:\n %s", decoded_output)

        # Calculate softmax over first generated token logits
        first_token_logits = outputs.scores[0][0]  # shape: (vocab_size,)
        # top_gen_ids = gen_ids[:5]  # First few generated tokens
        # print("Generated token IDs:", top_gen_ids)
        # print("Decoded tokens:", [self.tokenizer.decode([tid]) for tid in top_gen_ids])
        # logger.debug(
        #     "First token logits: %s", first_token_logits
        # )
        # Apply softmax to get probabilities
        # Get top 10 probabilities and their corresponding token indices
        probs = torch.topk(F.softmax(first_token_logits, dim=-1), 10) #(values, indices)
        topk_values, topk_indices = probs

        # Convert indices to list for easier matching
        topk_indices_list = topk_indices.tolist()
        topk_values_list = topk_values.tolist()

        # Initialize with fallback values
        yes_prob = 0.0
        no_prob = 0.0

        # Find index of yes_token_id and extract its probability
        if yes_token_id in topk_indices_list:
            yes_index = topk_indices_list.index(yes_token_id)
            yes_prob = topk_values_list[yes_index]

        if no_token_id in topk_indices_list:
            no_index = topk_indices_list.index(no_token_id)
            no_prob = topk_values_list[no_index]


        # Fallback if yes and no tokens were not picked up. They are inlcuded in the vocab but
        # have a value a -inf as logits
        if yes_prob == 0.0 and no_prob == 0.0:
            logger.warning(
                "Yes or No token probabilities are zero. Defaulting to 0.5."
            )
            yes_prob = 0.5
            no_prob = 0.5

        if yes_prob > no_prob:
            probability = yes_prob
        else:
            probability = 1 - no_prob
        logger.debug(
            "Yes token ID: %s | No token ID: %s", yes_token_id, no_token_id
        )
        logger.debug(
            "Top 10 token probs: %s", probs
        )
        logger.debug(
            "Yes token probability: %.4f | No token probability: %.4f",
            yes_prob,
            no_prob,
        )

        # Extract dict from the decoded output (e.g., via regex or JSON parsing)
        try:
            parsed = extract_dict(decoded_output)
            # logger.debug("Parsed output: %s", parsed)
        except Exception as e:
            logger.warning(f"Failed to parse output: {decoded_output}")
            parsed = {"diagnosis": None, "explanation": decoded_output}

        # Add diagnosis probability based on first token
        parsed["probability"] = round(probability, 4)

        logger.info(
            f"Tokenization time: {token_time:.4f}s | Inference time: {infer_time:.4f}s | Tokens: {num_prompt_tokens}"
        )

        return {
            "generated_text": parsed,
            "token_time": token_time,
            "infer_time": infer_time,
            "num_tokens": num_prompt_tokens,
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
        self.trainer = Llama4Trainer(
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


class Llama4Trainer:
    """Trainer class for Llama4Model."""

    def __init__(
        self, model: Llama4Model, train_loader, val_loader, test_loader
    ) -> None:
        """
        Initialize the Llama4 trainer. Finetruning is not implemented yet.
        This is a wrapper for inference only.

        Args:
            model (Llama4Model): The Llama4 model to be trained.
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
                self.llama_model.parameters(), lr=self.params.get("lr", 1e-4)
            )
            num_epochs = self.params.get("num_epochs", 1)

            self.llama_model.train()
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
                        f'  "probability": {round(probability, 4)},\n'
                        '  "explanation": "N/A"\n'
                        "}\n\n"
                    )

                    encoded = self.encode_prompt_target(
                        inputs,
                        target_output,
                        max_len=self.model.tokenizer.model_max_length,
                    )

                    optimizer.zero_grad()
                    #TODO: Should be optimized for diagnosis or probability -> need to adapt
                    outputs = self.llama_model(
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

                self.llama_model.save_pretrained(self.model_save_dir)
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
            if verbose > 1:
                logger.info("Generated label: %s", generated_text["diagnosis"])
                logger.info("Generated explanation: %s \n", generated_text["explanation"])
            if verbose > 2:
                logger.info("Input prompt: %s \n", X_input)

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
            "Batch evaluation is not implemented for Llama4Model. Use evaluate_single instead."
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
