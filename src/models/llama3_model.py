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
                          BitsAndBytesConfig)

import wandb
from src.eval.metrics import MetricsTracker
from src.models.pulsetemplate_model import PulseTemplateModel
from src.util.model_util import extract_dict, prompt_template_hf

warnings.filterwarnings(
    "ignore",
    message="Position ids are not supported for parameter efficient tuning. Ignoring position ids.",
)

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
        ]
        # Check if all required parameters exist in config
        missing_params = [param for param in required_params if param not in params]
        if missing_params:
            raise KeyError(f"Required parameters missing from config: {missing_params}")

        self.params: Dict[str, Any] = params
        self.params["save_test_set"] = kwargs.get("save_test_set", False)

        self.model_id: str = self.params.get(
            "model_id", "meta-llama/Llama-3.1-8B-Instruct"
        )
        self.max_length: int = self.params.get("max_length", 5120)

        self.tokenizer: Optional[Any] = None
        self.llama_model: Optional[Any] = None

        self.quantization_config = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_threshold=6.0, llm_int8_has_fp16_weight=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Number of GPUs: %d", torch.cuda.device_count())

    def _load_model(self) -> None:
        """Loads the tokenizer and model weights and initializes HF pipeline."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, use_fast=False, padding_side="left"
            )
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
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

            logger.info("Successfully loaded Llama3 model: %s", self.model_id)
        except Exception as e:
            logger.error("Failed to load Llama3 model: %s", e)
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

        # Tokenize with chat template
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

        # Convert "yes" and "no" tokens to ids
        yes_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("yes"))[0]
        no_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("no"))[0]

        logger.debug("Yes token ID: %s", yes_token_id)
        logger.debug("No token ID: %s", no_token_id)

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

        # Get generated token ids (excluding prompt) and convert to a Python list
        generated_token_ids_list = outputs.sequences[0][num_prompt_tokens:].tolist()
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_token_ids_list)

        decoded_output = self.tokenizer.decode(
            generated_token_ids_list, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        logger.debug("Decoded output:\n %s", decoded_output)

        parsed = extract_dict(decoded_output)
        classification = parsed.get("classification", "unknown")
        probability = 0.5  # Default probability

        if classification == "yes":
            try:
                yes_index = generated_token_ids_list.index(yes_token_id)
                if yes_index < len(outputs.scores):
                    logits = outputs.scores[yes_index][0]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    probability = round(probs[yes_token_id].item(), 4)
                    logger.debug(f"'yes' token found at index {yes_index} with probability {probability:.4f}")
            except ValueError:
                logger.warning("'yes' token not found in generated sequence.")
        elif classification == "no":
            try:
                no_index = generated_token_ids_list.index(no_token_id)
                if no_index < len(outputs.scores):
                    logits = outputs.scores[no_index][0]
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    probability = round(probs[no_token_id].item(), 4)
                    logger.debug(f"'no' token found at index {no_index} with probability {probability:.4f}")
            except ValueError:
                logger.warning("'no' token not found in generated sequence.")
        else:
            logger.warning(f"Classification is '{classification}', cannot determine probability.")

        parsed["probability"] = probability

        logger.info(
            f"Tokenization time: {token_time:.4f}s | Inference time: {infer_time:.4f}s | Tokens: {num_prompt_tokens}"
        )

        return {
            "generated_text": parsed,
            "token_time": token_time,
            "infer_time": infer_time,
            "num_tokens": num_prompt_tokens,
        }



    # def infer_llm(self, input_text: str) -> Dict[str, Any]:
    #     """Runs the HF model on the input and extracts diagnosis, explanation, and probability."""
    #     logger.info("---------------------------------------------")

    #     if not isinstance(input_text, str):
    #         input_text = str(input_text)

    #     # Format input using prompt template
    #     input_text = prompt_template_hf(input_text)

    #     # Tokenize with chat template
    #     chat_prompt = self.tokenizer.apply_chat_template(
    #         input_text, tokenize=False, add_generation_prompt=True
    #     )

    #     token_start = time.perf_counter()
    #     tokenized_inputs = self.tokenizer(
    #         chat_prompt,
    #         return_tensors="pt",
    #     )
    #     token_time = time.perf_counter() - token_start
        
    #     num_prompt_tokens = tokenized_inputs["input_ids"].size(1)

    #     # yes_token_id = self.tokenizer.convert_tokens_to_ids("yes")
    #     # no_token_id = self.tokenizer.convert_tokens_to_ids("no")
    #     yes_token = self.tokenizer.tokenize("yes")[0]
    #     no_token = self.tokenizer.tokenize("no")[0]
    #     logger.debug("Yes token ID: %s", yes_token)
    #     logger.debug("No token ID: %s", no_token)
        
    #     input_ids = tokenized_inputs["input_ids"].to(self.device)
    #     attention_mask = tokenized_inputs["attention_mask"].to(self.device)

    #     # Generate output with scores
    #     infer_start = time.perf_counter()
    #     with torch.no_grad():
    #         outputs = self.llama_model.generate(
    #             input_ids=input_ids,
    #             attention_mask=attention_mask,
    #             max_new_tokens=self.params.max_new_tokens,
    #             return_dict_in_generate=True,
    #             output_scores=True,
    #             output_hidden_states=False,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #             eos_token_id=self.tokenizer.eos_token_id,
    #         )

    #     infer_time = time.perf_counter() - infer_start

    #     # Get generated token ids (excluding prompt)
    #     # gen_ids = outputs.sequences[0][num_prompt_tokens:]
    #     generated_token_ids = outputs.sequences[0][num_prompt_tokens:]  # only new tokens
    #     generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_token_ids)

    #     yes_index = None
    #     no_index = None

    #     for i, tok in enumerate(generated_tokens):
    #         if tok == yes_token and yes_index is None:
    #             yes_index = i
    #         elif tok == no_token and no_index is None:
    #             no_index = i

    #     if yes_index is not None:
    #         logits = outputs.scores[yes_index][0]  # logits for the token at that index
    #         probs = F.softmax(logits, dim=-1)
    #         yes_prob = probs[self.tokenizer.convert_tokens_to_ids(yes_token)].item()
    #     else:
    #         yes_prob = None

    #     if no_index is not None:
    #         logits = outputs.scores[no_index][0]
    #         probs = F.softmax(logits, dim=-1)
    #         no_prob = probs[self.tokenizer.convert_tokens_to_ids(no_token)].item()
    #     else:
    #         no_prob = None

    #     logger.debug("Yes probability: %s", yes_prob)
    #     logger.debug("No probability: %s", no_prob)

    #     # answer_token_index = None
    #     # for i, token_id in enumerate(gen_ids):
    #     #     if token_id == yes_token_id or token_id == no_token_id:
    #     #         # Find the index of the first token that matches yes_token_id or no_token_id
    #     #         # This is the token we will use to calculate the probability
    #     #         answer_token_index = i
    #     #         break
    #     # if answer_token_index is None:
    #     #     logger.warning(
    #     #         "No yes_token_id or no_token_id found in generated tokens. Defaulting to 0.5."
    #     #     )
    #     #     answer_token_index = 0

    #     # Decode the full generated string
    #     decoded_output = self.tokenizer.decode(
    #         generated_token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #     )
    #     # logger.debug("Decoded output:\n %s", decoded_output)

    #     # # Calculate sigmoid over first generated token logits
    #     # # first_token_logits = outputs.scores[answer_token_index][0]  # shape: (vocab_size,)
    #     # first_token_logits = outputs.scores[0][0]  # First generated token
    #     # print(first_token_logits[yes_token_id], first_token_logits[no_token_id])

    #     # yes_logit = first_token_logits[yes_token_id]
    #     # no_logit = first_token_logits[no_token_id]

    #     # logits = torch.tensor([yes_logit, no_logit])
    #     # probs = F.softmax(logits, dim=-1)
    #     # yes_prob = probs[0].item()
    #     # no_prob = probs[1].item()

        
    #     # # Apply sigmoid to get probabilities
    #     # # Get top 10 probabilities and their corresponding token indices
    #     # token_probs = F.softmax(first_token_logits, dim=-1)
    #     # topk_values, topk_indices = torch.topk(token_probs, 10)
        
    #     # logger.debug("Top k probs: %s", token_probs)

    #     # # Convert indices to list for easier matching
    #     # topk_indices_list = topk_indices.tolist()
    #     # topk_values_list = topk_values.tolist()

    #     # # Initialize with fallback values
    #     # yes_prob = 0.0
    #     # no_prob = 0.0

    #     # # # Find index of yes_token_id and extract its probability
    #     # # if yes_token_id in topk_indices_list:
    #     # #     yes_index = topk_indices_list.index(yes_token_id)
    #     # #     yes_prob = topk_values_list[yes_index]

    #     # # if no_token_id in topk_indices_list:
    #     # #     no_index = topk_indices_list.index(no_token_id)
    #     # #     no_prob = topk_values_list[no_index]

    #     # yes_prob = token_probs[yes_token_id].item()
    #     # no_prob = token_probs[no_token_id].item()


    #     # Fallback if yes and no tokens were not picked up. They are inlcuded in the vocab but
    #     # have a value a -inf as logits
    #     # if yes_prob == 0.0 and no_prob == 0.0:
    #     #     logger.warning(
    #     #         "Yes or No token probabilities are zero. Defaulting to 0.5."
    #     #     )
    #     #     yes_prob = 0.5
    #     #     no_prob = 0.5

    #     if yes_prob > no_prob:
    #         probability = yes_prob
    #     else:
    #         probability = 1 - no_prob
    #     logger.debug(
    #         "Yes token ID: %s | No token ID: %s", yes_token, no_token
    #     )
    #     # logger.debug(
    #     #     "Top 10 token probs: %s", (topk_values, topk_indices)
    #     # )
    #     logger.debug(
    #         "Yes token probability: %.4f | No token probability: %.4f",
    #         yes_prob,
    #         no_prob,
    #     )

    #     # Extract dict from the decoded output (e.g., via regex or JSON parsing)
    #     try:
    #         parsed = extract_dict(decoded_output)
    #         # logger.debug("Parsed output: %s", parsed)
    #     except Exception as e:
    #         logger.warning(f"Failed to parse output: {decoded_output}")
    #         parsed = {"diagnosing": None, "classification": "na", "explanation": decoded_output}

    #     # Add diagnosis probability based on first token
    #     parsed["probability"] = round(probability, 4)

    #     logger.info(
    #         f"Tokenization time: {token_time:.4f}s | Inference time: {infer_time:.4f}s | Tokens: {num_prompt_tokens}"
    #     )

    #     return {
    #         "generated_text": parsed,
    #         "token_time": token_time,
    #         "infer_time": infer_time,
    #         "num_tokens": num_prompt_tokens,
    #     }
    
    def calculate_tokens(self, input_text: str) -> Dict[str, Any]:
        """
        Runs the full inference without loading the model and calculates the number of input and output tokens.
        Assuming num_output_tokens = max_new_tokens.

        Args:
            input_text: The input text to be tokenized.
        Returns:
            A dictionary containing the number of input and output tokens.
        """

        # Format input using prompt template
        input_text = prompt_template_hf(input_text)

        # Tokenize with chat template
        chat_prompt = self.tokenizer.apply_chat_template(
            input_text, tokenize=False, add_generation_prompt=True
        )
        tokenized_inputs = self.tokenizer(
            chat_prompt,
            return_tensors="pt",
        )
        num_input_tokens = tokenized_inputs["input_ids"].size(1)

        return {
            "num_input_tokens": num_input_tokens,
            "num_output_tokens": self.params.max_new_tokens,
        }


    def set_trainer(
        self,
        trainer_name: str,
        train_dataloader: Any,
        val_dataloader: Any,
        test_dataloader: Any,
        **kwargs: Any,
    ) -> None:
        """Sets the associated trainer instance.

        Args:
            trainer_name: Name of the trainer class.
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            test_dataloader: DataLoader for test data.
        """
        self.trainer = Llama3Trainer(
            self, train_dataloader, val_dataloader, test_dataloader, **kwargs
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
        self, model: Llama3Model, train_loader, val_loader, test_loader, **kwargs
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
        if kwargs.get("disable_model_load", False):
            logger.info("Skipping model loading for debugging purposes.")
        else:
            model._load_model()  #

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
                logger.info("Diagnosis for: %s", generated_text["diagnosing"])
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
    

    def estimate_nr_tokens(self) -> int:
        """Estimates the number of tokens for a task-dataset combination.

        Returns:
            The estimated number of tokens.
        """
        logger.info("Estimating number of tokens for the dataset...")
        # Load the tokenizer
        self.model.tokenizer = AutoTokenizer.from_pretrained(
                self.model.model_id, use_fast=False, padding_side="left"
        )

        test_loader = self.test_loader
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        num_input_tokens = 0
        num_output_tokens = 0

        for X, y in zip(test_loader[0].iterrows(), test_loader[1].iterrows()):
            X_input = X[1].iloc[0]
            token_dict = self.model.calculate_tokens(X_input)
            num_input_tokens = token_dict["num_input_tokens"]
            num_output_tokens = token_dict["num_output_tokens"]
            total_input_tokens += num_input_tokens
            total_output_tokens += num_output_tokens
            total_tokens += num_input_tokens + num_output_tokens
            logger.debug(
                "Input tokens: %s | Output tokens: %s",
                num_input_tokens,
                num_output_tokens,
            )

        logger.info(f"Total tokens for the task {self.model.task_name} dataset {self.model.dataset_name}: {total_tokens}")
        logger.info("Total input tokens: %s", total_input_tokens)
        logger.info("Total output tokens: %s", total_output_tokens)
        logger.info("Average input tokens: %s", total_input_tokens / len(test_loader[0]))
        logger.info("Average output tokens: %s", total_output_tokens / len(test_loader[0]))
        return total_tokens



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
