import ast
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logger = logging.getLogger("PULSE_logger")

# ------------------------------------
# Util functions for convDL models
# ------------------------------------


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation loss stops improving.
    Args:
        patience (int): Number of epochs to wait after the last improvement in validation loss.
        verbose (bool): If True, logs messages about early stopping progress.
        delta (float): Minimum change in validation loss to qualify as an improvement. Default is 0.0001.
            This value is chosen to prevent stopping due to minor fluctuations in validation loss.
    """

    def __init__(self, patience=10, verbose=True, delta=0.0001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.info(
                    "EarlyStopping counter: %d out of %d",
                    self.counter,
                    self.patience,
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                logger.debug(
                    "Score improved (%.6f --> %.6f). Saving model state...",
                    self.best_score,
                    score,
                )
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        """Load the best model state into the provided model."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            if self.verbose:
                logger.info("Loaded best model state from early stopping")


def save_torch_model(model_name: str, model: Any, save_dir: str) -> None:
    """Save the trained torch model to disk.

    Args:
        model_name: Name of the model to be saved
        model: The PyTorch model object to save
        save_dir: Parent directory where the model should be saved
    """

    try:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.pth")

        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), model_path)
        else:
            torch.save(model, model_path)

        logger.info("Model '%s' saved to %s", model_name, model_path)
    except Exception as e:
        logger.error("Failed to save model '%s': %s", model_name, str(e))


def save_sklearn_model(model_name: str, model: Any, save_dir: str) -> None:
    """
    Save the trained sklearn model to disk.

    Args:
        model_name: Name of the model to be saved
        model: The sklearn model object to save
        save_dir: Directory where the model should be saved
    """

    try:
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_name}.joblib")

        joblib.dump(model, model_path)

        logger.info("Model '%s' saved to %s", model_name, model_path)
    except Exception as e:
        logger.error("Failed to save model '%s': %s", model_name, str(e))


def prepare_data_for_model_convml(
    train_loader, val_loader, test_loader
) -> Dict[str, Any]:
    """
    Prepare data for conventional machine learning models by converting PyTorch tensors
    from dataloaders to numpy arrays while preserving feature names.

    Args:
        train_dataloader: DataLoader containing the training data or list of data in debug mode
        test_dataloader: DataLoader containing the test data or list of data in debug mode

    Returns:
        dict: Dictionary containing:
            - X_train: numpy array of training features
            - y_train: numpy array of training labels
            - X_val: numpy array of validation features (if available)
            - y_val: numpy array of validation labels (if available)
            - X_test: numpy array of test features
            - y_test: numpy array of test labels
            - feature_names: list of feature names (if available)
    """

    # Extract data from dataloaders
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    feature_names = []

    if isinstance(train_loader[0], pd.DataFrame):
        # If DataLoader is a DataFrame, extract features and labels directly
        X_train = np.array(train_loader[0].values)
        y_train = np.array(train_loader[1].values).squeeze()
        X_val = np.array(val_loader[0].values)
        y_val = np.array(val_loader[1].values).squeeze()
        X_test = np.array(test_loader[0].values)
        y_test = np.array(test_loader[1].values).squeeze()
        feature_names = list(train_loader[0].columns)

    else:
        # Convert lists to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

    # Log shapes
    logger.debug(
        "Prepared data shapes - X_train: %s, y_train: %s", X_train.shape, y_train.shape
    )
    logger.debug(
        "Prepared data shapes - X_val: %s, y_val: %s", X_val.shape, y_val.shape
    )
    logger.debug(
        "Prepared data shapes - X_test: %s, y_test: %s", X_test.shape, y_test.shape
    )

    # Return all processed data
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


def prepare_data_for_model_convdl(
    data_loader,
    config: Dict,
    model_name: Optional[str] = None,
    task_name: Optional[str] = None,
) -> Any:
    """
    Prepare data for conventional deep learning models by returning a configured data converter.

    Args:
        data_loader: DataLoader containing the input data
        config: Configuration dictionary with preprocessing settings
        model_name: Name of the model to determine format requirements
        task_name: Name of the current task (e.g., "mortality", "aki")

    Returns:
        WindowedDataTo3D: Configured converter instance ready to transform batches
    """

    # Import the converter
    from src.preprocessing.preprocessing_advanced.windowing import WindowedDataTo3D

    # Create converter with model name and config
    converter = WindowedDataTo3D(
        model_name=model_name, config=config, task_name=task_name
    )

    try:
        # Get a batch to inspect shape
        features, _ = next(iter(data_loader))

        # Configure converter based on data shape
        if len(features.shape) == 3:
            # Data is already 3D
            converter.needs_conversion = False
            logger.info("Input data is already 3D, no conversion needed")
        else:
            # Check if windowing is enabled in config
            windowing_enabled = False
            if task_name == "mortality":
                windowing_enabled = True
            elif "preprocessing_advanced" in config:
                preprocessing_advanced = config["preprocessing_advanced"]
                if "windowing" in preprocessing_advanced:
                    windowing_config = preprocessing_advanced["windowing"]
                    if "enabled" in windowing_config:
                        windowing_enabled = bool(windowing_config["enabled"])

            # Configure the converter based on windowing status
            converter.configure_conversion(windowing_enabled, features.shape)

            if windowing_enabled:
                logger.info("Will use 3D windowed conversion for batches")
            else:
                logger.info("Will use simple reshaping for batches")

    except Exception as e:
        logger.error("Error preparing data converter: %s", e)

    return converter


@DeprecationWarning
def calculate_pos_weight(train_loader):
    """
    Calculate positive class weight for imbalanced binary classification data.

    Args:
        train_loader: DataLoader containing the training data

    Returns:
        float: Weight for positive class (ratio of negative to positive samples)
    """
    try:
        all_labels = []
        for _, labels in train_loader:
            all_labels.extend(labels.cpu().numpy().flatten())

        all_labels = np.array(all_labels)
        neg_count = np.sum(all_labels == 0)
        pos_count = np.sum(all_labels == 1)

        if pos_count == 0:
            logger.warning("No positive samples found, using pos_weight=1.0")
            return 1.0

        weight = neg_count / pos_count
        logger.info(
            "Class imbalance - Neg: %d, Pos: %d, Weight: %f",
            neg_count,
            pos_count,
            weight,
        )
        return weight

    except Exception as e:
        logger.error("Error calculating class weights: %s", e)
        return 1.0


def initialize_weights(module):
    """Apply Xavier initialization to model weights."""
    if isinstance(module, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LSTM, nn.GRU)):
        for name, param in module.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_normal_(param)
            elif "weight_hh" in name:
                nn.init.xavier_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


# ------------------------------------
# Util functions for LLMs
# ------------------------------------


def prompt_template_hf(
    input_text: str, custom_system_message=None, model=None
) -> List[Dict[str, str]]:
    """
    Create a chat-based prompt compatible with Hugging Face's apply_chat_template.

    Args:
        input_text: The text to analyze.
        model: Optional model name for specific formatting.
        custom_system_message: Optional custom system message to override the default.

    Returns:
        A list of chat messages (dicts) for the LLM.
    """
    system_message = custom_system_message or (
        "You are a helpful assistant and experienced medical professional analyzing ICU time-series data "
        "to determine the presence of a critical condition.\n\n"
        "Your response must strictly follow this format:\n"
        "Output a valid JSON object with three keys: 'diagnosis', 'probability' and 'explanation'.\n\n"
        "1. 'diagnosis' a string with either diganosis or not-diagnosis\n"
        "2. 'probability' a value between 0 and 1. where 0 means not-diagnosis and 1 means diagnosis.\n"
        "3. 'explanation' should be a string providing a brief explanation of your diagnosis.\n\n"
        "Here is a positive example:\n"
        "{\n"
        '  "diagnosis": "sepsis",\n'
        '  "probability": "0.76",\n'
        '  "explanation": "lactate is 4.2 mmol/L (above normal <2.0); blood pressure is low (MAP 62 mmHg), which are signs of sepsis."\n'
        "}\n\n"
        "Here is a negative example:\n"
        "{\n"
        '  "diagnosis": "not-sepsis",\n'
        '  "probability": "0.01",\n'
        '  "explanation": "lactate is 1.2 mmol/L (normal <2.0); blood pressure is normal (MAP 80 mmHg), which are not signs of sepsis."\n'
        "}\n\n"
        "Do not include any other text or explanations outside of the JSON object.\n"
        "Think about the probability of your prediction carefully before answering.\n"
    )

    # Apply model-specific formatting if needed
    if model == "Gemma3Model":
        formatted_prompt = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Text:\n{input_text}"}],
            },
        ]
    elif model == "GPTModel":
        formatted_prompt = [
            {
                "role": "developer",
                "content": system_message,
            },
            {
                "role": "user",
                "content": input_text,
            },
        ]
    elif model == "MeditronModel":
        formatted_prompt = [
            f"<|im_start|>system\n{system_message}<|im_end|>\n"
            f"<|im_start|>user\n{input_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ]
    elif model == "DeepseekR1Model":
        # avoid using a system prompt. including it all in the user prompt
        formatted_prompt = [
            {
                "role": "user",
                "content": f"{system_message} Text:\n{input_text} <think>\n",
            },
        ]
    else:
        formatted_prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
        ]

    return formatted_prompt


def extract_last_json_block(text: str) -> Optional[str]:
    """Extract the last balanced JSON object from the input string."""
    stack = []
    start_idx = None

    for i, c in enumerate(reversed(text)):
        idx = len(text) - 1 - i
        if c == "}":
            if not stack:
                start_idx = idx
            stack.append("}")
        elif c == "{":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    return text[idx : start_idx + 1]
    return None


def extract_dict(output_text: str) -> Optional[Dict[str, str]]:
    """Extract and parse the last JSON-like object from the model's output text and return it as a dictionary.

    Args:
        output_text: The raw string returned by the language model.

    Returns:
        A dictionary parsed from the JSON string, or a default JSON object if no JSON was found.
    """
    default_json = {
        "diagnosis": "unknown",
        "probability": 0.5,
        "explanation": "No explanation provided.",
    }

    json_text = extract_last_json_block(output_text)
    if not json_text:
        logger.warning("No JSON object found in assistant output. Returning default.")
        return default_json

    # Heuristic fix: unterminated string
    if json_text.count('"') % 2 != 0:
        json_text += '"'
        logger.debug("Fixed unterminated string by adding closing quote.")

    # Heuristic fix: missing final brace
    if not json_text.endswith("}"):
        json_text += "}"
        logger.debug("Fixed unclosed JSON object by adding closing brace.")

    # Escape newlines in quoted strings
    def escape_newlines_in_strings(s: str) -> str:
        def repl(m):
            return m.group(0).replace("\n", "\\n").replace("\r", "\\r")

        return re.sub(r'"(.*?)"', repl, s, flags=re.DOTALL)

    json_text_clean = escape_newlines_in_strings(json_text)

    try:
        return json.loads(json_text_clean)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}\nRaw: {json_text_clean}")
        return default_json
