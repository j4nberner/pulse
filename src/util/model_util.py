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
    data_loader: Any,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for conventional machine learning models by converting PyTorch tensors
    from dataloaders to numpy arrays while preserving feature names.

    Args:
        data_loader: DataLoader containing the data or list of data in debug mode

    Returns:
        Tuple containing:
            - X: Numpy array of features
            - y: Numpy array of labels
            - feature_names: List of feature names
    """

    # Extract data from dataloaders
    X, y = [], []
    feature_names = []

    if isinstance(data_loader[0], pd.DataFrame):
        # If DataLoader is a DataFrame, extract features and labels directly
        df = data_loader[0].apply(pd.to_numeric, errors="coerce")
        X = np.array(df.values)
        y = np.array(data_loader[1].values).squeeze()
        feature_names = list(df.columns)

    else:
        # Convert lists to numpy arrays
        X = np.array(X)
        y = np.array(y)

    # Log shapes
    logger.debug("Prepared data shapes - X: %s, y: %s", X.shape, y.shape)

    # Return all processed data
    return X, y, feature_names


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
    from src.preprocessing.preprocessing_advanced.windowing import \
        WindowedDataTo3D

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
        "2. 'probability' a value between 0 and 1. where 0 means not diagnosed and 1 means diagnosed.\n"
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

    if output_text.count("{") > output_text.count("}"):
        output_text = output_text + "}"

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

    # Check if the correct keys are present
    if not all(
        key in json_text_clean for key in ["diagnosis", "probability", "explanation"]
    ):
        logger.warning(
            "JSON object does not contain all required keys. Returning default."
        )
        return default_json

    try:
        return json.loads(json_text_clean)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse JSON: {e}\nRaw: {json_text_clean}")
        return default_json

def sys_msg_smpls() -> list[str]:
    """Return a list of sample system messages."""
    sys_msg_list = []
    sys_msg_list.append((
        "You are a helpful assistant and experienced medical professional analyzing ICU time-series data "
        "to determine the presence of a critical condition.\n\n"
        "Your response must strictly follow this format:\n"
        "Output a valid JSON object with three keys: 'diagnosis', 'probability' and 'explanation'.\n\n"
        "1. 'diagnosis' a string with either diganosis or not-diagnosis\n"
        "2. 'probability' a value between 0 and 1. where 0 means not diagnosed and 1 means diagnosed.\n"
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
    ))
    sys_msg_list.append(
        "As a medical AI, analyze ICU data for **Acute Kidney Injury (AKI)**, **Sepsis**, or **In-hospital Mortality**. "
        "Provide a single diagnosis or its absence for the most relevant condition in valid JSON format only.\n\n"
        "--- Output Format ---\n"
        "{\n"
        '  "diagnosis": "STRING",     // e.g., "aki", "not-sepsis", "mortality"\n'
        '  "probability": FLOAT,      // 0.0 to 1.0\n'
        '  "explanation": "STRING"    // Brief clinical rationale\n'
        "}\n\n"
        "--- Examples of Expected Output ---\n"
        "**Example (Sepsis):**\n"
        "{\n"
        '  "diagnosis": "sepsis",\n'
        '  "probability": 0.76,\n'
        '  "explanation": "Lactate 4.2 mmol/L; MAP 62 mmHg, consistent with sepsis."\n'
        "}\n\n"
        "**Example (Not Mortality):**\n"
        "{\n"
        '  "diagnosis": "not-mortality",\n'
        '  "probability": 0.08,\n'
        '  "explanation": "Stable vitals and organ function, low SOFA score."\n'
        "}\n\n"
        "**Example (AKI):**\n"
        "{\n"
        '  "diagnosis": "aki",\n'
        '  "probability": 0.82,\n'
        '  "explanation": "Creatinine increased from 1.0 to 2.3 mg/dL within 48h; low urine output."\n'
        "}"
    )
    sys_msg_list.append(
        "You are an experienced medical AI analyzing ICU time-series data to identify **Acute Kidney Injury (AKI)**, **Sepsis**, or **In-hospital Mortality**.\n"
        "Your goal is to provide a specific diagnosis or indicate its absence for the most pertinent condition based on the patient's data.\n\n"
        "Your output must be a valid JSON object with the following keys and format, with NO additional text:\n\n"
        "--- JSON Schema ---\n"
        "{\n"
        '  "diagnosis": "string"  // Must be one of: "aki", "not-aki", "sepsis", "not-sepsis", "mortality", "not-mortality"\n'
        '  "probability": "float" // Value between 0.0 (no diagnosis) and 1.0 (definite diagnosis)\n'
        '  "explanation": "string" // Concise clinical reasoning for the diagnosis and probability\n'
        "}\n\n"
        "--- Guidance & Examples ---\n"
        "Prioritize the most impactful or evident condition in your diagnosis. For instance:\n\n"
        "**Example 1 (AKI):**\n"
        "{\n"
        '  "diagnosis": "aki",\n'
        '  "probability": 0.82,\n'
        '  "explanation": "Serum creatinine increased from 1.0 to 2.3 mg/dL within 48 hours; urine output is low (0.3 mL/kg/h), indicative of AKI."\n'
        "}\n\n"
        "**Example 2 (Not Sepsis):**\n"
        "{\n"
        '  "diagnosis": "not-sepsis",\n'
        '  "probability": 0.01,\n'
        '  "explanation": "Lactate is 1.2 mmol/L (normal); blood pressure and other vital signs are stable, showing no signs of sepsis."\n'
        "}\n\n"
        "**Example 3 (Mortality):**\n"
        "{\n"
        '  "diagnosis": "mortality",\n'
        '  "probability": 0.91,\n'
        '  "explanation": "Patient displays multi-organ failure and a high SOFA score, suggesting a severe prognosis and high risk of mortality."\n'
        "}"
    )
    sys_msg_list.append(
        "You are an expert medical AI assistant specialized in analyzing ICU time-series data "
        "to diagnose critical conditions. Your task is to analyze the provided patient data "
        "and determine the presence of one of the following conditions: **Acute Kidney Injury (AKI)**, **Sepsis**, or **In-hospital Mortality**. "
        "If a condition is not present, indicate its 'not-diagnosis' counterpart.\n\n"
        "Your response MUST be a valid JSON object with three keys: 'diagnosis', 'probability', and 'explanation'. "
        "Do not include any other text or explanations outside of the JSON object.\n\n"
        "--- JSON Structure ---\n"
        "1. 'diagnosis': A string. Choose one from: 'aki', 'not-aki', 'sepsis', 'not-sepsis', 'mortality', or 'not-mortality'.\n"
        "2. 'probability': A float value between 0.0 and 1.0, representing your confidence in the diagnosis (0.0 for not diagnosed, 1.0 for definite diagnosis).\n"
        "3. 'explanation': A concise string providing a brief, medically sound justification for your diagnosis and probability.\n\n"
        "--- Examples ---\n"
        "**Positive Sepsis Example:**\n"
        "{\n"
        '  "diagnosis": "sepsis",\n'
        '  "probability": 0.76,\n'
        '  "explanation": "Lactate is 4.2 mmol/L (above normal <2.0); MAP 62 mmHg, indicating hypoperfusion and organ dysfunction consistent with sepsis."\n'
        "}\n\n"
        "**Negative AKI Example:**\n"
        "{\n"
        '  "diagnosis": "not-aki",\n'
        '  "probability": 0.05,\n'
        '  "explanation": "Serum creatinine is stable at 0.9 mg/dL; urine output is adequate at 1.0 mL/kg/h, showing no evidence of acute kidney injury."\n'
        "}\n\n"
        "**Positive Mortality Example:**\n"
        "{\n"
        '  "diagnosis": "mortality",\n'
        '  "probability": 0.91,\n'
        '  "explanation": "Patient exhibits persistent refractory hypotension and progressive multi-organ failure, indicating high risk of in-hospital mortality."\n'
        "}\n\n"
        "Think critically about the provided patient data and the most relevant condition before generating your output."
    )

    return sys_msg_list
