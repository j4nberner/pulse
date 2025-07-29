import json
import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

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

    def __init__(self, patience: int = 10, verbose: bool = True, delta: float = 0.0001):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module):
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

        return model


def save_torch_model(model_name: str, model: Any, save_dir: str) -> str:
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
    except (OSError, torch.TorchError) as e:
        logger.error("Failed to save model '%s': %s", model_name, str(e))

    return model_path


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
    except (
        OSError,
        joblib.externals.loky.process_executor.TerminatedWorkerError,
        joblib.externals.loky.process_executor.BrokenProcessPool,
    ) as e:
        logger.error("Failed to save model '%s': %s", model_name, str(e))


def prepare_data_for_model_convml(
    data_loader: Any,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
    task_name: Optional[str] = None,
    architecture_type: Optional[str] = None,
) -> Any:
    """
    Prepare data for conventional deep learning models by returning a configured data converter.

    Args:
        data_loader: DataLoader containing the input data
        config: Configuration dictionary with preprocessing settings
        task_name: Name of the current task (e.g., "mortality", "aki")
        architecture_type: Base architecture of the convDL model ("CNN" or "RNN") to determine array format

    Returns:
        WindowedDataTo3D: Configured converter instance ready to transform batches
    """

    # Import the converter
    from src.preprocessing.preprocessing_advanced.windowing import WindowedDataTo3D

    # Create converter with architecture type and config
    converter = WindowedDataTo3D(
        architecture_type=architecture_type, config=config, task_name=task_name
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


def normalize_probability(prob_value: float) -> float:
    """Normalize probability to a 0.0-1.0 range."""
    return prob_value / 100.0


def prompt_template_hf(
    input_text: str, custom_system_message=None, model=None, task=None
) -> List[Dict[str, str]]:
    """
    Create a chat-based prompt compatible with Hugging Face's apply_chat_template.

    Args:
        input_text: The text to analyze.
        model: Optional model name for specific formatting.
        custom_system_message: Optional custom system message to override the default.
        task: The medical task/condition (e.g., "mortality", "aki", "sepsis") for dynamic system message generation.

    Returns:
        A list of chat messages (dicts) for the LLM.
    """
    if custom_system_message:
        system_message = custom_system_message
    elif task:
        # Use sample 2 (index 1) from system_message_samples which includes task-specific examples
        system_message = system_message_samples(task, sample_index=1)[0]
    else:
        raise ValueError(
            "Either 'custom_system_message' or 'task' parameter must be provided"
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
    elif model == "Gemini2p5flashModel" or model == "Gemini2p5proModel":
        formatted_prompt = [
            {"role": "user", "parts": [{"text": f"{system_message} \n\n{input_text}"}]}
        ]
    elif model == "ClaudeSonnet4Model":
        formatted_prompt = [
            {"role": "user", "content": input_text},
        ]
        return formatted_prompt, system_message
    elif model == "Grok4Model":
        return input_text, system_message
    elif model == "OpenAIo3Model":
        formatted_prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text},
        ]
    else:
        formatted_prompt = [
            {"role": "assistant", "content": system_message},
            {"role": "user", "content": input_text},
        ]

    return formatted_prompt


def system_message_samples(task: str, sample_index: int = None) -> list[str]:
    """
    Generate a controlled experimental set of system messages for testing prompt engineering techniques.

    Args:
        task: The specific medical condition to diagnose (e.g., "mortality", "aki", "sepsis").
        sample_index: If specified (0-4), returns only that specific sample. If None, returns all samples.

    Returns:
        A list of system messages (all 5 if sample_index is None, or single item list if sample_index specified).

    Experimental Design:
        This function implements a controlled experiment to isolate the impact of specific system message
        engineering techniques on LLM performance for medical diagnosis tasks. Each sample builds
        cumulatively on the previous one, allowing for precise attribution of performance changes.

    System Message Progression (Cumulative):
        Sample 1: Baseline - Basic instructions only
            - Tests fundamental instruction following capability
            - Establishes performance floor without any enhancements

        Sample 2: Sample 1 + Task-specific Examples
            - Adds positive and negative diagnostic examples with clinical details
            - Tests impact of few-shot learning and demonstration-based guidance
            - Uses real clinical parameters (vital signs, lab values, FiO2, urine output)

        Sample 3: Sample 2 + Probability Calibration Guidelines
            - Adds explicit probability range interpretations (0-20, 20-40, etc.)
            - Tests impact of confidence calibration guidance
            - Improves probability estimation accuracy and consistency

        Sample 4: Sample 3 + ICU Context Awareness
            - Adds contextual guidance about ICU patient baseline abnormalities
            - Tests impact of domain-specific contextual understanding
            - Helps model account for critically ill patient characteristics

        Sample 5: Sample 4 + Detailed JSON Schema
            - Adds explicit schema definition with field constraints
            - Tests impact of structured format specification
            - Ensures consistent output format compliance

    Experimental Logic:
        By comparing performance between consecutive samples, researchers can isolate the
        specific contribution of each prompt engineering technique:
        - Sample 2 vs 1: Effect of examples
        - Sample 3 vs 2: Effect of probability calibration
        - Sample 4 vs 3: Effect of ICU context
        - Sample 5 vs 4: Effect of schema structure
    """
    sys_msg_list = []

    # Define task-specific descriptions
    def get_task_description(task_name):
        if task_name.lower() == "mortality":
            return "predicting patient mortality during the ICU stay"
        elif task_name.lower() == "aki":
            return "detecting acute kidney injury (AKI) ≥ stage 1 according to KDIGO 2012 criteria"
        elif task_name.lower() == "sepsis":
            return "detecting sepsis per Sepsis-3 definition (Singer 2016) with SOFA score increase ≥2 points with suspected infection"
        else:
            return f"determining the presence of {task_name}"

    # Define task-specific examples
    def get_task_examples(task_name):
        if task_name.lower() == "sepsis":
            return {
                "positive": {
                    "diagnosis": "sepsis",
                    "probability": 82,
                    "explanation": "Patient shows sepsis criteria: temperature 38.9°C, heart rate 115 bpm, WBC 16,000/μL, lactate 4.1 mmol/L (elevated >2.0), and hypotension with MAP 58 mmHg despite fluid resuscitation.",
                },
                "negative": {
                    "diagnosis": "not-sepsis",
                    "probability": 12,
                    "explanation": "Patient shows no signs of sepsis: temperature 37.2°C, heart rate 88 bpm, normal WBC 7,200/μL, lactate 1.6 mmol/L (normal <2.0), and adequate blood pressure with MAP 78 mmHg.",
                },
            }
        elif task_name.lower() == "aki":
            return {
                "positive": {
                    "diagnosis": "aki",
                    "probability": 89,
                    "explanation": "Acute kidney injury evident: serum creatinine increased from baseline 1.1 to 2.7 mg/dL within 24 hours (>2x increase), urine output decreased to 0.3 mL/kg/h over 6 hours, meeting KDIGO Stage 2 criteria.",
                },
                "negative": {
                    "diagnosis": "not-aki",
                    "probability": 8,
                    "explanation": "Kidney function stable: creatinine 1.3 mg/dL (minimal change from baseline 1.2), adequate urine output at 1.1 mL/kg/h, no signs of acute kidney injury.",
                },
            }
        elif task_name.lower() == "mortality":
            return {
                "positive": {
                    "diagnosis": "mortality",
                    "probability": 91,
                    "explanation": "Critical condition: multi-organ failure with high lactate 6.2 mmol/L, requiring mechanical ventilation (FiO2 80%), hypotension, and oliguria <0.2 mL/kg/h despite treatment.",
                },
                "negative": {
                    "diagnosis": "not-mortality",
                    "probability": 15,
                    "explanation": "Improving trajectory: lactate normalizing to 2.1 mmol/L, weaning from ventilator support (FiO2 40%), stable hemodynamics, and adequate urine output 0.8 mL/kg/h.",
                },
            }
        else:
            return {
                "positive": {
                    "diagnosis": task_name,
                    "probability": 78,
                    "explanation": f"Clinical indicators strongly suggest {task_name} based on abnormal vital signs, laboratory values, and physiological parameters.",
                },
                "negative": {
                    "diagnosis": f"not-{task_name}",
                    "probability": 15,
                    "explanation": f"Clinical indicators do not support {task_name} diagnosis with stable vital signs and normal laboratory parameters.",
                },
            }

    examples = get_task_examples(task)
    task_description = get_task_description(task)

    # Base components that build up progressively
    base_instruction = (
        f"You are a helpful assistant and experienced medical professional analyzing ICU time-series data "
        f"for {task_description}.\n\n"
        "Your response must strictly follow this format:\n"
        "Output a valid JSON object with three keys: 'diagnosis', 'probability' and 'explanation'.\n\n"
        "1. 'diagnosis' a string with either 'diagnosis' or 'not-diagnosis'\n"
        "2. 'probability' an integer between 0 and 100, where 0 means not diagnosed and 100 means diagnosed.\n"
        "3. 'explanation' should be a string providing a brief explanation of your diagnosis.\n\n"
    )

    closing = (
        "Do not include any other text or explanations outside of the JSON object.\n"
        "Think about the probability of your prediction carefully before answering.\n"
    )

    examples_section = (
        "Here is a positive example:\n"
        "{\n"
        f'  "diagnosis": "{examples["positive"]["diagnosis"]}",\n'
        f'  "probability": "{examples["positive"]["probability"]}",\n'
        f'  "explanation": "{examples["positive"]["explanation"]}"\n'
        "}\n\n"
        "Here is a negative example:\n"
        "{\n"
        f'  "diagnosis": "{examples["negative"]["diagnosis"]}",\n'
        f'  "probability": "{examples["negative"]["probability"]}",\n'
        f'  "explanation": "{examples["negative"]["explanation"]}"\n'
        "}\n\n"
    )

    icu_context = (
        "Note: ICU patients often present with abnormal baseline values due to their critical condition. "
        "Consider the clinical context and severity of deviations when assessing for the target condition.\n\n"
    )

    schema_section = (
        "--- JSON Schema ---\n"
        "{\n"
        f'  "diagnosis": "string"  // Must be either "{task}" or "not-{task}"\n'
        '  "probability": integer // Value between 0 (no diagnosis) and 100 (definite diagnosis)\n'
        '  "explanation": "string" // Concise clinical reasoning for the diagnosis and probability\n'
        "}\n\n"
    )

    probability_guidelines = (
        "CRITICAL: Probability calibration guidelines:\n"
        "- 0-20: Very unlikely, clear absence of condition with normal parameters\n"
        "- 20-40: Unlikely, some concerning signs but insufficient evidence\n"
        "- 40-60: Uncertain, mixed evidence or borderline findings\n"
        "- 60-80: Likely, multiple indicators support diagnosis\n"
        "- 80-100: Very likely, strong evidence with clear clinical criteria met\n\n"
    )

    # Build the specific sample or all samples based on sample_index
    if sample_index is not None:
        if sample_index == 0:
            return [base_instruction + closing]
        elif sample_index == 1:
            return [base_instruction + examples_section + closing]
        elif sample_index == 2:
            return [
                base_instruction + examples_section + probability_guidelines + closing
            ]
        elif sample_index == 3:
            return [
                base_instruction
                + examples_section
                + probability_guidelines
                + icu_context
                + closing
            ]
        elif sample_index == 4:
            return [
                base_instruction
                + examples_section
                + probability_guidelines
                + schema_section
                + icu_context
                + closing
            ]
        else:
            raise ValueError(
                f"sample_index must be between 0 and 4, got {sample_index}"
            )

    # If no specific index requested, build all samples (for backward compatibility)
    sys_msg_list = []

    # Sample 1: Baseline (no examples)
    sys_msg_list.append(base_instruction + closing)

    # Sample 2: Sample 1 + Examples
    sys_msg_list.append(base_instruction + examples_section + closing)

    # Sample 3: Sample 2 + Probability Calibration
    sys_msg_list.append(
        base_instruction + examples_section + probability_guidelines + closing
    )

    # Sample 4: Sample 3 + ICU Context
    sys_msg_list.append(
        base_instruction
        + examples_section
        + probability_guidelines
        + icu_context
        + closing
    )

    # Sample 5: Sample 4 + Detailed Schema
    sys_msg_list.append(
        base_instruction
        + examples_section
        + probability_guidelines
        + schema_section
        + icu_context
        + closing
    )

    return sys_msg_list


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


def fix_json_formatting(json_text: str) -> str:
    """Fix common JSON formatting issues."""
    # Remove lines that are just a quoted brace (e.g., '"}"') or similar junk
    if any(
        re.match(r'^\s*"\s*}\s*"\s*,?\s*$', line)
        or re.match(r'^\s*"\s*}\s*"\s*$', line)
        for line in json_text.splitlines()
    ):
        lines = json_text.splitlines()
        filtered_lines = [
            line
            for line in lines
            if not re.match(r'^\s*"\s*}\s*"\s*,?\s*$', line)
            and not re.match(r'^\s*"\s*}\s*"\s*$', line)
        ]
        json_text = "\n".join(filtered_lines)
        logger.debug(
            "Fixed junk lines (quoted braces) by removing them from JSON text."
        )

    # Remove incomplete key at the end (e.g., "conf)
    incomplete_key_pattern = r',\s*\n?\s*"(?:[^"]*)$'
    if re.search(incomplete_key_pattern, json_text):
        json_text = re.sub(incomplete_key_pattern, "", json_text)
        logger.debug("Fixed incomplete key at the end of JSON text.")

    # Fix unterminated strings
    if json_text.count('"') % 2 != 0:
        json_text += '"'
        logger.debug("Fixed unterminated string by adding closing quote.")

    # Fix missing value after colon, replacing any '"key":}' or '"key":' with '"key": ""}'
    if re.search(r'(":[ \t]*)}', json_text) or re.search(r'(":[ \t]*)$', json_text):
        json_text = re.sub(r'(":[ \t]*)}', r'\1""}', json_text)
        json_text = re.sub(r'(":[ \t]*)$', r'\1""', json_text)
        logger.debug("Fixed missing value after colon by inserting empty string.")

    # Fix unclosed arrays (relevant e.g. for clinical workflow agent)
    open_brackets = json_text.count("[")
    close_brackets = json_text.count("]")
    if open_brackets > close_brackets:
        json_text += "]" * (open_brackets - close_brackets)
        logger.debug("Fixed unclosed array by adding closing bracket(s).")

    # Fix missing final brace
    if not json_text.endswith("}"):
        json_text += "}"
        logger.debug("Fixed unclosed JSON object by adding closing brace.")

    # Remove trailing commas before closing braces/brackets
    if re.search(r",\s*([\]}])", json_text):
        json_text = re.sub(r",\s*([\]}])", r"\1", json_text)
        logger.debug("Fixed trailing comma before closing braces")

    # Escape newlines in quoted strings
    def escape_newlines_in_strings(s: str) -> str:
        def repl(m):
            return m.group(0).replace("\n", "\\n").replace("\r", "\\r")

        return re.sub(r'"(.*?)"', repl, s, flags=re.DOTALL)

    json_text = escape_newlines_in_strings(json_text)

    return json_text


def parse_llm_output(
    output_text: str,
    required_keys: Optional[List[str]] = None,
    default_values: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """Extract and parse JSON objects from LLM output with robust error handling.

    Args:
        output_text: Raw text output from the language model
        required_keys: Keys that must be present in the output
        default_values: Default values to use if parsing fails

    Returns:
        Dictionary with parsed content and normalized values
    """
    if required_keys is None:
        required_keys = ["diagnosis", "probability", "explanation"]
    if default_values is None:
        default_values = {
            "diagnosis": "unknown",
            "probability": 0.5,
            "explanation": "No explanation provided.",
        }

    # Handle empty input
    if not output_text or not isinstance(output_text, str):
        logger.warning("Empty or non-string output received")
        return default_values

    # Balance braces if needed
    if output_text.count("{") > output_text.count("}"):
        output_text = output_text + "}" * (
            output_text.count("{") - output_text.count("}")
        )

    # Extract JSON block
    json_text = extract_last_json_block(output_text)
    if not json_text:
        logger.warning("No JSON object found in output. Returning default.")
        return default_values

    # Fix common JSON formatting issues
    json_text = fix_json_formatting(json_text)

    # Check if the required keys are present
    if not all(key in json_text for key in required_keys):
        logger.warning(
            "JSON object missing required keys %s. Returning default.",
            required_keys,
        )
        return default_values

    try:
        parsed = json.loads(json_text)

        # Process probability field - convert from various formats to 0.0-1.0 float
        if "probability" in parsed:
            try:
                prob_raw = parsed["probability"]

                # Handle string values (strip quotes and convert)
                if isinstance(prob_raw, str):
                    # Remove any surrounding quotes and whitespace
                    prob_raw = prob_raw.strip().strip("\"'")
                    # Try to extract numeric value from string
                    numeric_match = re.search(r"(\d+\.?\d*)", prob_raw)
                    if numeric_match:
                        prob_raw = numeric_match.group(1)
                    else:
                        logger.warning(
                            "No numeric value found in probability string: %s", prob_raw
                        )
                        parsed["probability"] = 0.5
                        return parsed

                # Convert to float
                prob_value = float(prob_raw)

                # Normalize to 0.0-1.0 range
                if 0 <= prob_value <= 1:
                    # Already in correct range
                    parsed["probability"] = prob_value
                elif 0 <= prob_value <= 100:
                    # Convert from 0-100 range to 0.0-1.0 range
                    parsed["probability"] = prob_value / 100.0
                else:
                    # Out of expected range, clamp and warn
                    logger.warning(
                        "Probability value %s out of expected range. Clamping to [0,1]",
                        prob_value,
                    )
                    if prob_value > 100:
                        # Assume it was meant to be 0-100 range
                        parsed["probability"] = min(1.0, prob_value / 100.0)
                    else:
                        # Negative or very small, clamp to 0
                        parsed["probability"] = max(0.0, prob_value)

            except (ValueError, TypeError) as e:
                logger.warning(
                    "Failed to convert probability '%s' to float: %s. Defaulting to 0.5",
                    parsed.get("probability", "None"),
                    e,
                )
                parsed["probability"] = 0.5

        return parsed

    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON: %s\nRaw: %s", e, json_text)
        return default_values


@DeprecationWarning
def extract_dict(output_text: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the last JSON-like object from the model's output text and return it as a dictionary.

    Args:
        output_text: The raw string returned by the language model.

    Returns:
        A dictionary parsed from the JSON string, or a default JSON object if no JSON was found.
    """
    warnings.warn(
        "extract_dict is deprecated and will be removed in a future release. "
        "Use parse_llm_output instead.",
        DeprecationWarning,
        stacklevel=2,
    )
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
        parsed = json.loads(json_text_clean)

        # Convert probability using helper function
        if "probability" in parsed:
            try:
                prob_value = float(parsed["probability"])
                parsed["probability"] = normalize_probability(prob_value)
            except (ValueError, TypeError):
                parsed["probability"] = 0.5

        return parsed
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse JSON: %s\nRaw: %s", e, json_text_clean)
        return default_json
