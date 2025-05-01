import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.util.data_util import (
    get_feature,
    get_feature_name,
    get_feature_reference_range,
    get_feature_uom,
)

logger = logging.getLogger("PULSE_logger")


def few_shot_paper_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess ICU data into prompts using few-shot format and centralized JSON prompt template.
    According to the paper "Large Language Models are Few-Shot Health Learners"
    Paper: https://arxiv.org/pdf/2305.15525

    Args:
        X (List[pd.DataFrame]): [X_eval, X_train] (test/val features + training examples).
        y (List[pd.DataFrame]): [y_eval, y_train] (corresponding labels).
        info_dict (Dict[str, Any]): Task metadata (e.g. task name, model ID, etc).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Prompt DataFrame and label DataFrame.
    """
    task = info_dict.get("task", "unknown_task")
    dataset = info_dict.get("dataset_name", "unknown_dataset")
    model_id = info_dict.get("model_name", "unknown_model")
    num_shots = 0 #TODO: info_dict.get("num_shots", 0) rout from kwargs
    mode = info_dict.get("mode", "train")

    logger.info(
        "Preprocessing model '%s' on dataset '%s', task '%s'", model_id, dataset, task
    )

    X_input = X[0].filter(regex=r"^(?!.*_na(_\d+)?$)")
    y_input = y[0]

    # Extract unique feature base names (e.g., "hr" from "hr_1")
    # and prepare feature descriptions for the reference section
    base_features = []
    for col in X_input.columns:
        parts = col.split("_")
        base_features.append(get_feature(parts[0]))

    # Optional few-shot examples
    X_train, y_train = None, None
    if mode != "train" and len(X) > 1:
        X_train = X[1].filter(regex=r"^(?!.*_na(_\d+)?$)")
        y_train = y[1]

    prompts = []
    for idx, row in X_input.iterrows():
        # 1. Build the real query
        query_features = ", ".join(
            f"{base_features[i][0]}: {val} {base_features[i][1]}"
            for i, val in enumerate(row.values)
            if pd.notna(val)
        )
        query_prompt = f"Patient ICU features: {query_features}"
        # query_prompt = prompt_template(query_input_text)

        # 2. Build few-shot examples
        few_shot_texts = []
        if num_shots > 0 and X_train is not None:
            indices = np.random.choice(
                len(X_train), size=min(num_shots, len(X_train)), replace=False
            )
            for i in indices:
                train_features = ", ".join(
                    f"{base_features[i][0]}: {val} {base_features[i][1]}"
                    for i, val in enumerate(X_train.iloc[i].values)
                    if pd.notna(val)
                )
                label = (
                    y_train.iloc[i].values[1]
                    if y_train.shape[1] > 1
                    else y_train.iloc[i].values[0]
                )
                label_text = task if label == 1 else f"not-{task}"

                few_shot_prompt = wrap_for_few_shot_template(train_features, label_text)
                few_shot_texts.append(few_shot_prompt)

        # 3. Combine few-shot + query
        full_prompt = "\n\n".join(few_shot_texts + [query_prompt])
        prompts.append(full_prompt)

    X_processed = pd.DataFrame({"text": prompts})
    return X_processed, y_input


# --------------------------------
# Helper functions
# --------------------------------


def prepare_feature_descriptions(base_features, X_cols):
    """Prepare feature descriptions with name, unit of measurement, and reference range.

    Args:
        base_features: Set of base feature names
        X_cols: DataFrame columns to check for additional features

    Returns:
        Feature descriptions text as a formatted string
    """
    # Generate feature descriptions for the reference section
    feature_descriptions = []
    for feature in sorted(base_features):  # Sort for consistent order
        feature_name = get_feature_name(feature)
        uom = get_feature_uom(feature)
        range_values = get_feature_reference_range(feature)

        if range_values:  # Check if the range exists (not empty tuple)
            range_str = f"{range_values[0]} - {range_values[1]}"
            feature_descriptions.append(
                f"- {feature_name}: Unit: {uom}. Reference range: {range_str}."
            )
        else:
            feature_descriptions.append(
                f"- {feature_name}: Unit: {uom}. Reference range: /."
            )

    # Add weight and height to feature descriptions if they exist in the columns
    if "weight" in X_cols:
        weight_name = get_feature_name("weight")
        weight_uom = get_feature_uom("weight")
        feature_descriptions.append(
            f"- {weight_name}: Unit: {weight_uom}. Reference range: /."
        )

    if "height" in X_cols:
        height_name = get_feature_name("height")
        height_uom = get_feature_uom("height")
        feature_descriptions.append(
            f"- {height_name}: Unit: {height_uom}. Reference range: /."
        )

    # Join all feature descriptions into a single string
    return "\n".join(feature_descriptions)


def format_patient_data(row, base_features, X_cols, data_window):
    """Format patient data for prompting.

    Args:
        row: Patient data row
        base_features: Set of base feature names
        X_cols: DataFrame columns to extract feature columns from

    Returns:
        Tuple of (patient_info, patient_features_text)
    """
    # Extract patient demographic info
    sex = row.get("sex", "unknown")
    age = row.get("age", "unknown")
    patient_info = f"The patient is a {sex}, aged {age} years."

    # Format feature values
    patient_features = []

    # Process dynamic features (those with time series)
    for feature in sorted(base_features):
        # Get columns for this feature (e.g., hr_1, hr_2, etc.)
        feature_cols = [col for col in X_cols if col.startswith(f"{feature}_")]

        # Filter to only include columns with numeric indices
        feature_cols = [col for col in feature_cols if col.split("_")[1].isdigit()]

        # Print warning if the number of feature columns doesn't match the data window
        if len(feature_cols) != data_window:
            logger.warning(
                f"Feature '{feature}' has {len(feature_cols)} columns, but expected {data_window} columns."
            )

        # Sort columns by time point
        if feature_cols:  # Only sort if there are valid columns
            feature_cols.sort(key=lambda x: int(x.split("_")[1]))

        # Extract values for this feature across all time points
        values = [str(row[col]) for col in feature_cols]
        values_str = f'"{", ".join(values)}"'

        # Use the proper feature name from dictionary
        feature_name = get_feature_name(feature)
        patient_features.append(f"- {feature_name}: {values_str}")

    # Get number of time points from dynamic features
    num_timepoints = len(feature_cols) if "feature_cols" in locals() else 6

    # Process static features (weight and height) - repeat value for all time points
    if "weight" in row.index and not pd.isna(row["weight"]):
        weight_value = str(row["weight"])
        weight_values = [weight_value] * num_timepoints
        weight_str = f'"{", ".join(weight_values)}"'
        weight_name = get_feature_name("weight")
        patient_features.append(f"- {weight_name}: {weight_str}")

    if "height" in row.index and not pd.isna(row["height"]):
        height_value = str(row["height"])
        height_values = [height_value] * num_timepoints
        height_str = f'"{", ".join(height_values)}"'
        height_name = get_feature_name("height")
        patient_features.append(f"- {height_name}: {height_str}")

    # Join patient features into a string
    patient_features_text = "\n".join(patient_features)

    return patient_info, patient_features_text


def wrap_for_few_shot_template(input_text: str, label_text: str) -> str:

    # Replace the final JSON object with a static answer for few-shot example
    result_json = {
        "diagnosis": label_text,
        "probability": "1.0",
        "explanation": "This is a known example explanation.",
    }

    prompt = (
    "This is an example. Analyze the following text and determine the most likely diagnosis.\n"
    "Text:\n"
    f"{input_text}"
    "Result:\n"
    f"{result_json}")
    return prompt.strip()