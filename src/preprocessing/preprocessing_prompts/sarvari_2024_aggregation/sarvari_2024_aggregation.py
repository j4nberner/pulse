import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.data_util import get_feature

logger = logging.getLogger("PULSE_logger")


def sarvari_aggregation_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Preprocess ICU data into prompts as per the paper
    "A systematic evaluation of the performance of GPT-4 and PaLM2 to diagnose comorbidities in MIMIC-IV patients".
    Paper: https://doi.org/10.1002/hcs2.79

    Args:
        X (List[pd.DataFrame]): [X_eval, X_train] (test/val features + training examples).
        y (List[pd.DataFrame]): [y_eval, y_train] (corresponding labels).
        info_dict (Dict[str, Any]): Task metadata (e.g. task name, model ID, etc).

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys "X" (prompt DataFrame) and "y" (label DataFrame).
    """
    pp = PreprocessorAdvanced()
    task = info_dict.get("task", "unknown_task")
    dataset = info_dict.get("dataset_name", "unknown_dataset")
    model_id = info_dict.get("model_name", "unknown_model")
    num_shots = info_dict.get("num_shots", 0)
    mode = info_dict.get("mode", "train")

    logger.info(
        "Preprocessing model '%s' on dataset '%s', task '%s'", model_id, dataset, task
    )

    # Filter out columns with "_na" suffixes
    X_input = X[0].filter(regex=r"^(?!.*_na(_\d+)?$)")
    y_input = y[0]
    # Optional few-shot examples
    X_train, y_train = None, None
    if mode != "train" and len(X) > 1:
        X_train = X[1].filter(regex=r"^(?!.*_na(_\d+)?$)")
        y_train = y[1]

    # Extract unique feature base names (e.g., "hr" from "hr_1")
    # and prepare feature descriptions for the reference section
    base_features = []
    for col in X_input.columns:
        parts = col.split("_")
        base_features.append(get_feature(parts[0]))

    # 1. Build the main query from test data
    # Aggregate features per data window
    X_input_aggregated = pp.aggregate_feature_windows(X_input)
    prompts = build_sarvari_query(X_input_aggregated, y=None, example=False, task=task)
    wrapped_prompts = _wrap_for_few_shot_template([""], prompts, task=task)

    # 2. Build few-shot examples from training data
    if num_shots > 0 and mode != "train":
        idx = np.random.choice(
            len(X_train), size=min(num_shots, len(X_train)), replace=False
        )
        X_train_aggregated = pp.aggregate_feature_windows(X_train.iloc[idx])
        few_shot_examples = build_sarvari_query(
            X_train_aggregated, y=y_train.iloc[idx], example=True, task=task
        )

        # 3. Combine few-shot + query
        combined_prompt = _wrap_for_few_shot_template(
            few_shot_examples, prompts, task=task
        )
        X_processed = pd.DataFrame({"text": combined_prompt})
    else:
        # 3. No few-shot examples, just use the main query
        X_processed = pd.DataFrame({"text": wrapped_prompts})

    return {
        "X": X_processed,
        "y": y_input,
    }


def build_sarvari_query(
    df: pd.DataFrame,
    y: pd.DataFrame = None,
    example: bool = False,
    task: str = "unknown_task",
) -> list[str]:
    """
    Generate prompt strings for each row in the DataFrame using real feature names and including patient info.

    Args:
        df: DataFrame with static and aggregated features (e.g., 'wbc_min', 'wbc_max', 'wbc_mean')
        y (pd.DataFrame, optional): DataFrame containing labels. Defaults to None.
        example (bool, optional): Flag to indicate if it's a few-shot example. Defaults to False.

    Returns:
        List[str]: List of query strings.
    """
    prompts = []
    df = df.reset_index()  # Reset index to ensure correct mapping with y

    # Identify aggregated feature basenames
    feature_basenames = sorted(
        set(
            col.rsplit("_", 1)[0]
            for col in df.columns
            if col.endswith(("_min", "_max", "_mean"))
        )
    )

    # Identify static columns (not part of the aggregation)
    static_cols = [
        col
        for col in df.columns
        if not any(col.startswith(f"{base}_") for base in feature_basenames)
    ]

    for idx, row in df.iterrows():
        lines = []

        # Add Prefix
        if example:
            lines.append("For example, if the patient data mentions:\n")
        else:
            lines.append("Patient data:\n")

        # Add patient static info at the top
        patient_info = ", ".join(f"{col}: {row[col]}" for col in static_cols)
        lines.append(f"Patient Info — {patient_info}")

        # Add feature stats
        for base in feature_basenames:
            try:
                feature_name, unit, ref_range = get_feature(base)
            except Exception:
                feature_name, unit, ref_range = base, "", None

            min_val = row.get(f"{base}_min", None)
            max_val = row.get(f"{base}_max", None)
            mean_val = row.get(f"{base}_mean", None)

            # Handle missing values
            if pd.isna(min_val) and pd.isna(max_val) and pd.isna(mean_val):
                continue

            line = f"{feature_name} (unit: {unit}): min={min_val:.3f}, max={max_val:.3f}, mean={mean_val:.3f}"
            lines.append(line)

        # Add label if provided
        if y is not None:
            y = y.reset_index()
            # Ensure y has the same index as df

            if y.shape[0] != df.shape[0]:
                raise ValueError(
                    "The number of rows in y must match the number of rows in df."
                )

            label = y.iloc[idx].get("label", None)
            diagnosis = ["yes"] if label == 1 else ["no"]
            if label is not None:
                lines.append(
                    "Then your answer may be: \n"
                    "{\n"
                    f' "diagnosis": "{diagnosis[0]}",\n'
                    ' "probability": "50"\n'
                    '  "explanation": "<a brief explanation for the prediction>"\n'
                    "}\n\n"
                )

        # Add suffix
        lines.append("")

        prompt = "\n".join(lines)
        prompts.append(prompt)

    return prompts


def _wrap_for_few_shot_template(
    few_shot_examples: list, prompts: list, task
) -> list[str]:
    """
    Wraps the few-shot examples and the main query into prompt strings.

    Args:
        few_shot_examples (list): List of few-shot examples.
        prompts (list): List of main query prompts.

    Returns:
        list[str]: List of combined prompt strings.
    """
    combined_prompts = []
    prefix = (
        f"Suggest a diagnosis of {task} for the following patient data. Reply with {task} or not-{task}.\n"
        "Give exact numbers and/or text quotes from the data that made you think of each of the diagnoses.\n"
        "Before finalizing your answer check if you haven't missed any abnormal data points. \n"
    )

    for prompt in prompts:
        # Add prefix to each prompt
        combined_prompt = [prefix]
        # Add few-shot examples to the prompt
        combined_prompt.extend(few_shot_examples)
        # Add the main query prompt
        combined_prompt.append(prompt)
        # Join the combined prompt into a single string
        combined_prompt_str = "\n".join(combined_prompt)

        # Append to the list of combined prompts
        combined_prompts.append(combined_prompt_str.strip())

    return combined_prompts
