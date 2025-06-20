import logging
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.util.data_util import get_feature

logger = logging.getLogger("PULSE_logger")


def liu_2023_few_shot_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Dict[str, pd.DataFrame]:
    """
    Preprocess ICU data into prompts using few-shot format and centralized JSON prompt template.
    According to the paper "Large Language Models are Few-Shot Health Learners"
    Paper: https://arxiv.org/pdf/2305.15525

    Args:
        X (List[pd.DataFrame]): [X_eval, X_train] (test/val features + training examples).
        y (List[pd.DataFrame]): [y_eval, y_train] (corresponding labels).
        info_dict (Dict[str, Any]): Task metadata (e.g. task name, model ID, etc).

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with keys "X" (prompt DataFrame) and "y" (label DataFrame).
    """
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
    base_features = {}

    for col in X_input.columns:
        if "_" in col and col.split("_")[-1].isdigit():
            parts = col.split("_")
            base_features[parts[0]] = get_feature(parts[0])
        else:
            base_features[col] = get_feature(col)

    # 1. Build the main query from test data
    # X_inputs_raw = pp.raw_feature_windows(X_input)
    prompts = build_liu_query(
        X_input, task=task, feature_basenames=base_features, example=False
    )

    # 2. Build few-shot examples from training data if available
    few_shot_examples = []
    if X_train is not None and y_train is not None:
        idx = np.random.choice(
            len(X_train), size=min(num_shots, len(X_train)), replace=False
        )
        # X_train_raw = pp.raw_feature_windows(X_input)
        few_shot_examples = build_liu_query(
            X_train.iloc[idx],
            y=y_train.iloc[idx],
            example=True,
            task=task,
            feature_basenames=base_features,
        )

    # 3. Combine few-shot + query
    combined_prompt = _wrap_for_few_shot_template(few_shot_examples, prompts, task=task)

    X_processed = pd.DataFrame({"text": combined_prompt})

    return {
        "X": X_processed,
        "y": y_input,
    }


def build_liu_query(
    df: pd.DataFrame, task: str, feature_basenames: dict, y=None, example=False
) -> list[str]:
    """
    Build the query for the Liu 2023 paper.

    Args:
        df (pd.DataFrame): DataFrame containing the features.
        task (str): The task to be performed (e.g., "sepsis", "AKI").
        feature_basenames (dict): Dict of feature base names and their descriptions.
        y (pd.DataFrame, optional): DataFrame containing the labels. Defaults to None.
        example (bool, optional): Whether to create an example prompt. Defaults to False.

    Returns:
        list[str]: List of prompt strings.
    """
    prompts = []
    df = df.reset_index(drop=True)

    # --- Preprocessing: Identify scalar and grouped columns ---
    grouped_cols = {}
    scalar_cols = []

    for col in df.columns:
        match = re.match(r"^(.+)_([0-9]+)$", col)
        if match:
            base = match.group(1)
            grouped_cols.setdefault(base, []).append(col)
        else:
            scalar_cols.append(col)

    # Sort grouped columns
    for base in grouped_cols:
        grouped_cols[base].sort(key=lambda x: int(x.split("_")[-1]))

    # reset index of y to match df
    if y is not None:
        y = y.reset_index(drop=True)

    # --- Iterate over each row once ---

    for idx, row in df.iterrows():
        lines = []

        # Prefix
        if example:
            lines.append(
                f"Example Question: Classify the following ICU patient data as either {task} or not-{task}\n"
            )
        else:
            lines.append(
                f"Question: Classify the following ICU patient data as either {task} or not-{task}\n"
            )

        # Create row_dict with grouped values
        row_dict = {}

        for col in scalar_cols:
            row_dict[col] = row[col]

        for base, cols in grouped_cols.items():
            row_dict[base] = row[cols].tolist()

        # Add feature stats
        for feature in feature_basenames.items():
            # Get feature name, unit, and reference range
            value_list = row_dict.get(feature[0], None)
            if value_list is not None:
                if feature[1][1] == "":
                    line = f"{feature[1][0]}: {value_list}"
                else:
                    line = f"{feature[1][0]} {feature[1][1]}: {value_list}"
                lines.append(line)

        # Add label (if provided)
        if y is not None:
            label = y.iloc[idx].get("label", None)
            diagnosis = task if label == 1 else f"not-{task}"
            if label is not None:
                lines.append(
                    "Answer: \n"
                    "{\n"
                    f' "diagnosis": "{diagnosis}",\n'
                    '  "classification": "<the score of your diagnosis between 0 and 100>",\n'
                    '  "explanation": "<a brief explanation for the prediction>"\n'
                    "}\n"
                )

        lines.append("")  # Suffix blank line
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
    prefix = ""

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
