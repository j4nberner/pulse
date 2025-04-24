# https://arxiv.org/pdf/2305.15525

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from src.util.model_util import apply_model_prompt_format
from src.util.data_util import get_feature_name

logger = logging.getLogger("PULSE_logger")


def few_shot_paper_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess input data into a text-based prompt format suitable for LLM models,
    adhering to LangChain guidelines.

    Args:
        X (List[pd.DataFrame]): Input features.
        y (List[pd.DataFrame]): Target labels.
        info_dict (Dict[str, Any]): Additional task-specific information such as
                                    'task', 'dataset', and 'model_name'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed feature prompts and unchanged labels.
    """
    task = info_dict.get("task", "unknown_task")
    dataset = info_dict.get("dataset_name", "unknown_dataset")
    model_id = info_dict.get("model_name", "unknown_model")
    num_shots = info_dict.get("shots", 0)
    mode = info_dict.get(
        "mode", "train"
    )  # Few-shot examples are only used in validation and test mode

    logger.info(
        f"Starting preprocessing for model '{model_id}' on dataset '{dataset}' and task '{task}'."
    )

    # Remove all columns with _na suffix
    X_in = X[0].filter(regex="^(?!.*_na$)")  # input data

    if mode != "train":
        X_train = X[1].filter(regex="^(?!.*_na$)")  # few shot examples
        y_train = y[1].filter(regex="^(?!.*_na$)")  # few shot examples

    prompts = []
    feature_names = [get_feature_name(name) for name in X_in.columns.tolist()]

    # Define the prompt template
    prompt_template = PromptTemplate(
        input_variables=["features", "label"],
        template="Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   Features:\n{features}\nA: {label}",
    )

    for idx, row in X_in.iterrows():
        # Format the features for this instance
        feature_string = ", ".join(
            [
                f"{feature_names[idx]}: {value}"
                for idx, value in enumerate(row.values)
                if pd.notna(value)
            ]
        )

        # Prepare few-shot examples if applicable
        examples = []
        if num_shots > 0 and mode != "train":
            # Randomly select num_shots examples from the training set
            random_indices = np.random.choice(
                len(X_train), size=min(num_shots, len(X_train)), replace=False
            )
            for shot_idx in random_indices:
                shot_features = ", ".join(
                    [
                        f"{col} {value}"
                        for col, value in X_train.iloc[shot_idx].items()
                        if pd.notna(value)
                    ]
                )
                shot_label = (
                    "not-" + task if y_train.iloc[shot_idx].values[1] == 0 else task
                )
                examples.append(
                    {"features": shot_features, "label": shot_label, "task": task}
                )  # Ensure 'task' is included

            # Create the FewShotPromptTemplate
            few_shot_prompt = FewShotPromptTemplate(
                examples=examples,
                example_prompt=prompt_template,
                prefix="Below are examples of ICU data classified as '{task}' or 'not-{task}':",
                suffix="Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   Features:\n{features}\nA:",
                input_variables=["features", "task"],
                example_separator="\n\n",
            )
        else:
            # If no few-shot examples, use a simple prompt template
            few_shot_prompt = PromptTemplate(
                input_variables=["features", "task"],
                template="Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   Features:\n{features}\nA:",
            )

        # Generate the prompt
        prompt = few_shot_prompt.format(features=feature_string, task=task)

        # Reformat prompt according to model-specific requirements
        formatted_prompt = apply_model_prompt_format(model_id, prompt)
        prompts.append(formatted_prompt)

    X_processed = pd.DataFrame({"text": prompts})

    logger.info(
        f"Converted {len(prompts)} samples to text prompt format for model '{model_id}'."
    )

    return X_processed, y_in
