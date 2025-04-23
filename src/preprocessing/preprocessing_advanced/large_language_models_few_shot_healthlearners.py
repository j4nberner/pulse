# https://arxiv.org/pdf/2305.15525

import logging
from typing import Dict, Any, Tuple
import pandas as pd
from src.util.model_util import apply_model_prompt_format

logger = logging.getLogger("PULSE_logger")


def few_shot_paper_preprocessor(
    X: pd.DataFrame, y: pd.DataFrame, info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess input data into a text-based prompt format suitable for LLM models,
    as described in the Large Language Models are Few-Shot
    Health Learners.

    Promt Structure:
    Q: Classify the given ICU data sequence as either <diagnosis> or <not-diagnosis>.:
       <feature_name> <value>, <feature_name> <value>...
    A: <diagnosis>



    Args:
        X (pd.DataFrame): Input features.
        y (pd.DataFrame): Target labels.
        info_dict (Dict[str, Any]): Additional task-specific information such as
                                    'task', 'dataset', and 'model_name'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed feature prompts and unchanged labels.
    """
    task = info_dict.get("task", "unknown_task")
    dataset = info_dict.get("dataset", "unknown_dataset")
    model_id = info_dict.get("model_name", "unknown_model")

    logger.info(
        f"Starting preprocessing for model '{model_id}' on dataset '{dataset}' and task '{task}'."
    )

    prompts = []

    for idx, row in X.iterrows():
        # Extract corresponding label for this row
        label_value = y.iloc[idx].values[1] if not y.empty else None

        # Format the features for this instance
        feature_string = ", ".join(
            [f"{col} {value}" for col, value in row.items() if pd.notna(value)]
        )

        # Get the number of examples to include
        num_shots = info_dict.get("shots", 0)

        # Build the prompt
        if num_shots > 0 and idx >= num_shots:
            # Include few-shot examples
            examples = []
            for shot_idx in range(num_shots):
                shot_features = ", ".join(
                    [
                        f"{col} {value}"
                        for col, value in X.iloc[shot_idx].items()
                        if pd.notna(value)
                    ]
                )
                shot_label = y.iloc[shot_idx].values[0]
                examples.append(
                    f"Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   {shot_features}\nA: {'not-' if shot_label == 0 else ''}{task}"
                )

            few_shot_examples = "\n\n".join(examples)
            prompt = f"{few_shot_examples}\n\nQ: Classify the given ICU data sequence as either {task} or not-{task}:\n   {feature_string}\nA:"
        else:
            # Zero-shot prompt
            prompt = f"Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   {feature_string}\nA:"

        # Reformat prompt according to model-specific requirements
        formatted_prompt = apply_model_prompt_format(model_id, prompt)
        prompts.append(formatted_prompt)

    X_processed = pd.DataFrame({"text": prompts})

    logger.info(
        f"Converted {len(prompts)} samples to text prompt format for model '{model_id}'."
    )

    return X_processed, y
