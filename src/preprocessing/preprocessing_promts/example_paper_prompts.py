import logging
from typing import Dict, Any, Tuple
import pandas as pd
from src.util.model_util import apply_model_prompt_format

logger = logging.getLogger("PULSE_logger")


def example_paper_preprocessor(
    X: pd.DataFrame, y: pd.DataFrame, info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess input data into a text-based prompt format suitable for LLM models,
    as described in the example paper.

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
        # Compose patient data section
        patient_info = "\n".join(f"{col}: {val}" for col, val in row.items())

        # Construct full prompt
        prompt = f"Dataset: {dataset}\nTask: {task}\nPatient data:\n{patient_info}"

        # Reformat prompt according to model-specific requirements
        formatted_prompt = apply_model_prompt_format(model_id, prompt)
        prompts.append(formatted_prompt)

    X_processed = pd.DataFrame({"text": prompts})

    logger.info(
        f"Converted {len(prompts)} samples to text prompt format for model '{model_id}'."
    )

    return X_processed, y
