import logging
from typing import Dict, Any, Tuple
import pandas as pd
from src.util.model_util import apply_model_prompt_format

logger = logging.getLogger("PULSE_logger")


def example_paper_preprocessor(
    X: pd.DataFrame, y: pd.DataFrame, info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply example advanced preprocessing to the data.

    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.DataFrame): Label DataFrame
        info_dict (Dict[str, Any]): Dictionary containing information to pass along.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed features and labels
    """
    # Example preprocessing for method in the paper
    logger.info(f"Applying Example paper specific preprocessing")
    task = info_dict.get("task", "unknown_task")
    dataset = info_dict.get("dataset", "unknown_dataset")
    model_id = info_dict.get("model_name", "unknown_model")

    # Create text prompts from the features
    processed_X = []

    for _, row in X.iterrows():
        # Extract feature values and names
        feature_texts = []
        for col_name, value in row.items():
            feature_texts.append(f"{col_name}: {value}")

        # Format as a prompt as declared in paper
        prompt = f"Dataset: {dataset}\nTask: {task}\n"
        prompt += "Patient data:\n" + "\n".join(feature_texts)

        # Reformat for specific LLM
        prompt = apply_model_prompt_format(model_id, prompt)

        processed_X.append(prompt)

    # Convert to DataFrame with a text column
    X_processed = pd.DataFrame({"text": processed_X})

    logger.info(f"Converted {len(processed_X)} samples to text format for Llama3")
    X = X_processed

    return X, y
