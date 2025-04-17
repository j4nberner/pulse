import logging
from typing import Dict, Any, Tuple
import pandas as pd

logger = logging.getLogger("PULSE_logger")


def apply_llama3_preprocessing(
    X: pd.DataFrame, y: pd.DataFrame, info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Llama3-specific preprocessing to the data.

    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.DataFrame): Label DataFrame
        info_dict (Dict[str, Any]): Dictionary containing dataset and task information

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Preprocessed features and labels
    """
    # Example preprocessing for Llama3
    logger.info(f"Applying Llama3-specific preprocessing")
    task = info_dict.get("task", "unknown_task")
    dataset = info_dict.get("dataset", "unknown_dataset")

    # Create text prompts from the features
    processed_X = []

    for _, row in X.iterrows():
        # Extract feature values and names
        feature_texts = []
        for col_name, value in row.items():
            feature_texts.append(f"{col_name}: {value}")

        # Format as a prompt
        prompt = f"Dataset: {dataset}\nTask: {task}\n"
        prompt += "Patient data:\n" + "\n".join(feature_texts)

        processed_X.append(prompt)

    # Convert to DataFrame with a text column
    X_processed = pd.DataFrame({"text": processed_X})

    logger.info(f"Converted {len(processed_X)} samples to text format for Llama3")
    X = X_processed

    return X, y
