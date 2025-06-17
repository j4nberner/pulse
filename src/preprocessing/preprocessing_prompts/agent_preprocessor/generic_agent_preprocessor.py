import gc
import logging
from typing import Any, Dict, List

import pandas as pd
import torch

logger = logging.getLogger("PULSE_logger")


def generic_agent_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generic preprocessor for all agent-based processing.
    
    This preprocessor handles all agent types since the actual prompt creation
    and multi-step reasoning logic is implemented within the respective agent classes.

    Args:
        X: Input features list
        y: Target labels list
        info_dict: Additional information dictionary

    Returns:
        Dictionary with processed data and agent flag
    """
    task = info_dict.get("task")
    dataset = info_dict.get("dataset_name")
    model_name = info_dict.get("model_name")
    mode = info_dict.get("mode")

    logger.info(
        "'%s'-mode: Starting agent-based prompt preprocessing for model '%s', dataset '%s', task '%s'.",
        mode,
        model_name,
        dataset,
        task,
    )

    # Handle different input formats
    if isinstance(X, list) and len(X) > 0:
        X_in = X[0]
    else:
        X_in = X

    if isinstance(y, list) and len(y) > 0:
        y_in = y[0]
    else:
        y_in = y

    try:
        if mode == "test":
            logger.info("Setting up agent preprocessing for test mode")

            # Return the data with agent flag
            # The actual agent processing will happen in PulseLLMModel.generate()
            return {
                "X": X_in,
                "y": y_in,
                "is_agent": True,
            }
        else:
            logger.debug(f"Skipping agent processing for %s mode", mode)
            return {
                "X": X_in,
                "y": y_in,
                "is_agent": False,
            }

    finally:
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()