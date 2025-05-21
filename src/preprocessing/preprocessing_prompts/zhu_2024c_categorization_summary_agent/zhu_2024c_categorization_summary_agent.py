import logging
import os
import gc
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from src.preprocessing.preprocessing_prompts.zhu_2024c_categorization_summary_agent.zhu_2024c_agent import (
    Zhu2024cAgent,
)

logger = logging.getLogger("PULSE_logger")


def zhu_2024c_categorization_summary_agent_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess input data using the improved Zhu 2024c agent with multistep reasoning.

    Args:
        X: Input features list (first item is current data, rest may be few-shot examples)
        y: Target labels list
        info_dict: Additional information dictionary with task, dataset, etc.

    Returns:
        Tuple of processed features and labels
    """
    task = info_dict.get("task")
    dataset = info_dict.get("dataset_name")
    model_id = info_dict.get("model_name")
    mode = info_dict.get("mode")  # train/val/test

    # Extract the model_id from the model yaml with fallback
    llm_model_id = info_dict.get("llm_model_id", "meta-llama/Llama-3.1-8B-Instruct")

    # Get output directory for logs from info_dict if available
    output_dir = info_dict.get("output_dir", None)
    if output_dir:
        # Create a specific directory for agent logs
        agent_log_dir = os.path.join(output_dir, "agent_logs")
        os.makedirs(agent_log_dir, exist_ok=True)
    else:
        agent_log_dir = None

    logger.info(
        "'%s'-mode: Starting agent-based prompt preprocessing for model '%s', dataset '%s', task '%s'.",
        mode,
        model_id,
        dataset,
        task,
    )

    # Handle different input formats
    if isinstance(X, list) and len(X) > 0:
        X_in = X[0]  # input data
    else:
        X_in = X  # assuming X is already a DataFrame

    if isinstance(y, list) and len(y) > 0:
        y_in = y[0]  # labels
    else:
        y_in = y  # assuming y is already a DataFrame

    try:
        # Only run the agent in test mode
        if mode == "test":
            logger.info("Running full agent pipeline for test mode")
            # Create the agent with correct parameters
            agent = Zhu2024cAgent(
                model_id=llm_model_id,
                task_name=task,
                dataset_name=dataset,
                output_dir=agent_log_dir,
            )

            # Process the data through the agent
            X_processed, y_processed = agent.process_batch(X_in, y_in)

            logger.debug(
                "Processed %s samples using agent-based approach for model '%s'.",
                len(X_processed),
                model_id,
            )
        else:
            # For train/val modes, simply skip processing
            logger.debug(f"Skipping agent processing for {mode} mode")
            X_processed = X_in
            y_processed = y_in

        return X_processed, y_processed

    finally:
        # Memory cleanup
        if "agent" in locals():
            del agent

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
