import json
import logging
import os
import gc
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from src.preprocessing.preprocessing_prompts.zhu_2024c_categorization_summary_agent.zhu_2024c_agent_class import (
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
    model_name = info_dict.get("model_name")
    mode = info_dict.get("mode")  # train/val/test
    model = info_dict.get("model_instance")  # Get the model instance
    output_dir = info_dict.get("output_dir", None)
    agent_log_dir = os.path.join(output_dir, "agent_logs")
    os.makedirs(agent_log_dir, exist_ok=True)

    logger.info(
        "'%s'-mode: Starting agent-based prompt preprocessing for model '%s', dataset '%s', task '%s'.",
        mode,
        model_name,
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

            # Check if we have a model instance
            if model is None:
                logger.error(
                    "No model instance provided, cannot run agent-based preprocessing"
                )
                return X_in, y_in

            # Create the agent with correct parameters
            agent = Zhu2024cAgent(
                model=model,
                task_name=task,
                dataset_name=dataset,
                output_dir=agent_log_dir,
            )

            # Process the data through the agent
            X_processed, y_processed = agent.process_batch(X_in, y_in)

            # Add flag to indicate this is a final prediction
            X_processed["is_agent_prediction"] = True

            # Store reference to the model in info_dict to ensure it's reused during evaluation
            # This prevents reloading during evaluate_single() method of the model without changing the non-agent pipeline
            info_dict["loaded_model"] = model

            # Also add essential info for evaluation
            # (Assuming diagnosis output is in JSON format with these fields)
            try:
                # Extract key fields from JSON predictions
                for idx in X_processed.index:
                    try:
                        text = X_processed.at[idx, "text"]
                        data = json.loads(text)
                        # Ensure probability is a float
                        if "probability" in data:
                            if isinstance(data["probability"], str):
                                data["probability"] = float(data["probability"])
                            X_processed.at[idx, "text"] = json.dumps(data)
                    except:
                        logger.warning(f"Failed to parse agent output as JSON for index {idx}")
            except Exception as e:
                logger.error(f"Error extracting agent prediction data: {e}")

            logger.debug(
                "Processed %s samples using agent-based approach for model '%s'.",
                len(X_processed),
                model_name,
            )
            return X_processed, y_processed

        else:
            # For train/val modes, simply skip processing
            logger.debug(f"Skipping agent processing for {mode} mode")
            return X_in, y_in

    finally:
        # Memory cleanup
        if "agent" in locals():
            del agent

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
