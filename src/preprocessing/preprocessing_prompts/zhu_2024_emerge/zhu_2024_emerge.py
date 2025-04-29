# https://dl.acm.org/doi/pdf/10.1145/3627673.3679582

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from src.util.model_util import apply_model_prompt_format
from src.util.data_util import get_feature_name, get_feature_uom, get_feature_reference_range

logger = logging.getLogger("PULSE_logger")

def zhu_2024_emerge_preprocessor(
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
    task = info_dict.get("task")
    dataset = info_dict.get("dataset_name")
    model_id = info_dict.get("model_name")
    mode = info_dict.get("mode") # train/val/test, few-shot examples are only used in validation and test mode

    logger.info(
        f"'{mode}'-mode: Starting prompt preprocessing for model '{model_id}', dataset '{dataset}', task '{task}'."
    )

    prompts = []
    X_in = X[0]  # input data
    y_in = y[0]  # labels

    # Define the prompt template for clinical assessment
    main_prompt_template = PromptTemplate(
        input_variables=[],
        template="""

    """

# Reformat prompt according to model-specific requirements
    formatted_prompt = apply_model_prompt_format(model_id, prompt)
    prompts.append(formatted_prompt)

    # Create dataframe with prompts
    X_processed = pd.DataFrame({"prompt": prompts})
    
    logger.info(
        f"Converted {len(prompts)} samples to text prompt format for model '{model_id}'."
    )

    return X_processed, y_in