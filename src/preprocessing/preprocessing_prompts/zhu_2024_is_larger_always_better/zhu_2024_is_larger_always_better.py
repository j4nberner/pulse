# https://arxiv.org/abs/2407.18525

import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from src.util.model_util import apply_model_prompt_format
from src.util.data_util import get_feature_name, get_feature_uom, get_feature_reference_range

logger = logging.getLogger("PULSE_logger")

def zhu_2024_is_larger_always_better_preprocessor(
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
    mode = info_dict.get("mode", "train") # Few-shot examples are only used in test mode

    logger.info(
        f"Starting prompt preprocessing for model '{model_id}' on dataset '{dataset}' and task '{task}'."
    )

    prompts = []
    X_in = X[0]  # input data
    # X_in = X[0].filter(regex="^(?!.*_na$)") # Remove all columns with _na suffix
    y_in = y[0]  # labels
    # X_train = X[1]  # few shot examples
    # y_train = y[1]  # few shot examples
    
    # Extract unique feature base names (e.g., "hr" from "hr_1")
    base_features = set()
    for col in X_in.columns:
        parts = col.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            base_features.add(parts[0])
    
    # Generate feature descriptions for the reference section
    feature_descriptions = []
    for feature in sorted(base_features):  # Sort for consistent order
        feature_name = get_feature_name(feature)
        uom = get_feature_uom(feature)
        range_values = get_feature_reference_range(feature)
        
        if range_values:  # Check if the range exists (not empty tuple)
            range_str = f"{range_values[0]} - {range_values[1]}"
            feature_descriptions.append(f"•\t{feature_name}: Unit: {uom}. Reference range: {range_str}.")
        else:
            feature_descriptions.append(f"•\t{feature_name}: Unit: {uom}. Reference range: /.")
    
    # Add weight and height to feature descriptions if they exist in the dictionary
    if "weight" in X_in.columns:
        weight_name = get_feature_name("weight")
        weight_uom = get_feature_uom("weight")
        feature_descriptions.append(f"•\t{weight_name}: Unit: {weight_uom}. Reference range: /.")
    
    if "height" in X_in.columns:
        height_name = get_feature_name("height")
        height_uom = get_feature_uom("height")
        feature_descriptions.append(f"•\t{height_name}: Unit: {height_uom}. Reference range: /.")
    
    # Join all feature descriptions into a single string
    feature_descriptions_text = "\n".join(feature_descriptions)

    # Define the prompt template
    example_prompt = PromptTemplate(
        input_variables=["features", "label"],
        template="Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   Features:\n{features}\nA: {label}",
    )

    # Define the prompt template for clinical assessment
    main_prompt_template = PromptTemplate(
        input_variables=["task", "feature_descriptions_text", "patient_info", "patient_features_text"],
        template="""You are an experienced doctor in Intensive Care Unit (ICU) treatment.
    
    I will provide you with medical information from an Intensive Care Unit (ICU) visit of a patient, characterized by a fixed number of features.
    
    Present multiple hours data of a patient in one batch. Represent each feature within this data as a string of values, separated by commas.
    
    Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay/having {task} at the end of the data batch. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death/{task}.
    
    In situations where the data does not allow for a reasonable conclusion, respond with the phrase "I do not know" without any additional explanation.
    
    {feature_descriptions_text}
    
    Here is an example of input information:
    Example #1:
    Input information of a patient:
    The patient is a female, aged 52 years. 
    The patient has data from 6 hours that occurred at 0, 1, 2, 3, 4, 5. 
    Details of the features for each visit are as follows:
    •\tHeart Rate: "73, 77, 86, 81, 95, 92"
    •\t…
    RESPONSE:
    0.3
    
    Input information of a patient:
    {patient_info}
    The patient has data from 6 hours that occurred at 0, 1, 2, 3, 4, 5. 
    Details of the features for each visit are as follows:
    {patient_features_text}
    
    
    Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of {task}. Do not include any additional explanation.
    RESPONSE:"""
    )

    # Process each row to create individual prompts
    for idx, row in X_in.iterrows():
        # Extract patient demographic info
        sex = row.get('sex', 'unknown')
        age = row.get('age', 'unknown')
        patient_info = f"The patient is a {sex}, aged {age} years."
        
        # Format feature values for this patient
        patient_features = []
        
        # Process dynamic features (those with time series)
        for feature in sorted(base_features):
            # Get columns for this feature (e.g., hr_1, hr_2, etc.)
            feature_cols = [col for col in X_in.columns if col.startswith(f"{feature}_")]
                        
            # Filter to only include columns with numeric indices
            feature_cols = [col for col in feature_cols if col.split('_')[1].isdigit()]
                        
            # Sort columns by time point
            if feature_cols:  # Only sort if there are valid columns
                feature_cols.sort(key=lambda x: int(x.split('_')[1]))
            
            # Extract values for this feature across all time points
            values = [str(row[col]) for col in feature_cols]
            values_str = f'"{", ".join(values)}"'
            
            # Use the proper feature name from dictionary
            feature_name = get_feature_name(feature)
            patient_features.append(f"•\t{feature_name}: {values_str}")
        
        # Get number of time points from dynamic features
        num_timepoints = len(feature_cols) if 'feature_cols' in locals() else 6  # Default to 6 if no dynamic features
        
        # Process static features (weight and height) - repeat value for all time points
        if 'weight' in X_in.columns and not pd.isna(row['weight']):
            weight_value = str(row['weight'])
            weight_values = [weight_value] * num_timepoints
            weight_str = f'"{", ".join(weight_values)}"'
            weight_name = get_feature_name("weight")
            patient_features.append(f"•\t{weight_name}: {weight_str}")
            
        if 'height' in X_in.columns and not pd.isna(row['height']):
            height_value = str(row['height'])
            height_values = [height_value] * num_timepoints
            height_str = f'"{", ".join(height_values)}"'
            height_name = get_feature_name("height")
            patient_features.append(f"•\t{height_name}: {height_str}")
        
        # Join patient features into a string
        patient_features_text = "\n".join(patient_features)
        
        # Create final prompt for this patient
        prompt = main_prompt_template.format(
            task=task,
            feature_descriptions_text=feature_descriptions_text,
            patient_info=patient_info,
            patient_features_text=patient_features_text
        )
        
        # Reformat prompt according to model-specific requirements
        formatted_prompt = apply_model_prompt_format(model_id, prompt)
        prompts.append(formatted_prompt)
        
    # Create dataframe with prompts
    X_processed = pd.DataFrame({"prompt": prompts})
    
    logger.info(
        f"Converted {len(prompts)} samples to text prompt format for model '{model_id}'."
    )
    
    # Print the first prompt for testing purposes
    if len(prompts) > 0:
        logger.info("Sample prompt (first row of data):")
        logger.info(X_processed["prompt"].iloc[0])

    return X_processed, y_in    