# https://arxiv.org/abs/2407.18525
# zhu_2024_is_larger_always_better() implements the best setting prompt template used for mortality prediction on the MIMIC-IV dataset
# should always use 1 shot!

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
    task = info_dict.get("task")
    dataset = info_dict.get("dataset_name")
    model_id = info_dict.get("model_name")
    # num_shots = info_dict.get("shots")
    num_shots = 1  # Fixed value of num_shots=1 for this specific implementation
    mode = info_dict.get("mode") # train/val/test, few-shot examples are only used in validation and test mode

    # Set the data window based on task
    if task == "mortality":
        data_window = 25  # Fixed value for mortality task
    else:
        # For aki or sepsis, use the value from config
        data_window = info_dict.get("data_window")

    logger.info(
        f"'{mode}'-mode: Starting prompt preprocessing for model '{model_id}', dataset '{dataset}', task '{task}' with '{num_shots}' shots and '{data_window}' h data window'."
    )

    prompts = []
    X_in = X[0]  # input data
    y_in = y[0]  # labels

    # Extract few-shot examples if provided
    X_train = X[1] if len(X) > 1 else None
    y_train = y[1] if len(y) > 1 else None

    # Define the prompt template for clinical assessment
    main_prompt_template = PromptTemplate(
        input_variables=["task_description", "likelihood_description", "feature_descriptions_text", "patient_info", "patient_features_text", "data_window", "time_points_str"],
        template="""You are an experienced doctor in Intensive Care Unit (ICU) treatment.
    
    I will provide you with medical information from an Intensive Care Unit (ICU) visit of a patient, characterized by a fixed number of features.

    Present multiple hours data of a patient in one batch. Represent each feature within this data as a string of values, separated by commas.

    Your task is to assess the provided medical data and analyze the health records from ICU visits {task_description}. Please respond with only a floating-point number between 0 and 1, where a higher number suggests {likelihood_description}.

    In situations where the data does not allow for a reasonable conclusion, respond with the phrase "I do not know" without any additional explanation.

    {feature_descriptions_text}

    {few_shot_examples}
    Input information of a patient:
    {patient_info}
    The patient has data from {data_window} hours that occurred at {time_points_str}. 
    Details of the features for each visit are as follows:
    {patient_features_text}

    Please respond with only a floating-point number between 0 and 1, where a higher number suggests {likelihood_description}. Do not include any additional explanation.
    RESPONSE:
    """
    )

    # Few shot example template
    few_shot_example_template = """Here is an example of input information:
    Example #{index}:
    Input information of a patient:
    {patient_info}
    The patient has data from {data_window} hours that occurred at {time_points_str}. 
    Details of the features for each visit are as follows:
    {patient_features_text}
    RESPONSE:
    {label}
    """

    # Create task-specific description
    if task == "mortality":
        task_description = "to determine the likelihood of the patient not surviving their hospital stay"
        likelihood_description = "a greater likelihood of death"
    elif task == "aki":
        task_description = f"to determine the likelihood of the patient having acute kidney injury at the end of the data batch"
        likelihood_description = f"a greater likelihood of acute kidney injury"
    else:  # task == "sepsis"
        task_description = f"to determine the likelihood of the patient having sepsis at the end of the data batch"
        likelihood_description = f"a greater likelihood of sepsis"

    # Extract unique feature base names (e.g., "hr" from "hr_1")
    base_features = set()
    for col in X_in.columns:
        parts = col.split('_')
        if len(parts) > 1 and parts[1].isdigit():
            base_features.add(parts[0])
    
    # Join all feature descriptions into a single string
    feature_descriptions_text = prepare_feature_descriptions(base_features, X_in.columns)

    # Process each row to create individual prompts
    for idx, row in X_in.iterrows():
        # Format the patient data (using the helper function)
        patient_info, patient_features_text = format_patient_data(row, base_features, X_in.columns, data_window)

    # Generate the time points string
    time_points = list(range(data_window))
    time_points_str = ", ".join(map(str, time_points))

    # Generate few-shot examples if needed
    few_shot_examples_text = ""
    if num_shots > 0 and mode != "train" and X_train is not None and y_train is not None:
        # Randomly select examples from training data
        random_indices = np.random.choice(
            len(X_train), size=min(num_shots, len(X_train)), replace=False
        )
        
        few_shot_examples = []
        for i, idx in enumerate(random_indices):
            # Get example data
            example_row = X_train.iloc[idx]
            example_label = y_train.iloc[idx]
            
            # Format example patient data (using the helper function)
            example_patient_info, example_patient_features_text = format_patient_data(
                example_row, base_features, X_in.columns, data_window
            )

            # Format the example using the template
            example_text = few_shot_example_template.format(
                index=i+1,
                patient_info=example_patient_info,
                patient_features_text=example_patient_features_text,
                data_window=data_window,
                time_points_str=time_points_str,
                label=str(int(example_label["label"]))
            )
            few_shot_examples.append(example_text)
        
        # Join all examples
        few_shot_examples_text = "\n".join(few_shot_examples)
    
    # Create final prompt for this patient
    prompt = main_prompt_template.format(
        task_description=task_description,
        likelihood_description=likelihood_description,
        feature_descriptions_text=feature_descriptions_text,
        patient_info=patient_info,
        patient_features_text=patient_features_text,
        data_window=data_window,
        time_points_str=time_points_str,
        few_shot_examples=few_shot_examples_text
    )
    
    # Reformat prompt according to model-specific requirements
    formatted_prompt = apply_model_prompt_format(model_id, prompt)
    prompts.append(formatted_prompt)

    # Create dataframe with prompts
    X_processed = pd.DataFrame({"prompt": prompts})
    
    logger.info(
        f"Converted {len(prompts)} samples to text prompt format for model '{model_id}'."
    )

    return X_processed, y_in    

# --------------------------------
# Helper functions
# --------------------------------

def prepare_feature_descriptions(base_features, X_cols):
    """Prepare feature descriptions with name, unit of measurement, and reference range.
    
    Args:
        base_features: Set of base feature names
        X_cols: DataFrame columns to check for additional features
        
    Returns:
        Feature descriptions text as a formatted string
    """
    # Generate feature descriptions for the reference section
    feature_descriptions = []
    for feature in sorted(base_features):  # Sort for consistent order
        feature_name = get_feature_name(feature)
        uom = get_feature_uom(feature)
        range_values = get_feature_reference_range(feature)
        
        if range_values:  # Check if the range exists (not empty tuple)
            range_str = f"{range_values[0]} - {range_values[1]}"
            feature_descriptions.append(f"- {feature_name}: Unit: {uom}. Reference range: {range_str}.")
        else:
            feature_descriptions.append(f"- {feature_name}: Unit: {uom}. Reference range: /.")
    
    # Add weight and height to feature descriptions if they exist in the columns
    if "weight" in X_cols:
        weight_name = get_feature_name("weight")
        weight_uom = get_feature_uom("weight")
        feature_descriptions.append(f"- {weight_name}: Unit: {weight_uom}. Reference range: /.")
    
    if "height" in X_cols:
        height_name = get_feature_name("height")
        height_uom = get_feature_uom("height")
        feature_descriptions.append(f"- {height_name}: Unit: {height_uom}. Reference range: /.")
    
    # Join all feature descriptions into a single string
    return "\n".join(feature_descriptions)

def format_patient_data(row, base_features, X_cols, data_window):
    """Format patient data for prompting.
    
    Args:
        row: Patient data row
        base_features: Set of base feature names
        X_cols: DataFrame columns to extract feature columns from
        
    Returns:
        Tuple of (patient_info, patient_features_text)
    """
    # Extract patient demographic info
    sex = row.get('sex', 'unknown')
    age = row.get('age', 'unknown')
    patient_info = f"The patient is a {sex}, aged {age} years."
    
    # Format feature values
    patient_features = []
    
    # Process dynamic features (those with time series)
    for feature in sorted(base_features):
        # Get columns for this feature (e.g., hr_1, hr_2, etc.)
        feature_cols = [col for col in X_cols if col.startswith(f"{feature}_")]
                    
        # Filter to only include columns with numeric indices
        feature_cols = [col for col in feature_cols if col.split('_')[1].isdigit()]

        # Print warning if the number of feature columns doesn't match the data window
        if len(feature_cols) != data_window:
            logger.warning(
                f"Feature '{feature}' has {len(feature_cols)} columns, but expected {data_window} columns."
            )
                    
        # Sort columns by time point
        if feature_cols:  # Only sort if there are valid columns
            feature_cols.sort(key=lambda x: int(x.split('_')[1]))
        
        # Extract values for this feature across all time points
        values = [str(row[col]) for col in feature_cols]
        values_str = f'"{", ".join(values)}"'
        
        # Use the proper feature name from dictionary
        feature_name = get_feature_name(feature)
        patient_features.append(f"- {feature_name}: {values_str}")
    
    # Get number of time points from dynamic features
    num_timepoints = len(feature_cols) if 'feature_cols' in locals() else 6
    
    # Process static features (weight and height) - repeat value for all time points
    if 'weight' in row.index and not pd.isna(row['weight']):
        weight_value = str(row['weight'])
        weight_values = [weight_value] * num_timepoints
        weight_str = f'"{", ".join(weight_values)}"'
        weight_name = get_feature_name("weight")
        patient_features.append(f"- {weight_name}: {weight_str}")
        
    if 'height' in row.index and not pd.isna(row['height']):
        height_value = str(row['height'])
        height_values = [height_value] * num_timepoints
        height_str = f'"{", ".join(height_values)}"'
        height_name = get_feature_name("height")
        patient_features.append(f"- {height_name}: {height_str}")
    
    # Join patient features into a string
    patient_features_text = "\n".join(patient_features)
    
    return patient_info, patient_features_text