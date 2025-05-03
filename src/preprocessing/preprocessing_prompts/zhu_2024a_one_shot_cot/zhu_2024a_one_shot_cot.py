import logging
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import textwrap
from langchain.prompts import PromptTemplate

from src.preprocessing.preprocessing_advanced.preprocessing_advanced import (
    PreprocessorAdvanced,
)

logger = logging.getLogger("PULSE_logger")


def zhu_2024a_one_shot_cot_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess ICU data into prompts using few-shot format and centralized JSON prompt template.
    According to the paper "Prompting Large Language Models for Zero-Shot Clinical Prediction with Structured Longitudinal Electronic Health Record Data"
    Paper: https://arxiv.org/pdf/2402.01713"
    Implements the Chain of Thought prompt template used for mortality prediction on the MIMIC-IV dataset

        Args:
            X (List[pd.DataFrame]): Input features.
            y (List[pd.DataFrame]): Target labels.
            info_dict (Dict[str, Any]): Additional task-specific information such as
                                        'task', 'dataset', and 'model_name'.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Prompt DataFrame and label DataFrame.
    """
    preprocessor_advanced = PreprocessorAdvanced()

    task = info_dict.get("task")
    dataset = info_dict.get("dataset_name")
    model_id = info_dict.get("model_name")
    num_shots = 4  # Fixed value of num_shots=1 for this specific implementation
    mode = info_dict.get(
        "mode"
    )  # train/val/test, few-shot examples are only used in validation and test mode

    # Set the data window based on task
    if task == "mortality":
        data_window = 25  # Fixed value for mortality task
    else:
        # For aki or sepsis, use the value from config
        data_window = info_dict.get("data_window")

    logger.info(
        "'%s'-mode: Starting prompt preprocessing for model '%s', dataset '%s', task '%s' with '%s' shots and '%s' h data window'.",
        mode,
        model_id,
        dataset,
        task,
        num_shots,
        data_window,
    )

    prompts = []
    X_in = X[0]  # input data
    y_in = y[0]  # labels

    # Extract few-shot examples if provided
    X_train = X[1] if len(X) > 1 else None
    y_train = y[1] if len(y) > 1 else None

    # Define the prompt template for clinical assessment
    main_prompt_template = PromptTemplate(
        input_variables=[
            "task",
            "task_description",
            "likelihood_description",
            "feature_descriptions_text",
            "few_shot_examples",
            "patient_info",
            "patient_features_text",
            "data_window",
            "time_points_str",
        ],
        template=textwrap.dedent(
            """
    You are an experienced doctor in Intensive Care Unit (ICU) treatment.
    
    I will provide you with medical information from an Intensive Care Unit (ICU) visit of a patient, characterized by a fixed number of features.

    Present multiple hours data of a patient in one batch. Represent each feature within this data as a string of values, separated by commas.

    Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the {task_description}.

    {feature_descriptions_text}

    {few_shot_examples}

    Input information of a patient:
    {patient_info}
    The patient has data from {data_window} hours that occurred at {time_points_str}. 
    Details of the features for each visit are as follows:
    {patient_features_text}

    Please follow the Chain-of-Thought Analysis Process:

    1. Analyze the data step by step, For example:
        - Blood pressure shows a slight downward trend, indicating...
        - Heart rate is stable, suggesting...
        - Lab results indicate [specific condition or lack thereof]...
        - The patient underwent [specific intervention], which could mean...

    2. Make Intermediate Conclusions:
        - Draw intermediate conclusions from each piece of data. For example:
            - If a patient’s blood pressure is consistently low, it might indicate poor cardiovascular function.
            - The patient’s cardiovascular function is [conclusion].
            - [Other intermediate conclusions based on data].
    
    3. Aggregate the Findings:
        - After analyzing each piece of data, aggregate these findings to form a comprehensive view of the patient’s condition.
        - Summarize key points from the initial analysis and intermediate conclusions.
    
    Aggregated Findings:
        - Considering the patient’s vital signs and lab results, the overall health status is...
    
    4. Final Assessment:
        - Conclude with an assessment of the {task_description}.
        - Follow the instructions to provide output. The probability should be provided as a floating-point number between 0 and 1, where a higher number suggests a greater {task_description}.
        {{'diagnosis': '{task}' or 'not-{task}', 'probability': 0.XX, 'explanation': 'This is a brief summary of aggregated findings.'}}

    Example Chain-of-Thought Analysis:

    1. Analyze the data step by step:
        - Blood pressure shows a slight downward trend, which might indicate a gradual decline in cardiovascular stability.
        - Heart rate is stable, which is a good sign, suggesting no immediate cardiac distress.
        - Elevated white blood cell count could indicate an infection or an inflammatory process in the body.
        - Low potassium levels might affect heart rhythm and overall muscle function.
    
    2. Make Intermediate Conclusions:
        - The decreasing blood pressure could be a sign of worsening heart function or infection-related hypotension.
        - Stable heart rate is reassuring but does not completely rule out underlying issues.
        - Possible infection, considering the elevated white blood cell count.
        - Potassium levels need to be corrected to prevent cardiac complications.
    
    3. Aggregate the Findings:
        - The patient is possibly facing a cardiovascular challenge, compounded by an infection and electrolyte imbalance.
    
    Aggregated Findings:
        - Considering the downward trend in blood pressure, stable heart rate, signs of infection, and electrolyte imbalance, the patient’s overall health status seems to be moderately compromised.
    
    4. Final Assessment:
        {{'diagnosis': '{task}' or 'not-{task}', 'probability': 0.65, 'explanation': 'Moderately compromised condition due to decreasing blood pressure, stable heart rate, signs of infection and electrolyte imbalance.'}}

    RESPONSE:
    """
        ).strip(),
    )

    # Few shot example template
    few_shot_example_template = textwrap.dedent(
        """
    Here is an example of input information:
    Example #{index}:
    Input information of a patient:
    {patient_info}
    The patient has data from {data_window} hours that occurred at {time_points_str}. 
    Details of the features for each visit are as follows:
    {patient_features_text}
    RESPONSE:
    {result_json}

    """
    ).strip()

    # Create task-specific description
    # TODO: delete likelihood_description
    if task == "mortality":
        task_description = "likelihood of the patient not surviving their hospital stay"
        likelihood_description = "a greater likelihood of death"
    elif task == "aki":
        task_description = "likelihood of the patient having acute kidney injury at the end of the data batch"
        likelihood_description = "a greater likelihood of acute kidney injury"
    else:  # task == "sepsis"
        task_description = (
            "likelihood of the patient having sepsis at the end of the data batch"
        )
        likelihood_description = "a greater likelihood of sepsis"

    # Extract unique feature base names (e.g., "hr" from "hr_1")
    base_features = set()
    for col in X_in.columns:
        parts = col.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            base_features.add(parts[0])

    # Join all feature descriptions into a single string
    feature_descriptions_text = preprocessor_advanced.prepare_feature_descriptions(
        base_features, X_in.columns
    )

    # Generate the time points string
    time_points = list(range(data_window))
    time_points_str = ", ".join(map(str, time_points))

    # Generate few-shot examples if needed
    few_shot_examples_text = ""
    if (
        num_shots > 0
        and mode != "train"
        and X_train is not None
        and y_train is not None
    ):
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
            example_patient_info, example_patient_features_text = (
                preprocessor_advanced.format_patient_data(
                    example_row, base_features, X_in.columns, data_window
                )
            )

            label_value = float(example_label.values[0])
            label_text = task if label_value == 1 else f"not-{task}"
            result_json = {
                "diagnosis": label_text,
                "probability": label_value,
                "explanation": "This is a known example.",
            }

            # Format the example using the template
            example_text = few_shot_example_template.format(
                index=i + 1,
                patient_info=example_patient_info,
                patient_features_text=example_patient_features_text,
                data_window=data_window,
                time_points_str=time_points_str,
                result_json=result_json,
            )
            few_shot_examples.append(example_text)

        # Join all examples
        few_shot_examples_text = "\n\n".join(few_shot_examples)

    # Process each row to create individual prompts
    for idx, row in X_in.iterrows():
        # Format the patient data (using the helper function)
        patient_info, patient_features_text = preprocessor_advanced.format_patient_data(
            row, base_features, X_in.columns, data_window
        )

        # Create final prompt for this patient
        prompt = main_prompt_template.format(
            task=task,
            task_description=task_description,
            likelihood_description=likelihood_description,
            feature_descriptions_text=feature_descriptions_text,
            patient_info=patient_info,
            patient_features_text=patient_features_text,
            data_window=data_window,
            time_points_str=time_points_str,
            few_shot_examples=few_shot_examples_text,
        )

        prompts.append(prompt)

    # Create dataframe with prompts
    X_processed = pd.DataFrame({"text": prompts})

    logger.info(
        "Converted %s samples to text prompt format for model '%s'.",
        len(prompts),
        model_id,
    )

    return X_processed, y_in
