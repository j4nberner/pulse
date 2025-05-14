import json
import logging
import textwrap
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate

from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced

logger = logging.getLogger("PULSE_logger")


def zhu_2024b_one_shot_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess ICU data into prompts using few-shot format and centralized JSON prompt template.
    According to Zhu et al. 2024, "Is larger always better? Evaluating and prompting large language models for non-generative medical tasks"
    Paper: https://arxiv.org/abs/2407.18525
    Implements the best setting prompt template used for mortality prediction on the MIMIC-IV dataset

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
    num_shots = 1  # Fixed value of num_shots=1 for this specific implementation
    mode = info_dict.get("mode")  # train/val/test

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
            result_json = json.dumps(
                {
                    "diagnosis": label_text,
                    "probability": "<the probability of your estimation as a float (1.0 is very sure)>",
                    "explanation": "<a brief explanation for the prediction.>",
                },
                indent=2,
            )

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

    logger.debug(
        "Converted %s samples to text prompt format for model '%s'.",
        len(prompts),
        model_id,
    )

    return X_processed, y_in
