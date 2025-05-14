import logging
import textwrap
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate

from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.data_util import get_feature_name

logger = logging.getLogger("PULSE_logger")


def zhu_2024c_categorization_summary_preprocessor(
    X: List[pd.DataFrame], y: List[pd.DataFrame], info_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess input data into a text-based prompt format suitable for LLM models,
    adhering to LangChain guidelines.

    Preprocess ICU data into prompts using zero-shot format, categorization of features into too low/high, summary generation and centralized JSON prompt template.
    Adapted from Zhu et al. 2024, "EMERGE: Enhancing Multimodal Electronic Health Records Predictive Modeling with Retrieval-Augmented Generation"
    Paper: https://dl.acm.org/doi/pdf/10.1145/3627673.3679582
    Implements a simplified version of the EMERGE framework without RAG and multi-model fusion

    Args:
        X (List[pd.DataFrame]): Input features.
        y (List[pd.DataFrame]): Target labels.
        info_dict (Dict[str, Any]): Additional task-specific information such as
                                    'task', 'dataset', and 'model_name'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed feature prompts and unchanged labels.
    """
    preprocessor_advanced = PreprocessorAdvanced()
    task = info_dict.get("task")
    dataset = info_dict.get("dataset_name")
    model_id = info_dict.get("model_name")
    mode = info_dict.get("mode")  # train/val/test

    logger.info(
        "'%s'-mode: Starting prompt preprocessing for model '%s', dataset '%s', task '%s'.",
        mode,
        model_id,
        dataset,
        task,
    )

    prompts = []
    X_in = X  # input data
    y_in = y  # labels

    # TODO: add base_feature extraction to preprocessor_advanced
    # Extract base feature names
    base_features = set()
    for col in X_in.columns:
        if "_" in col and col.split("_")[-1].isdigit():
            base_name = col.split("_")[0]
            base_features.add(base_name)

    # Define the prompt template for clinical assessment
    main_prompt_template = PromptTemplate(
        input_variables=[
            "complication_name",
            "prediction_description",
            "patient_features_categorized",
            "task_info",
        ],
        template=textwrap.dedent(
            """
As an experienced clinical professor, you have been provided with the following information to assist in summarizing a patient's health status:
- Potential abnormal features exhibited by the patient
- Definition and description of a common ICU complication: {complication_name}
Using this information, please create a concise and clear summary of the patient's health status. Your summary should be informative and beneficial for {prediction_description}. Please provide your summary directly without any additional explanations.

Potential abnormal features:  
{patient_features_categorized}

Disease definition and description: 
{task_info}
"""
        ).strip(),
    )

    # Create task-specific content with inline logic
    if task == "mortality":
        complication_name = "death"
        prediction_description = "the prediction of ICU mortality"
        task_info = "Mortality refers to the occurrence of death within a specific population and time period. In the context of ICU patients, the task involves analyzing information from the first 25 hours of a patient's ICU stay to predict whether the patient will survive the remainder of their stay. This prediction task supports early risk assessment and clinical decision-making in critical care settings."
    elif task == "aki":
        complication_name = "acute kidney injury"
        prediction_description = "prediction of the onset of acute kidney injury"
        task_info = "Acute kidney injury (AKI) is a subset of acute kidney diseases and disorders (AKD), characterized by a rapid decline in kidney function occurring within 7 days, with health implications. According to KDIGO criteria, AKI is diagnosed when there is an increase in serum creatinine to ≥1.5 times baseline within the prior 7 days, or an increase in serum creatinine by ≥0.3 mg/dL (≥26.5 µmol/L) within 48 hours, or urine output <0.5 mL/kg/h for 6–12 hours. The most common causes of AKI include sepsis, ischemia from hypotension or shock, and nephrotoxic exposures such as certain medications or contrast agents."
    else:  # task == "sepsis"
        complication_name = "sepsis"
        prediction_description = "prediction of the onset of sepsis"
        task_info = "Sepsis is a life-threatening condition characterized by organ dysfunction resulting from a dysregulated host response to infection. It is diagnosed when a suspected or confirmed infection is accompanied by an acute increase of two or more points in the patient's Sequential Organ Failure Assessment (SOFA) score relative to their baseline. The SOFA score evaluates six physiological parameters: the ratio of partial pressure of oxygen to the fraction of inspired oxygen, mean arterial pressure, serum bilirubin concentration, platelet count, serum creatinine level, and the Glasgow Coma Score. A complication of sepsis is septic shock, which is marked by a drop in blood pressure and elevated lactate levels. Indicators of suspected infection may include positive blood cultures or the initiation of antibiotic therapy."

    # Process all rows at once to categorize features
    categorized_features_df = preprocessor_advanced.categorize_features(
        X_in, base_features, X_in.columns
    )

    # Process each row to create individual prompts
    for idx, row in X_in.iterrows():
        # Get this patient's abnormal features
        patient_categorized = categorized_features_df.loc[idx]

        # Find abnormal features (too low = -1, too high = 1)
        abnormal_indices = patient_categorized[
            (patient_categorized == -1) | (patient_categorized == 1)
        ].index

        # Format abnormal features for the prompt
        abnormal_descriptions = []
        for feature in abnormal_indices:
            # Get feature name from dictionary
            feature_name = get_feature_name(feature)

            # Determine if too high or too low
            category = "too high" if patient_categorized[feature] == 1 else "too low"

            # Create description
            description = f"{feature_name} {category}"
            abnormal_descriptions.append(description)

        # Join abnormal features with commas
        patient_features_categorized = ", ".join(abnormal_descriptions)

        # If no abnormal features, indicate that
        if not abnormal_descriptions:
            patient_features_categorized = "No abnormal features detected."

        # Create final prompt for this patient
        prompt = main_prompt_template.format(
            complication_name=complication_name,
            prediction_description=prediction_description,
            patient_features_categorized=patient_features_categorized,
            task_info=task_info,
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
