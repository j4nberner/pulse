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

    logger.info(
        f"Starting preprocessing for model '{model_id}' on dataset '{dataset}' and task '{task}'."
    )

    prompts = []
    X_in = X[0]  # input data
    y_in = y[0]  # labels
    X_train = X[1]  # few shot examples
    y_train = y[1]  # few shot examples

    feature_names = [get_feature_name(name) for name in X_in.columns.tolist()]
    uom = [get_feature_uom(name) for name in X_in.columns.tolist()]
    reference_ranges = [
        get_feature_reference_range(name) for name in X_in.columns.tolist()
    ]

    # Define the prompt template
    example_prompt = PromptTemplate(
        input_variables=["features", "label"],
        template="Q: Classify the given ICU data sequence as either {task} or not-{task}:\n   Features:\n{features}\nA: {label}",
    )

    # Define the prompt template for clinical assessment
    main_prompt_template = PromptTemplate(
        input_variables=["task"],
        template="""You are an experienced doctor in Intensive Care Unit (ICU) treatment.
    
    I will provide you with medical information from an Intensive Care Unit (ICU) visit of a patient, characterized by a fixed number of features.
    
    Present multiple hours data of a patient in one batch. Represent each feature within this data as a string of values, separated by commas.
    
    Your task is to assess the provided medical data and analyze the health records from ICU visits to determine the likelihood of the patient not surviving their hospital stay/having {task} at the end of the data batch. Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of death/{task}.
    
    In situations where the data does not allow for a reasonable conclusion, respond with the phrase "I do not know" without any additional explanation.
    
    •\tHeart Rate: Unit: bpm. Reference range: 60 - 100.
    •\tSystolic Blood Pressure: Unit: mmHg. Reference range: 90 - 120.
    •\tDiastolic Blood Pressure: Unit: mmHg. Reference range: 60 - 80.
    •\tMean Arterial Pressure (MAP): Unit: mmHg. Reference range: 65 - 100.
    •\tOxygen Saturation: Unit: %. Reference range: 95 - 100.
    •\tRespiratory Rate: Unit: breaths/min. Reference range: 12 - 20.
    •\tTemperature: Unit: °C. Reference range: 36.5 - 37.5.
    •\tpH Level: Unit: /. Reference range: 7.35 - 7.45.
    •\tPartial Pressure of Oxygen (PaO2): Unit: mmHg. Reference range: 75 - 100.
    •\tPartial Pressure of Carbon Dioxide (PaCO2): Unit: mmHg. Reference range: 35 - 45.
    •\tBase Excess: Unit: mmol/L. Reference range: -2 - 2.
    •\tBicarbonate: Unit: mmol/L. Reference range: 22 - 29.
    •\tFraction of Inspired Oxygen (FiO2): Unit: %. Reference range: 21 - 100.
    •\tInternational Normalized Ratio (INR): Unit: /. Reference range: 0.8 - 1.2.
    •\tPartial Thromboplastin Time (PTT): Unit: sec. Reference range: 25 - 35.
    •\tFibrinogen: Unit: mg/dL. Reference range: 200 - 400.
    •\tSodium: Unit: mmol/L. Reference range: 135 - 145.
    •\tPotassium: Unit: mmol/L. Reference range: 3.5 - 5.
    •\tChloride: Unit: mmol/L. Reference range: 96 - 106.
    •\tCalcium: Unit: mg/dL. Reference range: 8.5 - 10.5.
    •\tIonized Calcium: Unit: mmol/L. Reference range: 1.1 - 1.3.
    •\tMagnesium: Unit: mg/dL. Reference range: 1.7 - 2.2.
    •\tPhosphate: Unit: mg/dL. Reference range: 2.5 - 4.5.
    •\tGlucose: Unit: mg/dL. Reference range: 70 - 140.
    •\tLactate: Unit: mmol/L. Reference range: 0.5 - 2.
    •\tAlbumin: Unit: g/dL. Reference range: 3.5 - 5.
    •\tAlkaline Phosphatase: Unit: U/L. Reference range: 44 - 147.
    •\tAlanine Aminotransferase (ALT): Unit: U/L. Reference range: 7 - 56.
    •\tAspartate Aminotransferase (AST): Unit: U/L. Reference range: 10 - 40.
    •\tTotal Bilirubin: Unit: mg/dL. Reference range: 0.1 - 1.2.
    •\tDirect Bilirubin: Unit: mg/dL. Reference range: 0 - 0.3.
    •\tBlood Urea Nitrogen (BUN): Unit: mg/dL. Reference range: 7 - 20.
    •\tCreatinine: Unit: mg/dL. Reference range: 0.6 - 1.3.
    •\tUrine Output: Unit: mL/h. Reference range: 30 - 50.
    •\tHemoglobin: Unit: g/dL. Reference range: 12.5 - 17.5.
    •\tMean Corpuscular Hemoglobin (MCH): Unit: pg. Reference range: 27 - 33.
    •\tMean Corpuscular Hemoglobin Concentration (MCHC): Unit: g/dL. Reference range: 32 - 36.
    •\tMean Corpuscular Volume (MCV): Unit: fL. Reference range: 80 - 100.
    •\tPlatelets: Unit: 1000/µL. Reference range: 150 - 450.
    •\tWhite Blood Cell Count (WBC): Unit: 1000/µL. Reference range: 4 - 11.
    •\tNeutrophils: Unit: %. Reference range: 55 - 70.
    •\tBand Neutrophils: Unit: %. Reference range: 0 - 6.
    •\tLymphocytes: Unit: %. Reference range: 20 - 40.
    •\tC-Reactive Protein (CRP): Unit: mg/L. Reference range: 0 - 10.
    •\tMethemoglobin: Unit: %. Reference range: 0 - 2.
    •\tCreatine Kinase (CK): Unit: U/L. Reference range: 30 - 200.
    •\tCreatine Kinase-MB (CK-MB): Unit: ng/mL. Reference range: 0 - 5.
    •\tTroponin T: Unit: ng/mL. Reference range: 0 - 14.
    •\tHeight: Unit: cm. Reference range: /.
    •\tWeight: Unit: kg. Reference range: /.
    
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
    The patient is a male, aged 50.0 years. 
    The patient has data from 6 hours that occurred at 0, 1, 2, 3, 4, 5. 
    Details of the features for each visit are as follows:
    •\tHeart Rate: "75.34, 84.21, 92.45, 68.72, 87.15, 78.93"
    •\tSystolic Blood Pressure: "112.24, 105.78, 118.36, 94.52, 110.87, 115.19"
    •\tDiastolic Blood Pressure: "72.41, 65.29, 78.56, 74.83, 68.12, 76.97"
    •\tMean Arterial Pressure (MAP): "85.68, 78.45, 92.13, 72.87, 82.35, 89.54"
    •\tOxygen Saturation: "98.32, 96.45, 97.78, 95.23, 99.07, 97.64"
    •\tRespiratory Rate: "16.25, 14.78, 18.32, 13.46, 19.58, 15.91"
    •\tTemperature: "36.82, 37.14, 36.63, 37.35, 36.98, 37.05"
    •\tpH Level: "7.38, 7.41, 7.36, 7.43, 7.40, 7.37"
    •\tPartial Pressure of Oxygen (PaO2): "88.76, 92.35, 85.43, 98.67, 78.29, 94.14"
    •\tPartial Pressure of Carbon Dioxide (PaCO2): "38.42, 42.17, 36.85, 40.29, 44.53, 39.76"
    •\tBase Excess: "0.24, -1.38, 1.45, -2.17, 2.06, 0.58"
    •\tBicarbonate: "25.36, 23.48, 27.29, 24.75, 26.34, 28.17"
    •\tFraction of Inspired Oxygen (FiO2): "21.00, 24.35, 28.72, 21.00, 35.45, 30.18"
    •\tInternational Normalized Ratio (INR): "0.92, 1.03, 1.15, 0.95, 1.18, 1.07"
    •\tPartial Thromboplastin Time (PTT): "28.46, 32.19, 27.85, 30.42, 33.67, 29.38"
    •\tFibrinogen: "310.27, 275.64, 350.42, 240.93, 380.51, 290.75"
    •\tSodium: "138.47, 142.23, 136.85, 140.32, 139.58, 143.15"
    •\tPotassium: "3.94, 4.27, 3.73, 4.56, 4.08, 4.82"
    •\tChloride: "101.37, 98.45, 103.28, 100.56, 105.18, 97.82"
    •\tCalcium: "9.24, 9.82, 8.94, 10.12, 9.53, 9.07"
    •\tIonized Calcium: "1.22, 1.15, 1.27, 1.19, 1.24, 1.28"
    •\tMagnesium: "1.93, 2.12, 1.84, 2.05, 1.88, 2.17"
    •\tPhosphate: "3.24, 3.85, 2.93, 4.06, 3.52, 3.07"
    •\tGlucose: "95.42, 110.87, 85.29, 120.63, 105.48, 92.75"
    •\tLactate: "1.23, 0.85, 1.54, 1.08, 1.82, 0.73"
    •\tAlbumin: "4.23, 3.87, 4.56, 3.62, 4.05, 4.34"
    •\tAlkaline Phosphatase: "78.56, 95.32, 120.75, 60.43, 135.28, 85.96"
    •\tAlanine Aminotransferase (ALT): "25.47, 18.35, 35.82, 15.63, 42.18, 29.54"
    •\tAspartate Aminotransferase (AST): "22.34, 18.76, 32.45, 15.27, 38.93, 25.18"
    •\tTotal Bilirubin: "0.54, 0.82, 0.37, 1.05, 0.73, 0.45"
    •\tDirect Bilirubin: "0.12, 0.06, 0.17, 0.21, 0.09, 0.14"
    •\tBlood Urea Nitrogen (BUN): "15.32, 10.78, 18.45, 8.96, 16.23, 12.57"
    •\tCreatinine: "0.93, 1.14, 0.73, 1.25, 0.86, 1.07"
    •\tUrine Output: "45.36, 38.87, 42.53, 35.29, 48.64, 40.12"
    •\tHemoglobin: "14.53, 13.82, 15.24, 13.27, 14.86, 16.08"
    •\tMean Corpuscular Hemoglobin (MCH): "29.45, 31.27, 28.63, 32.15, 30.56, 27.84"
    •\tMean Corpuscular Hemoglobin Concentration (MCHC): "33.35, 34.22, 35.08, 32.47, 34.75, 33.63"
    •\tMean Corpuscular Volume (MCV): "88.24, 92.75, 85.36, 95.48, 90.32, 87.69"
    •\tPlatelets: "230.46, 310.85, 180.27, 270.53, 350.92, 210.39"
    •\tWhite Blood Cell Count (WBC): "7.53, 6.28, 8.57, 5.56, 9.24, 6.83"
    •\tNeutrophils: "60.47, 65.23, 58.84, 67.35, 62.18, 59.76"
    •\tBand Neutrophils: "2.34, 3.45, 1.27, 4.18, 2.56, 3.09"
    •\tLymphocytes: "32.45, 25.82, 35.47, 28.63, 30.92, 33.18"
    •\tC-Reactive Protein (CRP): "3.56, 1.83, 5.24, 2.53, 8.07, 4.28"
    •\tMethemoglobin: "0.52, 1.05, 0.37, 1.58, 0.84, 0.43"
    •\tCreatine Kinase (CK): "95.36, 120.47, 80.28, 150.93, 110.52, 65.79"
    •\tCreatine Kinase-MB (CK-MB): "2.53, 1.86, 3.24, 0.95, 2.87, 1.58"
    •\tTroponin T: "6.24, 3.17, 8.45, 2.09, 9.68, 5.32"
    •\tHeight: "183.00, 183.00, 183.00, 183.00, 183.00, 183.00"
    •\tWeight: "78.50, 78.50, 78.50, 78.50, 78.50, 78.50"
    
    
    Please respond with only a floating-point number between 0 and 1, where a higher number suggests a greater likelihood of {task}. Do not include any additional explanation.
    RESPONSE:"""
    )