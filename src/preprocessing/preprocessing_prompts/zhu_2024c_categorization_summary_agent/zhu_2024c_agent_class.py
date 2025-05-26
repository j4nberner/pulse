import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from langchain.prompts import PromptTemplate

from src.preprocessing.preprocessing_prompts.agents.pulsetemplate_agent import (
    PulseTemplateAgent,
)
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import (
    PreprocessorAdvanced,
)
from src.util.data_util import get_feature_name
from src.util.model_util import extract_dict

logger = logging.getLogger("PULSE_logger")


class Zhu2024cAgent(PulseTemplateAgent):
    """
    Implementation of Zhu 2024c agent using multi-step approach.

    Steps:
    1. Analyze and summarize abnormal patient features
    2. Use this summary to produce a final diagnosis
    """

    def __init__(
        self,
        model: Any,  # Now accepts a model instance
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        # Initialize parent class with model
        super().__init__(
            model=model,
            task_name=task_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            **kwargs,
        )

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Initialize task content
        self.task_content = self._get_task_specific_content()

        # Define steps
        self._define_steps()

    def _get_task_specific_content(self) -> Dict[str, str]:
        """Get task-specific content for prompts."""
        task = self.task_name
        if task == "mortality":
            return {
                "complication_name": "death",
                "prediction_description": "the prediction of ICU mortality",
                "task_info": "Mortality refers to the occurrence of death within a specific population and time period. In the context of ICU patients, the task involves analyzing information from the first 25 hours of a patient's ICU stay to predict whether the patient will survive the remainder of their stay. This prediction task supports early risk assessment and clinical decision-making in critical care settings.",
            }
        elif task == "aki":
            return {
                "complication_name": "acute kidney injury",
                "prediction_description": "prediction of the onset of acute kidney injury",
                "task_info": "Acute kidney injury (AKI) is a subset of acute kidney diseases and disorders (AKD), characterized by a rapid decline in kidney function occurring within 7 days, with health implications. According to KDIGO criteria, AKI is diagnosed when there is an increase in serum creatinine to ≥1.5 times baseline within the prior 7 days, or an increase in serum creatinine by ≥0.3 mg/dL (≥26.5 µmol/L) within 48 hours, or urine output <0.5 mL/kg/h for 6–12 hours. The most common causes of AKI include sepsis, ischemia from hypotension or shock, and nephrotoxic exposures such as certain medications or contrast agents.",
            }
        elif task == "sepsis":
            return {
                "complication_name": "sepsis",
                "prediction_description": "prediction of the onset of sepsis",
                "task_info": "Sepsis is a life-threatening condition characterized by organ dysfunction resulting from a dysregulated host response to infection. It is diagnosed when a suspected or confirmed infection is accompanied by an acute increase of two or more points in the patient's Sequential Organ Failure Assessment (SOFA) score relative to their baseline. The SOFA score evaluates six physiological parameters: the ratio of partial pressure of oxygen to the fraction of inspired oxygen, mean arterial pressure, serum bilirubin concentration, platelet count, serum creatinine level, and the Glasgow Coma Score. A complication of sepsis is septic shock, which is marked by a drop in blood pressure and elevated lactate levels. Indicators of suspected infection may include positive blood cultures or the initiation of antibiotic therapy.",
            }
        return {}

    def _define_steps(self) -> None:
        """Define the reasoning steps for this agent."""
        # Step 1: Feature Analysis
        self.add_step(
            name="feature_analysis",
            system_message="You are a helpful assistant and medical professional that analyzes ICU time-series data and predicts whether a patient will develop a specific diagnosis. Descriptions of ICU complications are to be understood as an aid to understand the patient's condition but do not necessarily mean that the patient has developed said complications. Do not give recommendations. Provide your analysis as plain text paragraphs.",
            prompt_template=self._create_summary_prompt_template(),
            input_formatter=self._process_patient_features,
            output_processor=None,
            parse_json=False,
        )

        # Step 2: Diagnosis
        self.add_step(
            name="diagnosis",
            system_message=(
                "You are a helpful assistant and experienced medical professional analyzing ICU time-series data "
                "to determine the presence of a critical condition.\n\n"
                "Your response must strictly follow this format:\n"
                "Output a valid JSON object with three keys: 'diagnosis', 'probability' and 'explanation'.\n\n"
                "1. 'diagnosis' a string with either diganosis or not-diagnosis\n"
                "2. 'probability' a value between 0 and 1. where 0 means not-diagnosis and 1 means diagnosis.\n"
                "3. 'explanation' should be a string providing a brief explanation of your diagnosis.\n\n"
                "Here is a positive example:\n"
                "{\n"
                '  "diagnosis": "sepsis",\n'
                '  "probability": "0.76",\n'
                '  "explanation": "lactate is 4.2 mmol/L (above normal <2.0); blood pressure is low (MAP 62 mmHg), which are signs of sepsis."\n'
                "}\n\n"
                "Here is a negative example:\n"
                "{\n"
                '  "diagnosis": "not-sepsis",\n'
                '  "probability": "0.01",\n'
                '  "explanation": "lactate is 1.2 mmol/L (normal <2.0); blood pressure is normal (MAP 80 mmHg), which are not signs of sepsis."\n'
                "}\n\n"
                "Do not include any other text or explanations outside of the JSON object.\n"
                "Think about the probability of your prediction carefully before answering.\n"
            ),  # Use default benchmark system message for the prediction step
            prompt_template=self._create_diagnosis_prompt_template(),
            input_formatter=None,
            output_processor=None,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient's data through all reasoning steps."""
        # Reset memory for this patient
        self.memory.reset()

        # Explicitly set the current sample ID
        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        logger.debug(
            f"Setting current sample ID: {sample_id} (type: {type(sample_id)})"
        )
        self.memory.set_current_sample(sample_id)

        # Initialize state
        state = {
            "patient_data": patient_data,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
        }

        # Step 1: Feature Analysis (Summary)
        try:
            feature_result = self.run_step("feature_analysis", patient_data, state)

            # Store the actual output text in state
            if isinstance(feature_result["output"], dict):
                feature_summary = feature_result["output"].get("output", "")
            else:
                feature_summary = feature_result["output"]

            state["feature_analysis_output"] = feature_summary
            logger.debug(f"Feature summary: {feature_summary[:100]}...")
        except Exception as e:
            logger.error(f"Error in feature analysis step: {e}", exc_info=True)
            state["feature_analysis_output"] = "Error generating patient summary."

        # Step 2: Diagnosis using the summary from step 1
        try:
            # Get the summary from step 1
            summary = state.get("feature_analysis_output", "No summary available")

            # Run the diagnosis step with the summary as input
            diagnosis_result = self.run_step("diagnosis", summary, state)

            # Extract the output properly
            if isinstance(diagnosis_result["output"], dict):
                diagnosis_output = diagnosis_result["output"]
            else:
                diagnosis_output = diagnosis_result["output"]

            state["diagnosis_output"] = diagnosis_output
            logger.debug(f"Diagnosis result: {str(diagnosis_output)[:100]}...")

            # Extract JSON if needed
            if isinstance(diagnosis_output, str):
                try:
                    extracted = extract_dict(diagnosis_output)
                    if extracted:
                        state["diagnosis_data"] = extracted
                except Exception as e:
                    logger.warning(f"Error extracting diagnosis data: {e}")
            elif isinstance(diagnosis_output, dict):
                state["diagnosis_data"] = diagnosis_output
        except Exception as e:
            logger.error(f"Error in diagnosis step: {e}", exc_info=True)
            state["diagnosis_output"] = "Error generating diagnosis."

        # Return final output
        return self._create_final_output(state)

    def _create_summary_prompt_template(self):
        """Create a function that formats the summary prompt."""

        def format_summary_prompt(feature_data, state):
            prompt = f"""As an experienced clinical professor, you have been provided with the following information to assist in summarizing a patient's health status:
    - Potential abnormal features exhibited by the patient
    - Definition and description of a common ICU complication: {self.task_content['complication_name']}

    Using this information, please create a concise and clear summary of the patient's health status. Your summary should be informative and beneficial for {self.task_content['prediction_description']}. Please provide your summary directly without any additional explanations.

    Potential abnormal features:  
    {feature_data}

    Disease definition and description: 
    {self.task_content['task_info']}
    """
            return prompt

        return format_summary_prompt

    def _create_diagnosis_prompt_template(self):
        """Create a function that formats the diagnosis prompt."""

        def format_diagnosis_prompt(summary, state):
            # Get summary from previous step
            summary = state.get("feature_analysis_output", "No summary available")

            prompt = f"""Based on the following patient summary, determine if the patient is likely to develop {self.task_content['complication_name']}:

Patient Summary:
{summary}

Please provide your assessment following the required format."""

            return prompt

        return format_diagnosis_prompt

    def _process_patient_features(
        self, state: Dict[str, Any], patient_data: pd.Series
    ) -> str:
        """Process patient features to extract abnormal values."""
        # First convert patient_data to the right format if needed
        if not isinstance(patient_data, pd.Series):
            logger.debug(f"Patient data type: {type(patient_data)}")
            if isinstance(patient_data, pd.DataFrame):
                if not patient_data.empty:
                    patient_data = patient_data.iloc[0]
                    logger.debug("Converted DataFrame to Series")
                else:
                    return "No patient data available for analysis."
            elif isinstance(patient_data, dict):
                patient_data = pd.Series(patient_data)
                logger.debug("Converted dict to Series")
            elif hasattr(patient_data, "__iter__") and not isinstance(
                patient_data, str
            ):
                # Try to iterate to debug
                logger.debug("Iterating patient data:")
                for item in patient_data:
                    logger.debug(f"Item: {type(item)}")
                return "Patient data format not supported"
            else:
                logger.error(f"Unsupported patient data type: {type(patient_data)}")
                return "Error: Patient data format not supported"

        # Now that patient_data should be a Series, create a DataFrame for further processing
        patient_df = pd.DataFrame([patient_data])
        logger.debug(f"Created patient_df with shape: {patient_df.shape}")

        # # Create a DataFrame from the Series to use with categorize_features
        # patient_df = pd.DataFrame([patient_data])

        # Extract base feature names
        base_features = set()
        for col in patient_data.index:
            if isinstance(col, str) and "_" in col and col.split("_")[-1].isdigit():
                base_name = "_".join(col.split("_")[:-1])
                base_features.add(base_name)

        # Categorize features
        try:
            categorized_features_df = self.preprocessor_advanced.categorize_features(
                patient_df, base_features, patient_df.columns
            )

            # Format abnormal features
            patient_categorized = categorized_features_df.iloc[0]
            abnormal_indices = patient_categorized[
                (patient_categorized == -1) | (patient_categorized == 1)
            ].index

            # Format for prompt
            abnormal_descriptions = []
            for feature in abnormal_indices:
                # Get feature name
                if (
                    isinstance(feature, str)
                    and "_" in feature
                    and feature.split("_")[-1].isdigit()
                ):
                    feature_abbreviation = "_".join(feature.split("_")[:-1])
                else:
                    feature_abbreviation = feature

                # Get human-readable feature name from the dictionary
                feature_name = get_feature_name(feature_abbreviation)

                # Determine category
                category = (
                    "too high" if patient_categorized[feature] == 1 else "too low"
                )

                # Create description
                description = f"{feature_name} {category}"
                abnormal_descriptions.append(description)

            patient_features = ", ".join(abnormal_descriptions)
            if not abnormal_descriptions:
                patient_features = "No abnormal features detected."
        except Exception as e:
            logger.error(f"Error processing patient features: {e}", exc_info=True)
            patient_features = f"Error processing patient features: {str(e)}"

        return patient_features

    def _create_final_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final output from the agent's state."""
        # Get the diagnosis output from the second step
        diagnosis_output = state.get("diagnosis_output", "")

        # Ensure diagnosis_output is a string (crucial fix)
        if not isinstance(diagnosis_output, str):
            try:
                # Convert dict/object to string if needed
                diagnosis_output = json.dumps(diagnosis_output, indent=2)
            except:
                diagnosis_output = str(diagnosis_output)

        # Try to extract structured data if available
        try:
            parsed = extract_dict(diagnosis_output)
            if not parsed:
                parsed = {
                    "diagnosis": None,
                    "probability": 0.5,
                    "explanation": diagnosis_output,
                }
        except Exception as e:
            logger.warning(f"Failed to parse diagnosis output: {e}")
            parsed = {
                "diagnosis": None,
                "probability": 0.5,
                "explanation": diagnosis_output,
            }

        # Return the diagnosis output as the final result
        return {
            "output": diagnosis_output,  # Now guaranteed to be a string
            "parsed": parsed,
            "state": state,
        }
