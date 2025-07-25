import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.agent_util import get_task_specific_content
from src.util.data_util import get_feature_name

logger = logging.getLogger("PULSE_logger")


class Zhu2024cAgent(PulseAgent):
    """
    Implementation of Zhu 2024c agent using multi-step approach.

    Steps:
    1. Analyze and summarize abnormal patient features
    2. Use this summary to produce a final prediction
    """

    def __init__(
        self,
        model: Any,
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics_tracker: Optional[Any] = None,
        **kwargs,
    ):
        # Initialize parent class with model
        super().__init__(
            model=model,
            task_name=task_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            metrics_tracker=metrics_tracker,
            **kwargs,
        )

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Initialize task content
        self.task_content = get_task_specific_content(self.task_name)

        # Define steps
        self._define_steps()

    def _update_task_specific_content(self) -> None:
        """Update task-specific content when task changes."""
        self.task_content = get_task_specific_content(self.task_name)

        # Redefine steps with updated task content
        self.steps = []  # Clear existing steps
        self._define_steps()

    # ------------------------------------------
    # Agent Step Methods
    # ------------------------------------------

    def _define_steps(self) -> None:
        """Define the reasoning steps for this agent."""
        # Step 1: Feature Analysis
        self.add_step(
            name="feature_analysis",
            system_message="You are an objective medical data analyst. Analyze the provided ICU time-series data patterns without bias toward any particular outcome. Most patients do not develop serious complications. Focus on factual observations and provide balanced analysis as plain text paragraphs.",
            prompt_template=self._create_summary_prompt_template(),
            input_formatter=self._process_patient_features,
            output_processor=None,
            parse_json=False,
        )

        # Step 2: Final Prediction
        self.add_step(
            name="final_prediction",
            system_message=None,
            prompt_template=self._create_final_prediction_prompt_template(),
            input_formatter=None,
            output_processor=None,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient's data through all reasoning steps."""
        # Update task context if needed (ensures task_content is current)
        if hasattr(self.model, "task_name") and hasattr(self.model, "dataset_name"):
            self.update_task_context(self.model.task_name, self.model.dataset_name)

        # Reset memory for this patient
        self.memory.reset()

        # Explicitly set the current sample ID
        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        logger.debug("Setting current sample ID: %s", sample_id)
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
            #TODO: @sophiafe Uncomment. Throws error
            # logger.debug(f"Feature summary: {feature_summary[:100]}...")
        except Exception as e:
            logger.error(f"Error in feature analysis step: {e}", exc_info=True)
            state["feature_analysis_output"] = "Error generating patient summary."

        # Step 2: Final Prediction using the summary from step 1
        try:
            summary = state.get("feature_analysis_output", "No summary available")
            final_prediction_result = self.run_step("final_prediction", summary, state)

            # Output is already parsed because parse_json=True in the step definition
            final_output = final_prediction_result["output"]

            # Get token metrics from agent memory (aggregated from all steps)
            all_steps = self.memory.samples.get(str(sample_id), [])
            total_input_tokens = sum(step.num_input_tokens for step in all_steps)
            total_output_tokens = sum(step.num_output_tokens for step in all_steps)
            total_token_time = sum(step.token_time for step in all_steps)
            total_infer_time = sum(step.infer_time for step in all_steps)

            # Return in same format as standard pipeline
            return {
                "generated_text": final_output,
                "token_time": total_token_time,
                "infer_time": total_infer_time,
                "num_input_tokens": total_input_tokens,
                "num_output_tokens": total_output_tokens,
            }

        except Exception as e:
            logger.error(f"Error in final prediction step: {e}", exc_info=True)
            state["final_output"] = "Error generating final prediction output."

    # ------------------------------------------
    # Prompt Template and Feature Processing Methods
    # ------------------------------------------

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
    {self.task_content['task_info_long']}
    """
            return prompt

        return format_summary_prompt

    def _create_final_prediction_prompt_template(self):
        """Create a function that formats the final_prediction prompt."""

        def format_final_prediction_prompt(summary, state):
            # Get summary from previous step
            summary = state.get("feature_analysis_output", "No summary available")

            prompt = f"""Based on the following patient summary, determine if the patient is likely to develop {self.task_content['complication_name']}:

Patient Summary:
{summary}

Please provide your assessment following the required format."""

            return prompt

        return format_final_prediction_prompt

    def _process_patient_features(
        self, state: Dict[str, Any], patient_data: pd.Series
    ) -> str:
        """Process patient features to extract abnormal values."""

        patient_df = pd.DataFrame([patient_data])
        logger.debug(f"Created patient_df with shape: {patient_df.shape}")

        # Extract base feature names
        base_features = set()
        for col in patient_data.index:
            if isinstance(col, str) and "_" in col and col.split("_")[-1].isdigit():
                base_name = "_".join(col.split("_")[:-1])
                base_features.add(base_name)

        # Categorize features
        try:
            categorized_features_df = self.preprocessor_advanced.categorize_features(
                df=patient_df,
                base_features=base_features,
                X_cols=patient_df.columns,
                num_categories=3,
                for_llm=True,
            )

            # Format abnormal features
            patient_categorized = categorized_features_df.iloc[0]
            abnormal_indices = patient_categorized[
                (patient_categorized == "too low") | (patient_categorized == "too high")
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

                # Get category description
                category = patient_categorized[feature]

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
