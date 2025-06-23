import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import (
    PreprocessorAdvanced,
)
from src.util.agent_util import (
    create_error_response,
    extract_confidence,
    extract_requested_labs,
    filter_na_columns,
    format_clinical_data,
    format_clinical_text,
    get_available_labs,
    get_available_vitals,
    get_lab_groups_available,
    get_monitoring_period_hours,
    get_task_specific_content,
    validate_features,
    validate_lab_request,
)
from src.util.data_util import (
    get_all_feature_groups,
    get_feature_group_keys,
    get_feature_group_title,
    get_feature_name,
)

logger = logging.getLogger("PULSE_logger")


class ClinicalWorkflowAgent(PulseAgent):
    """
    Advanced agent that mimics clinical decision-making workflow.

    Workflow:
    1. Initial assessment with vital signs only
    2. Iterative lab ordering based on clinical reasoning
    3. Prediction updates after each new information
    4. Stops when confidence threshold is reached or max iterations
    """

    def __init__(
        self,
        model: Any,
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics_tracker: Optional[Any] = None,
        confidence_threshold: float = 0.85,
        max_iterations: int = 5,
        min_iterations: int = 1,
        **kwargs,
    ):
        super().__init__(
            model=model,
            task_name=task_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            metrics_tracker=metrics_tracker,
            **kwargs,
        )

        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_iterations = min_iterations
        self.task_content = get_task_specific_content(self.task_name)

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Use data_util feature groups
        self.vital_signs = set(get_feature_group_keys("vitals"))
        self.lab_groups = get_all_feature_groups()

        self._define_steps()

    def _define_steps(self) -> None:
        """Define the clinical workflow steps."""

        # Step 1: Initial vital signs assessment
        self.add_step(
            name="initial_assessment",
            system_message="You are an experienced ICU physician making an initial patient assessment based on vital signs. Provide clinical reasoning and initial probability estimate.",
            prompt_template=self._create_initial_assessment_template(),
            input_formatter=self._format_vital_signs,
            output_processor=None,
            parse_json=True,
        )

        # Step 2: Lab ordering decision
        self.add_step(
            name="lab_ordering",
            system_message="You are deciding which additional tests to order based on your clinical assessment. Focus on the most clinically relevant tests that will help confirm or rule out your differential diagnosis.",
            prompt_template=self._create_lab_ordering_template(),
            input_formatter=self._format_state_only,
            output_processor=None,
            parse_json=True,
        )

        # Step 3: Updated assessment with new labs
        self.add_step(
            name="updated_assessment",
            system_message="You are updating your clinical assessment with new laboratory results. Integrate this information with your previous assessment.",
            prompt_template=self._create_updated_assessment_template(),
            input_formatter=self._format_lab_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 4: Final decision - uses default system message from model_util.py
        self.add_step(
            name="final_prediction",
            system_message=None,  # This will use the default system message
            prompt_template=self._create_final_prediction_template(),
            input_formatter=self._format_final_data,
            output_processor=None,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient through the clinical workflow."""
        # Reset memory
        self.memory.reset()

        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        self.memory.set_current_sample(sample_id)

        # Store original data with _na columns for uncertainty analysis
        original_patient_data = patient_data.copy()

        # Filter out _na columns
        patient_data = filter_na_columns(patient_data)

        # Initialize state
        state = {
            "patient_data": patient_data,
            "original_patient_data": original_patient_data,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "available_features": set(patient_data.index),
            "used_features": set(),
            "assessment_history": [],
            "current_confidence": 0.0,
            "iteration": 0,
        }

        try:
            # Step 1: Initial assessment with vital signs
            initial_result = self.run_step("initial_assessment", patient_data, state)
            initial_output = initial_result["output"]

            # Handle error cases from input formatter
            if (
                isinstance(initial_output, str)
                and "Error formatting input" in initial_output
            ):
                logger.error("Initial assessment formatting failed: %s", initial_output)
                return create_error_response("Input formatting failed")

            # Check if parsing was successful
            if initial_output.get("diagnosis") == "unknown":
                logger.warning(
                    "Initial assessment JSON parsing likely failed - got fallback response"
                )

            # Extract confidence with fallback logic
            confidence_normalized = extract_confidence(initial_output)

            state["assessment_history"].append(
                {
                    "step": "initial_assessment",
                    "reasoning": initial_output.get(
                        "explanation", initial_output.get("reasoning", "")
                    ),
                    "probability": initial_output.get("probability", 50),
                    "confidence": initial_output.get("confidence", 50),
                }
            )

            state["current_confidence"] = confidence_normalized
            state["used_features"].update(
                get_available_vitals(state["available_features"])
            )

            logger.info(
                "Initial assessment complete. Confidence: %.3f, Threshold: %.3f",
                confidence_normalized,
                self.confidence_threshold,
            )

            # Iterative lab ordering and assessment
            while (
                state["current_confidence"] < self.confidence_threshold
                or state["iteration"] < self.min_iterations
            ) and state["iteration"] < self.max_iterations:

                state["iteration"] += 1
                logger.info(
                    "Starting iteration %d, current confidence: %.3f",
                    state["iteration"],
                    state["current_confidence"],
                )

                # Step 2: Decide on lab orders
                lab_order_result = self.run_step("lab_ordering", None, state)
                lab_order_output = lab_order_result["output"]

                # Extract requested labs with validation
                requested_labs = extract_requested_labs(lab_order_output)
                if requested_labs is None:
                    break

                if not requested_labs:
                    logger.info("No additional labs requested, stopping iteration")
                    break

                logger.info("Requested labs: %s", requested_labs)

                # Validate requested labs against data_util
                valid_requested_labs = validate_features(requested_labs)
                if not valid_requested_labs:
                    logger.info("No valid labs requested, stopping iteration")
                    break

                # Additional validation: ensure no already-used features are requested
                valid_requested_labs = validate_lab_request(valid_requested_labs, state)
                if not valid_requested_labs:
                    logger.info(
                        "No new labs requested (all already analyzed), stopping iteration"
                    )
                    break

                logger.info("Valid requested labs: %s", valid_requested_labs)

                # Get available requested labs
                available_labs = get_available_labs(
                    valid_requested_labs, state["available_features"]
                )

                if not available_labs:
                    logger.info(
                        "No requested labs available in data, stopping iteration"
                    )
                    break

                logger.info("Available labs for analysis: %s", available_labs)

                # Add safety check: ensure we're not re-using already used features
                already_used = available_labs.intersection(state["used_features"])
                if already_used:
                    logger.warning(
                        "Attempted to re-use already analyzed features: %s",
                        already_used,
                    )
                    available_labs = available_labs - state["used_features"]
                    if not available_labs:
                        logger.info(
                            "No new labs after removing duplicates, stopping iteration"
                        )
                        break

                state["used_features"].update(available_labs)

                # Step 3: Updated assessment with new labs
                updated_result = self.run_step(
                    "updated_assessment", available_labs, state
                )
                updated_output = updated_result["output"]

                state["assessment_history"].append(
                    {
                        "step": f"iteration_{state['iteration']}",
                        "new_tests": available_labs,
                        "reasoning": updated_output.get(
                            "explanation", updated_output.get("reasoning", "")
                        ),
                        "probability": updated_output.get("probability", 50),
                        "confidence": updated_output.get("confidence", 50),
                    }
                )

                # Extract confidence with fallback logic
                confidence_normalized = extract_confidence(updated_output)
                state["current_confidence"] = confidence_normalized
                logger.info(
                    "Updated assessment complete. New confidence: %.3f",
                    confidence_normalized,
                )

            logger.info(
                "Workflow complete after %d iterations. Final confidence: %.3f",
                state["iteration"],
                state["current_confidence"],
            )

            # Step 4: Final decision using default system message
            final_result = self.run_step("final_prediction", None, state)
            final_output = final_result["output"]

            # Aggregate token metrics
            all_steps = self.memory.samples.get(str(sample_id), [])
            total_input_tokens = sum(step.num_input_tokens for step in all_steps)
            total_output_tokens = sum(step.num_output_tokens for step in all_steps)
            total_token_time = sum(step.token_time for step in all_steps)
            total_infer_time = sum(step.infer_time for step in all_steps)

            return {
                "generated_text": final_output,
                "token_time": total_token_time,
                "infer_time": total_infer_time,
                "num_input_tokens": total_input_tokens,
                "num_output_tokens": total_output_tokens,
            }

        except Exception as e:
            logger.error("Error in clinical workflow: %s", e, exc_info=True)
            return {
                "generated_text": {"error": f"Clinical workflow error: {str(e)}"},
                "token_time": 0,
                "infer_time": 0,
                "num_input_tokens": 0,
                "num_output_tokens": 0,
            }

    def _format_state_only(self, input_data: Any, state: Dict[str, Any]) -> None:
        """Input formatter for lab ordering - returns None since template uses only state."""
        return None

    def _format_vital_signs(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format vital signs data for initial assessment using aggregate_feature_windows."""
        patient_data = input_data
        original_patient_data = state["original_patient_data"]
        vitals_keys = get_feature_group_keys("vitals")
        available_vitals = {
            vital
            for vital in vitals_keys
            if any(feat.startswith(vital) for feat in patient_data.index)
        }

        return format_clinical_data(
            patient_data=patient_data,
            feature_keys=available_vitals,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=True,
            include_temporal_patterns=True,
            include_uncertainty=True,
            original_patient_data=original_patient_data,
        )

    def _format_lab_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format lab data for updated assessment."""
        available_labs = input_data
        patient_data = state["patient_data"]
        original_patient_data = state["original_patient_data"]
        return format_clinical_data(
            patient_data=patient_data,
            feature_keys=available_labs,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=False,
            include_temporal_patterns=True,
            include_uncertainty=True,
            original_patient_data=original_patient_data,
        )

    def _format_final_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format comprehensive data for final decision."""
        patient_data = state["patient_data"]
        original_patient_data = state["original_patient_data"]

        # Get demographics
        demographics = {}
        if "age" in patient_data.index:
            demographics["age"] = patient_data["age"]
        if "sex" in patient_data.index:
            demographics["sex"] = patient_data["sex"]
        if "weight" in patient_data.index:
            demographics["weight"] = patient_data["weight"]

        # Get all used clinical data with temporal patterns and uncertainty
        all_clinical_data = format_clinical_data(
            patient_data=patient_data,
            feature_keys=state["used_features"],
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=False,
            include_temporal_patterns=True,
            include_uncertainty=True,
            original_patient_data=original_patient_data,
        )

        return {
            "demographics": demographics,
            "clinical_data": all_clinical_data,
            "assessment_history": state["assessment_history"],
        }

    def _create_initial_assessment_template(self):
        """Template for initial vital signs assessment."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            vital_signs = formatted_data["vital_signs"]

            # Get monitoring period from the data
            monitoring_hours = get_monitoring_period_hours(state["patient_data"])

            # Format patient demographics
            demo_text = []
            if "age" in demographics:
                demo_text.append(f"Age: {demographics['age']} years")
            if "sex" in demographics:
                demo_text.append(f"Sex: {demographics['sex']}")
            if "weight" in demographics:
                demo_text.append(f"Weight: {demographics['weight']} kg")

            demographics_str = (
                ", ".join(demo_text) if demo_text else "Demographics: Not available"
            )

            # Format vital signs using helper method
            vitals_text = format_clinical_text(vital_signs)
            vitals_str = "\n".join(vitals_text)

            return f"""You are evaluating an ICU patient for risk of {self.task_content['complication_name']}.

Patient Demographics:
{demographics_str}

Current vital signs (over {monitoring_hours}-hour monitoring period):
{vitals_str}

Clinical Context:
{self.task_content['task_info']}

Based on these vital signs, patient demographics, and temporal patterns (where available), provide your initial clinical assessment.

Pay attention to temporal patterns in the data:
- Trend direction (stable, slowly/moderately/rapidly increasing/decreasing)
- Value normality (normal, slightly/very low/high)
- Clinical significance of combined trend and abnormality patterns

Respond in JSON format:
{{
    "diagnosis": "preliminary-{self.task_content['task_name']}-risk",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['task_name']} will not occur and 100 means {self.task_content['task_name']} will definitely occur),
    "explanation": "Your detailed clinical reasoning including differential diagnosis and temporal pattern assessment (MAX 200 words)",
    "confidence": XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment)
}}

IMPORTANT: With only vital signs available, confidence should typically be 50-70. Higher confidence (>75) should only be used when clinical picture is very clear."""

        return format_prompt

    def _create_lab_ordering_template(self):
        """Template for deciding which labs to order."""

        def format_prompt(_, state):
            previous_assessment = state["assessment_history"][-1]

            # Get available tests by clinical group using data_util
            available_by_group = get_lab_groups_available(state["available_features"])

            # Filter out already used features and organize already used ones
            filtered_available = {}
            already_used_by_group = {}
            total_unused_features = 0

            for group_name, features in available_by_group.items():
                unused_features = [
                    f for f in features if f not in state["used_features"]
                ]
                used_features = [f for f in features if f in state["used_features"]]

                if unused_features:
                    filtered_available[group_name] = unused_features
                    total_unused_features += len(unused_features)

                if used_features:
                    already_used_by_group[group_name] = used_features

            # Debug logging to track filtering
            logger.debug("Used features so far: %s", list(state["used_features"]))
            logger.debug("Total unused features available: %d", total_unused_features)
            for group_name, features in filtered_available.items():
                logger.debug(
                    "Group %s: %d unused features: %s",
                    group_name,
                    len(features),
                    features,
                )

            # If no unused features available, stop ordering
            if total_unused_features == 0:
                logger.info(
                    "No unused lab features available - all tests already ordered"
                )
                return f"""All available laboratory tests have already been ordered and analyzed. 

Previous Assessment:
- Reasoning: {previous_assessment.get('explanation', previous_assessment.get('reasoning', ''))}
- Current probability: {int(float(previous_assessment['probability']) * 100) if isinstance(previous_assessment['probability'], (int, float)) and previous_assessment['probability'] <= 1 else int(previous_assessment['probability'])}%
- Current confidence: {int(float(previous_assessment.get('confidence', previous_assessment['probability'])) * 100) if isinstance(previous_assessment.get('confidence', previous_assessment['probability']), (int, float)) and previous_assessment.get('confidence', previous_assessment['probability']) <= 1 else int(previous_assessment.get('confidence', previous_assessment['probability']))}%

Since no additional tests are available, respond with an empty test list.

Respond in JSON format:
{{
    "diagnosis": "lab-ordering-complete",
    "probability": {int(float(previous_assessment['probability']) * 100) if isinstance(previous_assessment['probability'], (int, float)) and previous_assessment['probability'] <= 1 else int(previous_assessment['probability'])},
    "explanation": "All available laboratory tests have been ordered and analyzed. No additional tests are available.",
    "requested_tests": []
}}"""

            # Format test list showing only unused features
            test_list = []
            for group_name, features in filtered_available.items():
                group_title = get_feature_group_title(group_name)
                test_list.append(f"\n{group_title.upper()}:")
                for feature_key in features:
                    full_name = get_feature_name(feature_key)
                    test_list.append(f"  - {feature_key}: {full_name}")

            # Format already analyzed tests (concise summary)
            analyzed_summary = []
            if already_used_by_group:
                all_analyzed_features = []
                for group_name, features in already_used_by_group.items():
                    feature_names = [get_feature_name(f) for f in features]
                    all_analyzed_features.extend(feature_names)
                analyzed_summary.append(
                    f"\nTests previously analyzed: {', '.join(all_analyzed_features)}"
                )

            return f"""Based on your previous assessment, decide which additional tests to order.

Previous Assessment:
- Reasoning: {previous_assessment.get('explanation', previous_assessment.get('reasoning', ''))}
- Current probability: {int(float(previous_assessment['probability']) * 100) if isinstance(previous_assessment['probability'], (int, float)) and previous_assessment['probability'] <= 1 else int(previous_assessment['probability'])}%
- Current confidence: {int(float(previous_assessment.get('confidence', previous_assessment['probability'])) * 100) if isinstance(previous_assessment.get('confidence', previous_assessment['probability']), (int, float)) and previous_assessment.get('confidence', previous_assessment['probability']) <= 1 else int(previous_assessment.get('confidence', previous_assessment['probability']))}%
{''.join(analyzed_summary)}

Available tests to order:
{''.join(test_list)}

Clinical goal: Determine risk of {self.task_content['complication_name']}

Guidelines for test selection:
- Use only the EXACT abbreviations shown in the list above (e.g., "crea", "bun", "wbc", "ph", "pco2"). Do NOT use full names, group names, or name variations.
- Select a maximum of 2-6 of the most clinically relevant tests.
- Focus on tests that will help confirm or rule out your differential diagnosis.
- Prioritize tests that directly assess organ function relevant to your suspected diagnosis.

Respond in JSON format:
{{
    "diagnosis": "lab-ordering-decision",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['task_name']} will not occur and 100 means {self.task_content['task_name']} will definitely occur),
    "explanation": "Why you want these specific tests and how they will help your decision-making (MAX 200 words)",
    "requested_tests": ["test1", "test2", "test3"]
}}
REMEMBER: Only use exact abbreviations from the list above."""

        return format_prompt

    def _create_updated_assessment_template(self):
        """Template for updated assessment with new lab results."""

        def format_prompt(formatted_lab_data, state):
            # Get monitoring period from the data
            monitoring_hours = get_monitoring_period_hours(state["patient_data"])

            # Format lab results using helper method
            formatted_labs = format_clinical_text(formatted_lab_data)
            labs_str = "\n".join(formatted_labs)

            previous_assessment = state["assessment_history"][-1]

            return f"""Update your clinical assessment with new laboratory results.

Previous Assessment:
- Reasoning: {previous_assessment.get('explanation', previous_assessment.get('reasoning', ''))}
- Previous probability: {int(float(previous_assessment['probability']) * 100) if isinstance(previous_assessment['probability'], (int, float)) and previous_assessment['probability'] <= 1 else int(previous_assessment['probability'])}%
- Previous confidence: {int(float(previous_assessment.get('confidence', previous_assessment['probability'])) * 100) if isinstance(previous_assessment.get('confidence', previous_assessment['probability']), (int, float)) and previous_assessment.get('confidence', previous_assessment['probability']) <= 1 else int(previous_assessment.get('confidence', previous_assessment['probability']))}%

New Laboratory Results (over {monitoring_hours}-hour monitoring period):
{labs_str}

Respond in JSON format:
{{
    "diagnosis": "updated-{self.task_content['task_name']}-assessment",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['task_name']} will not occur and 100 means {self.task_content['task_name']} will definitely occur),
    "explanation": "How the new labs change your assessment and interpretation of abnormal values (MAX 200 words)",
    "confidence": XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment)
}}"""

        return format_prompt

    def _create_final_prediction_template(self):
        """Template for final clinical decision using standard format."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            clinical_data = formatted_data["clinical_data"]
            assessment_history = formatted_data["assessment_history"]

            # Get monitoring period from the data
            monitoring_hours = get_monitoring_period_hours(state["patient_data"])

            # Format demographics
            demo_text = []
            if "age" in demographics:
                demo_text.append(f"Age: {demographics['age']} years")
            if "sex" in demographics:
                demo_text.append(f"Sex: {demographics['sex']}")
            if "weight" in demographics:
                demo_text.append(f"Weight: {demographics['weight']} kg")
            demographics_str = (
                ", ".join(demo_text) if demo_text else "Demographics: Not available"
            )

            # Format all clinical data using helper method
            clinical_text = format_clinical_text(clinical_data)
            clinical_str = "\n".join(clinical_text)

            # Format assessment progression
            assessment_summary = []
            for i, assessment in enumerate(assessment_history):
                # Convert probability and confidence to float, then to 0-100 scale
                try:
                    probability = float(assessment["probability"])
                    confidence = float(
                        assessment.get("confidence", assessment["probability"])
                    )
                    # Convert to 0-100 scale if stored as 0-1
                    if probability <= 1:
                        probability = probability * 100
                    if confidence <= 1:
                        confidence = confidence * 100
                except (ValueError, TypeError):
                    probability = 50.0
                    confidence = 50.0

                assessment_summary.append(
                    f"Step {i+1} ({assessment['step']}): Probability={probability:.1f}%, "
                    f"Confidence={confidence:.1f}%"
                )

            return f"""Patient Demographics:
{demographics_str}

Clinical Data Summary (over {monitoring_hours}-hour monitoring period):
{clinical_str}

Assessment Progression:
{chr(10).join(assessment_summary)}

Task: Determine if this ICU patient will develop {self.task_content['complication_name']}.

Clinical Context: {self.task_content['task_info']}"""

        return format_prompt
