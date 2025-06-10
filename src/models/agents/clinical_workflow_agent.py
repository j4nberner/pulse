import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
import numpy as np
from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import (
    PreprocessorAdvanced,
)
from src.util.data_util import (
    get_feature_name,
    get_feature_reference_range,
    get_feature_uom,
    get_all_feature_groups,
    get_feature_group_keys,
    get_feature_group_title,
    validate_feature_exists,
    get_common_feature_aliases,
    get_clinical_group_aliases,
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
        self.task_content = self._get_task_specific_content()

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

        # Filter out _na columns
        patient_data = self._filter_na_columns(patient_data)

        # Initialize state
        state = {
            "patient_data": patient_data,
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
                return self._create_error_response("Input formatting failed")

            # Check if parsing was successful
            if initial_output.get("diagnosis") == "unknown":
                logger.warning(
                    "Initial assessment JSON parsing likely failed - got fallback response"
                )

            # Extract confidence with fallback logic
            confidence_normalized = self._extract_confidence(initial_output)

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
                self._get_available_vitals(state["available_features"])
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
                requested_labs = self._extract_requested_labs(lab_order_output)
                if requested_labs is None:
                    break

                if not requested_labs:
                    logger.info("No additional labs requested, stopping iteration")
                    break

                logger.info("Requested labs: %s", requested_labs)

                # Validate requested labs against data_util
                valid_requested_labs = self._validate_features(requested_labs)
                if not valid_requested_labs:
                    logger.info("No valid labs requested, stopping iteration")
                    break

                # Additional validation: ensure no already-used features are requested
                valid_requested_labs = self._validate_lab_request(
                    valid_requested_labs, state
                )
                if not valid_requested_labs:
                    logger.info(
                        "No new labs requested (all already analyzed), stopping iteration"
                    )
                    break

                logger.info("Valid requested labs: %s", valid_requested_labs)

                # Get available requested labs
                available_labs = self._get_available_labs(
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
                confidence_normalized = self._extract_confidence(updated_output)
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

    def _extract_confidence(self, output: Dict[str, Any]) -> float:
        """Extract confidence value from LLM output with fallback logic."""
        if "confidence" in output:
            confidence = output.get("confidence", 50)
            # Handle string values from LLM output
            if isinstance(confidence, str):
                try:
                    confidence = float(confidence)
                except ValueError:
                    confidence = 50
            return confidence / 100.0
        else:
            # Use probability as confidence indicator when confidence not provided
            probability = output.get("probability", 50)
            # Handle string values from LLM output
            if isinstance(probability, str):
                try:
                    probability = float(probability)
                except ValueError:
                    probability = 50
            return probability / 100.0

    def _extract_requested_labs(
        self, lab_order_output: Dict[str, Any]
    ) -> Optional[List[str]]:
        """Extract requested labs from lab ordering output with validation."""
        if not isinstance(lab_order_output, dict):
            logger.info("Lab ordering failed to return dict, stopping iteration")
            return None

        if "requested_tests" in lab_order_output:
            return lab_order_output.get("requested_tests", [])
        elif lab_order_output.get("diagnosis") == "unknown":
            # This is the fallback dict from failed parsing
            logger.info("Lab ordering failed to parse JSON, stopping iteration")
            return None
        else:
            # Unexpected dict format
            logger.info("Lab ordering returned unexpected format, stopping iteration")
            return None

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "generated_text": {"error": error_message},
            "token_time": 0,
            "infer_time": 0,
            "num_input_tokens": 0,
            "num_output_tokens": 0,
        }

    def _get_available_vitals(self, available_features: Set[str]) -> Set[str]:
        """Get available vital signs from the data using data_util groups."""
        available_vitals = set()
        vitals_keys = get_feature_group_keys("vitals")

        for vital in vitals_keys:
            if any(feat.startswith(vital) for feat in available_features):
                available_vitals.add(vital)
        return available_vitals

    def _get_available_labs(
        self, requested_labs: List[str], available_features: Set[str]
    ) -> Set[str]:
        """Get available requested labs from the data."""
        available_labs = set()
        for lab in requested_labs:
            if any(feat.startswith(lab) for feat in available_features):
                available_labs.add(lab)
        return available_labs

    def _get_lab_groups_available(
        self, available_features: Set[str]
    ) -> Dict[str, List[str]]:
        """Get available lab tests organized by clinical groups."""
        available_by_group = {}

        for group_name, group_dict in self.lab_groups.items():
            if group_name == "vitals":  # Skip vitals as they're handled separately
                continue

            available_in_group = []
            for feature_key in group_dict.keys():
                if any(feat.startswith(feature_key) for feat in available_features):
                    available_in_group.append(feature_key)

            if available_in_group:
                available_by_group[group_name] = available_in_group

        return available_by_group

    def _validate_features(self, feature_list: List[str]) -> List[str]:
        """Validate that requested features exist in the data_util feature dictionary."""
        valid_features = []

        # Get mappings from data_util
        group_mappings = {}

        # Add official group mappings from data_util
        for group_key in [
            "bga",
            "coag",
            "electrolytes_met",
            "liver_kidney",
            "hematology_immune",
            "cardiac",
        ]:
            group_features = get_feature_group_keys(group_key)
            group_title = get_feature_group_title(group_key).lower()

            # Add both the key and the display title as mappings
            group_mappings[group_key] = group_features
            group_mappings[group_title] = group_features

        # Add clinical aliases from data_util
        clinical_aliases = get_clinical_group_aliases()
        for feature_tuple, aliases in clinical_aliases.items():
            feature_list_from_tuple = list(feature_tuple)
            for alias in aliases:
                group_mappings[alias] = feature_list_from_tuple

        # Get individual feature name mappings from data_util
        name_mappings = get_common_feature_aliases()

        for feature in feature_list:
            feature_lower = feature.lower().strip()

            # First check exact match (highest priority)
            if validate_feature_exists(feature):
                valid_features.append(feature)
                continue

            # Check lowercase exact match
            if validate_feature_exists(feature_lower):
                valid_features.append(feature_lower)
                continue

            # Check if it's a group name that needs expansion
            if feature_lower in group_mappings:
                group_features = group_mappings[feature_lower]
                added_features = []
                for group_feature in group_features:
                    if validate_feature_exists(group_feature):
                        valid_features.append(group_feature)
                        added_features.append(group_feature)
                if added_features:
                    logger.info(
                        "Expanded group '%s' to features: %s", feature, added_features
                    )
                continue

            # Check if it needs individual mapping
            if feature_lower in name_mappings:
                mapped_feature = name_mappings[feature_lower]
                if validate_feature_exists(mapped_feature):
                    valid_features.append(mapped_feature)
                    logger.info("Mapped '%s' to '%s'", feature, mapped_feature)
                    continue
                else:
                    logger.warning(
                        "Mapped feature '%s' not found in data_util", mapped_feature
                    )
                    continue

            # Check for partial matches in available feature groups
            found_match = False
            for group_name, group_dict in self.lab_groups.items():
                for feature_key in group_dict.keys():
                    if (
                        feature_lower in feature_key.lower()
                        or feature_key.lower() in feature_lower
                    ):
                        if validate_feature_exists(feature_key):
                            valid_features.append(feature_key)
                            logger.info(
                                "Partial match: mapped '%s' to '%s'",
                                feature,
                                feature_key,
                            )
                            found_match = True
                            break
                if found_match:
                    break

            if not found_match:
                logger.warning("Feature '%s' not found in data_util", feature)

        return valid_features

    def _format_clinical_data(
        self,
        patient_data: pd.Series,
        feature_keys: Set[str],
        include_demographics: bool = False,
        include_temporal_patterns: bool = False,
    ) -> Dict[str, Any]:
        """
        Format clinical data (vital signs or lab results) using aggregate_feature_windows.

        Args:
            patient_data: Patient data series
            feature_keys: Set of feature keys to format
            include_demographics: Whether to include demographics in output
            include_temporal_patterns: Whether to include temporal trend analysis

        Returns:
            Dictionary with formatted clinical data
        """
        # Convert patient data to DataFrame for preprocessing
        patient_df = pd.DataFrame([patient_data])

        # Use aggregate_feature_windows to get min, max, mean for each feature
        aggregated_df = self.preprocessor_advanced.aggregate_feature_windows(patient_df)
        aggregated_row = aggregated_df.iloc[0]

        # Prepare result dictionary
        result = {}

        # Extract patient demographics if requested
        if include_demographics:
            patient_demographics = {}
            if "age" in patient_data.index:
                patient_demographics["age"] = patient_data["age"]
            if "sex" in patient_data.index:
                patient_demographics["sex"] = patient_data["sex"]
            if "weight" in patient_data.index:
                patient_demographics["weight"] = patient_data["weight"]
            result["demographics"] = patient_demographics

        # Format clinical features with aggregated values using data_util
        clinical_data = {}

        for feature_key in feature_keys:
            # Check if this feature has aggregated columns in the data
            if any(col.startswith(f"{feature_key}_") for col in aggregated_row.index):
                min_val = aggregated_row.get(f"{feature_key}_min", None)
                max_val = aggregated_row.get(f"{feature_key}_max", None)
                mean_val = aggregated_row.get(f"{feature_key}_mean", None)

                # Only include features with non-NaN mean values
                if not pd.isna(mean_val):
                    feature_name = get_feature_name(feature_key)
                    unit = get_feature_uom(feature_key)
                    normal_range = get_feature_reference_range(feature_key)

                    clinical_data[feature_key] = {
                        "name": feature_name,
                        "min": min_val if not pd.isna(min_val) else mean_val,
                        "max": max_val if not pd.isna(max_val) else mean_val,
                        "mean": mean_val,
                        "unit": unit,
                        "normal_range": normal_range,
                    }

                    # Add temporal pattern analysis if requested
                    if include_temporal_patterns:
                        temporal_pattern = self._analyze_temporal_pattern(
                            patient_data, feature_key, min_val, max_val, mean_val
                        )
                        clinical_data[feature_key][
                            "temporal_pattern"
                        ] = temporal_pattern

        # Store clinical data under appropriate key
        if include_demographics:
            result["vital_signs"] = clinical_data
        else:
            result = clinical_data

        return result

    def _format_state_only(self, input_data: Any, state: Dict[str, Any]) -> None:
        """Input formatter for lab ordering - returns None since template uses only state."""
        return None

    def _format_vital_signs(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format vital signs data for initial assessment using aggregate_feature_windows."""
        patient_data = input_data
        vitals_keys = get_feature_group_keys("vitals")
        available_vitals = {
            vital
            for vital in vitals_keys
            if any(feat.startswith(vital) for feat in patient_data.index)
        }

        return self._format_clinical_data(
            patient_data=patient_data,
            feature_keys=available_vitals,
            include_demographics=True,
            include_temporal_patterns=True,
        )

    def _format_lab_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format lab data for updated assessment."""
        available_labs = input_data
        patient_data = state["patient_data"]
        return self._format_clinical_data(
            patient_data=patient_data,
            feature_keys=available_labs,
            include_demographics=False,
            include_temporal_patterns=True,
        )

    def _format_final_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format comprehensive data for final decision."""
        patient_data = state["patient_data"]

        # Get demographics
        demographics = {}
        if "age" in patient_data.index:
            demographics["age"] = patient_data["age"]
        if "sex" in patient_data.index:
            demographics["sex"] = patient_data["sex"]
        if "weight" in patient_data.index:
            demographics["weight"] = patient_data["weight"]

        # Get all used clinical data with temporal patterns
        all_clinical_data = self._format_clinical_data(
            patient_data=patient_data,
            feature_keys=state["used_features"],
            include_demographics=False,
            include_temporal_patterns=True,
        )

        return {
            "demographics": demographics,
            "clinical_data": all_clinical_data,
            "assessment_history": state["assessment_history"],
        }

    def _format_clinical_text(self, clinical_data: Dict[str, Dict]) -> List[str]:
        """
        Format clinical data dictionary into human-readable text lines.

        Args:
            clinical_data: Dictionary with feature data containing name, min, max, mean, unit, normal_range

        Returns:
            List of formatted text lines
        """
        formatted_lines = []

        for feature_key, data in clinical_data.items():
            name = data["name"]
            min_val = data["min"]
            max_val = data["max"]
            mean_val = data["mean"]
            unit = data["unit"]
            normal_range = data["normal_range"]

            # Format value range or single value
            if abs(min_val - max_val) < 0.01:  # Essentially the same value
                value_str = f"{mean_val:.2f}"
            else:
                value_str = f"{min_val:.2f}-{max_val:.2f} (avg: {mean_val:.2f})"

            # Add normal range if available
            if unit and normal_range != (0, 0):
                normal_str = f" [normal: {normal_range[0]}-{normal_range[1]} {unit}]"
                base_text = f"- {name}: {value_str} {unit}{normal_str}"
            else:
                unit_str = f" {unit}" if unit else ""
                base_text = f"- {name}: {value_str}{unit_str}"

            # Add temporal pattern information if available
            if "temporal_pattern" in data:
                temporal_pattern = data["temporal_pattern"]
                base_text += f" ({temporal_pattern})"

            formatted_lines.append(base_text)

        return formatted_lines

    def _create_initial_assessment_template(self):
        """Template for initial vital signs assessment."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            vital_signs = formatted_data["vital_signs"]

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
            vitals_text = self._format_clinical_text(vital_signs)
            vitals_str = "\n".join(vitals_text)

            return f"""You are evaluating an ICU patient for risk of {self.task_content['complication_name']}.

Patient Demographics:
{demographics_str}

Current vital signs (over monitoring period):
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
    "diagnosis": "preliminary-{self.task_content['complication_name']}-risk",
    "probability": "XX (integer between 0 and 100, where 0 means {self.task_content['complication_name']} will not occur and 100 means {self.task_content['complication_name']} will definitely occur)",
    "explanation": "Your detailed clinical reasoning including differential diagnosis and temporal pattern assessment",
    "confidence": "XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment)"
}}

IMPORTANT: With only vital signs available, confidence should typically be 50-70. Higher confidence (>75) should only be used when clinical picture is very clear."""

        return format_prompt

    def _create_lab_ordering_template(self):
        """Template for deciding which labs to order."""

        def format_prompt(_, state):
            previous_assessment = state["assessment_history"][-1]

            # Get available tests by clinical group using data_util
            available_by_group = self._get_lab_groups_available(
                state["available_features"]
            )

            # Filter out already used features - CRITICAL for preventing re-ordering
            filtered_available = {}
            total_unused_features = 0

            for group_name, features in available_by_group.items():
                unused_features = [
                    f for f in features if f not in state["used_features"]
                ]
                if unused_features:
                    filtered_available[group_name] = unused_features
                    total_unused_features += len(unused_features)

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
- Current probability: {previous_assessment['probability']}%
- Current confidence: {previous_assessment.get('confidence', previous_assessment['probability'])}%

Since no additional tests are available, respond with an empty test list.

Respond in JSON format:
{{
    "diagnosis": "lab-ordering-complete",
    "probability": "{previous_assessment['probability']}",
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

            return f"""Based on your previous assessment, decide which additional tests to order.

Previous Assessment:
- Reasoning: {previous_assessment.get('explanation', previous_assessment.get('reasoning', ''))}
- Current probability: {previous_assessment['probability']}%
- Current confidence: {previous_assessment.get('confidence', previous_assessment['probability'])}%

Available tests to order (excluding already ordered tests):
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
    "probability": "XX (integer between 0 and 100, where 0 means {self.task_content['complication_name']} will not occur and 100 means {self.task_content['complication_name']} will definitely occur)",
    "explanation": "Why you want these specific tests and how they will help your decision-making",
    "requested_tests": ["test1", "test2", "test3"]
}}
REMEMBER: Only use exact abbreviations from the list above."""

        return format_prompt

    def _create_updated_assessment_template(self):
        """Template for updated assessment with new lab results."""

        def format_prompt(formatted_lab_data, state):
            # Format lab results using helper method
            formatted_labs = self._format_clinical_text(formatted_lab_data)
            labs_str = "\n".join(formatted_labs)

            previous_assessment = state["assessment_history"][-1]

            return f"""Update your clinical assessment with new laboratory results.

Previous Assessment:
- Reasoning: {previous_assessment.get('explanation', previous_assessment.get('reasoning', ''))}
- Previous probability: {previous_assessment['probability']}%
- Previous confidence: {previous_assessment.get('confidence', previous_assessment['probability'])}%

New Laboratory Results (over monitoring period):
{labs_str}

Respond in JSON format:
{{
    "diagnosis": "updated-{self.task_content['complication_name']}-assessment",
    "probability": "XX (integer between 0 and 100, where 0 means {self.task_content['complication_name']} will not occur and 100 means {self.task_content['complication_name']} will definitely occur)",
    "explanation": "How the new labs change your assessment and interpretation of abnormal values",
    "confidence": "XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment)"
}}"""

        return format_prompt

    def _create_final_prediction_template(self):
        """Template for final clinical decision using standard format."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            clinical_data = formatted_data["clinical_data"]
            assessment_history = formatted_data["assessment_history"]

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
            clinical_text = self._format_clinical_text(clinical_data)
            clinical_str = "\n".join(clinical_text)

            # Format assessment progression
            assessment_summary = []
            for i, assessment in enumerate(assessment_history):
                # Convert probability and confidence to float
                try:
                    probability = float(assessment["probability"])
                    confidence = float(
                        assessment.get("confidence", assessment["probability"])
                    )
                except (ValueError, TypeError):
                    probability = 50.0
                    confidence = 50.0

                assessment_summary.append(
                    f"Step {i+1} ({assessment['step']}): Probability={probability:.1f}%, "
                    f"Confidence={confidence:.1f}%"
                )

            return f"""Patient Demographics:
{demographics_str}

Clinical Data Summary (over monitoring period):
{clinical_str}

Assessment Progression:
{chr(10).join(assessment_summary)}

Task: Determine if this ICU patient will develop {self.task_content['complication_name']}.

Clinical Context: {self.task_content['task_info']}"""

        return format_prompt

    def _get_task_specific_content(self) -> Dict[str, str]:
        """Get task-specific content for prompts."""
        task = self.task_name
        if task == "mortality":
            return {
                "complication_name": "mortality",
                "task_info": "ICU mortality refers to death occurring during the ICU stay. Key risk factors include hemodynamic instability, respiratory failure, multi-organ dysfunction, and severe metabolic derangements.",
            }
        elif task == "aki":
            return {
                "complication_name": "aki",
                "task_info": "Acute kidney injury (AKI) is defined by rapid decline in kidney function with increased creatinine (≥1.5x baseline or ≥0.3 mg/dL increase in 48h) or decreased urine output (<0.5 mL/kg/h for 6-12h). Common causes include sepsis, hypotension, and nephrotoxins.",
            }
        elif task == "sepsis":
            return {
                "complication_name": "sepsis",
                "task_info": "Sepsis is life-threatening organ dysfunction caused by dysregulated host response to infection. Diagnosed by SOFA score increase ≥2 points with suspected infection. Key indicators include fever, tachycardia, tachypnea, altered mental status, and laboratory abnormalities.",
            }
        return {
            "complication_name": "complications",
            "task_info": "General ICU complications assessment.",
        }

    def _filter_na_columns(self, patient_data: pd.Series) -> pd.Series:
        """Filter out columns with '_na' suffixes like Sarvari preprocessor does."""
        # Convert to DataFrame for regex filtering, then back to Series
        temp_df = pd.DataFrame([patient_data])
        filtered_df = temp_df.filter(regex=r"^(?!.*_na(_\d+)?$)")
        return filtered_df.iloc[0]

    def _analyze_temporal_pattern(
        self,
        patient_data: pd.Series,
        feature_key: str,
        min_val: float,
        max_val: float,
        mean_val: float,
    ) -> str:
        """
        Simple temporal pattern analysis returning brief text assessment.

        Args:
            patient_data: Full patient data series
            feature_key: The feature to analyze (e.g., 'hr', 'sbp')
            min_val: Minimum value over monitoring period
            max_val: Maximum value over monitoring period
            mean_val: Mean value over monitoring period

        Returns:
            Brief text description of trend and normality
        """
        try:
            # Extract time-windowed values for this feature
            time_series_values = []
            for col in patient_data.index:
                if col.startswith(f"{feature_key}_") and col.split("_")[-1].isdigit():
                    val = patient_data[col]
                    if not pd.isna(val):
                        time_series_values.append(val)

            if len(time_series_values) < 3:
                return "stable trend"

            # Linear regression for trend analysis
            n = len(time_series_values)
            x = np.arange(n)  # Time points
            y = np.array(time_series_values)

            # Calculate linear regression slope
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            denominator = np.sum((x - x_mean) ** 2)

            if denominator == 0:
                slope = 0
            else:
                slope = numerator / denominator

            # Normalize slope by mean value to get relative change rate
            relative_slope = (slope / y_mean * 100) if y_mean != 0 else 0

            # Determine trend direction and strength
            if abs(relative_slope) < 2:  # Less than 2% change per time unit
                trend = "stable"
            elif relative_slope >= 8:  # Strong increase (>=8% per time unit)
                trend = "rapidly increasing"
            elif relative_slope >= 4:  # Moderate increase (4-8% per time unit)
                trend = "moderately increasing"
            elif relative_slope > 0:  # Mild increase (2-4% per time unit)
                trend = "slowly increasing"
            elif relative_slope <= -8:  # Strong decrease (<=-8% per time unit)
                trend = "rapidly decreasing"
            elif relative_slope <= -4:  # Moderate decrease (-8 to -4% per time unit)
                trend = "moderately decreasing"
            else:  # Mild decrease (-4 to -2% per time unit)
                trend = "slowly decreasing"

            # Use categorize_features for abnormality assessment
            patient_df = pd.DataFrame([patient_data])
            categorized_df = self.preprocessor_advanced.categorize_features(
                df=patient_df,
                base_features={feature_key},
                X_cols=patient_data.index,
                num_categories=5,
                for_llm=True,
            )

            status = (
                categorized_df[feature_key].iloc[0]
                if feature_key in categorized_df.columns
                else "unknown range"
            )

            return f"{trend}, {status}"

        except Exception as e:
            logger.warning(
                "Error analyzing temporal pattern for %s: %s", feature_key, e
            )
            return "stable trend"

    def _validate_lab_request(
        self, requested_labs: List[str], state: Dict[str, Any]
    ) -> List[str]:
        """
        Validate lab requests to ensure no already-used features are requested again.

        Args:
            requested_labs: List of requested lab abbreviations
            state: Current workflow state

        Returns:
            List of valid, unused lab abbreviations
        """
        valid_labs = []
        already_used_requests = []

        for lab in requested_labs:
            if lab in state["used_features"]:
                already_used_requests.append(lab)
            else:
                valid_labs.append(lab)

        if already_used_requests:
            logger.warning(
                "Model requested already analyzed features: %s. These will be ignored.",
                already_used_requests,
            )
            logger.info("Valid new lab requests: %s", valid_labs)

        return valid_labs
