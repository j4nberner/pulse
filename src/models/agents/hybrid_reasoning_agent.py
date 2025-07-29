import glob
import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.agent_util import (create_error_response, filter_na_columns,
                                 format_clinical_data, format_clinical_text,
                                 format_demographics_str,
                                 get_monitoring_period_hours,
                                 get_task_specific_content,
                                 parse_numeric_value)
from src.util.data_util import (features_dict, get_feature_name,
                                get_priority_features_for_task)

logger = logging.getLogger("PULSE_logger")


class HybridReasoningAgent(PulseAgent):
    """
    Hybrid AI-Clinical Reasoning Agent that combines pretrained XGBoost predictions
    with LLM-based clinical reasoning using feature importance guidance.

    Workflow:
    1. ML Risk Stratification (XGBoost prediction + feature importance)
    2. Clinical Context Integration (interpret ML findings clinically)
    3. Detailed Investigation (conditional based on confidence/agreement)
    4. Confidence-Weighted Synthesis (combine ML + clinical reasoning)
    """

    def __init__(
        self,
        model: Any,
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics_tracker: Optional[Any] = None,
        ml_confidence_threshold: float = 0.8,
        agreement_threshold: float = 0.2,
        top_features_count: int = 10,
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

        self.ml_confidence_threshold = ml_confidence_threshold
        self.agreement_threshold = agreement_threshold
        self.top_features_count = top_features_count
        self.task_content = get_task_specific_content(self.task_name)

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Initialize XGBoost model
        self.xgb_model = None
        self.xgb_feature_names = None
        self._load_xgb_model()

        self._define_steps()

    def _update_task_specific_content(self) -> None:
        """Update task-specific content when task changes."""
        self.task_content = get_task_specific_content(self.task_name)

        # Redefine steps with updated task content
        self.steps = []  # Clear existing steps
        self._define_steps()

    def _load_xgb_model(self) -> None:
        """Load the pretrained XGBoost model for the current task."""
        try:
            # Construct path to pretrained model
            agents_dir = os.path.dirname(os.path.abspath(__file__))
            pretrained_dir = os.path.join(agents_dir, "pretrained_models")

            # Find XGBoost model with the specific naming pattern
            pattern = os.path.join(
                pretrained_dir, f"XGBoost_{self.task_name}_{self.dataset_name}_*.joblib"
            )
            matching_files = glob.glob(pattern)

            if not matching_files:
                logger.warning(
                    "No XGBoost model found for task %s, dataset %s in %s",
                    self.task_name,
                    self.dataset_name,
                    pretrained_dir,
                )
                return

            # Use the most recent model
            model_path = sorted(matching_files)[-1]
            if len(matching_files) > 1:
                logger.info("Found multiple models, using most recent: %s", model_path)

            # Load the model
            self.xgb_model = joblib.load(model_path)

            # Get feature names - simplified approach
            if hasattr(self.xgb_model, "_pulse_feature_names"):
                self.xgb_feature_names = list(self.xgb_model._pulse_feature_names)
                logger.info(
                    "Loaded %d feature names from model", len(self.xgb_feature_names)
                )
            else:
                logger.warning(
                    "No feature names found in model - will use data column order"
                )
                self.xgb_feature_names = None

            logger.info("Successfully loaded XGBoost model from %s", model_path)

        except Exception as e:
            logger.error("Failed to load XGBoost model: %s", e, exc_info=True)
            self.xgb_model = None

    def _preprocess_patient_data(self, patient_data: pd.Series) -> pd.DataFrame:
        """
        Preprocess patient data for XGBoost prediction.
        This replicates the exact preprocessing from training.
        """
        # Convert to DataFrame
        patient_df = pd.DataFrame([patient_data])

        # Apply categorical encoding (same as training)
        if "sex" in patient_df.columns:
            patient_df = patient_df.copy()
            patient_df["sex"] = (
                patient_df["sex"].map({"Male": 1, "Female": 0}).fillna(-1)
            )

        return patient_df

    def _define_steps(self) -> None:
        """Define the hybrid reasoning workflow steps."""

        # Step 1: ML Risk Stratification
        self.add_step(
            name="ml_interpretation",
            system_message="You are an AI-assisted clinical decision support specialist. Analyze the ML model's risk prediction and feature importance to provide clinical interpretation of the AI assessment.",
            prompt_template=self._create_ml_stratification_template(),
            input_formatter=self._format_ml_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 2: Clinical Context Integration
        self.add_step(
            name="clinical_assessment",
            system_message="You are an experienced ICU physician evaluating how AI predictions align with clinical expectations. Assess agreement between ML findings and clinical reasoning.",
            prompt_template=self._create_clinical_integration_template(),
            input_formatter=self._format_integration_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 3: Detailed Investigation (conditional)
        self.add_step(
            name="detailed_investigation",
            system_message="You are conducting a focused clinical investigation of discrepant or uncertain findings. Analyze the most important clinical parameters in detail.",
            prompt_template=self._create_detailed_investigation_template(),
            input_formatter=self._format_investigation_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 4: Final Prediction
        self.add_step(
            name="final_prediction",
            system_message=None,  # Uses default system message
            prompt_template=self._create_synthesis_template(),
            input_formatter=self._format_synthesis_data,
            output_processor=None,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient through the hybrid reasoning workflow."""
        # Update task context if needed (ensures task_content is current)
        if hasattr(self.model, "task_name") and hasattr(self.model, "dataset_name"):
            self.update_task_context(self.model.task_name, self.model.dataset_name)

        # Reset memory
        self.memory.reset()

        # Keep original data with _na columns for XGBoost
        original_patient_data = patient_data.copy()

        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        self.memory.set_current_sample(sample_id, original_patient_data)

        # Filter out _na columns for clinical reasoning
        filtered_patient_data = filter_na_columns(patient_data)

        # Initialize state
        state = {
            "patient_data": filtered_patient_data,
            "original_patient_data": original_patient_data,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "ml_prediction": None,
            "ml_confidence": 0.0,
            "feature_importance": {},
            "top_features": [],
            "clinical_assessment": None,
            "agreement": None,
            "needs_investigation": False,
        }

        try:
            # Step 1: ML Risk Stratification
            ml_result = self.run_step("ml_interpretation", original_patient_data, state)
            ml_output = ml_result["output"]

            if isinstance(ml_output, str) and "Error" in ml_output:
                logger.error("ML stratification failed: %s", ml_output)
                return create_error_response("ML risk stratification failed")

            # Verify the LLM step completed successfully
            logger.debug("ML stratification LLM response received successfully")
            logger.debug(
                "State after ML step - ML prediction: %d, ML confidence: %.3f",
                state["ml_prediction"],
                state["ml_confidence"],
            )

            # Step 2: Clinical Context Integration
            integration_result = self.run_step("clinical_assessment", None, state)
            integration_output = integration_result["output"]

            if isinstance(integration_output, str) and "Error" in integration_output:
                logger.error("Clinical integration failed: %s", integration_output)
                return create_error_response("Clinical integration failed")

            state["clinical_assessment"] = integration_output

            # Store the converted clinical probability for later use
            clinical_prob_raw = integration_output.get("probability", 50)
            clinical_prob = parse_numeric_value(clinical_prob_raw, 50)
            if isinstance(clinical_prob, float) and clinical_prob < 1:
                clinical_prob = int(clinical_prob * 100)
            else:
                clinical_prob = int(clinical_prob)
            state["clinical_probability"] = clinical_prob

            # Determine if detailed investigation is needed
            ml_conf = int(state["ml_confidence"] * 100)
            ml_prob = state["ml_prediction"]  # Already an integer

            # Check agreement between ML and clinical assessment (using integer percentages)
            prob_difference = abs(ml_prob - clinical_prob)
            low_ml_confidence = ml_conf < int(self.ml_confidence_threshold * 100)
            high_disagreement = prob_difference > int(self.agreement_threshold * 100)

            state["agreement"] = {
                "probability_difference": prob_difference,
                "ml_confidence": ml_conf,
                "clinical_confidence": int(
                    parse_numeric_value(integration_output.get("confidence", 0), 0)
                    * 100
                ),
            }

            state["needs_investigation"] = low_ml_confidence or high_disagreement

            logger.info(
                "ML vs Clinical: %d vs %d (diff=%d), ML conf=%d, Investigation needed: %s",
                ml_prob,
                clinical_prob,
                prob_difference,
                ml_conf,
                state["needs_investigation"],
            )

            # Step 3: Detailed Investigation (conditional)
            if state["needs_investigation"]:
                investigation_result = self.run_step(
                    "detailed_investigation", None, state
                )
                investigation_output = investigation_result["output"]
                state["investigation_results"] = investigation_output
                logger.info(
                    "Conducted detailed investigation due to disagreement/uncertainty"
                )
            else:
                state["investigation_results"] = None
                logger.info(
                    "Skipped detailed investigation - high confidence and agreement"
                )

            # Step 4: Confidence-Weighted Synthesis
            # Apply dampening logic before synthesis to ensure template gets dampened values
            if state["needs_investigation"] and state.get("investigation_results"):
                ml_prob = state.get("ml_prediction", 50) / 100.0
                ml_conf = state.get("ml_confidence", 0.5)

                # Apply confidence-weighted dampening
                if ml_conf >= 0.9:
                    max_deviation = 0.2
                elif ml_conf >= 0.8:
                    max_deviation = 0.35
                elif ml_conf >= 0.7:
                    max_deviation = 0.5
                elif ml_conf >= 0.6:
                    max_deviation = 0.65
                else:
                    max_deviation = 0.8

                investigation_prob = parse_numeric_value(
                    state["investigation_results"].get("probability", 0.5), 0.5
                )
                if (
                    isinstance(investigation_prob, (int, float))
                    and investigation_prob > 1
                ):
                    investigation_prob = (
                        investigation_prob / 100.0
                    )  # Convert percentage to decimal

                deviation = abs(investigation_prob - ml_prob)
                if deviation > max_deviation:
                    if investigation_prob > ml_prob:
                        dampened_prob = ml_prob + max_deviation
                    else:
                        dampened_prob = max(ml_prob - max_deviation, 0)

                    # Create dampened investigation results for synthesis
                    dampened_investigation = state["investigation_results"].copy()
                    dampened_investigation["probability"] = dampened_prob
                    state["dampened_investigation_results"] = dampened_investigation
                    logger.info(
                        "Applied dampening: %.1f%% -> %.1f%%",
                        investigation_prob * 100,
                        dampened_prob * 100,
                    )

            synthesis_result = self.run_step("final_prediction", None, state)
            synthesis_output = synthesis_result["output"]

            # Aggregate token metrics
            all_steps = self.memory.samples.get(str(sample_id), [])
            total_input_tokens = sum(step.num_input_tokens for step in all_steps)
            total_output_tokens = sum(step.num_output_tokens for step in all_steps)
            total_token_time = sum(step.token_time for step in all_steps)
            total_infer_time = sum(step.infer_time for step in all_steps)

            return {
                "generated_text": synthesis_output,
                "token_time": total_token_time,
                "infer_time": total_infer_time,
                "num_input_tokens": total_input_tokens,
                "num_output_tokens": total_output_tokens,
            }

        except Exception as e:
            logger.error("Error in hybrid reasoning workflow: %s", e, exc_info=True)
            return create_error_response(f"Hybrid reasoning error: {str(e)}")

    def update_task_context(self, task_name, dataset_name):
        """Update task and dataset context, and reload XGBoost model if needed."""
        if (self.task_name != task_name) or (self.dataset_name != dataset_name):
            self.task_name = task_name
            self.dataset_name = dataset_name
            self.task_content = get_task_specific_content(self.task_name)
            # Delete previous XGBoost model to free up memory
            if hasattr(self, "xgb_model") and self.xgb_model is not None:
                del self.xgb_model
                self.xgb_model = None
            self._load_xgb_model()  # Ensure the correct model is loaded
            self._define_steps()

    def _run_xgb_prediction(
        self, patient_data: pd.Series
    ) -> Tuple[float, float, Dict[str, float]]:
        """Run XGBoost prediction and extract feature importance."""
        if self.xgb_model is None:
            logger.warning("XGBoost model not available, using fallback values")
            return 0.5, 0.5, {}

        try:
            # Preprocess data
            patient_df = self._preprocess_patient_data(patient_data)

            # Handle feature alignment
            if self.xgb_feature_names is not None:
                # Use stored feature names (preferred path)
                expected_features = self.xgb_feature_names

                # Add missing features with default values
                for feature in expected_features:
                    if feature not in patient_df.columns:
                        patient_df[feature] = 0
                        logger.debug("Added missing feature '%s' with value 0", feature)

                # Reorder columns to match training order
                patient_df = patient_df[expected_features]
                feature_names_for_importance = expected_features

                logger.debug(
                    "Using stored feature names (%d features)", len(expected_features)
                )

            else:
                # Fallback: use current column order
                feature_names_for_importance = list(patient_df.columns)
                logger.warning(
                    "Using fallback feature mapping (%d features)",
                    len(feature_names_for_importance),
                )

            # Make prediction
            if hasattr(self.xgb_model, "predict_proba"):
                prediction_proba = self.xgb_model.predict_proba(patient_df)[0]
                probability = (
                    prediction_proba[1]
                    if len(prediction_proba) > 1
                    else prediction_proba[0]
                )
            else:
                probability = self.xgb_model.predict(patient_df)[0]

            # Calculate confidence
            confidence = abs(probability - 0.5) * 2

            # Get feature importance
            feature_importance = {}
            if hasattr(self.xgb_model, "feature_importances_"):
                importance_scores = self.xgb_model.feature_importances_

                if len(feature_names_for_importance) == len(importance_scores):
                    feature_importance = dict(
                        zip(feature_names_for_importance, importance_scores)
                    )
                else:
                    logger.warning(
                        "Feature count mismatch: %d names vs %d scores",
                        len(feature_names_for_importance),
                        len(importance_scores),
                    )
                    # Create generic names as final fallback
                    generic_names = [
                        f"feature_{i}" for i in range(len(importance_scores))
                    ]
                    feature_importance = dict(zip(generic_names, importance_scores))

            logger.debug(
                "XGBoost prediction: prob=%.3f, conf=%.3f, features=%d",
                probability,
                confidence,
                len(feature_importance),
            )

            return float(probability), float(confidence), feature_importance

        except Exception as e:
            logger.error("Error in XGBoost prediction: %s", e, exc_info=True)
            return 0.5, 0.5, {}

    def _format_ml_data(self, state: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
        """Format data for ML risk stratification step."""
        patient_data = input_data  # This is the original_patient_data with _na columns

        # Run XGBoost prediction
        ml_prob, ml_conf, feature_importance = self._run_xgb_prediction(patient_data)

        # Store in state for later use
        state["ml_prediction"] = int(ml_prob * 100)  # Convert to integer percentage
        state["ml_confidence"] = ml_conf
        state["feature_importance"] = feature_importance

        # Get top important features with improved clinical naming and base feature deduplication
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )

            # Deduplicate by base feature name, keeping the most important time window for each
            base_feature_map = {}
            for feat_name, importance in sorted_features:
                # Extract base feature name
                if feat_name.endswith("_na"):
                    base_feature = feat_name.replace("_na", "").split("_")[0]
                else:
                    base_feature = (
                        feat_name.split("_")[0] if "_" in feat_name else feat_name
                    )

                # Keep only the most important instance of each base feature
                if (
                    base_feature not in base_feature_map
                    or importance > base_feature_map[base_feature][1]
                ):
                    base_feature_map[base_feature] = (feat_name, importance)

            # Sort by importance and take top N unique base features
            unique_features = sorted(
                base_feature_map.values(), key=lambda x: x[1], reverse=True
            )
            top_unique_features = unique_features[: self.top_features_count]

            # Format features for clinical display
            formatted_features = []
            for feat_name, importance in top_unique_features:
                # Try to get clinical name
                if feat_name.endswith("_na"):
                    # Missingness indicator
                    base_feature = feat_name.replace("_na", "").split("_")[0]
                    clinical_name = get_feature_name(base_feature)
                    if clinical_name and clinical_name != "Unknown":
                        display_name = f"Missing: {clinical_name}"
                    else:
                        display_name = f"Missing: {feat_name}"
                else:
                    # Regular feature - use clinical name only
                    base_feature = (
                        feat_name.split("_")[0] if "_" in feat_name else feat_name
                    )
                    clinical_name = get_feature_name(base_feature)
                    if clinical_name and clinical_name != "Unknown":
                        display_name = clinical_name
                    else:
                        display_name = feat_name

                formatted_features.append((display_name, importance))

            state["top_features"] = formatted_features
        else:
            state["top_features"] = []

        logger.info(
            "ML prediction: %.1f%% (conf: %.1f%%), top features: %s",
            ml_prob * 100,
            ml_conf * 100,
            [f[0] for f in state["top_features"][:10]],
        )

        return {
            "ml_probability": int(ml_prob * 100),
            "ml_confidence": int(ml_conf * 100),
            "top_features": state["top_features"],
            "patient_data": state["patient_data"],
        }

    def _format_integration_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format data for clinical context integration."""
        # Extract clinical feature keys from formatted features
        top_feature_keys = set()

        for feat_display_name, importance in state["top_features"]:
            # Extract base feature name from display name
            if feat_display_name.startswith("Missing: "):
                # Handle missing data patterns - extract the clinical name and map back to abbreviation
                clinical_name = feat_display_name.replace("Missing: ", "")
                # Find matching abbreviation from features_dict
                base_feature = None
                for abbrev, (full_name, _, _) in features_dict.items():
                    if full_name == clinical_name:
                        base_feature = abbrev
                        break
                if not base_feature:
                    # Fallback to simple conversion
                    base_feature = clinical_name.lower().replace(" ", "_")
            else:
                # Regular clinical name - find matching abbreviation
                base_feature = None
                for abbrev, (full_name, _, _) in features_dict.items():
                    if full_name == feat_display_name:
                        base_feature = abbrev
                        break
                if not base_feature:
                    # Fallback to simple conversion
                    base_feature = feat_display_name.lower().replace(" ", "_")

            # Skip _na features for clinical data formatting
            if base_feature and not base_feature.endswith("_na"):
                top_feature_keys.add(base_feature)

        clinical_data = format_clinical_data(
            patient_data=state["patient_data"],
            feature_keys=top_feature_keys,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=True,
            include_temporal_patterns=True,
            include_uncertainty=True,
            original_patient_data=state["original_patient_data"],
        )

        return {
            "ml_results": {
                "probability": state["ml_prediction"],  # Already an integer
                "confidence": int(state["ml_confidence"] * 100),
                "top_features": state["top_features"][:10],
            },
            "clinical_data": clinical_data,
        }

    def _format_investigation_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format data for detailed investigation with comprehensive task-specific features."""
        # Start with features already analyzed in Step 2
        step2_features = set()

        # Add all top important features for detailed analysis
        for feat_display_name, importance in state["top_features"]:
            # Extract base feature name from display name
            if feat_display_name.startswith("Missing: "):
                # Handle missing data patterns - extract the clinical name and map back to abbreviation
                clinical_name = feat_display_name.replace("Missing: ", "")
                # Find matching abbreviation from features_dict
                base_feature = None
                for abbrev, (full_name, _, _) in features_dict.items():
                    if full_name == clinical_name:
                        base_feature = abbrev
                        break
                if not base_feature:
                    # Fallback to simple conversion
                    base_feature = clinical_name.lower().replace(" ", "_")
            else:
                # Regular clinical name - find matching abbreviation
                base_feature = None
                for abbrev, (full_name, _, _) in features_dict.items():
                    if full_name == feat_display_name:
                        base_feature = abbrev
                        break
                if not base_feature:
                    # Fallback to simple conversion
                    base_feature = feat_display_name.lower().replace(" ", "_")

            # Skip _na features for clinical data formatting
            if base_feature and not base_feature.endswith("_na"):
                step2_features.add(base_feature)

        # Add comprehensive task-specific features for deeper investigation
        task_specific_features = get_priority_features_for_task(self.task_name)

        # Identify newly introduced features
        new_features = task_specific_features - step2_features
        all_investigation_features = step2_features | task_specific_features

        logger.debug(
            "Investigation features: step2=%d, task-specific=%d, new=%d, total=%d",
            len(step2_features),
            len(task_specific_features),
            len(new_features),
            len(all_investigation_features),
        )

        clinical_data = format_clinical_data(
            patient_data=state["patient_data"],
            feature_keys=all_investigation_features,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=False,
            include_temporal_patterns=True,
            include_uncertainty=True,
            original_patient_data=state["original_patient_data"],
        )

        return {
            "focus_features": list(all_investigation_features),
            "step2_features": list(step2_features),
            "new_features": list(new_features),
            "clinical_data": clinical_data,
            "disagreement_context": state["agreement"],
            "ml_assessment": state["ml_prediction"],  # Already an integer
            "clinical_assessment": state[
                "clinical_probability"
            ],  # Use the converted value from state
        }

    def _format_synthesis_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format data for final synthesis."""
        patient_data = state["patient_data"]

        # Get demographics
        demographics = {}
        if "age" in patient_data.index:
            demographics["age"] = patient_data["age"]
        if "sex" in patient_data.index:
            demographics["sex"] = patient_data["sex"]
        if "weight" in patient_data.index:
            demographics["weight"] = patient_data["weight"]

        return {
            "demographics": demographics,
            "ml_assessment": {
                "probability": state["ml_prediction"],  # Already an integer
                "confidence": int(state["ml_confidence"] * 100),
                "key_features": state["top_features"][:10],
            },
            "clinical_assessment": state["clinical_assessment"],
            "investigation_conducted": state["needs_investigation"],
            "investigation_results": state.get(
                "dampened_investigation_results", state.get("investigation_results")
            ),
        }

    def _create_ml_stratification_template(self):
        """Template for ML risk stratification."""

        def format_prompt(formatted_data, state):
            ml_prob = formatted_data["ml_probability"]
            ml_conf = formatted_data["ml_confidence"]
            top_features = formatted_data["top_features"]

            # Format top features with clinical interpretation
            features_text = []
            for feat_display_name, importance in top_features:
                features_text.append(f"- {feat_display_name}: {importance:.3f}")

            features_str = (
                "\n".join(features_text)
                if features_text
                else "No significant features identified"
            )

            return f"""An XGBoost model has analyzed this ICU patient's data for {self.task_content['task_name']} risk prediction.

XGBoost Model Assessment:
- Predicted probability of {self.task_content['task_name']}: {ml_prob:.0f}%
- Model confidence: {ml_conf:.0f}%

Top Important Features (by XGBoost Feature Importance):
{features_str}

Clinical Context:
{self.task_content['task_info']}

Task: 
Provide clinical interpretation of the XGBoost model's assessment.

Consider:
- What do these important features suggest clinically?
- XGBoost models excel at capturing non-linear relationships and feature interactions - are there potential interactions between these features?
- Are there any missingness patterns (_na features) that might indicate data quality issues?
- Does the XGBoost prediction align with typical clinical presentation patterns?
- What clinical reasoning might explain this risk level?

Respond in JSON format:
{{
    "diagnosis": "ai-clinical-interpretation",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['task_name']} will not occur and 100 means {self.task_content['task_name']} will definitely occur; your clinical interpretation of the appropriate risk level based on XGBoost findings),
    "explanation": "Clinical interpretation of XGBoost model findings, including significance of important features and any data quality considerations (MAX 190 words)",
    "confidence": XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment; your confidence in interpreting the XGBoost model results)
}}"""

        return format_prompt

    def _create_clinical_integration_template(self):
        """Template for clinical context integration."""

        def format_prompt(formatted_data, state):
            ml_results = formatted_data["ml_results"]
            clinical_data = formatted_data["clinical_data"]

            # Format demographics
            demographics = clinical_data.get("demographics", {})
            demographics_str = format_demographics_str(demographics)

            # Get monitoring period from the data
            monitoring_hours = get_monitoring_period_hours(state["patient_data"])

            # Format clinical data for top features
            vital_signs = clinical_data.get("vital_signs", {})
            clinical_text = format_clinical_text(vital_signs)
            clinical_str = (
                "\n".join(clinical_text)
                if clinical_text
                else "No clinical data available for key features"
            )

            # Format top ML features
            ml_features_text = []
            for feat_display_name, importance in ml_results["top_features"]:
                # Extract just the display name without importance scores or bullet points
                ml_features_text.append(feat_display_name)

            ml_features_str = ", ".join(ml_features_text)

            return f"""Compare XGBoost model assessment with clinical evaluation for {self.task_content['task_name']} risk.

Patient Demographics:
{demographics_str}

XGBoost Model Results:
- XGBoost predicted probability: {ml_results['probability']:.0f}%
- XGBoost confidence: {ml_results['confidence']:.0f}%
- XGBoost identified key factors: {ml_features_str}

Clinical Data for Key Factors (Over {monitoring_hours}-Hour Monitoring Period):
{clinical_str}

Clinical Context:
{self.task_content['task_info']}

Clinical Assessment Task:
Based on your clinical expertise and the actual patient data, provide your independent assessment.

Important Limitations:
- Available data is limited to laboratory values and vital signs only
- No information on: medications, physical examination, patient history, imaging, microbiology
- Your confidence should reflect these data limitations

Consider:
- Do the clinical values support or contradict the XGBoost prediction?
- XGBoost models can capture complex feature interactions - are there clinical patterns the model might have missed or overemphasized?
- How do temporal trends affect your clinical judgment compared to the XGBoost assessment?
- What is your confidence in this clinical assessment given the limited available data?

Respond in JSON format:
{{
    "diagnosis": "clinical-{self.task_content['task_name']}-assessment",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['task_name']} will not occur and 100 means {self.task_content['task_name']} will definitely occur; your independent clinical assessment of {self.task_content['task_name']} risk),
    "explanation": "Your clinical reasoning based on patient data, noting agreement/disagreement with XGBoost assessment (MAX 180 words)",
    "confidence": XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment; confidence reflects your certainty in your own reasoning based on the available data),
    "ai_agreement": "agree/partial/disagree (how well your clinical assessment aligns with the XGBoost prediction)"
}}

Warning: 
Any significant deviation from the XGBoost-predicted probability for {self.task_content['task_name']} must be backed by a strong clinical justification."""

        return format_prompt

    def _create_detailed_investigation_template(self):
        """Template for detailed investigation."""

        def format_prompt(formatted_data, state):
            focus_features = formatted_data["focus_features"]
            step2_features = formatted_data["step2_features"]
            new_features = formatted_data["new_features"]
            clinical_data = formatted_data["clinical_data"]
            disagreement = formatted_data["disagreement_context"]
            ml_prob = formatted_data["ml_assessment"]
            clinical_prob = formatted_data["clinical_assessment"]

            # Get monitoring period from the data
            monitoring_hours = get_monitoring_period_hours(state["patient_data"])

            # Format clinical data for investigation
            clinical_text = format_clinical_text(clinical_data)
            clinical_str = "\n".join(clinical_text)

            # Create feature breakdown text
            feature_breakdown = ""
            if step2_features and new_features:
                # Convert to full names for display
                step2_display = [get_feature_name(f) for f in sorted(step2_features)]
                new_display = [get_feature_name(f) for f in sorted(new_features)]

                feature_breakdown = f"""
Feature Analysis Breakdown:
- Previously analyzed features: {', '.join(step2_display)}
- Newly introduced features: {', '.join(new_display)}
- Total features for investigation: {len(focus_features)}

Focus:
Pay special attention to newly introduced features that may explain the disagreement."""
            elif new_features:
                # Convert to full names for display
                new_display = [get_feature_name(f) for f in sorted(new_features)]

                feature_breakdown = f"""
Feature Analysis Breakdown:
- Newly introduced features: {', '.join(new_display)}
- Total features for investigation: {len(focus_features)}

Focus:
These are additional clinical parameters not previously considered."""
            else:
                feature_breakdown = f"""
Feature Analysis Breakdown:
- All features were previously analyzed in initial assessment
- Total features for investigation: {len(focus_features)}

Focus:
Look for subtle patterns and interactions in the existing data."""

            return f"""Detailed Clinical Investigation

Disagreement Detected:
- XGBoost model prediction: {ml_prob:.0f}%
- Clinical assessment: {clinical_prob:.0f}%
- Probability difference: {disagreement['probability_difference']:.0f}
- XGBoost confidence: {disagreement['ml_confidence']:.0f}%
{feature_breakdown}

Detailed Clinical Data (Over {monitoring_hours}-Hour Monitoring Period):
{clinical_str}

Clinical Context:
{self.task_content['task_info']}

Investigation Task:
Conduct a focused analysis to resolve the disagreement between XGBoost and clinical assessments.

Analyze:
- Do newly introduced features provide additional insight into the disagreement?
- Are there subtle clinical patterns that explain the disagreement?
- Could temporal trends provide additional insight into the disagreement?
- Are there interactions between parameters that affect risk assessment (XGBoost models excel at capturing such interactions)?
- Which assessment (XGBoost or initial clinical) appears more reliable given the comprehensive data?

Respond in JSON format:
{{
    "diagnosis": "detailed-investigation-{self.task_content['task_name']}",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['task_name']} will not occur and 100 means {self.task_content['task_name']} will definitely occur; refined probability assessment after detailed investigation),
    "explanation": "Analysis explaining the disagreement and your refined assessment based on thorough investigation (MAX 190 words)",
    "confidence": XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your assessment; confidence reflects your certainty in your own reasoning based on the available data)
}}"""

        return format_prompt

    def _create_synthesis_template(self):
        """Template for confidence-weighted synthesis."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            ml_assessment = formatted_data["ml_assessment"]
            clinical_assessment = formatted_data["clinical_assessment"]
            investigation_conducted = formatted_data["investigation_conducted"]
            investigation_results = formatted_data.get("investigation_results")

            # Format demographics
            demographics_str = format_demographics_str(demographics)

            # Format key AI features
            ai_features = []
            for feat_display_name, importance in ml_assessment["key_features"]:
                ai_features.append(feat_display_name)

            investigation_text = ""
            if investigation_conducted and investigation_results:
                # Format investigation probability correctly as integer percentage
                investigation_prob = investigation_results.get("probability", "N/A")
                if isinstance(investigation_prob, (int, float)):
                    # Convert to integer percentage if it's a decimal
                    if investigation_prob < 1:
                        investigation_prob = int(investigation_prob * 100)
                    else:
                        investigation_prob = int(investigation_prob)
                    investigation_prob_text = f"{investigation_prob}%"
                else:
                    investigation_prob_text = f"{investigation_prob}%"

                investigation_text = f"""
Detailed Investigation Results:
- Refined probability: {investigation_prob_text}
- Investigation findings: {investigation_results.get('explanation', 'No details available')}"""
            else:
                investigation_text = "\nDetailed Investigation:\nNot required (high confidence and agreement)"

            # Format clinical probability for display
            clinical_prob_display = clinical_assessment.get("probability", "N/A")
            if isinstance(clinical_prob_display, (int, float)):
                if clinical_prob_display < 1:
                    clinical_prob_display = clinical_prob_display * 100
                clinical_prob_text = f"{clinical_prob_display:.0f}%"
            else:
                clinical_prob_text = f"{clinical_prob_display}%"

            return f"""Patient Demographics:
{demographics_str}

Hybrid XGBoost-Clinical Assessment Summary:

XGBoost Model Assessment:
- XGBoost prediction: {ml_assessment['probability']:.0f}%
- XGBoost confidence: {ml_assessment['confidence']:.0f}%
- Key XGBoost factors: {', '.join(ai_features)}

Clinical Assessment:
- Clinical prediction: {clinical_prob_text}
- Clinical confidence: {clinical_assessment.get('confidence', 'N/A')}%
- AI agreement level: {clinical_assessment.get('ai_agreement', 'unclear')}
- Clinical reasoning: {clinical_assessment.get('explanation', 'No clinical reasoning provided')}
{investigation_text}

Synthesis Guidance:
- XGBoost confidence ({ml_assessment['confidence']:.0f}%) means {ml_assessment['confidence']:.0f}% certain the risk is {ml_assessment['probability']:.0f}%
- High-confidence predictions (>80%) should strongly anchor your assessment
- Clinical assessment has limited context (no medications, physical exam, full history)
- Justify any major deviation from high-confidence ML predictions

Clinical Context: 
{self.task_content['task_info']}"""

        return format_prompt

    def _prepare_step_metadata(
        self, step_name: str, state: Dict[str, Any], output: Any
    ) -> Dict[str, Any]:
        """Prepare step-specific metadata for hybrid reasoning workflow."""
        additional_metadata = {}

        # Base metadata for all steps
        additional_metadata.update(
            {
                "metadata_total_features_available": len(
                    state.get("available_features", set())
                ),
                "metadata_data_completeness_score": self._calculate_data_completeness(
                    state.get("patient_data")
                ),
            }
        )

        if step_name == "ml_interpretation":
            # Log ML model results that aren't captured elsewhere
            additional_metadata.update(
                {
                    "metadata_ml_prediction": state.get("ml_prediction", 0),
                    "metadata_ml_confidence": int(state.get("ml_confidence", 0) * 100),
                    "metadata_ml_top_unique_features": str(
                        [
                            f[0]
                            for f in state.get("top_features", [])[
                                : self.top_features_count
                            ]
                        ]
                    ),
                    "metadata_xgb_model_available": self.xgb_model is not None,
                    "metadata_xgb_feature_count": (
                        len(self.xgb_feature_names) if self.xgb_feature_names else 0
                    ),
                }
            )

        elif step_name == "clinical_assessment":
            # Log agreement analysis that isn't captured elsewhere
            ml_prediction = state.get("ml_prediction", 0)
            clinical_prob = state.get("clinical_probability", 0)

            additional_metadata.update(
                {
                    "metadata_ml_vs_clinical_diff": abs(ml_prediction - clinical_prob),
                    "metadata_ai_agreement": (
                        output.get("ai_agreement") if isinstance(output, dict) else None
                    ),
                    "metadata_ml_confidence_adequate": state.get("ml_confidence", 0)
                    >= self.ml_confidence_threshold,
                }
            )

        elif step_name == "detailed_investigation":
            # Log investigation trigger logic that isn't captured elsewhere
            agreement = state.get("agreement", {})
            additional_metadata.update(
                {
                    "metadata_investigation_triggered": state.get(
                        "needs_investigation", False
                    ),
                    "metadata_ml_confidence_low": agreement.get("ml_confidence", 0)
                    < int(self.ml_confidence_threshold * 100),
                    "metadata_high_disagreement": agreement.get(
                        "probability_difference", 0
                    )
                    > int(self.agreement_threshold * 100),
                    "metadata_agreement_threshold": int(self.agreement_threshold * 100),
                }
            )

        elif step_name == "final_prediction":
            # Calculate objective synthesis using mathematical formula
            # Convert ML prediction from percentage to 0-1 range for consistent calculation
            ml_prob = state.get("ml_prediction", 50) / 100.0
            ml_conf = state.get("ml_confidence", 0.5)

            clinical_assessment = state.get("clinical_assessment", {})
            clinical_prob = parse_numeric_value(
                clinical_assessment.get("probability", 0.5), 0.5
            )
            clinical_conf = (
                parse_numeric_value(clinical_assessment.get("confidence", 0), 0) / 100.0
            )

            # Use dampened investigation results if available
            final_clinical_prob = clinical_prob
            if state.get("needs_investigation", False):
                if state.get("dampened_investigation_results"):
                    # Use dampened results
                    final_clinical_prob = parse_numeric_value(
                        state["dampened_investigation_results"].get(
                            "probability", clinical_prob
                        ),
                        clinical_prob,
                    )
                    clinical_conf = (
                        parse_numeric_value(
                            state["dampened_investigation_results"].get(
                                "confidence", 0
                            ),
                            0,
                        )
                        / 100.0
                    )
                elif state.get("investigation_results"):
                    # Use original results
                    investigation_results = state["investigation_results"]
                    final_clinical_prob = parse_numeric_value(
                        investigation_results.get("probability", clinical_prob),
                        clinical_prob,
                    )
                    clinical_conf = (
                        parse_numeric_value(
                            investigation_results.get("confidence", 0), 0
                        )
                        / 100.0
                    )

            # Calculate confidence-weighted average
            total_weight = ml_conf + clinical_conf
            if total_weight > 0:
                objective_synthesis = (
                    ml_prob * ml_conf + final_clinical_prob * clinical_conf
                ) / total_weight
            else:
                objective_synthesis = (
                    ml_prob + final_clinical_prob
                ) / 2  # Fallback to simple average

            # Log metadata
            additional_metadata.update(
                {
                    "metadata_objective_synthesis": objective_synthesis,
                    "metadata_dampened_clinical": final_clinical_prob,
                    "metadata_max_allowed_deviation": (
                        0.35 if ml_conf >= 0.8 else 0.5
                    ),  # Simplified for logging
                    "metadata_dampening_applied": state.get(
                        "dampened_investigation_results"
                    )
                    is not None,
                    "metadata_original_clinical": (
                        parse_numeric_value(
                            state.get("investigation_results", {}).get(
                                "probability", clinical_prob
                            ),
                            clinical_prob,
                        )
                        if state.get("needs_investigation")
                        else clinical_prob
                    ),
                }
            )

        return additional_metadata

    def _calculate_data_completeness(self, patient_data: Any) -> float:
        """Calculate data completeness score."""
        try:
            if patient_data is not None and hasattr(patient_data, "index"):
                non_na_count = patient_data.count()
                total_count = len(patient_data)
                return float(non_na_count / total_count) if total_count > 0 else 0.0
            return 1.0
        except Exception as e:
            logger.warning("Error calculating data completeness: %s", e)
            return 0.0
