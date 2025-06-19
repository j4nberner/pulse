import glob
import logging
import os
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd

from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.agent_util import (create_error_response, extract_confidence,
                                 filter_na_columns, format_clinical_data,
                                 format_clinical_text)
from src.util.data_util import get_feature_name

logger = logging.getLogger("PULSE_logger")


class HybridReasoningAgent(PulseAgent):
    """
    Hybrid AI-Clinical Reasoning Agent that combines pretrained XGBoost predictions
    with LLM-based clinical reasoning using feature importance guidance.

    Workflow:
    1. ML Risk Stratification (XGBoost prediction + feature importance)
    2. Clinical Context Integration (interpret ML findings clinically)
    3. Targeted Investigation (conditional based on confidence/agreement)
    4. Confidence-Weighted Synthesis (combine ML + clinical reasoning)
    """

    def __init__(
        self,
        model: Any,
        task_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        metrics_tracker: Optional[Any] = None,
        ml_confidence_threshold: float = 0.7,
        agreement_threshold: float = 0.2,
        top_features_count: int = 8,
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
        self.task_content = self._get_task_specific_content()

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Initialize XGBoost model
        self.xgb_model = None
        self.xgb_feature_names = None
        self._load_xgb_model()

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
            name="ml_risk_stratification",
            system_message="You are an AI-assisted clinical decision support specialist. Analyze the ML model's risk prediction and feature importance to provide clinical interpretation of the AI assessment.",
            prompt_template=self._create_ml_stratification_template(),
            input_formatter=self._format_ml_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 2: Clinical Context Integration
        self.add_step(
            name="clinical_context_integration",
            system_message="You are an experienced ICU physician evaluating how AI predictions align with clinical expectations. Assess agreement between ML findings and clinical reasoning.",
            prompt_template=self._create_clinical_integration_template(),
            input_formatter=self._format_integration_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 3: Targeted Investigation (conditional)
        self.add_step(
            name="targeted_investigation",
            system_message="You are conducting a focused clinical investigation of discrepant or uncertain findings. Analyze the most important clinical parameters in detail.",
            prompt_template=self._create_targeted_investigation_template(),
            input_formatter=self._format_investigation_data,
            output_processor=None,
            parse_json=True,
        )

        # Step 4: Confidence-Weighted Synthesis
        self.add_step(
            name="confidence_weighted_synthesis",
            system_message=None,  # Uses default system message
            prompt_template=self._create_synthesis_template(),
            input_formatter=self._format_synthesis_data,
            output_processor=None,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient through the hybrid reasoning workflow."""
        # Reset memory
        self.memory.reset()

        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        self.memory.set_current_sample(sample_id)

        # Keep original data with _na columns for XGBoost
        original_patient_data = patient_data.copy()

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
            ml_result = self.run_step(
                "ml_risk_stratification", original_patient_data, state
            )
            ml_output = ml_result["output"]

            if isinstance(ml_output, str) and "Error" in ml_output:
                logger.error("ML stratification failed: %s", ml_output)
                return create_error_response("ML risk stratification failed")

            # Update state with ML results
            state["ml_prediction"] = ml_output.get("probability", 50)
            state["ml_confidence"] = extract_confidence(ml_output)

            # Step 2: Clinical Context Integration
            integration_result = self.run_step(
                "clinical_context_integration", None, state
            )
            integration_output = integration_result["output"]

            if isinstance(integration_output, str) and "Error" in integration_output:
                logger.error("Clinical integration failed: %s", integration_output)
                return create_error_response("Clinical integration failed")

            state["clinical_assessment"] = integration_output

            # Determine if detailed investigation is needed
            ml_conf = state["ml_confidence"]
            clinical_prob = integration_output.get("probability", 50) / 100.0
            ml_prob = state["ml_prediction"] / 100.0

            # Check agreement between ML and clinical assessment
            prob_difference = abs(ml_prob - clinical_prob)
            low_ml_confidence = ml_conf < self.ml_confidence_threshold
            high_disagreement = prob_difference > self.agreement_threshold

            state["agreement"] = {
                "probability_difference": prob_difference,
                "ml_confidence": ml_conf,
                "clinical_confidence": extract_confidence(integration_output),
            }

            state["needs_investigation"] = low_ml_confidence or high_disagreement

            logger.info(
                "ML vs Clinical: %.3f vs %.3f (diff=%.3f), ML conf=%.3f, Investigation needed: %s",
                ml_prob,
                clinical_prob,
                prob_difference,
                ml_conf,
                state["needs_investigation"],
            )

            # Step 3: Targeted Investigation (conditional)
            if state["needs_investigation"]:
                investigation_result = self.run_step(
                    "targeted_investigation", None, state
                )
                investigation_output = investigation_result["output"]
                state["investigation_results"] = investigation_output
                logger.info(
                    "Conducted targeted investigation due to disagreement/uncertainty"
                )
            else:
                state["investigation_results"] = None
                logger.info(
                    "Skipped targeted investigation - high confidence and agreement"
                )

            # Step 4: Confidence-Weighted Synthesis
            synthesis_result = self.run_step(
                "confidence_weighted_synthesis", None, state
            )
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
        state["ml_prediction"] = ml_prob * 100  # Convert to percentage
        state["ml_confidence"] = ml_conf
        state["feature_importance"] = feature_importance

        # Get top important features with improved clinical naming
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )

            # Format features for clinical display
            formatted_features = []
            for feat_name, importance in sorted_features[: self.top_features_count]:
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
                    # Regular feature
                    base_feature = (
                        feat_name.split("_")[0] if "_" in feat_name else feat_name
                    )
                    clinical_name = get_feature_name(base_feature)
                    if clinical_name and clinical_name != "Unknown":
                        display_name = f"{clinical_name} ({feat_name})"
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
            [f[0] for f in state["top_features"][:3]],
        )

        return {
            "ml_probability": ml_prob * 100,
            "ml_confidence": ml_conf * 100,
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
                # Handle missing data patterns
                clinical_name = feat_display_name.replace("Missing: ", "")
                # Try to reverse-lookup the feature key
                base_feature = clinical_name.lower().replace(" ", "_")
            elif "(" in feat_display_name and feat_display_name.endswith(")"):
                # Extract actual feature name from "Clinical Name (feat_name)" format
                feat_name = feat_display_name.split("(")[-1].rstrip(")")
                base_feature = (
                    feat_name.split("_")[0] if "_" in feat_name else feat_name
                )
            else:
                # Direct feature name
                base_feature = (
                    feat_display_name.split("_")[0]
                    if "_" in feat_display_name
                    else feat_display_name
                )

            # Skip _na features for clinical data formatting
            if not base_feature.endswith("_na"):
                top_feature_keys.add(base_feature)

        clinical_data = format_clinical_data(
            patient_data=state["patient_data"],
            feature_keys=top_feature_keys,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=True,
            include_temporal_patterns=True,
        )

        return {
            "ml_results": {
                "probability": state["ml_prediction"],
                "confidence": state["ml_confidence"] * 100,
                "top_features": state["top_features"][:5],  # Show top 5 for brevity
            },
            "clinical_data": clinical_data,
        }

    def _format_investigation_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format data for targeted investigation."""
        # Focus on discrepant features - those with high importance but potential clinical disagreement
        investigation_features = set()

        # Add all top important features for detailed analysis
        for feat_display_name, importance in state["top_features"]:
            # Extract base feature name from display name
            if feat_display_name.startswith("Missing: "):
                clinical_name = feat_display_name.replace("Missing: ", "")
                base_feature = clinical_name.lower().replace(" ", "_")
            elif "(" in feat_display_name and feat_display_name.endswith(")"):
                feat_name = feat_display_name.split("(")[-1].rstrip(")")
                base_feature = (
                    feat_name.split("_")[0] if "_" in feat_name else feat_name
                )
            else:
                base_feature = (
                    feat_display_name.split("_")[0]
                    if "_" in feat_display_name
                    else feat_display_name
                )

            # Skip _na features for clinical data formatting
            if not base_feature.endswith("_na"):
                investigation_features.add(base_feature)

        clinical_data = format_clinical_data(
            patient_data=state["patient_data"],
            feature_keys=investigation_features,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=False,
            include_temporal_patterns=True,
        )

        return {
            "focus_features": list(investigation_features),
            "clinical_data": clinical_data,
            "disagreement_context": state["agreement"],
            "ml_assessment": state["ml_prediction"],
            "clinical_assessment": state["clinical_assessment"].get("probability", 50),
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
                "probability": state["ml_prediction"],
                "confidence": state["ml_confidence"] * 100,
                "key_features": state["top_features"][:3],
            },
            "clinical_assessment": state["clinical_assessment"],
            "investigation_conducted": state["needs_investigation"],
            "investigation_results": state.get("investigation_results"),
            "agreement_analysis": state["agreement"],
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

            return f"""An AI model has analyzed this ICU patient's data for {self.task_content['complication_name']} risk prediction.

AI MODEL ASSESSMENT:
- Predicted probability of {self.task_content['complication_name']}: {ml_prob:.1f}%
- Model confidence: {ml_conf:.1f}%

TOP IMPORTANT FEATURES (by AI model importance):
{features_str}

Clinical Context:
{self.task_content['task_info']}

TASK: Provide clinical interpretation of the AI model's assessment.

Consider:
- What do these important features suggest clinically?
- Are there any missingness patterns (_na features) that might indicate data quality issues?
- Does the AI prediction align with typical clinical presentation patterns?
- What clinical reasoning might explain this risk level?

Respond in JSON format:
{{
    "diagnosis": "ai-clinical-interpretation",
    "probability": XX (your clinical interpretation of the appropriate risk level based on AI findings, integer 0-100),
    "explanation": "Clinical interpretation of AI model findings, including significance of important features and any data quality considerations",
    "confidence": XX (your confidence in interpreting the AI model results, integer 0-100)
}}"""

        return format_prompt

    def _create_clinical_integration_template(self):
        """Template for clinical context integration."""

        def format_prompt(formatted_data, state):
            ml_results = formatted_data["ml_results"]
            clinical_data = formatted_data["clinical_data"]

            # Format demographics
            demographics = clinical_data.get("demographics", {})
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
                ml_features_text.append(f"- {feat_display_name}")

            ml_features_str = "\n".join(ml_features_text)

            return f"""Compare AI model assessment with clinical evaluation for {self.task_content['complication_name']} risk.

Patient Demographics:
{demographics_str}

AI MODEL RESULTS:
- AI predicted probability: {ml_results['probability']:.1f}%
- AI confidence: {ml_results['confidence']:.1f}%
- AI identified key factors:
{ml_features_str}

CLINICAL DATA FOR KEY FACTORS:
{clinical_str}

CLINICAL ASSESSMENT TASK:
Based on your clinical expertise and the actual patient data, provide your independent assessment.

Consider:
- Do the clinical values support or contradict the AI prediction?
- Are there clinical patterns the AI might have missed or overemphasized?
- How do temporal trends affect your clinical judgment?
- What is your confidence in this clinical assessment?

Respond in JSON format:
{{
    "diagnosis": "clinical-{self.task_content['complication_name']}-assessment",
    "probability": XX (your independent clinical assessment of {self.task_content['complication_name']} risk, integer 0-100),
    "explanation": "Your clinical reasoning based on patient data, noting agreement/disagreement with AI assessment",
    "confidence": XX (your confidence in this clinical assessment, integer 0-100),
    "ai_agreement": "agree/partial/disagree (how well your clinical assessment aligns with the AI prediction)"
}}"""

        return format_prompt

    def _create_targeted_investigation_template(self):
        """Template for targeted investigation."""

        def format_prompt(formatted_data, state):
            focus_features = formatted_data["focus_features"]
            clinical_data = formatted_data["clinical_data"]
            disagreement = formatted_data["disagreement_context"]
            ml_prob = formatted_data["ml_assessment"]
            clinical_prob = formatted_data["clinical_assessment"]

            # Format clinical data for investigation
            clinical_text = format_clinical_text(clinical_data)
            clinical_str = "\n".join(clinical_text)

            return f"""TARGETED CLINICAL INVESTIGATION

DISAGREEMENT DETECTED:
- AI model prediction: {ml_prob:.1f}%
- Clinical assessment: {clinical_prob:.1f}%
- Probability difference: {disagreement['probability_difference']:.3f}
- AI confidence: {disagreement['ml_confidence']:.3f}

FOCUS AREAS FOR INVESTIGATION:
Clinical parameters requiring detailed analysis: {', '.join(focus_features)}

DETAILED CLINICAL DATA:
{clinical_str}

INVESTIGATION TASK:
Conduct a focused analysis to resolve the disagreement between AI and clinical assessments.

Analyze:
- Are there subtle clinical patterns that explain the disagreement?
- Could temporal trends provide additional insight?
- Are there interactions between parameters that affect risk assessment?
- Which assessment (AI or initial clinical) appears more reliable given the detailed data?

Respond in JSON format:
{{
    "diagnosis": "targeted-investigation-{self.task_content['complication_name']}",
    "probability": XX (refined probability assessment after detailed investigation, integer 0-100),
    "explanation": "Detailed analysis explaining the disagreement and your refined assessment based on thorough investigation",
    "confidence": XX (confidence in this refined assessment, integer 0-100),
    "resolution": "favor-ai/favor-clinical/synthesis (which approach your investigation supports)"
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
            agreement_analysis = formatted_data["agreement_analysis"]

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

            # Format key AI features
            ai_features = []
            for feat_display_name, importance in ml_assessment["key_features"]:
                # Use the already formatted display name
                ai_features.append(feat_display_name)

            investigation_text = ""
            if investigation_conducted and investigation_results:
                investigation_text = f"""
DETAILED INVESTIGATION RESULTS:
- Refined probability: {investigation_results.get('probability', 'N/A')}%
- Investigation findings: {investigation_results.get('explanation', 'No details available')}
- Resolution approach: {investigation_results.get('resolution', 'unclear')}"""
            else:
                investigation_text = "\nDETAILED INVESTIGATION: Not required (high confidence and agreement)"

            return f"""Patient Demographics:
{demographics_str}

HYBRID AI-CLINICAL ASSESSMENT SUMMARY:

AI MODEL ASSESSMENT:
- AI prediction: {ml_assessment['probability']:.1f}%
- AI confidence: {ml_assessment['confidence']:.1f}%
- Key AI factors: {', '.join(ai_features)}

CLINICAL ASSESSMENT:
- Clinical prediction: {clinical_assessment.get('probability', 'N/A')}%
- Clinical confidence: {clinical_assessment.get('confidence', 'N/A')}%
- AI agreement level: {clinical_assessment.get('ai_agreement', 'unclear')}

AGREEMENT ANALYSIS:
- Probability difference: {agreement_analysis['probability_difference']:.3f}
- AI confidence: {agreement_analysis['ml_confidence']:.3f}
- Clinical confidence: {agreement_analysis['clinical_confidence']:.3f}{investigation_text}

SYNTHESIS TASK:
Provide a final assessment that intelligently combines AI model predictions with clinical reasoning.

Guidelines for synthesis:
- Weight assessments by their respective confidence levels
- Consider the agreement/disagreement between AI and clinical approaches
- If investigation was conducted, incorporate those refined findings
- Provide a probability that reflects the best integration of both approaches
- Explain how you balanced AI predictions with clinical judgment

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
