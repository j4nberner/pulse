import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.models.agents.pulse_agent import PulseAgent
from src.preprocessing.preprocessing_advanced.preprocessing_advanced import \
    PreprocessorAdvanced
from src.util.agent_util import (create_error_response, filter_na_columns,
                                 format_clinical_data, format_clinical_text,
                                 get_specialist_features, get_specialist_system_message)

logger = logging.getLogger("PULSE_logger")


class CollaborativeReasoningAgent(PulseAgent):
    """
    Multi-specialist collaborative reasoning agent with uncertainty integration.

    Architecture:
    1. Phase 1: Parallel specialist analysis (Hemodynamic, Metabolic, Hematologic)
    2. Phase 2: Synthesis with uncertainty integration (no inter-agent communication)

    Each specialist analyzes their domain independently, then a synthesis step
    combines their expertise with uncertainty considerations.
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
        super().__init__(
            model=model,
            task_name=task_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            metrics_tracker=metrics_tracker,
            **kwargs,
        )

        self.task_content = self._get_task_specific_content()

        # Initialize preprocessing tools
        self.preprocessor_advanced = PreprocessorAdvanced()

        # Define specialist types
        self.specialist_types = ["hemodynamic", "metabolic", "hematologic"]

        self._define_steps()

    def _define_steps(self) -> None:
        """Define the collaborative reasoning workflow steps."""

        # Phase 1: Specialist assessments (parallel)
        for specialist_type in self.specialist_types:
            self.add_step(
                name=f"{specialist_type}_assessment",
                system_message=get_specialist_system_message(
                    specialist_type, self.task_name
                ),
                prompt_template=self._create_specialist_assessment_template(
                    specialist_type
                ),
                input_formatter=self._format_specialist_data,
                output_processor=None,
                parse_json=True,
            )

        # Phase 2: Synthesis step
        self.add_step(
            name="synthesis",
            system_message=None,  # Uses default system message
            prompt_template=self._create_synthesis_template(),
            input_formatter=self._format_synthesis_data,
            output_processor=None,
            parse_json=True,
        )

    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient through the collaborative reasoning workflow."""
        # Reset memory
        self.memory.reset()

        sample_id = patient_data.name if hasattr(patient_data, "name") else "default"
        self.memory.set_current_sample(sample_id)

        # Keep original data with _na columns for uncertainty analysis
        original_patient_data = patient_data.copy()

        # Filter out _na columns for regular processing
        filtered_patient_data = filter_na_columns(patient_data)

        # Initialize state
        state = {
            "patient_data": filtered_patient_data,
            "original_patient_data": original_patient_data,
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "available_features": set(filtered_patient_data.index),
            "specialist_assessments": {},
            "overall_confidence": 0.0,
        }

        try:
            # Phase 1: Run all specialist assessments in parallel
            for specialist_type in self.specialist_types:
                # Get features for this specialist
                specialist_features = get_specialist_features(
                    specialist_type, state["available_features"]
                )

                if not specialist_features:
                    logger.warning(
                        "No features available for %s specialist", specialist_type
                    )
                    # Create default assessment for missing specialist
                    state["specialist_assessments"][specialist_type] = {
                        "diagnosis": f"insufficient-data-{specialist_type}",
                        "probability": 50,
                        "confidence": 10,
                        "explanation": f"Insufficient {specialist_type} data available for assessment",
                        "features_analyzed": [],
                    }
                    continue

                # Run specialist assessment
                specialist_result = self.run_step(
                    f"{specialist_type}_assessment", specialist_features, state
                )
                specialist_output = specialist_result["output"]

                # Handle error cases
                if isinstance(specialist_output, str) and "Error" in specialist_output:
                    logger.error(
                        "%s assessment failed: %s", specialist_type, specialist_output
                    )
                    state["specialist_assessments"][specialist_type] = {
                        "diagnosis": f"error-{specialist_type}",
                        "probability": 50,
                        "confidence": 10,
                        "explanation": f"Error in {specialist_type} assessment: {specialist_output}",
                        "features_analyzed": list(specialist_features),
                    }
                    continue

                # Store specialist assessment with additional metadata
                state["specialist_assessments"][specialist_type] = {
                    **specialist_output,
                    "features_analyzed": list(specialist_features),
                }

                logger.info(
                    "%s assessment complete. Probability: %s, Confidence: %s",
                    specialist_type,
                    specialist_output.get("probability", "unknown"),
                    specialist_output.get("confidence", "unknown"),
                )

            # Phase 2: Synthesis
            synthesis_result = self.run_step("synthesis", None, state)
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
            logger.error(
                "Error in collaborative reasoning workflow: %s", e, exc_info=True
            )
            return create_error_response(f"Collaborative reasoning error: {str(e)}")

    def _format_specialist_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format specialist-specific data for assessment."""
        specialist_features = input_data  # Set of feature keys for this specialist
        patient_data = state["patient_data"]
        original_patient_data = state["original_patient_data"]

        return format_clinical_data(
            patient_data=patient_data,
            feature_keys=specialist_features,
            preprocessor_advanced=self.preprocessor_advanced,
            include_demographics=True,
            include_temporal_patterns=True,
            include_uncertainty=True,
            original_patient_data=original_patient_data,
        )

    def _format_synthesis_data(
        self, state: Dict[str, Any], input_data: Any
    ) -> Dict[str, Any]:
        """Format data for synthesis step."""
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
            "specialist_assessments": state["specialist_assessments"],
            "task_name": self.task_name,
        }

    def _create_specialist_assessment_template(self, specialist_type: str):
        """Template for specialist assessment."""

        def format_prompt(formatted_data, state):
            # Extract demographics and clinical data
            demographics = formatted_data.get("demographics", {})

            # Clinical data might be under 'vital_signs' key or directly
            if "vital_signs" in formatted_data:
                clinical_data = formatted_data["vital_signs"]
            else:
                clinical_data = {
                    k: v for k, v in formatted_data.items() if k != "demographics"
                }

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

            # Format clinical data
            clinical_text = format_clinical_text(clinical_data)
            clinical_str = (
                "\n".join(clinical_text)
                if clinical_text
                else "No data available for this domain"
            )

            # Get specialist focus description
            specialist_focus = {
                "hemodynamic": "cardiovascular stability, perfusion status, and cardiac function",
                "metabolic": "acid-base balance, electrolyte status, and organ function",
                "hematologic": "blood counts, immune response, and coagulation status",
            }

            focus_desc = specialist_focus.get(
                specialist_type, "specialized clinical parameters"
            )

            return f"""You are a {specialist_type} specialist analyzing ICU patient data for {self.task_content['complication_name']} risk.

Patient Demographics:
{demographics_str}

{specialist_type.title()} Data (over monitoring period):
{clinical_str}

Clinical Context:
{self.task_content['task_info']}

Specialist Focus:
As a {specialist_type} specialist, focus on {focus_desc}. Provide your domain-specific assessment based on your expertise.

Pay attention to:
- Temporal patterns (trend direction and clinical significance)
- Data uncertainty and completeness
- Domain-specific abnormalities and their severity
- Clinical relationships between parameters in your domain

Respond in JSON format:
{{
    "diagnosis": "specialist-{self.task_content['complication_name']}-assessment",
    "probability": XX (integer between 0 and 100, where 0 means {self.task_content['complication_name']} will not occur and 100 means {self.task_content['complication_name']} will definitely occur),
    "explanation": "Your concise {specialist_type} specialist assessment including key findings, temporal patterns, and clinical significance (MAX 150 words)",
    "confidence": XX (integer between 0 and 100, where 0 means not confident at all and 100 means very confident in your {specialist_type} assessment)
}}

IMPORTANT: Base your confidence on data completeness, clarity of findings, and strength of evidence in your domain."""

        return format_prompt

    def _create_synthesis_template(self):
        """Template for synthesis step combining all specialist assessments."""

        def format_prompt(formatted_data, state):
            demographics = formatted_data["demographics"]
            specialist_assessments = formatted_data["specialist_assessments"]

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

            # Format specialist assessments
            assessment_summary = []
            total_confidence = 0
            valid_assessments = 0

            for specialist_type, assessment in specialist_assessments.items():
                probability = assessment.get("probability", 50)
                confidence = assessment.get("confidence", 50)
                explanation = assessment.get("explanation", "No explanation provided")

                # Count valid assessments for confidence calculation
                if not assessment.get("diagnosis", "").startswith(
                    ("error-", "insufficient-data-")
                ):
                    total_confidence += float(confidence)
                    valid_assessments += 1

                assessment_summary.append(
                    f"\n{specialist_type.upper()} SPECIALIST:\n"
                    f"- Probability: {probability}%\n"
                    f"- Confidence: {confidence}%\n"
                    f"- Assessment: {explanation}\n"
                )

            # Calculate average confidence for weighting
            avg_confidence = total_confidence / max(valid_assessments, 1)

            assessment_str = "".join(assessment_summary)

            return f"""Patient Demographics:
{demographics_str}

SPECIALIST ASSESSMENTS:
{assessment_str}

Clinical Context:
{self.task_content['task_info']}

SYNTHESIS TASK:
As the coordinating physician, integrate the specialist assessments to provide a final clinical decision regarding {self.task_content['complication_name']} risk.

Consider:
- Agreement/disagreement between specialists
- Confidence levels of each assessment
- Clinical coherence of the overall picture
- Data quality and completeness across domains
- Physiological relationships between domains

Guidelines:
- Weight assessments by specialist confidence levels
- Higher agreement between confident specialists increases overall confidence
- Conflicting assessments from confident specialists require careful interpretation
- Consider the clinical context and typical presentation patterns

Average specialist confidence: {avg_confidence:.1f}%"""

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
