import logging
from datetime import datetime
from typing import Any, Dict, Optional


from src.util.agent_util import (get_monitoring_period_hours,
                                 get_specialist_features)

logger = logging.getLogger("PULSE_logger")

# ------------------------------------
# Memory Management for Agent Reasoning Steps
# ------------------------------------


class StepMemory:
    """Memory of a single reasoning step."""

    def __init__(self, step_number: int, step_name: str):
        self.step_number = step_number
        self.step_name = step_name
        self.system_message = None
        self.input = None
        self.output = None
        self.num_input_tokens = 0
        self.num_output_tokens = 0
        self.token_time = 0.0
        self.infer_time = 0.0
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "step_name": self.step_name,
            "system_message": self.system_message,
            "input": self.input,
            "output": self.output,
            "num_input_tokens": self.num_input_tokens,
            "num_output_tokens": self.num_output_tokens,
            "token_time": self.token_time,
            "infer_time": self.infer_time,
            "timestamp": self.timestamp,
        }


class AgentMemoryManager:
    """Manager for agent reasoning steps and autonomous decision tracking."""

    def __init__(
        self,
        agent_id: str,
        output_dir: Optional[str] = None,
        metrics_tracker: Optional[Any] = None,
    ):
        self.agent_id = agent_id
        self.samples = {}  # Change from steps to samples dictionary
        self.current_sample_id = None  # Track current sample being processed
        self.total_samples = 0  # Track total expected samples
        self.metrics_tracker = metrics_tracker  # Store reference
        self.current_target_label = None  # Store current sample's target
        self.agent_instance = None  # Store reference to agent for metadata

    def set_current_sample(
        self, sample_id: Any, patient_data: Optional[Any] = None
    ) -> None:
        """Set the current sample being processed and optionally add sample-level metadata."""
        self.current_sample_id = str(sample_id)
        if self.current_sample_id not in self.samples:
            self.samples[self.current_sample_id] = []

        # Add sample-level metadata if patient data provided
        if patient_data is not None:
            self._add_sample_metadata(patient_data)

    def set_current_sample_target(self, target_label: float) -> None:
        """Set the target label for the current sample."""
        self.current_target_label = target_label

    def add_step(
        self,
        step_name: str,
        input_data: Any,
        output_data: Any,
        system_message: str = None,
        num_input_tokens: int = 0,
        num_output_tokens: int = 0,
        token_time: float = 0.0,
        infer_time: float = 0.0,
        additional_metadata: Optional[Dict[str, Any]] = None,
    ) -> StepMemory:
        """Add a reasoning step to memory and MetricsTracker."""
        if self.current_sample_id is None:
            logger.warning("No current sample set, using default")
            self.set_current_sample("default")

        steps = self.samples[self.current_sample_id]
        step = StepMemory(len(steps) + 1, step_name)
        step.input = input_data
        step.output = output_data
        step.system_message = system_message
        step.num_input_tokens = num_input_tokens
        step.num_output_tokens = num_output_tokens
        step.token_time = token_time
        step.infer_time = infer_time

        steps.append(step)

        # Add to MetricsTracker if available
        if self.metrics_tracker:
            # Extract metrics from final prediction output if it's parsed
            predicted_probability = None
            predicted_diagnosis = ""
            predicted_explanation = ""

            # Additional fields for lab ordering steps
            requested_tests = ""
            confidence = None

            # For steps with parsed json output, dict keys will be tracked individually
            if isinstance(output_data, dict):
                predicted_probability = output_data.get("probability", None)
                predicted_diagnosis = output_data.get("diagnosis", "")
                predicted_explanation = output_data.get("explanation", "")
                confidence = output_data.get("confidence", None)

                # Extract lab ordering specific information
                if step_name == "lab_ordering":
                    requested_tests_list = output_data.get("requested_tests", [])
                    requested_tests = (
                        ",".join(requested_tests_list) if requested_tests_list else ""
                    )

            metadata_item = {
                "Sample ID": str(self.current_sample_id),
                "Step Number": step.step_number,
                "Step Name": step_name,
                "System Message": system_message or "",
                "Input Prompt": str(input_data),
                "Output": str(output_data),
                "Target Label": self.current_target_label or 0,
                "Predicted Probability": predicted_probability,
                "Predicted Diagnosis": predicted_diagnosis,
                "Predicted Explanation": predicted_explanation,
                "Requested Tests": requested_tests,
                "Confidence": confidence,
                "Tokenization Time": token_time,
                "Inference Time": infer_time,
                "Input Tokens": num_input_tokens,
                "Output Tokens": num_output_tokens,
            }

            # Add any additional metadata fields (for agent-specific data)
            if additional_metadata:
                metadata_item.update(additional_metadata)

            self.metrics_tracker.add_metadata_item(metadata_item)

        return step

    def get_final_step(self, sample_id: Any) -> Optional[StepMemory]:
        """Get the final step for a specific sample."""
        # Convert sample_id to string for consistent comparison
        str_sample_id = str(sample_id)

        if str_sample_id not in self.samples:
            logger.warning(
                "Sample ID %s not found in samples dict. Available samples: %s",
                sample_id,
                list(self.samples.keys()),
            )
            return None

        if not self.samples[str_sample_id]:
            logger.warning("No steps found for sample ID %s", sample_id)
            return None

        return self.samples[str_sample_id][-1]

    def reset(self) -> None:
        """Reset memory for the current sample."""
        if (
            self.current_sample_id is not None
            and self.current_sample_id in self.samples
        ):
            # Clear just the current sample's steps
            self.samples[self.current_sample_id] = []

    def _add_sample_metadata(self, patient_data: Any) -> None:
        """Add sample-level metadata row."""

        # Calculate data quality metrics
        data_completeness = self._calculate_data_completeness(patient_data)
        imputation_percentage = self._calculate_imputation_percentage(patient_data)

        sample_metadata = {
            "Sample ID": str(self.current_sample_id),
            "Step Name": "SAMPLE_METADATA",
            "Step Number": 0,
            "Target Label": self.current_target_label or 0,
            # Patient Demographics & Clinical Context
            "metadata_patient_age": (
                getattr(patient_data, "age", None)
                if hasattr(patient_data, "age")
                else patient_data.get("age") if hasattr(patient_data, "get") else None
            ),
            "metadata_patient_sex": (
                getattr(patient_data, "sex", None)
                if hasattr(patient_data, "sex")
                else patient_data.get("sex") if hasattr(patient_data, "get") else None
            ),
            "metadata_patient_weight": (
                getattr(patient_data, "weight", None)
                if hasattr(patient_data, "weight")
                else (
                    patient_data.get("weight") if hasattr(patient_data, "get") else None
                )
            ),
            "metadata_patient_height": (
                getattr(patient_data, "height", None)
                if hasattr(patient_data, "height")
                else (
                    patient_data.get("height") if hasattr(patient_data, "get") else None
                )
            ),
            "metadata_monitoring_hours": (
                get_monitoring_period_hours(patient_data)
                if hasattr(patient_data, "index")
                else 0
            ),
            # Data Quality Metrics
            "metadata_total_features_available": (
                len(set(patient_data.index)) if hasattr(patient_data, "index") else 0
            ),
            "metadata_data_completeness_score": data_completeness,
            "metadata_imputation_percentage": imputation_percentage,
            # Agent-specific metadata based on agent type from agent_id
            "metadata_agent_type": (
                self.agent_id.split("_")[0] if "_" in self.agent_id else "Unknown"
            ),
            # Timestamps
            "metadata_sample_start_time": datetime.now().isoformat(),
            # Placeholders for system message, input, output that aren't applicable to sample metadata
            "System Message": "SAMPLE_METADATA_ROW",
            "Input Prompt": "",
            "Output": "",
            "Predicted Probability": None,
            "Predicted Diagnosis": "",
            "Predicted Explanation": "",
            "Requested Tests": "",
            "Confidence": None,
            "Tokenization Time": 0,
            "Inference Time": 0,
            "Input Tokens": 0,
            "Output Tokens": 0,
        }

        # Add domain-specific metadata for CollaborativeReasoningAgent
        if "CollaborativeReasoningAgent" in self.agent_id and hasattr(
            patient_data, "index"
        ):
            available_features = set(patient_data.index)
            try:
                sample_metadata.update(
                    {
                        "metadata_hemodynamic_features_available": len(
                            get_specialist_features("hemodynamic", available_features)
                        ),
                        "metadata_metabolic_features_available": len(
                            get_specialist_features("metabolic", available_features)
                        ),
                        "metadata_hematologic_features_available": len(
                            get_specialist_features("hematologic", available_features)
                        ),
                    }
                )
            except Exception:
                # Fallback if specialist feature calculation fails
                sample_metadata.update(
                    {
                        "metadata_hemodynamic_features_available": 0,
                        "metadata_metabolic_features_available": 0,
                        "metadata_hematologic_features_available": 0,
                    }
                )

        # Add to metrics tracker
        if self.metrics_tracker:
            self.metrics_tracker.add_metadata_item(sample_metadata)

    def _calculate_data_completeness(self, patient_data: Any) -> float:
        """Calculate overall data completeness (0-1)."""
        try:
            if hasattr(patient_data, "index"):
                # For pandas Series - count non-NaN values
                non_na_count = patient_data.count()
                total_count = len(patient_data)
                return float(non_na_count / total_count) if total_count > 0 else 0.0
            return 1.0  # Fallback
        except Exception as e:
            logger.warning("Error calculating data completeness: %s", e)
            return 0.0

    def _calculate_imputation_percentage(self, patient_data: Any) -> float:
        """Calculate percentage of data that was imputed using _na indicators."""
        try:
            if hasattr(patient_data, "index"):
                # Count _na indicator columns
                na_columns = [col for col in patient_data.index if "_na" in col]
                if not na_columns:
                    return 0.0

                # Count how many _na indicators show imputation (value = 1)
                imputed_count = 0
                for col in na_columns:
                    if patient_data[col] == 1.0:
                        imputed_count += 1

                return float(imputed_count / len(na_columns)) if na_columns else 0.0
            return 0.0
        except Exception as e:
            logger.warning(f"Error calculating imputation percentage: {e}")
            return 0.0
