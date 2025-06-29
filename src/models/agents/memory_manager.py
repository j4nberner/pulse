import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

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

        # Autonomous decision tracking (replaces MetaCognitiveTracker)
        self.decision_points = {}  # Sample ID -> list of decision points
        self.adaptation_history = {}  # Sample ID -> list of adaptations

    def set_current_sample(self, sample_id: Any) -> None:
        """Set the current sample being processed."""
        self.current_sample_id = str(sample_id)
        if self.current_sample_id not in self.samples:
            self.samples[self.current_sample_id] = []
        # Initialize decision tracking for this sample
        if self.current_sample_id not in self.decision_points:
            self.decision_points[self.current_sample_id] = []
        if self.current_sample_id not in self.adaptation_history:
            self.adaptation_history[self.current_sample_id] = []

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

            self.metrics_tracker.add_metadata_item(
                {
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
            )

        return step

    def log_decision_point(
        self, step: str, options: List[Any], chosen: str, reasoning: str
    ) -> None:
        """Log autonomous decision points for validation."""
        if self.current_sample_id is None:
            self.set_current_sample("default")

        decision_data = {
            "step": step,
            "timestamp": time.time(),
            "options": options,
            "chosen": chosen,
            "reasoning": reasoning,
        }
        self.decision_points[self.current_sample_id].append(decision_data)

        # Also log to metadata CSV if metrics tracker available
        if self.metrics_tracker:
            self.metrics_tracker.add_metadata_item(
                {
                    "Sample ID": str(self.current_sample_id),
                    "Step Number": len(self.samples.get(self.current_sample_id, []))
                    + 1,
                    "Step Name": f"AUTONOMOUS_DECISION_{step}",
                    "System Message": "Autonomous decision-making",
                    "Input Prompt": f"Options: {options}",
                    "Output": f"Chosen: {chosen}",
                    "Target Label": self.current_target_label or 0,
                    "Predicted Probability": None,
                    "Predicted Diagnosis": "",
                    "Predicted Explanation": reasoning,
                    "Requested Tests": "",
                    "Confidence": None,
                    "Tokenization Time": 0,
                    "Inference Time": 0,
                    "Input Tokens": 0,
                    "Output Tokens": 0,
                    "Decision_Options": str(options),
                    "Decision_Chosen": chosen,
                    "Decision_Reasoning": reasoning,
                    "Decision_Timestamp": decision_data["timestamp"],
                }
            )

    def log_adaptation(self, trigger: str, change: Dict[str, Any]) -> None:
        """Log when agent adapts its approach."""
        if self.current_sample_id is None:
            self.set_current_sample("default")

        adaptation_data = {
            "trigger": trigger,
            "change": change,
            "timestamp": time.time(),
        }
        self.adaptation_history[self.current_sample_id].append(adaptation_data)

        # Also log to metadata CSV if metrics tracker available
        if self.metrics_tracker:
            self.metrics_tracker.add_metadata_item(
                {
                    "Sample ID": str(self.current_sample_id),
                    "Step Number": len(self.samples.get(self.current_sample_id, []))
                    + 1,
                    "Step Name": f"AUTONOMOUS_ADAPTATION",
                    "System Message": "Agent adaptation",
                    "Input Prompt": f"Trigger: {trigger}",
                    "Output": f"Change: {change}",
                    "Target Label": self.current_target_label or 0,
                    "Predicted Probability": None,
                    "Predicted Diagnosis": "",
                    "Predicted Explanation": f"Adapted due to: {trigger}",
                    "Requested Tests": "",
                    "Confidence": None,
                    "Tokenization Time": 0,
                    "Inference Time": 0,
                    "Input Tokens": 0,
                    "Output Tokens": 0,
                    "Adaptation_Trigger": trigger,
                    "Adaptation_Change": str(change),
                    "Adaptation_Timestamp": adaptation_data["timestamp"],
                }
            )

    def get_react_trace(self, sample_id: Any = None) -> List[Dict[str, Any]]:
        """Export ReAct-style trace for validation."""
        target_sample_id = (
            str(sample_id) if sample_id is not None else self.current_sample_id
        )
        if target_sample_id is None or target_sample_id not in self.decision_points:
            return []

        trace = []
        for decision in self.decision_points[target_sample_id]:
            trace.append(
                {
                    "thought": decision["reasoning"],
                    "action": decision["chosen"],
                    "observation": f"Selected {decision['chosen']} from {decision['options']}",
                }
            )
        return trace

    def get_autonomy_summary(self, sample_id: Any = None) -> Dict[str, Any]:
        """Get summary of autonomous decisions for a sample."""
        target_sample_id = (
            str(sample_id) if sample_id is not None else self.current_sample_id
        )
        if target_sample_id is None:
            return {}

        return {
            "total_decisions": len(self.decision_points.get(target_sample_id, [])),
            "total_adaptations": len(self.adaptation_history.get(target_sample_id, [])),
            "decision_points": self.decision_points.get(target_sample_id, []),
            "adaptations": self.adaptation_history.get(target_sample_id, []),
            "react_trace": self.get_react_trace(target_sample_id),
        }

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
        else:
            # If no current sample, reset all
            self.samples = {}
            self.current_sample_id = None
