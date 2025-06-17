import logging
from datetime import datetime
from typing import Any, Dict, Optional

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
    """Manager for agent reasoning steps."""

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

    def set_current_sample(self, sample_id: Any) -> None:
        """Set the current sample being processed."""
        self.current_sample_id = str(sample_id)
        if self.current_sample_id not in self.samples:
            self.samples[self.current_sample_id] = []

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

            # For final_prediction steps, output will be a parsed dictionary
            if step_name == "final_prediction" and isinstance(output_data, dict):
                predicted_probability = output_data.get("probability", None)
                predicted_diagnosis = output_data.get("diagnosis", "")
                predicted_explanation = output_data.get("explanation", "")

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
                    "Tokenization Time": token_time,
                    "Inference Time": infer_time,
                    "Input Tokens": num_input_tokens,
                    "Output Tokens": num_output_tokens,
                }
            )

        return step
    
    def get_final_step(self, sample_id: Any) -> Optional[StepMemory]:
        """Get the final step for a specific sample."""
        # Convert sample_id to string for consistent comparison
        str_sample_id = str(sample_id)

        if str_sample_id not in self.samples:
            logger.warning(
                f"Sample ID {sample_id} not found in samples dict. Available samples: {list(self.samples.keys())}"
            )
            return None

        if not self.samples[str_sample_id]:
            logger.warning(f"No steps found for sample ID {sample_id}")
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


# ------------------------------------
# Other Util Functions
# ------------------------------------
