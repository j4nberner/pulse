import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("PULSE_logger")

# ------------------------------------
# Model Adapter for Agents
# ------------------------------------


class ModelAdapter:
    """Interface adapter for models to be used with agents."""

    def __init__(self, model: Any):
        """Initialize the adapter with a model instance."""
        self.model = model
        self.model_name = getattr(model, "__class__", type(model)).__name__

    def call_with_messages(
        self, messages: List[Dict[str, str]], parse_json: bool = True
    ) -> Dict[str, Any]:
        """Call the model with the provided messages in chat format."""
        try:
            # Extract system message and user message
            system_message = next(
                (m["content"] for m in messages if m["role"] == "system"), None
            )
            user_message = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )

            if not user_message:
                raise ValueError("No user message found in messages")

            # All models should have infer_llm method - use it with the system message
            if hasattr(self.model, "infer_llm"):
                result = self.model.infer_llm(
                    input_text=user_message,
                    custom_system_message=system_message,
                    force_raw_text=not parse_json,
                )
                return result
            else:
                raise ValueError(
                    f"Model {self.model_name} doesn't have infer_llm method"
                )

        except Exception as e:
            logger.error(f"Error calling model: {e}", exc_info=True)
            return {"output": f"Error: {str(e)}", "error": str(e)}


def get_model_instance(model_config: Dict[str, Any], **kwargs) -> Any:
    """
    Create a model instance based on the configuration.

    Args:
        model_config: Configuration dictionary for the model
        **kwargs: Additional parameters to pass to the model

    Returns:
        Model instance
    """
    model_name = model_config.get("name")
    params = model_config.get(
        "params", {}
    ).copy()  # Create a copy to avoid side effects
    model_type = params.get("type")

    # Add kwargs to params
    for key, value in kwargs.items():
        params[key] = value

    # Create the appropriate model type
    if model_name == "Llama3Model" or model_name == "Llama3":
        from src.models.llama3_model import Llama3Model

        return Llama3Model(
            params=params,
            pretrained_model_path=model_config.get("pretrained_model_path", None),
            wandb=kwargs.get("wandb", False),
            output_dir=kwargs.get("output_dir", None),
            inference_only=kwargs.get("inference_only", False),
            model_name=model_name,
        )
    elif model_name == "MistralModel":
        from src.models.mistral_model import MistralModel

        return MistralModel(
            params=params,
            pretrained_model_path=model_config.get("pretrained_model_path", None),
            wandb=kwargs.get("wandb", False),
            output_dir=kwargs.get("output_dir", None),
            inference_only=kwargs.get("inference_only", False),
        )
    elif model_name == "DeepseekR1Model":
        from src.models.deepseekr1_model import DeepseekR1Model

        return DeepseekR1Model(
            params=params,
            pretrained_model_path=model_config.get("pretrained_model_path", None),
            wandb=kwargs.get("wandb", False),
            output_dir=kwargs.get("output_dir", None),
            inference_only=kwargs.get("inference_only", True),
            model_name=model_name,
            task_name=kwargs.get("task_name"),
            dataset_name=kwargs.get("dataset_name"),
        )
    elif model_name == "Gemma3Model":
        from src.models.gemma3_model import Gemma3Model

        return Gemma3Model(
            params=params,
            pretrained_model_path=model_config.get("pretrained_model_path", None),
            wandb=kwargs.get("wandb", False),
            output_dir=kwargs.get("output_dir", None),
            inference_only=kwargs.get("inference_only", True),
            model_name=model_name,
            task_name=kwargs.get("task_name"),
            dataset_name=kwargs.get("dataset_name"),
        )

    # Add more model types as needed

    logger.error(f"Unknown model: {model_name}")
    return None


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
        enable_logging: bool = True,
    ):
        self.agent_id = agent_id
        self.samples = {}  # Change from steps to samples dictionary
        self.current_sample_id = None  # Track current sample being processed
        self.total_samples = 0  # Track total expected samples
        self.enable_logging = enable_logging
        self.output_dir = output_dir
        self.log_file = None

        if enable_logging and output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.log_file = os.path.join(
                    output_dir, f"agent_{agent_id}_{timestamp}.json"
                )
                # Initialize log file with empty samples array
                with open(self.log_file, "w") as f:
                    json.dump(
                        {"agent_id": agent_id, "total_samples": 0, "samples": []}, f
                    )
            except Exception as e:
                logger.warning(f"Failed to initialize log file: {e}")
                self.log_file = None

    def set_total_samples(self, total: int) -> None:
        """Set the total number of samples to be processed."""
        self.total_samples = total
        # Update the log file with total samples
        if self.enable_logging and self.log_file:
            try:
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                data["total_samples"] = total
                with open(self.log_file, "w") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                logger.warning(f"Failed to update total samples in log file: {e}")

    def set_current_sample(self, sample_id: Any) -> None:
        """Set the current sample being processed."""
        # Convert to string for consistency and to avoid comparison issues
        if sample_id is not None:
            self.current_sample_id = str(sample_id)
        else:
            self.current_sample_id = "default"
            logger.warning("Using 'default' as sample ID because None was provided")

        # Initialize empty steps list for this sample if it doesn't exist
        if self.current_sample_id not in self.samples:
            self.samples[self.current_sample_id] = []

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
        """Add a reasoning step to memory."""
        if self.current_sample_id is None:
            logger.warning("No current sample set, using default")
            self.set_current_sample("default")

        # Get the steps for the current sample
        steps = self.samples[self.current_sample_id]

        step = StepMemory(len(steps) + 1, step_name)
        step.input = input_data
        step.output = output_data
        step.system_message = system_message
        step.num_input_tokens = num_input_tokens
        step.num_output_tokens = num_output_tokens
        step.token_time = token_time
        step.infer_time = infer_time

        # Add to the current sample's steps
        steps.append(step)

        # Log step if enabled
        if self.enable_logging and self.log_file:
            self._log_step(step)

        return step

    def get_step(self, step_number: int) -> Optional[StepMemory]:
        """Get a specific step by number for the current sample."""
        if self.current_sample_id is None or self.current_sample_id not in self.samples:
            return None

        steps = self.samples[self.current_sample_id]
        if 0 <= step_number - 1 < len(steps):
            return steps[step_number - 1]
        return None

    def get_last_step(self) -> Optional[StepMemory]:
        """Get the last reasoning step for the current sample."""
        if self.current_sample_id is None or self.current_sample_id not in self.samples:
            return None

        steps = self.samples[self.current_sample_id]
        if steps:
            return steps[-1]
        return None

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

    def _log_step(self, step: StepMemory) -> None:
        """Log step to file for the current sample."""
        if self.current_sample_id is None:
            logger.warning("No current sample set for logging")
            return

        try:
            # Create or load the existing log file
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    data = json.load(f)
            else:
                data = {
                    "agent_id": self.agent_id,
                    "total_samples": self.total_samples,
                    "samples": [],
                }

            # Find the sample in the samples array or create it
            sample_entry = None
            for sample in data["samples"]:
                if sample.get("sample_id") == str(self.current_sample_id):
                    sample_entry = sample
                    break

            if sample_entry is None:
                # Sample not found, create new entry
                sample_entry = {
                    "sample_id": str(self.current_sample_id),
                    "steps": [],
                }
                data["samples"].append(sample_entry)

            # Add the step to the sample's steps
            sample_entry["steps"].append(step.to_dict())

            # Write back to the file
            with open(self.log_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to log agent step: {e}")


# ------------------------------------
# FileLogCallback
# ------------------------------------


class FileLogCallback:
    """Buffered file logger for agent steps"""

    def __init__(self, log_file_path, flush_interval=10):
        self.log_file_path = log_file_path
        self.buffer = []
        self.flush_interval = flush_interval
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        # Initialize file
        with open(log_file_path, "w") as f:
            json.dump({"entries": []}, f)
        logger.debug(f"Initialized agent log file: {log_file_path}")

    def __call__(self, memory_step, agent=None):
        """Called for each step"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "step_number": memory_step.get("step", 0),
                "tool": memory_step.get("tool", "unknown"),
                "input": memory_step.get("input", ""),
                "output": memory_step.get("output", ""),
            }

            # Add to buffer
            self.buffer.append(entry)

            # Flush to disk when buffer reaches threshold
            if len(self.buffer) >= self.flush_interval:
                self._flush_buffer()
        except Exception as e:
            logger.error(f"Error in FileLogCallback: {e}")

    def _flush_buffer(self):
        """Write buffered entries to file"""
        if not self.buffer:
            return

        try:
            # Read current data
            with open(self.log_file_path, "r") as f:
                data = json.load(f)

            # Add buffered entries
            data["entries"].extend(self.buffer)

            # Write updated data
            with open(self.log_file_path, "w") as f:
                json.dump(data, f, indent=2)

            # Clear buffer
            self.buffer = []
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")

    def __del__(self):
        """Ensure remaining entries are written when object is destroyed"""
        self._flush_buffer()


# ------------------------------------
# Other Util Functions
# ------------------------------------


def format_as_standard_prompt(
    system_message: str, reasoning_steps: List[Dict[str, str]], task_question: str
) -> str:
    """Format agent reasoning steps as a standard pipeline prompt.

    Args:
        system_message: System message to use
        reasoning_steps: List of reasoning steps with name, input, and output
        task_question: Final question for the task

    Returns:
        Formatted prompt string
    """
    formatted_prompt = f"System: {system_message}\n\nUser: "

    # Add each reasoning step
    for step in reasoning_steps:
        formatted_prompt += f"\n{step['name']}:\n{step['output']}\n"

    # Add the final task question
    formatted_prompt += f"\n{task_question}"

    return formatted_prompt


# TODO: This is in data_util?
def get_feature_name(column_name: str) -> str:
    """Extract human-readable feature name from column name."""
    if "_" in column_name and column_name.split("_")[-1].isdigit():
        return "_".join(column_name.split("_")[:-1])
    return column_name
