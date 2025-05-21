import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("PULSE_logger")


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
        self.current_sample_id = sample_id
        # Initialize empty steps list for this sample if it doesn't exist
        if sample_id not in self.samples:
            self.samples[sample_id] = []

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
