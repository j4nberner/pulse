import logging
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PULSE_logger")


class FileLogCallback:
    """Simple file logger for agent steps"""

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
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

            with open(self.log_file_path, "r+") as f:
                data = json.load(f)
                data["entries"].append(entry)
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error in FileLogCallback: {e}")


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


def get_feature_name(column_name: str) -> str:
    """Extract human-readable feature name from column name."""
    if "_" in column_name and column_name.split("_")[-1].isdigit():
        return "_".join(column_name.split("_")[:-1])
    return column_name
