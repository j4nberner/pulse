import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd

from src.util.agent_util import AgentMemoryManager
from src.util.model_util import prompt_template_hf

logger = logging.getLogger("PULSE_logger")


class PulseAgent(ABC):
    """Base template for all agents in the PULSE framework."""

    def __init__(
        self,
        model: Any,  # Now accepts an actual model instance
        task_name: str,
        dataset_name: str,
        output_dir: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        metrics_tracker: Optional[Any] = None,
        **kwargs,
    ):
        """Initialize the agent template.

        Args:
            model: The model instance to use for inference
            task_name: The current task (e.g., 'aki', 'mortality')
            dataset_name: The dataset being used (e.g., 'hirid')
            output_dir: Directory for logs and outputs
            steps: Predefined reasoning steps
            **kwargs: Additional arguments
        """
        # Store model info
        self.model = model
        self.model_name = getattr(
            model, "model_name", getattr(model, "__class__", type(model)).__name__
        )

        # Store task info
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.steps = steps or []
        self.kwargs = kwargs

        # Create memory manager
        agent_id = f"{self.__class__.__name__}_{task_name}_{dataset_name}"
        self.memory = AgentMemoryManager(agent_id, output_dir, metrics_tracker)

    def add_step(self, name: str, **step_params) -> None:
        """Add a reasoning step to the agent."""
        self.steps.append({"name": name, **step_params})

    @abstractmethod
    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient's data."""

    def run_step(
        self, step_name: str, input_data: Any, state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run a single agent step."""
        logger.debug(f"Running step: {step_name}")

        # Get step configuration
        step_config = next((s for s in self.steps if s["name"] == step_name), None)
        if not step_config:
            raise ValueError(f"Step '{step_name}' not found in agent steps")

        # Format input if a formatter is provided
        try:
            if step_config.get("input_formatter"):
                formatted_input = step_config["input_formatter"](state, input_data)
                logger.debug(f"Input formatted successfully for {step_name}")
            else:
                formatted_input = input_data
        except Exception as e:
            logger.error(
                f"Error in input formatter for step '{step_name}': {e}", exc_info=True
            )
            formatted_input = f"Error formatting input: {str(e)}"

        # Format the prompt using the provided template
        if callable(step_config.get("prompt_template")):
            prompt = step_config["prompt_template"](formatted_input, state)
        else:
            prompt = formatted_input

        # Get system message
        system_message = step_config.get("system_message")

        # For logging purposes, get the actual default system message when None
        log_system_message = system_message
        if system_message is None:
            # Get default system message from first message in prompt template
            default_prompt = prompt_template_hf("")[0]
            log_system_message = default_prompt["content"]

        # This flag controls whether to parse JSON output from the model
        # Set to True for the final prediction step
        parse_json = step_config.get("parse_json", False)
        start_time = time.time()

        try:
            # Call the model's generate method directly
            result = self.model._generate_standard(
                input_text=prompt,
                custom_system_message=system_message,  # None means use default
                parse_json=parse_json,
            )

            # Extract output
            output = result["generated_text"]

            if step_config.get("output_processor") and callable(
                step_config["output_processor"]
            ):
                output = step_config["output_processor"](output, state)

            # Add step to memory
            self.memory.add_step(
                step_name=step_name,
                input_data=prompt,
                output_data=output,
                system_message=log_system_message,
                num_input_tokens=result.get("num_input_tokens", 0),
                num_output_tokens=result.get("num_output_tokens", 0),
                token_time=result.get("token_time", 0),
                infer_time=time.time() - start_time,
            )

            return {"output": output, "result": result}

        except Exception as e:
            logger.error(f"Error executing step '{step_name}': {e}", exc_info=True)

            # Add error step to memory
            self.memory.add_step(
                step_name=step_name,
                input_data=prompt,
                output_data=f"Error: {str(e)}",
                system_message=log_system_message,
                infer_time=time.time() - start_time,
            )

            return {"output": f"Error: {str(e)}", "error": str(e)}
