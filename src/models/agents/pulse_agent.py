import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.models.agents.memory_manager import AgentMemoryManager
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

        # Log agent configuration metadata if metrics tracker available
        if metrics_tracker:
            self._log_agent_configuration()

    def add_step(self, name: str, **step_params) -> None:
        """Add a reasoning step to the agent."""
        self.steps.append({"name": name, **step_params})

    def update_task_context(self, task_name: str, dataset_name: str) -> None:
        """Update the agent's task and dataset context."""
        if self.task_name != task_name or self.dataset_name != dataset_name:
            logger.debug(
                f"Updating agent context from {self.task_name}/{self.dataset_name} to {task_name}/{dataset_name}"
            )
            self.task_name = task_name
            self.dataset_name = dataset_name

            # Update memory manager context
            if hasattr(self, "memory"):
                agent_id = f"{self.__class__.__name__}_{task_name}_{dataset_name}"
                self.memory.agent_id = agent_id

            # Call task-specific update if implemented by subclass
            if hasattr(self, "_update_task_specific_content"):
                self._update_task_specific_content()

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
            default_prompt = prompt_template_hf("", task=self.task_name)[0]
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

            # Prepare additional metadata if subclass provides it
            additional_metadata = None
            if hasattr(self, "_prepare_step_metadata"):
                additional_metadata = self._prepare_step_metadata(
                    step_name, state, output
                )

            # Add step to memory
            step_memory = self.memory.add_step(
                step_name=step_name,
                input_data=prompt,
                output_data=output,
                system_message=log_system_message,
                num_input_tokens=result.get("num_input_tokens", 0),
                num_output_tokens=result.get("num_output_tokens", 0),
                token_time=result.get("token_time", 0),
                infer_time=time.time() - start_time,
                additional_metadata=additional_metadata,
            )

            return {"output": output, "result": result}

        except Exception as e:
            logger.error(f"Error executing step '{step_name}': {e}", exc_info=True)

            # Add error step to memory
            step_memory = self.memory.add_step(
                step_name=step_name,
                input_data=prompt,
                output_data=f"Error: {str(e)}",
                system_message=log_system_message,
                infer_time=time.time() - start_time,
                additional_metadata=None,
            )

            return {"output": f"Error: {str(e)}", "error": str(e)}

    def _log_agent_configuration(self):
        """Log the agent's configuration metadata."""
        config_metadata = {
            "Sample ID": "AGENT_CONFIG",
            "Step Name": "CONFIGURATION",
            "Step Number": -1,
            "Target Label": 0,
            # Agent Configuration
            "metadata_agent_type": self.__class__.__name__,
            "metadata_model_name": self.model_name,
            "metadata_task_name": self.task_name,
            "metadata_dataset_name": self.dataset_name,
            "metadata_output_dir": self.output_dir,
            # Agent-specific configurations (will be None for agents that don't have them)
            "metadata_confidence_threshold": getattr(
                self, "confidence_threshold", None
            ),
            "metadata_max_iterations": getattr(self, "max_iterations", None),
            "metadata_min_iterations": getattr(self, "min_iterations", None),
            "metadata_ml_confidence_threshold": getattr(
                self, "ml_confidence_threshold", None
            ),
            "metadata_agreement_threshold": getattr(self, "agreement_threshold", None),
            "metadata_top_features_count": getattr(self, "top_features_count", None),
            "metadata_specialist_types": getattr(self, "specialist_types", None),
            # Additional parameters
            "metadata_additional_params": str(self.kwargs) if self.kwargs else "",
            "metadata_total_steps_defined": len(self.steps),
            "metadata_config_timestamp": datetime.now().isoformat(),
            # Standard row placeholders
            "System Message": "AGENT_CONFIGURATION",
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

        # Add to metrics tracker
        if hasattr(self.memory, "metrics_tracker") and self.memory.metrics_tracker:
            self.memory.metrics_tracker.add_metadata_item(config_metadata)

        logger.info(f"Agent configuration logged: {self.__class__.__name__}")
