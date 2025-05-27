import json
import logging
import os
from abc import ABC, abstractmethod
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.util.agent_util import AgentMemoryManager, ModelAdapter

logger = logging.getLogger("PULSE_logger")


class PulseTemplateAgent(ABC):
    """Base template for all agents in the PULSE framework."""

    def __init__(
        self,
        model: Any,  # Now accepts an actual model instance
        task_name: str,
        dataset_name: str,
        output_dir: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
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
        self.model_name = getattr(
            model, "model_name", getattr(model, "__class__", type(model)).__name__
        )
        self.model_adapter = ModelAdapter(model)

        # Store task info
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.steps = steps or []
        self.kwargs = kwargs

        # Create memory manager
        agent_id = f"{self.__class__.__name__}_{task_name}_{dataset_name}"
        self.memory = AgentMemoryManager(agent_id, output_dir)

    def add_step(self, name: str, **step_params) -> None:
        """Add a reasoning step to the agent."""
        self.steps.append({"name": name, **step_params})

    @abstractmethod
    def process_single(self, patient_data: pd.Series) -> Dict[str, Any]:
        """Process a single patient's data."""
        pass

    def process_batch(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Process a batch of patient data."""
        # Track processed prompts
        prompts = []
        processed_indices = []
        # Store additional metadata for evaluation
        is_agent_prediction = []
        token_counts_input = []
        token_counts_output = []

        logger.info(f"Processing {len(X)} patients with {self.__class__.__name__}")
        logger.debug(f"Input data columns: {X.columns}")

        # Set the total number of samples in the memory manager
        self.memory.set_total_samples(len(X))

        # Process each patient
        for i, (idx, row) in enumerate(X.iterrows()):
            try:
                # Process patient data through agent steps
                result = self.process_single(row)
                
                # Get parsed result if available
                parsed = result.get("parsed", {})
                
                if parsed and isinstance(parsed, dict):
                    # Use parsed structured output
                    prompts.append(json.dumps(parsed, ensure_ascii=False))
                    
                    # Get token metrics from final step - FIX HERE
                    final_step = self.memory.get_final_step(idx)
                    if not final_step:
                        logger.warning(f"Could not find final step for sample {idx} - using default token counts")
                    
                    # Use getattr instead of .get() for StepMemory objects
                    token_counts_input.append(
                        getattr(final_step, "num_input_tokens", 0) if final_step else 0
                    )
                    token_counts_output.append(
                        getattr(final_step, "num_output_tokens", 0) if final_step else 0
                    )
                else:
                    # Fallback to raw output if no structured data available
                    if isinstance(result["output"], dict):
                        prompts.append(json.dumps(result["output"], indent=2))
                    else:
                        prompts.append(str(result["output"]))
                    token_counts_input.append(0)  # Default values for fallback case
                    token_counts_output.append(0)

                # Mark as agent prediction to avoid second inference
                is_agent_prediction.append(True)
                processed_indices.append(idx)

            except Exception as e:
                logger.error(
                    f"Error processing patient {i+1}/{len(X)}: {e}", exc_info=True
                )
                # Add empty prompt or a placeholder for failed processing
                prompts.append("Error during agent processing")
                is_agent_prediction.append(
                    False
                )  # Don't flag errors as agent predictions
                token_counts_input.append(0)
                token_counts_output.append(0)
                processed_indices.append(idx)

        # Create DataFrame in SAME format as standard pipeline
        X_processed = pd.DataFrame(
            {
                "text": prompts,
                "is_agent_prediction": is_agent_prediction,
                "num_input_tokens": token_counts_input,
                "num_output_tokens": token_counts_output,
            },
            index=processed_indices,
        )

        # Debug log to verify structure
        logger.debug(f"Agent produced DataFrame with columns: {X_processed.columns}")
        if len(X_processed) > 0:
            logger.debug(f"Sample agent output: {X_processed['text'].iloc[0][:100]}")

        # Return the processed data and unchanged labels
        return X_processed, y

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
        system_message = step_config.get(
            "system_message", "You are a helpful assistant."
        )

        # Create step info
        sample_steps = []
        if self.memory.current_sample_id in self.memory.samples:
            sample_steps = self.memory.samples[self.memory.current_sample_id]

        step_info = {
            "step_number": len(sample_steps) + 1,
            "step_name": step_name,
            "system_message": system_message,
            "input": prompt,
        }

        # Prepare messages for the model
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]

        # Call the model
        parse_json = step_config.get("parse_json", False)
        start_time = time.time()

        try:
            # Use the model adapter to call the model
            result = self.model_adapter.call_with_messages(messages, parse_json)

            # Extract output properly - it's in the generated_text field for Llama3Model
            if "generated_text" in result:
                if parse_json and isinstance(result["generated_text"], dict):
                    output = result["generated_text"]
                else:
                    output = result["generated_text"]
            else:
                output = result.get("output", "")

            if step_config.get("output_processor") and callable(
                step_config["output_processor"]
            ):
                output = step_config["output_processor"](output, state)

            # Add step to memory
            self.memory.add_step(
                step_name=step_name,
                input_data=prompt,
                output_data=output,
                system_message=system_message,
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
                system_message=system_message,
                infer_time=time.time() - start_time,
            )

            return {"output": f"Error: {str(e)}", "error": str(e)}
