import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.preprocessing.preprocessing_prompts.agents.memory_manager import AgentMemoryManager

logger = logging.getLogger("PULSE_logger")


class PulseTemplateAgent(ABC):
    """Base template for all agents in the PULSE framework.

    This class defines the core interfaces and common functionality that all
    PULSE agents should implement, similar to PulseTemplateModel.
    """

    def __init__(
        self,
        model_id: str,
        task_name: str,
        dataset_name: str,
        output_dir: Optional[str] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ):
        """Initialize the agent template.

        Args:
            model_id: The model ID to use for inference
            task_name: The current task (e.g., 'aki', 'mortality')
            dataset_name: The dataset being used (e.g., 'hirid')
            output_dir: Directory for logs and outputs
            steps: Predefined reasoning steps
            **kwargs: Additional arguments passed to the model
        """
        self.model_id = model_id
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
        # Create output dataframe with 'prompt' column
        X_processed = pd.DataFrame(index=X.index)
    
        logger.info(f"Processing {len(X)} patients with {self.__class__.__name__}")
        logger.debug(f"Input data columns: {X.columns}")
        
        # Set the total number of samples in the memory manager
        self.memory.set_total_samples(len(X))
    
        # Process each patient
        for i, (idx, row) in enumerate(X.iterrows()):
            try:
                # Set the current sample in the memory manager
                self.memory.set_current_sample(idx)
                
                # Process patient with sample ID
                result = self.process_single(row)
    
                # Store the formatted prompt
                X_processed.loc[idx, "prompt"] = result["output"]
    
                if (i + 1) % 5 == 0 or i + 1 == len(X):
                    logger.info(f"Processed {i+1}/{len(X)} patients")
    
            except Exception as e:
                logger.error(
                    f"Error processing patient {i+1}/{len(X)}: {e}", exc_info=True
                )
                X_processed.loc[idx, "prompt"] = (
                    f"Error processing patient data: {str(e)}"
                )
    
        # Return the processed data and unchanged labels
        return X_processed, y
