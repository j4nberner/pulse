import sys
from typing import List, Dict, Any
import logging
import torch
from torch import nn

from . import get_model_class

# Set up logger
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages all models for ICU predictions. Loading, Api-Access, and Saving."""

    def __init__(self, models: List[dict]):
        """
        Initialize the ModelManager with model names. Verifies model attributes. Converts model names to model objects.

        Args:
            models: List of model configurations.
        """
        self.pipelines = {}
        self.models = []
        if not models:
            logger.error("No models specified.")
            sys.exit()

        self.models = self._prepare_models(models)

    def _prepare_models(self, model_configs) -> List[Any]:
        """
        Checks model configurations and converts them to actual model objects.

        Args:
            model_configs: List of model configurations.

        Returns:
            List[Any]: List of instantiated model objects.
        """

        logger.info("Preparing %d models...", len(model_configs))
        prepared_models = []

        for config in model_configs:
            model_name = config.get("name")
            if not model_name:
                logger.error("Model name is required.")
                continue

            try:
                # Create model instance from configuration. Associate trainer with model.
                model_cls = get_model_class(model_name)
                model = model_cls(
                    config.get("params", {})
                )  # **config.get("params", {})

                prepared_models.append(model)
                logger.info("Model '%s' prepared successfully", model_name)
            except Exception as e:
                logger.error("Failed to prepare model '%s': %s", model_name, str(e))

        if not prepared_models:
            logger.error("No valid models could be prepared.")
            sys.exit(1)

        return prepared_models

    def save_model(self, model_name: str, model: Any) -> None:
        """Save the trained model to disk."""
        # Placeholder - implement actual model saving logic
        logger.info("Saving model: %s", model_name)

    def load_model(self, model_name: str) -> Any:
        """Load a specific model."""
        logger.info("Loading model: %s", model_name)
        # Placeholder - implement actual model loading logic
        return {"name": model_name}
