import sys
from typing import List, Dict, Any
import logging
import torch
from torch import nn
import os

from . import get_model_class

# Set up logger
logger = logging.getLogger("PULSE_logger")


class ModelManager:
    """Manages all models for ICU predictions. Loading, Api-Access, and Saving."""

    def __init__(self, models: List[dict], **kwargs) -> None:
        """
        Initialize the ModelManager with model names. Verifies model attributes.
        Converts model names to model objects with specified parameters.

        Args:
            models: List of model configurations.
        """
        self.pipelines = {}
        self.models = []
        if not models:
            logger.error("No models specified.")
            sys.exit()

        self.wandb = kwargs.get("wandb", False)
        self.output_dir = kwargs.get("output_dir", "")
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
                logger.info(
                    "---------------Preparing model '%s'---------------", model_name
                )
                # Create model instance from configuration. Associate trainer with model.
                model_cls = get_model_class(model_name)
                model = model_cls(
                    config.get("params", {}),
                    wandb=self.wandb.get("enabled", False),
                    output_dir=self.output_dir,
                )  # Pass parameters to model constructor

                prepared_models.append(model)
                logger.info("Model '%s' prepared successfully", model_name)
            except Exception as e:
                logger.error("Failed to prepare model '%s': %s", model_name, str(e))

        if not prepared_models:
            logger.error("No valid models could be prepared.")
            sys.exit(1)

        return prepared_models
