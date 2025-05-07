import logging
import os
import sys
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf
from torch import nn

from . import get_model_class

# Set up logger
logger = logging.getLogger("PULSE_logger")


class ModelManager:
    """Manages all models for ICU predictions. Loading, Api-Access, and Saving."""

    def __init__(self, config: OmegaConf, **kwargs) -> None:
        """
        Initialize the ModelManager with model names. Verifies model attributes.
        Converts model names to model objects with specified parameters.

        Args:
            config: Omegaconf configuration object containing model settings.
            **kwargs: Additional keyword arguments.
        """
        self.pipelines = {}
        self.models = config.get("models", None)
        if not self.models:
            logger.error("No models specified.")
            sys.exit()

        self.wandb = config.get("wandb", {"enabled": False})
        self.output_dir = config.get("output_dir", "")
        self.model_configs = self.models
        self.prompt_configs = config.get("prompting", None)
        self.models = self._prepare_models()

    def _prepare_models(self) -> List[Any]:
        """
        Checks model configurations and converts them to actual model objects.

        Returns:
            List[Any]: List of instantiated model objects.
        """
        logger.info("Preparing %d models...", len(self.model_configs))
        prepared_models = []

        for _, config in self.model_configs.items():
            model_name = config.get("name")
            if not model_name:
                logger.error("Model name is required.")
                continue

            try:
                if self.prompt_configs.prompting_ids is not None:
                    for prompting_id in self.prompt_configs.prompting_ids:
                        config.params["prompting_id"] = prompting_id
                        logger.info(
                            "---------------Preparing model '%s'---------------",
                            model_name,
                        )
                        logger.info("Prompting Preprocessing ID: %s", prompting_id)
                        model = self._create_model_from_config(config)
                        prepared_models.append(model)
                        logger.info("Model '%s' prepared successfully", model_name)
                else:
                    logger.info(
                        "---------------Preparing model '%s'---------------", model_name
                    )
                    model = self._create_model_from_config(config)
                    prepared_models.append(model)
                    logger.info("Model '%s' prepared successfully", model_name)
            except Exception as e:
                logger.error("Failed to prepare model '%s': %s", model_name, str(e))

        if not prepared_models:
            logger.error("No valid models could be prepared.")
            sys.exit(1)

        return prepared_models

    def _create_model_from_config(self, config: Dict) -> Any:
        """
        Create a fresh model instance from a configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A new model instance
        """
        model_name = config.get("name")
        # Create model instance from configuration
        model_cls = get_model_class(model_name)
        model = model_cls(
            config.get("params", {}),
            wandb=self.wandb.get("enabled", False),
            output_dir=self.output_dir,
        )

        # Load model weights if path is specified
        if config.get("pretrained_model_path", None):
            try:
                model.load_model_weights(config["pretrained_model_path"])
                logger.info(
                    "Loaded pretrained model weights from %s",
                    config["pretrained_model_path"],
                )
            except Exception as e:
                logger.warning(
                    "Failed to load pretrained model weights from %s: %s. Defaulting to random initialization.",
                    config["pretrained_model_path"],
                    str(e),
                )

        return model

    def get_models_for_task(self, dataset_name: str) -> List[Any]:
        """
        Create fresh model instances for a specific task/dataset combination.

        Args:
            dataset_name: Name of the dataset being processed

        Returns:
            List[Any]: List of fresh model instances
        """
        logger.info(f"Creating fresh model instances for dataset: {dataset_name}")
        fresh_models = []

        for _, config in self.model_configs.items():
            try:
                # Create a new model instance from the saved config
                fresh_model = self._create_model_from_config(config)
                fresh_models.append(fresh_model)
            except Exception as e:
                model_name = config.get("name", "unknown")
                logger.error(
                    f"Failed to create fresh model '{model_name}' for dataset {dataset_name}: {str(e)}"
                )

        return fresh_models
