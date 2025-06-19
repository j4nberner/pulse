import logging
import sys
from typing import Any, Dict, List

from omegaconf import OmegaConf

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
        self.benchmark_settings = config.get("benchmark_settings", {})

        self.wandb = config.get("wandb", {"enabled": False})
        self.output_dir = config.get("output_dir", "")
        self.model_configs = self.models
        self.prompt_configs = config.get("prompting", None)

        self._model_cache = {}
        self._loaded_models = {}  # Track loaded model instances by model_name

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
        Create a updated model instance from a configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            A new model instance
        """
        model_name = config.get("name")
        cache_key = model_name

        # Check if we already have this model instance cached
        if cache_key in self._model_cache:
            logger.info("Using cached model instance: %s", cache_key)
            return self._model_cache[cache_key]

        # Create model as before
        model_cls = get_model_class(model_name)

        # Add random seed to params if not already present
        params = config.get("params", {})
        if "random_seed" not in params:
            params["random_seed"] = self.benchmark_settings.get("random_seed", 42)

        # Set parameters
        kwargs = {
            "params": params,
            "pretrained_model_path": config.get("pretrained_model_path", None),
            "wandb": self.wandb.get("enabled", False),
            "output_dir": self.output_dir,
            "model_name": model_name,
        }

        # Create the model
        model = model_cls(**kwargs)

        # Cache the model before returning
        self._model_cache[cache_key] = model
        logger.info("Created and cached model instance: %s", cache_key)


        return model

    def get_models_for_task(self, dataset_name: str) -> List[Any]:
        """
        Create updated model instances for a specific task/dataset combination.

        Args:
            dataset_name: Name of the dataset being processed

        Returns:
            List[Any]: List of updated model instances
        """
        logger.info("Creating updated model instances for dataset: %s", dataset_name)
        updated_models = []

        for _, config in self.model_configs.items():
            try:
                # Create a new model instance from the saved config
                updated_model = self._create_model_from_config(config)
                updated_models.append(updated_model)
            except Exception as e:
                model_name = config.get("name", "unknown")
                logger.error(
                    "Failed to create updated model '%s' for dataset %s: %s",
                    model_name,
                    dataset_name,
                    str(e),
                )

        return updated_models

