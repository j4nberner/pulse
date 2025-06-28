import logging
from typing import Any, Optional

logger = logging.getLogger("PULSE_logger")

# Agent registry mapping prompting_id patterns to agent classes
AGENT_REGISTRY = {}


def register_agent(prompting_pattern: str, agent_class):
    """Register an agent class for a specific prompting pattern."""
    AGENT_REGISTRY[prompting_pattern] = agent_class


def get_agent_class(prompting_id: str):
    """Get the appropriate agent class based on prompting_id."""
    for pattern, agent_class in AGENT_REGISTRY.items():
        if pattern in prompting_id:
            return agent_class
    return None


def create_agent_instance(
    prompting_id: str,
    model: Any,
    task_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    metrics_tracker: Optional[Any] = None,
    **kwargs
):
    """Create an agent instance based on prompting_id."""
    try:
        if not prompting_id:
            logger.error("Cannot create agent: prompting_id is not set")
            return None

        agent_class = get_agent_class(prompting_id)
        if not agent_class:
            logger.warning(
                "No agent implementation found for prompting_id: %s", prompting_id
            )
            return None

        # Create agent instance
        agent_instance = agent_class(
            model=model,
            task_name=task_name,
            dataset_name=dataset_name,
            output_dir=output_dir,
            metrics_tracker=metrics_tracker,
            **kwargs
        )

        logger.info("Initialized %s for %s", agent_class.__name__, prompting_id)
        return agent_instance

    except Exception as e:
        logger.error("Failed to create agent for %s: %s", prompting_id, e)
        return None


# Register available agents
try:
    from src.models.agents.zhu_2024c_agent import Zhu2024cAgent

    register_agent("zhu_2024c_categorization_summary_agent", Zhu2024cAgent)
except ImportError as e:
    logger.warning("Could not import Zhu2024cAgent: %s", e)

try:
    from src.models.agents.clinical_workflow_agent import ClinicalWorkflowAgent

    register_agent("clinical_workflow_agent", ClinicalWorkflowAgent)
except ImportError as e:
    logger.warning("Could not import ClinicalWorkflowAgent: %s", e)

try:
    from src.models.agents.collaborative_reasoning_agent import \
        CollaborativeReasoningAgent

    register_agent("collaborative_reasoning_agent", CollaborativeReasoningAgent)
except ImportError as e:
    logger.warning("Could not import CollaborativeReasoningAgent: %s", e)

try:
    from src.models.agents.hybrid_reasoning_agent import HybridReasoningAgent

    register_agent("hybrid_reasoning_agent", HybridReasoningAgent)
except ImportError as e:
    logger.warning("Could not import HybridReasoningAgent: %s", e)

# Add more agent registrations here as needed
# register_agent("other_agent_pattern", OtherAgentClass)
