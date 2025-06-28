from .agent_preprocessor.generic_agent_preprocessor import \
    generic_agent_preprocessor
from .liu_2023_few_shot_health_learners.liu_2023_large_language_models_few_shot_healthlearners import \
    liu_2023_few_shot_preprocessor
from .sarvari_2024_aggregation.sarvari_2024_aggregation import \
    sarvari_aggregation_preprocessor
from .zhu_2024a_cot.zhu_2024a_cot import zhu_2024a_cot_preprocessor
from .zhu_2024b_zero_one_shot.zhu_2024b_zero_one_shot import \
    zhu_2024b_zero_one_shot_preprocessor

preprocessor_method_dict = {
    "liu_2023_few_shot_preprocessor": liu_2023_few_shot_preprocessor,
    "zhu_2024a_cot_preprocessor": zhu_2024a_cot_preprocessor,
    "zhu_2024b_zero_shot_preprocessor": zhu_2024b_zero_one_shot_preprocessor,
    "zhu_2024b_one_shot_preprocessor": zhu_2024b_zero_one_shot_preprocessor,
    "sarvari_2024_aggregation_preprocessor": sarvari_aggregation_preprocessor,
    "zhu_2024c_categorization_summary_agent_preprocessor": generic_agent_preprocessor,
    "clinical_workflow_agent_preprocessor": generic_agent_preprocessor,
    "collaborative_reasoning_agent_preprocessor": generic_agent_preprocessor,
    "hybrid_reasoning_agent_preprocessor": generic_agent_preprocessor,
    # Add other preprocessor methods here as needed
}


def get_prompting_preprocessor(prompting_id: str):
    """
    Get the advanced preprocessor method based on the preprocessing ID.

    Args:
        prompting_id (str): The ID of the preprocessing method.
    """
    if prompting_id in preprocessor_method_dict:
        return preprocessor_method_dict[prompting_id]

    raise ValueError(f"Preprocessor {prompting_id} not found.")
