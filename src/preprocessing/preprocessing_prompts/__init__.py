from .few_shot_health_learners.large_language_models_few_shot_healthlearners import (
    few_shot_paper_preprocessor,
)
from .zhu_2024a_one_shot_cot.zhu_2024a_one_shot_cot import (
    zhu_2024a_one_shot_cot_preprocessor,
)
from .zhu_2024b_one_shot.zhu_2024b_one_shot import (
    zhu_2024b_one_shot_preprocessor,
)
from .zhu_2024c_categorization_summary.zhu_2024c_categorization_summary import (
    zhu_2024c_categorization_summary_preprocessor,
)


preprocessor_method_dict = {
    "few_shot_paper_preprocessor": few_shot_paper_preprocessor,
    "zhu_2024a_one_shot_cot_preprocessor": zhu_2024a_one_shot_cot_preprocessor,
    "zhu_2024b_one_shot_preprocessor": zhu_2024b_one_shot_preprocessor,
    "zhu_2024c_categorization_summary_preprocessor": zhu_2024c_categorization_summary_preprocessor,
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
    else:
        raise ValueError(f"Preprocessor {prompting_id} not found.")
