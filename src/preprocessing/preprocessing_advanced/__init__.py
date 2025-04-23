from .example_paper_prompts import example_paper_preprocessor
from .large_language_models_few_shot_healthlearners import few_shot_paper_preprocessor


preprocessor_method_dict = {
    "example_paper_preprocessor": example_paper_preprocessor,
    "few_shot_paper_preprocessor": few_shot_paper_preprocessor,
    # Add other preprocessor methods here as needed
}


def get_advanced_preprocessor(preprocessing_id: str):
    """
    Get the advanced preprocessor method based on the preprocessing ID.

    Args:
        preprocessing_id (str): The ID of the preprocessing method.
    """
    if preprocessing_id in preprocessor_method_dict:
        return preprocessor_method_dict[preprocessing_id]
    else:
        raise ValueError(f"Preprocessor {preprocessing_id} not found.")
