from .few_shot_health_learners.large_language_models_few_shot_healthlearners import \
    few_shot_paper_preprocessor

preprocessor_method_dict = {
    "few_shot_paper_preprocessor": few_shot_paper_preprocessor,
    # Add other preprocessor methods here as needed
}


def get_prompting_preprocessor(preprocessing_id: str):
    """
    Get the advanced preprocessor method based on the preprocessing ID.

    Args:
        preprocessing_id (str): The ID of the preprocessing method.
    """
    if preprocessing_id in preprocessor_method_dict:
        return preprocessor_method_dict[preprocessing_id]
    else:
        raise ValueError(f"Preprocessor {preprocessing_id} not found.")
