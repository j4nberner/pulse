# HybridReasoningAgent XGBoost Models Directory

This directory contains pretrained XGBoost models for the HybridReasoningAgent.

## Expected Naming Convention

The HybridReasoningAgent looks for XGBoost models with the following naming pattern:
`XGBoost_{task_name}_{dataset_name}_{timestamp}.joblib`

Where:
- `task_name`: The prediction task (e.g., "aki", "mortality", "sepsis")
- `dataset_name`: The dataset used (e.g., "eicu", "hirid")
- `timestamp`: The training timestamp in format YYYYMMDD_HHMMSS

## Example Files
- `XGBoost_aki_eicu_20250604_150110.joblib`
- `XGBoost_mortality_hirid_20250603_142030.joblib`
- `XGBoost_sepsis_eicu_20250605_093245.joblib`

## Model Requirements

The XGBoost models should:
- Be saved using `joblib.save(model, filepath)`
- Include feature importance attributes (automatically included in XGBoost models)
- Be trained on data that includes `_na` missingness indicators
- Have feature names accessible via `model.feature_names_in_` or `model.get_booster().feature_names`

## Multiple Model Handling

If multiple models exist for the same task and dataset combination (different timestamps), the agent will automatically use the most recent one based on filename sorting.

## Usage

The HybridReasoningAgent will automatically load the appropriate model based on the task and dataset names provided during initialization, using glob pattern matching to find the most recent model file.