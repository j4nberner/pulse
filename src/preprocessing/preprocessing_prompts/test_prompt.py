import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import logging
import sys
from pathlib import Path

# Add project root to path to enable imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from src.preprocessing.preprocessing_prompts.few_shot_health_learners.large_language_models_few_shot_healthlearners import few_shot_paper_preprocessor
from src.util.model_util import apply_model_prompt_format

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PULSE_test_logger")

def create_synthetic_data(num_samples: int = 5) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], Dict[str, Any]]:
    """
    Creates synthetic data for testing the prompt generation functionality.
    
    Args:
        num_samples: Number of samples to generate
        
    Returns:
        X: List of DataFrames containing feature data
        y: List of DataFrames containing labels
        info_dict: Dictionary with task information
    """
    # Create main input data
    features = ['heart_rate', 'blood_pressure', 'oxygen_level', 'temperature']
    X_main = pd.DataFrame({
        'heart_rate': np.random.randint(60, 100, num_samples),
        'blood_pressure': np.random.randint(110, 150, num_samples),
        'oxygen_level': np.random.randint(90, 100, num_samples),
        'temperature': np.random.uniform(36.0, 38.0, num_samples).round(1)
    })
    
    # Create few-shot examples
    X_train = pd.DataFrame({
        'heart_rate': np.random.randint(60, 100, 10),
        'blood_pressure': np.random.randint(110, 150, 10),
        'oxygen_level': np.random.randint(90, 100, 10),
        'temperature': np.random.uniform(36.0, 38.0, 10).round(1)
    })
    
    # Create labels (binary classification: sepsis or not-sepsis)
    y_main = pd.DataFrame({
        'not_sepsis': np.random.randint(0, 2, num_samples),
        'sepsis': np.random.randint(0, 2, num_samples)
    })
    y_train = pd.DataFrame({
        'not_sepsis': np.random.randint(0, 2, 10),
        'sepsis': np.random.randint(0, 2, 10)
    })
    
    # Fix the labels to be one-hot encoded
    for idx in range(len(y_main)):
        if y_main.iloc[idx]['not_sepsis'] == y_main.iloc[idx]['sepsis']:
            y_main.at[idx, 'not_sepsis'] = 1
            y_main.at[idx, 'sepsis'] = 0
    
    for idx in range(len(y_train)):
        if y_train.iloc[idx]['not_sepsis'] == y_train.iloc[idx]['sepsis']:
            y_train.at[idx, 'not_sepsis'] = 1
            y_train.at[idx, 'sepsis'] = 0
    
    # Create info dictionary
    info_dict = {
        "task": "sepsis",
        "dataset_name": "test_icu_data",
        "model_name": "gpt-3.5-turbo",
        "shots": 2,
        "mode": "test"
    }
    
    return [X_main, X_train], [y_main, y_train], info_dict

def test_prompt_generation():
    """Test the prompt generation with different configurations."""
    # Test with few-shot examples
    X, y, info_dict = create_synthetic_data(num_samples=3)
    X_processed, y_processed = few_shot_paper_preprocessor(X, y, info_dict)
    
    logger.info(f"Generated {len(X_processed)} prompts with few-shot examples")
    logger.info(f"Sample prompt:\n{X_processed['text'].iloc[0]}")
    
    # Test without few-shot examples
    info_dict["shots"] = 0
    X_processed_no_shots, y_processed = few_shot_paper_preprocessor(X, y, info_dict)
    
    logger.info(f"Generated {len(X_processed_no_shots)} prompts without few-shot examples")
    logger.info(f"Sample prompt:\n{X_processed_no_shots['text'].iloc[0]}")
    
    # Test with a different model
    info_dict["model_name"] = "claude-3-sonnet"
    info_dict["shots"] = 1
    X_processed_claude, y_processed = few_shot_paper_preprocessor(X, y, info_dict)
    
    logger.info(f"Generated {len(X_processed_claude)} prompts for Claude model")
    logger.info(f"Sample prompt:\n{X_processed_claude['text'].iloc[0]}")
    
    return X_processed, X_processed_no_shots, X_processed_claude

if __name__ == "__main__":
    logger.info("Starting test prompt generation...")
    with_shots, no_shots, claude_model = test_prompt_generation()
    logger.info("Test prompt generation completed successfully!")