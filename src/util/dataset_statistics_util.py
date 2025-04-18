
import os
import pandas as pd
import numpy as np
import gc
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
import sys
from omegaconf import OmegaConf

# Add project root to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.preprocessing.preprocessing_baseline.preprocessing_baseline import PreprocessorBaseline
from src.preprocessing.preprocessing_advanced.windowing import Windower
from src.data.dataloader import DatasetManager

# Set up logger
logger = logging.getLogger("PULSE_logger")

def calculate_dataset_statistics(
    X: pd.DataFrame, 
    y: pd.DataFrame, 
    task: str,
    dataset_name: str,
    set_type: str = "train",
    windowing_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Calculate statistics for the dataset.
    
    Args:
        X (pd.DataFrame): Features dataframe
        y (pd.DataFrame): Labels dataframe
        task (str): Task name ('mortality', 'aki', 'sepsis')
        dataset_name (str): Dataset name ('hirid', 'miiv', etc.)
        set_type (str, optional): Type of dataset ('train', 'val', 'test'). Defaults to "train".
        windowing_config (Dict, optional): Windowing configuration if applicable
        
    Returns:
        Dict[str, Any]: Dictionary of dataset statistics
    """
    try:
        # Extract windowing info if available
        windowing_info = "none"
        if windowing_config and windowing_config.get("enabled", False):
            dynamic_window = windowing_config.get("data_window", 0)
            prediction_window = windowing_config.get("prediction_window", 0)
            step_size = windowing_config.get("step_size", 0)
            windowing_info = f"{dynamic_window}dw_{prediction_window}pw_{step_size}sz"
        
        # For mortality task, use simplified statistics
        if task == "mortality":
            # Count total stays (each row is now a stay)
            total_stays = len(X)
            
            # Count cases (stays with positive label)
            positive_stays = y['label'].sum()
            negative_stays = total_stays - positive_stays
            
            # Format the statistics in a dictionary
            stats = {
                'Task': task,
                'Dataset': dataset_name,
                'Set': set_type,
                'Windowing': windowing_info,
                'Total Stays': total_stays,
                'Cases': positive_stays, 
                'Cases %': f"{(positive_stays/total_stays)*100:.1f}%",
                'Controls': negative_stays,
                'Controls %': f"{(negative_stays/total_stays)*100:.1f}%",
                'Total Rows': total_stays,  # Same as total_stays after reshaping
                'Rows of Cases': positive_stays,  # Same as positive_stays after reshaping
                'Rows of Cases %': f"{(positive_stays/total_stays)*100:.1f}%",
                'Rows of Controls': negative_stays,  # Same as negative_stays after reshaping
                'Rows of Controls %': f"{(negative_stays/total_stays)*100:.1f}%",
                'Positive Labels': positive_stays,  # Same as positive_stays after reshaping
                'Positive Labels %': f"{(positive_stays/total_stays)*100:.1f}%",
                'Negative Labels': negative_stays,  # Same as negative_stays after reshaping
                'Negative Labels %': f"{(negative_stays/total_stays)*100:.1f}%"
            }
            
            return stats
        
        # For other tasks, use the original statistics calculation
        else:
            # Check if stay_id is in the dataframes
            if 'stay_id' not in X.columns or 'stay_id' not in y.columns:
                logger.warning(f"stay_id column missing in data for {task}/{dataset_name}/{set_type}")
                # Create placeholder statistics if proper calculation isn't possible
                stats = {
                    'Task': task,
                    'Dataset': dataset_name,
                    'Set': set_type,
                    'Windowing': windowing_info,
                    'Total Stays': len(X), # Approximation
                    'Cases': y['label'].sum(),
                    'Cases %': f"{(y['label'].sum()/len(y))*100:.1f}%",
                    'Controls': len(y) - y['label'].sum(),
                    'Controls %': f"{((len(y) - y['label'].sum())/len(y))*100:.1f}%",
                    'Total Rows': len(X),
                    'Rows of Cases': y['label'].sum(),
                    'Rows of Cases %': f"{(y['label'].sum()/len(y))*100:.1f}%",
                    'Rows of Controls': len(y) - y['label'].sum(),
                    'Rows of Controls %': f"{((len(y) - y['label'].sum())/len(y))*100:.1f}%",
                    'Positive Labels': y['label'].sum(),
                    'Positive Labels %': f"{(y['label'].sum()/len(y))*100:.1f}%",
                    'Negative Labels': len(y) - y['label'].sum(), 
                    'Negative Labels %': f"{((len(y) - y['label'].sum())/len(y))*100:.1f}%"
                }
                return stats
            
            # Add row index for each stay_id to facilitate proper merging
            X_indexed = X.copy()
            y_indexed = y.copy()
            X_indexed['row_idx'] = X_indexed.groupby('stay_id').cumcount()
            y_indexed['row_idx'] = y_indexed.groupby('stay_id').cumcount()
            
            # Merge X and y temporarily for analysis
            merged_full = pd.merge(X_indexed, y_indexed, on=['stay_id', 'row_idx'])
            
            # Count total rows
            total_rows = len(merged_full)
            
            # Analyze by stay_id 
            stay_id_has_positive = merged_full.groupby('stay_id')['label'].any()
            
            # Count of stay_ids with and without positive labels
            positive_stays = stay_id_has_positive.sum()
            negative_stays = len(stay_id_has_positive) - positive_stays
            total_stays = len(stay_id_has_positive)
            
            # Create a mapping of stay_id to whether it has a positive label
            stay_id_positive_map = stay_id_has_positive.to_dict()
            
            # Add a column to merged_full indicating if the stay_id has a positive label anywhere
            merged_full['stay_has_positive'] = merged_full['stay_id'].map(stay_id_positive_map)
            
            # Calculate row counts for each category
            rows_of_controls = len(merged_full[~merged_full['stay_has_positive']])
            rows_of_cases = len(merged_full[merged_full['stay_has_positive']])
            negative_labels = len(merged_full[merged_full['label'] == False])
            positive_labels = len(merged_full[merged_full['label'] == True])
            
            # Format the statistics in a dictionary for further use
            stats = {
                'Task': task,
                'Dataset': dataset_name,
                'Set': set_type,
                'Windowing': windowing_info,
                'Total Stays': total_stays,
                'Cases': positive_stays, 
                'Cases %': f"{(positive_stays/total_stays)*100:.1f}%",
                'Controls': negative_stays,
                'Controls %': f"{(negative_stays/total_stays)*100:.1f}%",
                'Total Rows': total_rows,
                'Rows of Cases': rows_of_cases,
                'Rows of Cases %': f"{(rows_of_cases/total_rows)*100:.1f}%",
                'Rows of Controls': rows_of_controls,
                'Rows of Controls %': f"{(rows_of_controls/total_rows)*100:.1f}%",
                'Positive Labels': positive_labels,
                'Positive Labels %': f"{(positive_labels/total_rows)*100:.1f}%",
                'Negative Labels': negative_labels,
                'Negative Labels %': f"{(negative_labels/total_rows)*100:.1f}%"
            }
            
            return stats
    
    finally:
        # Clean up intermediate objects
        if task != "mortality":  # Skip cleanup for mortality as these objects won't exist
            if 'X_indexed' in locals():
                del X_indexed
            if 'y_indexed' in locals():
                del y_indexed
            if 'merged_full' in locals():
                del merged_full
            if 'stay_id_has_positive' in locals():
                del stay_id_has_positive
            if 'stay_id_positive_map' in locals():
                del stay_id_positive_map
            gc.collect()

def print_statistics(statistics_list: List[Dict[str, Any]]) -> None:
    """
    Print statistics in a formatted way for easy reading.
    
    Args:
        statistics_list (List[Dict[str, Any]]): List of statistics dictionaries
    """
    if not statistics_list:
        logger.info("No statistics available to display")
        return
    
    logger.info(f"{'.'*40}")
    logger.info(f"DATASET STATISTICS")
    
    # Print header row with slash delimiters
    logger.info(f"Task/Dataset/Set/Windowing/Total Stays/Cases/Controls/Total Rows/Rows of Cases/Rows of Controls/Positive Labels/Negative Labels")
    
    # Print each row of statistics with slash delimiters
    for stat in statistics_list:
        # Format numbers with commas for thousands
        total_stays = f"{stat['Total Stays']:,}"
        cases = f"{stat['Cases']:,}"
        controls = f"{stat['Controls']:,}"
        total_rows = f"{stat['Total Rows']:,}"
        rows_cases = f"{stat['Rows of Cases']:,}"
        rows_controls = f"{stat['Rows of Controls']:,}"
        pos_labels = f"{stat['Positive Labels']:,}"
        neg_labels = f"{stat['Negative Labels']:,}"
        
        # For mortality task, simplify display
        if stat['Task'] == 'mortality':
            logger.info(
                f"{stat['Task']}/"
                f"{stat['Dataset']}/"
                f"{stat['Set']}/"
                f"{stat['Windowing']}/"
                f"{total_stays}/"
                f"{cases} ({stat['Cases %']})/"
                f"{controls} ({stat['Controls %']})/"
                f"NA/NA/NA/NA/NA"
            )
        else:
            logger.info(
                f"{stat['Task']}/"
                f"{stat['Dataset']}/"
                f"{stat['Set']}/"
                f"{stat['Windowing']}/"
                f"{total_stays}/"
                f"{cases} ({stat['Cases %']})/"
                f"{controls} ({stat['Controls %']})/"
                f"{total_rows}/"
                f"{rows_cases} ({stat['Rows of Cases %']})/"
                f"{rows_controls} ({stat['Rows of Controls %']})/"
                f"{pos_labels} ({stat['Positive Labels %']})/"
                f"{neg_labels} ({stat['Negative Labels %']})"
            )
    
    logger.info(f"{'.'*40}")

def save_statistics_csv(
    statistics_list: List[Dict[str, Any]], 
    output_dir: str, 
    filename: str = "dataset_statistics.csv"
) -> str:
    """
    Save statistics as a CSV file with '/' as delimiter.
    
    Args:
        statistics_list (List[Dict[str, Any]]): List of statistics dictionaries
        output_dir (str): Directory to save the CSV file
        filename (str, optional): Name of the CSV file. Defaults to "dataset_statistics.csv".
        
    Returns:
        str: Path to the saved CSV file
    """
    if not statistics_list:
        logger.warning("No statistics available to save")
        return None
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for CSV
    rows = []
    
    # First row is header
    header = [
        "Task", "Dataset", "Set", "Windowing", 
        "Total Stays", "Cases", "Cases %", "Controls", "Controls %",
        "Total Rows", "Rows of Cases", "Rows of Cases %", "Rows of Controls", "Rows of Controls %",
        "Positive Labels", "Positive Labels %", "Negative Labels", "Negative Labels %"
    ]
    rows.append(header)
    
    # Add data rows
    for stat in statistics_list:
        # For mortality task, add NA for the row statistics
        if stat['Task'] == 'mortality':
            row = [
                stat['Task'], stat['Dataset'], stat['Set'], stat['Windowing'],
                str(stat['Total Stays']), str(stat['Cases']), stat['Cases %'], 
                str(stat['Controls']), stat['Controls %'],
                "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"
            ]
        else:
            row = [
                stat['Task'], stat['Dataset'], stat['Set'], stat['Windowing'],
                str(stat['Total Stays']), str(stat['Cases']), stat['Cases %'], 
                str(stat['Controls']), stat['Controls %'],
                str(stat['Total Rows']), str(stat['Rows of Cases']), stat['Rows of Cases %'],
                str(stat['Rows of Controls']), stat['Rows of Controls %'],
                str(stat['Positive Labels']), stat['Positive Labels %'],
                str(stat['Negative Labels']), stat['Negative Labels %']
            ]
        rows.append(row)
    
    # Create a CSV file with '/' as delimiter
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        for row in rows:
            f.write('/'.join(row) + '\n')
    
    logger.info(f"Statistics saved to CSV: {output_path}")
    return output_path

def get_dataset_statistics(config_path: str) -> List[Dict[str, Any]]:
    """
    Generate statistics for all datasets defined in the config file.
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        List[Dict[str, Any]]: List of statistics dictionaries
    """
    # Load configuration
    if config_path.endswith('.yaml'):
        from src.util.config_util import load_config_with_models
        config = load_config_with_models(config_path)
    else:
        config = OmegaConf.load(config_path)
    
    # Initialize dataset manager
    dm = DatasetManager(config)
    
    # Initialize lists for statistics
    all_statistics = []
    
    # Process each task and dataset combination
    for task in config.tasks:
        for dataset_name in config.datasets:
            dataset_id = f"{task}_{dataset_name}"
            
            # Load the dataset if not already loaded
            if not dm.datasets[dataset_id]["loaded"]:
                success = dm.load_dataset(dataset_id)
                if not success:
                    logger.error(f"Failed to load dataset: {dataset_id}")
                    continue

            data = dm.datasets[dataset_id]["data"]

            # Get windowing config if available
            windowing_config = None
            if hasattr(config, "preprocessing_advanced"):
                if hasattr(config.preprocessing_advanced, "windowing"):
                    windowing_config = config.preprocessing_advanced.windowing
            
            # Calculate statistics for each split
            train_stats = calculate_dataset_statistics(
                data["X_train"], data["y_train"], task, dataset_name, "train", windowing_config
            )
            val_stats = calculate_dataset_statistics(
                data["X_val"], data["y_val"], task, dataset_name, "val", windowing_config
            )
            test_stats = calculate_dataset_statistics(
                data["X_test"], data["y_test"], task, dataset_name, "test", windowing_config
            )
            
            # Add to the list of all statistics
            all_statistics.extend([train_stats, val_stats, test_stats])
            
            # If limit_test_set is specified, also get statistics for limited test set
            if config.get("limit_test_set", False):
                if 'stay_id' in data["X_test"].columns:
                    # Get unique stay_ids in ascending order
                    unique_stay_ids = sorted(data["X_test"]["stay_id"].unique())
                    # Take only the first 100 stay_ids (or all if less than 100)
                    selected_stay_ids = unique_stay_ids[:min(100, len(unique_stay_ids))]
                    # Filter X and y to include only the selected stay_ids
                    X_limited = data["X_test"][data["X_test"]["stay_id"].isin(selected_stay_ids)]
                    y_limited = data["y_test"][data["y_test"]["stay_id"].isin(selected_stay_ids)]
                    
                    # Calculate statistics for the limited test set
                    test_limited_stats = calculate_dataset_statistics(
                        X_limited, y_limited, task, dataset_name, "test_limited_100", windowing_config
                    )
                    all_statistics.append(test_limited_stats)
    
    return all_statistics

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate statistics for datasets defined in config")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--output-dir", type=str, help="Directory to save the statistics CSV file")
    parser.add_argument("--filename", type=str, default="dataset_statistics.csv", help="Name of the output CSV file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config.endswith('.yaml'):
        from src.util.config_util import load_config_with_models
        config = load_config_with_models(args.config)
    else:
        config = OmegaConf.load(args.config)
    
    # Set output directory (default is base_path/datasets/statistics)
    output_dir = args.output_dir
    if not output_dir:
        base_path = getattr(config, "base_path", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(base_path, "datasets/statistics")
    
    # Generate statistics
    statistics_list = get_dataset_statistics(args.config)
    
    # Print and save the statistics
    print_statistics(statistics_list)
    save_statistics_csv(statistics_list, output_dir, args.filename)
    
    logger.info("Statistics generation completed.")

if __name__ == "__main__":
    main()