import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
import logging
from src.preprocessing.preprocessing_baseline.preprocessing_baseline import PreprocessorBaseline

# Set up logger
logger = logging.getLogger("PULSE_logger")

# TODO: Add logging of windowed data (processed or loaded) shape

class Windower:
    """
    Class for creating windowed data with specified data window, prediction window, and step size.
    
    This class transforms time-series ICU data into fixed-width windows suitable for
    machine learning models. It handles:
    - Taking preprocessed dataframes and applying windowing transformation
    - Creating windows with configurable data window, prediction window, and step size
    - Saving windowed data to parquet files if specified
    - Loading previously windowed data
    """
    
    def __init__(self, base_path, save_data=False, debug_mode=False):
        """
        Initialize the Windower.
        
        Args:
            base_path (str): Base path for data directories
            save_data (bool): Whether to save windowed data
        """
        self.base_path = base_path
        self.save_data = save_data
        self.debug_mode = debug_mode

    def create_windows(self, data_dict, data_window, prediction_window, step_size=1):
        """
        Create windowed data with specified data window, prediction window, and step size.
        
        Args:
            data_dict (dict): Dictionary containing train, val, test data with X and y keys
            data_window (int): Size of the data window (number of time steps to include)
            prediction_window (int): Size of the prediction window (time to prediction)
            step_size (int): Step size for window shifting
            
        Returns:
            dict: Dictionary containing windowed data
        """
        results = {}
        
        for set_type in ['train', 'val', 'test']:
            X = data_dict[set_type]['X']
            y = data_dict[set_type]['y']
            
            X_np = X.values
            y_np = y['label'].values
            stay_id_np = X['stay_id'].values

            # Define static columns that will be preserved in order
            static_columns = ['stay_id', 'sex', 'age', 'height', 'weight']
            columns_index = [X.columns.get_loc(col) for col in static_columns if col in X.columns]
            non_static_columns = [i for i in range(X_np.shape[1]) if i not in columns_index]

            result_rows = []
            result_labels = []
            unique_stay_ids = np.unique(stay_id_np)
            
            # If in debug mode, limit to 100 stay_ids
            if self.debug_mode:
                if len(unique_stay_ids) > 100:
                    unique_stay_ids = unique_stay_ids[:100]
                    logger.info(f"DEBUG MODE: Limited to first 100 stay_ids for {set_type} set")

            logger.info(f"Processing {set_type} set with {len(unique_stay_ids)} stay_ids")
            # Update progress bar every 5000 stay_ids (or fewer in debug mode)
            miniters = 5000 if not self.debug_mode else 10
            for stay_id in tqdm(unique_stay_ids, mininterval=1.0, miniters=miniters, desc=f"{set_type} stay_ids"):
                mask = stay_id_np == stay_id
                X_stay = X_np[mask]
                y_stay = y_np[mask]
                
                # Minimum length depends on prediction window
                min_length = data_window + prediction_window
                if prediction_window == 0:
                    min_length = data_window
                    
                if len(X_stay) < min_length:
                    continue
                
                # Get static values for this stay
                static_row = {X.columns[col_idx]: X_stay[0, col_idx] for col_idx in columns_index}

                # Adjust the range based on prediction window
                max_start = len(X_stay) - min_length + 1
                
                # Use step_size for window sliding
                for start in range(0, max_start, step_size):
                    X_window = X_stay[start:start + data_window]
                    
                    # Skip if we don't have a full window (could happen with the last window)
                    if len(X_window) < data_window:
                        continue
                    
                    row = {f'{X.columns[col_idx]}_{hour}': X_window[hour, col_idx]
                           for col_idx in non_static_columns
                           for hour in range(data_window)}
                    
                    row.update(static_row)
                    result_rows.append(row)
                    
                    # For prediction_window=0, get label from the last position of data window
                    # For positive prediction_window, get label from the position after data window + prediction_window - 1
                    label_position = start + data_window - 1
                    if prediction_window > 0:
                        label_position = start + data_window + prediction_window - 1
                        
                    result_labels.append(y_stay[label_position])

            X_window = pd.DataFrame(result_rows)
            y_window = pd.DataFrame(result_labels, columns=['label'])

            # Ensure correct datatypes
            for col in X_window.columns:
                if '_' in col and col.split('_')[-1].isdigit():
                    # Extract the base column name (everything before the last underscore)
                    base_col = '_'.join(col.split('_')[:-1])
                    if base_col in X.columns:
                        X_window[col] = X_window[col].astype(X[base_col].dtype)
                else:
                    # For static columns
                    if col in X.columns:
                        X_window[col] = X_window[col].astype(X[col].dtype)
            
            y_window['label'] = y_window['label'].astype(int)
            
            # Reorder columns to have stay_id, sex, age, height, weight first
            all_columns = list(X_window.columns)
            ordered_static_columns = [col for col in static_columns if col in all_columns]
            other_columns = [col for col in all_columns if col not in ordered_static_columns]
            X_window = X_window[ordered_static_columns + other_columns]
            
            # Log the shape of the windowed dataset
            logger.info(f"Windowed {set_type} set - X shape: {X_window.shape}, y shape: {y_window.shape}")
            
            results[set_type] = {'X': X_window, 'y': y_window}
        
        # Force garbage collection to free memory
        gc.collect()
        
        return results  
  
    def save_windowed_data(self, results, task, dataset, data_window, prediction_window, step_size):
        """
        Save windowed data to parquet files.
        
        Args:
            results (dict): Dictionary containing windowed data
            task (str): Task name
            dataset (str): Dataset name
            data_window (int): Size of the data window
            prediction_window (int): Size of the prediction window
            step_size (int): Step size for window shifting
        """
        # Generate dynamic directory name based on preprocessing configuration
        preprocessor = PreprocessorBaseline(base_path=self.base_path)  # Temporary instance to access method
        config_dirname = preprocessor._generate_preprocessing_dirname()

        # Directory naming adjusted based on debug mode
        debug_suffix = "_debug" if self.debug_mode else ""
        save_directory = f'datasets/preprocessed_splits/{task}/{dataset}/{config_dirname}/{data_window}_dw_{prediction_window}_pw_{step_size}_sz{debug_suffix}'
        os.makedirs(os.path.join(self.base_path, save_directory), exist_ok=True)
        
        for set_type in ['train', 'val', 'test']:
            # Save files
            results[set_type]['X'].to_parquet(os.path.join(self.base_path, save_directory, f"X_{set_type}.parquet"))
            results[set_type]['y'].to_parquet(os.path.join(self.base_path, save_directory, f"y_{set_type}.parquet"))
        
        logger.info(f"All windowed data saved to {os.path.join(self.base_path, save_directory)}")

    def load_windowed_data(self, task, dataset, data_window, prediction_window, step_size):
        """
        Load previously windowed data from parquet files.
        
        Args:
            task (str): Task name
            dataset (str): Dataset name
            data_window (int): Size of the data window
            prediction_window (int): Size of the prediction window
            step_size (int): Step size for window shifting
            
        Returns:
            dict: Dictionary containing windowed data or None if loading fails
        """
        # Generate dynamic directory name based on preprocessing configuration
        preprocessor = PreprocessorBaseline(base_path=self.base_path)  # Temporary instance to access method
        config_dirname = preprocessor._generate_preprocessing_dirname()

        # Check for debug mode directory first
        debug_suffix = "_debug" if self.debug_mode else ""
        load_directory = f'datasets/preprocessed_splits/{task}/{dataset}/{config_dirname}/{data_window}_dw_{prediction_window}_pw_{step_size}_sz{debug_suffix}'
        full_path = os.path.join(self.base_path, load_directory)
        
        # If in debug mode but debug files don't exist, we don't want to fall back to regular files
        if self.debug_mode and not os.path.exists(full_path):
            logger.info(f"Debug-mode windowed data directory {full_path} does not exist")
            return None
        
        # If not in debug mode and regular files don't exist
        if not os.path.exists(full_path):
            logger.info(f"Windowed data directory {full_path} does not exist")
            return None
        
        results = {}
        
        try:
            for set_type in ['train', 'val', 'test']:
                X_path = os.path.join(full_path, f"X_{set_type}.parquet")
                y_path = os.path.join(full_path, f"y_{set_type}.parquet")
                
                if not os.path.exists(X_path) or not os.path.exists(y_path):
                    logger.error(f"Missing windowed data files in {full_path}")
                    return None
                
                X = pd.read_parquet(X_path)
                y = pd.read_parquet(y_path)
                
                results[set_type] = {'X': X, 'y': y}
                
                logger.info(f"Loaded windowed {set_type} set - X shape: {X.shape}, y shape: {y.shape}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error loading windowed data from {full_path}: {e}")
            return None
    
    def read_preprocessed_data(self, task, dataset):
        """
        Read preprocessed train, validation and test parquet files.
        
        Args:
            task (str): Task name
            dataset (str): Dataset name
            
        Returns:
            dict: Dictionary containing preprocessed data or None if loading fails
        """
        read_directory = f"datasets/preprocessed_splits/{task}/{dataset}/train_val_test_standardized"
        full_path = os.path.join(self.base_path, read_directory)
        
        if not os.path.exists(full_path):
            logger.error(f"Preprocessed data directory {full_path} does not exist")
            return None
        
        try:
            data_sets = {
                'train': {
                    'X': pd.read_parquet(os.path.join(full_path, "X_train.parquet")),
                    'y': pd.read_parquet(os.path.join(full_path, "y_train.parquet"))
                },
                'val': {
                    'X': pd.read_parquet(os.path.join(full_path, "X_val.parquet")),
                    'y': pd.read_parquet(os.path.join(full_path, "y_val.parquet"))
                },
                'test': {
                    'X': pd.read_parquet(os.path.join(full_path, "X_test.parquet")),
                    'y': pd.read_parquet(os.path.join(full_path, "y_test.parquet"))
                }
            }
            
            # Log shapes of loaded datasets
            for set_type in data_sets:
                logger.info(f"Loaded {set_type} set - X shape: {data_sets[set_type]['X'].shape}, y shape: {data_sets[set_type]['y'].shape}")
            
            return data_sets
        
        except Exception as e:
            logger.error(f"Error reading preprocessed data from {full_path}: {e}")
            return None
    
    def window_data(self, task, dataset, config, data_dict=None):
        """
        Apply windowing to a dataset, always trying to load presaved data first.
        
        Args:
            task (str): Task name
            dataset (str): Dataset name
            config (dict): Configuration for windowing
            data_dict (dict, optional): Dictionary containing data to window. If None, will load from preprocessed files.
            
        Returns:
            dict: Dictionary containing windowed data or original data if windowing is not applicable
        """
        
        # Extract windowing parameters from config
        data_window = config.get('data_window', 6)
        prediction_window = config.get('prediction_window', 0)
        step_size = config.get('step_size', 1)
        
        # ALWAYS try to load presaved windowed data first
        logger.info(f"Checking for presaved windowed data for {task}_{dataset} with data_window={data_window}, "
                f"prediction_window={prediction_window}, step_size={step_size}")
        windowed_data = self.load_windowed_data(task, dataset, data_window, prediction_window, step_size)
        
        if windowed_data is not None:
            logger.info(f"Using presaved windowed data for {task}_{dataset}")
            return windowed_data
        
        logger.info(f"No presaved windowed data found, will create new windows")
        
        # If data_dict is not provided, load preprocessed data
        if data_dict is None:
            logger.info(f"Loading preprocessed data for {task}_{dataset}")
            data_dict = self.read_preprocessed_data(task, dataset)
            if data_dict is None:
                logger.error(f"Could not load preprocessed data for {task}_{dataset}")
                return None
        
        logger.info(f"Applying windowing to {task}_{dataset} with data_window={data_window}, "
                f"prediction_window={prediction_window}, step_size={step_size}")
        
        # Create windows
        windowed_data = self.create_windows(data_dict, data_window, prediction_window, step_size)
        
        # Save windowed data if specified
        if self.save_data:
            logger.info(f"Saving windowed data for {task}_{dataset}")
            self.save_windowed_data(windowed_data, task, dataset, data_window, prediction_window, step_size)
        
        return windowed_data