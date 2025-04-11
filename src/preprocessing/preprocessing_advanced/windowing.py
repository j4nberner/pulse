import os
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import gc
import logging
from src.preprocessing.preprocessing_baseline.preprocessing_baseline import PreprocessorBaseline

# Set up logger
logger = logging.getLogger("PULSE_logger")

# TODO: create windower that returns 3d arrays

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
    
    def __init__(self, base_path, save_data=False, debug_mode=False, original_base_path=None, preprocessor_config=None):
        """
        Initialize the Windower.
        
        Args:
            base_path (str): Base path for data directories
            save_data (bool): Whether to save windowed data
            debug_mode (bool): Whether to run in debug mode (limited data)
            original_base_path (str): Original base path for permanent storage (for Slurm jobs)
        """
        self.base_path = base_path
        self.save_data = save_data
        self.debug_mode = debug_mode
        self.original_base_path = original_base_path
        self.preprocessor_config = preprocessor_config

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
            static_columns = ['stay_id', 'age', 'sex', 'height', 'weight']
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
        Save windowed data to parquet files, both in scratch and permanent storage if applicable.
        
        Args:
            results (dict): Dictionary containing windowed data
            task (str): Task name
            dataset (str): Dataset name
            data_window (int): Size of the data window
            prediction_window (int): Size of the prediction window
            step_size (int): Step size for window shifting
        """

        # Generate dynamic directory name based on preprocessing configuration
        preprocessor = PreprocessorBaseline(
            base_path=self.base_path,
            config=self.preprocessor_config
        )  # Pass config to temporary instance
        config_dirname = preprocessor._generate_preprocessing_dirname()

        # Directory naming adjusted based on debug mode
        debug_suffix = "_debug" if self.debug_mode else ""
        save_directory = f'datasets/preprocessed_splits/{task}/{dataset}/{config_dirname}/{data_window}_dw_{prediction_window}_pw_{step_size}_sz{debug_suffix}'
        os.makedirs(os.path.join(self.base_path, save_directory), exist_ok=True)
        
        # Save to current base_path (might be scratch)
        for set_type in ['train', 'val', 'test']:
            # Save files
            results[set_type]['X'].to_parquet(os.path.join(self.base_path, save_directory, f"X_{set_type}.parquet"))
            results[set_type]['y'].to_parquet(os.path.join(self.base_path, save_directory, f"y_{set_type}.parquet"))
        
        logger.info(f"Windowed data saved to {os.path.join(self.base_path, save_directory)}")
        
        # If original_base_path is set, also save to permanent storage
        if self.original_base_path:
            permanent_directory = os.path.join(self.original_base_path, save_directory)
            os.makedirs(permanent_directory, exist_ok=True)
            
            for set_type in ['train', 'val', 'test']:
                # Save files to permanent storage
                results[set_type]['X'].to_parquet(os.path.join(permanent_directory, f"X_{set_type}.parquet"))
                results[set_type]['y'].to_parquet(os.path.join(permanent_directory, f"y_{set_type}.parquet"))
            
            logger.info(f"Windowed data also saved to permanent storage: {permanent_directory}")

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
        preprocessor = PreprocessorBaseline(
            base_path=self.base_path, 
            config=self.preprocessor_config
        )  # Pass config to temporary instance
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
            
            logger.info(f"Loaded windowed data from {full_path}")

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
    
class WindowedDataTo3D:
    """
    Class for converting windowed ICU data from flattened 2D pandas DataFrames to 3D numpy arrays
    suitable for deep learning models.
    
    This class treats static features (like demographics) as time series by repeating their values
    across the time dimension, resulting in a single 3D array output.
    """
    
    def __init__(self, logger=None, model_name=None, config=None):
        """
        Initialize the WindowedDataTo3D converter.
        
        Args:
            logger: Logger instance for logging messages
            model_name (str, optional): Name of the model to determine array format
            config (dict, optional): Configuration with windowing parameters
        """
        self.logger = logger or logging.getLogger("PULSE_logger")
        
        # Dictionary mapping model names to their types (CNN or RNN)
        self.model_type_mapping = {
            # CNN type models
            "CNNModel": "CNN",
            
            # RNN type models
            "LSTMModel": "RNN",
            "InceptionTimeModel": "RNN",
            "GRUModel": "RNN"
        }
        
        # Set model type based on provided model name
        self.model_type = None
        if model_name:
            self.model_type = self.model_type_mapping.get(model_name)
            if not self.model_type:
                self.logger.warning(f"Unknown model name: {model_name}. Using RNN format as default.")
                self.model_type = "RNN"
                
        # Extract window size from config if available
        self.window_size = 6  # Default
        if config and "preprocessing_advanced" in config:
            if isinstance(config["preprocessing_advanced"], dict) and "windowing" in config["preprocessing_advanced"]:
                self.window_size = config["preprocessing_advanced"]["windowing"].get("data_window", 3)
                self.logger.info(f"Using window size {self.window_size} from config")
    
    def convert_to_3d(self, windowed_data, static_columns=None, id_column='stay_id', model_name=None):
        """
        Convert windowed data from 2D DataFrames to 3D numpy arrays, treating static features
        as time series by repeating them. The output shape is determined by the model type.
        
        Args:
            windowed_data (dict): Dictionary containing train, val, test data with X and y keys,
                                as produced by Windower.window_data()
            static_columns (list, optional): List of static columns to include as repeated time series
                                        Default: ['sex', 'age', 'height', 'weight']
            id_column (str): Column name to exclude from the resulting array
            model_name (str, optional): Name of the model to determine array format, 
                                        overrides the model_name provided during initialization
                                        
        Returns:
            dict: Dictionary containing the data as 3D arrays
        """
        if static_columns is None:
            static_columns = ['sex', 'age', 'height', 'weight']
        
        results = {}
        
        # Determine model type from model_name parameter or from initialization
        model_type = None
        if model_name:
            model_type = self.model_type_mapping.get(model_name)
            if not model_type:
                self.logger.warning(f"Unknown model name: {model_name}. Using RNN format as default.")
                model_type = "RNN"
        else:
            model_type = self.model_type
            
        # Default to RNN if model_type is still None
        if not model_type:
            model_type = "RNN"
            
        is_cnn = model_type == "CNN"
        
        for set_type in ['train', 'val', 'test']:
            if set_type not in windowed_data:
                self.logger.warning(f"Set type {set_type} not found in windowed data")
                continue
                
            X = windowed_data[set_type]['X']
            y = windowed_data[set_type]['y']
            
            # Drop the ID column
            X_without_id = X.drop(columns=[id_column], errors='ignore')
            
            # Find all time-dependent columns and organize them
            time_dependent_cols = {}
            time_indices_set = set()
            
            for col in X_without_id.columns:
                if col in static_columns:
                    # Skip static columns for now, will add them later
                    continue
                    
                # Extract feature name and time index using regex
                match = re.match(r'(.+)_(\d+)$', col)
                if match:
                    feature_name, time_idx = match.groups()
                    time_idx = int(time_idx)
                    time_indices_set.add(time_idx)
                    
                    if feature_name not in time_dependent_cols:
                        time_dependent_cols[feature_name] = []
                    
                    time_dependent_cols[feature_name].append((time_idx, col))
            
            # Get a sorted list of all unique time indices
            time_indices = sorted(time_indices_set)
            data_window = len(time_indices)
            
            # Get feature names sorted alphabetically
            feature_names = sorted(time_dependent_cols.keys())
            
            # Add static columns to the feature list
            available_static_cols = [col for col in static_columns if col in X_without_id.columns]
            all_feature_names = available_static_cols + feature_names
            
            # Create 3D array with appropriate dimensions based on model type
            samples = X_without_id.shape[0]
            n_features = len(all_feature_names)
            
            if is_cnn:
                # CNN format: (samples, features, time_steps)
                X_3d = np.zeros((samples, n_features, data_window), dtype=np.float32)
                self.logger.info(f"Creating 3D array for {set_type} set (CNN format): {samples} samples, {n_features} features, {data_window} time steps")
            else:
                # RNN format: (samples, time_steps, features)
                X_3d = np.zeros((samples, data_window, n_features), dtype=np.float32)
                self.logger.info(f"Creating 3D array for {set_type} set (RNN format): {samples} samples, {data_window} time steps, {n_features} features")
            
            # Fill the array based on model type
            if is_cnn:
                # Fill static features for CNN format
                for feature_idx, feature_name in enumerate(available_static_cols):
                    static_values = X_without_id[feature_name].values
                    # Broadcast static values across the time dimension
                    for t in range(data_window):
                        X_3d[:, feature_idx, t] = static_values
                
                # Fill time-dependent features for CNN format
                static_offset = len(available_static_cols)
                for feature_idx, feature_name in enumerate(feature_names):
                    for time_idx, col_name in sorted(time_dependent_cols[feature_name], key=lambda x: x[0]):
                        # Map the time_idx to its position in the time_indices list
                        t_pos = time_indices.index(time_idx)
                        X_3d[:, feature_idx + static_offset, t_pos] = X_without_id[col_name].values
            else:
                # Fill static features for RNN format
                for feature_idx, feature_name in enumerate(available_static_cols):
                    static_values = X_without_id[feature_name].values
                    # Broadcast static values across the time dimension
                    for t in range(data_window):
                        X_3d[:, t, feature_idx] = static_values
                
                # Fill time-dependent features for RNN format
                static_offset = len(available_static_cols)
                for feature_idx, feature_name in enumerate(feature_names):
                    for time_idx, col_name in sorted(time_dependent_cols[feature_name], key=lambda x: x[0]):
                        # Map the time_idx to its position in the time_indices list
                        t_pos = time_indices.index(time_idx)
                        X_3d[:, t_pos, feature_idx + static_offset] = X_without_id[col_name].values
            
            # Convert y to numpy array
            y_np = y['label'].values.astype(np.int32)
            
            # Store results
            results[set_type] = {
                'X': X_3d,                          # 3D array with appropriate shape for model
                'y': y_np,                          # 1D array with labels
                'feature_names': all_feature_names  # List of feature names
            }
            
            self.logger.info(f"Converted {set_type} set to 3D array with shape: {X_3d.shape}")
        
        return results

    def convert_batch_to_3d(self, batch_features, window_size=None, static_feature_count=4, id_column_index=0):
        """
        Convert a batch of features from 2D to 3D format suitable for temporal models.
        Works with tensors extracted directly from the dataloader.
        
        Args:
            batch_features (torch.Tensor): Batch of features in 2D format (batch_size, n_features)
            window_size (int, optional): Size of the time window. If None, will try to infer it
            static_feature_count (int): Number of static features (excluding id_column)
            id_column_index (int): Index of the ID column to exclude (typically 0 for stay_id)
            
        Returns:
            torch.Tensor: Batch of features in 3D format
        """
        import torch
        
        # If already 3D, return as is
        if len(batch_features.shape) == 3:
            return batch_features
        
        batch_size, n_features = batch_features.shape
        
        # Determine model type (CNN or RNN)
        is_cnn = self.model_type == "CNN"
        
        # Skip the ID column
        if id_column_index is not None:
            # Create a mask to exclude the ID column
            keep_mask = torch.ones(n_features, dtype=torch.bool)
            keep_mask[id_column_index] = False
            
            # Apply the mask to get features without ID
            features_no_id = batch_features[:, keep_mask]
            n_features = features_no_id.shape[1]
        else:
            features_no_id = batch_features
        
        # Extract static features (typically columns 0-3 after removing ID)
        static_features = features_no_id[:, :static_feature_count]
        
        # Extract dynamic features (everything after static features)
        dynamic_features = features_no_id[:, static_feature_count:]
        n_dynamic_features = dynamic_features.shape[1]
        
        # Try to use configured window size or infer if not provided
        if window_size is None:
            # Use stored window size from config if available
            if hasattr(self, 'window_size'):
                window_size = self.window_size
                self.logger.debug(f"Using window size {window_size} from config")
            else:
                # Otherwise try to infer from data
                possible_window_sizes = [3, 6, 9, 12, 18, 24]
                for ws in possible_window_sizes:
                    if n_dynamic_features % ws == 0:
                        window_size = ws
                        break
                
                # Default to 6 if we can't determine
                if window_size is None:
                    window_size = 3
                    self.logger.warning(f"Could not infer window size. Using default: {window_size}")
        
        # Calculate number of actual features
        n_actual_dynamic_features = n_dynamic_features // window_size
        
        try:
            # Reshape dynamic features based on window size
            if is_cnn:
                # For CNN: (batch, features, time)
                dynamic_3d = dynamic_features.reshape(batch_size, n_actual_dynamic_features, window_size)
                
                # Repeat static features for each time step
                static_3d = static_features.unsqueeze(-1).repeat(1, 1, window_size)
                
                # Combine on feature dimension (static first, then dynamic)
                return torch.cat([static_3d, dynamic_3d], dim=1)
            else:
                # For RNN: (batch, time, features)
                dynamic_3d = dynamic_features.reshape(batch_size, window_size, n_actual_dynamic_features)
                
                # Repeat static features for each time step
                static_3d = static_features.unsqueeze(1).repeat(1, window_size, 1)
                
                # Combine on feature dimension
                return torch.cat([static_3d, dynamic_3d], dim=2)
        
        except Exception as e:
            self.logger.warning(f"Error reshaping batch to 3D: {e}. Using simple approach.")
            
            # Fall back to simple reshape if the proper reshaping fails
            if is_cnn:
                return features_no_id.unsqueeze(-1)  # (batch, features, 1)
            else:
                return features_no_id.unsqueeze(1)   # (batch, 1, features)

