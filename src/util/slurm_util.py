import logging
import os
import shutil

from tqdm import tqdm

logger = logging.getLogger(__name__)

def is_on_slurm():
    """Check if we're running on a Slurm cluster."""
    job_id = os.getenv("SLURM_JOB_ID")
    return job_id is not None

def get_local_scratch_dir():
    """Get the scratch directory on the compute node."""
    # TMPDIR is usually set on Slurm compute nodes
    local_scratch_dir = os.getenv("TMPDIR")
    if not local_scratch_dir:
        # Fallback - some clusters might use a different env variable
        local_scratch_dir = os.getenv("SCRATCH")
    return local_scratch_dir

def copy_data_to_scratch(config):
    """
    Copy dataset files to local scratch space on compute node.
    
    Args:
        config (TrainConfig): The training configuration object
        
    Returns:
        tuple: (config, data_copied) where:
            - config is the updated config with modified dataset paths
            - data_copied is a boolean indicating if data was copied
    """
    data_copied = False
    
    if not is_on_slurm():
        logger.info("Not running on Slurm, using original data paths.")
        return config, data_copied
    
    # Get scratch directory
    scratch_dir = get_local_scratch_dir()
    if not scratch_dir:
        logger.warning("Scratch directory not found. Using original data paths.")
        return config, data_copied
    
    # Create a directory for this job in scratch
    job_id = os.getenv("SLURM_JOB_ID")
    username = os.getenv("USER", "user")
    scratch_data_dir = os.path.join(scratch_dir, f"tmp.{job_id}.{username}")
    os.makedirs(scratch_data_dir, exist_ok=True)
    
    logger.info(f"Scratch directory available at: {scratch_data_dir}")
    
    # Store the original base path for saving results to permanent storage
    original_base_path = config.base_path
    config.original_base_path = original_base_path  # Add this as a new attribute

    # Dictionary of directories to copy
    dirs_to_copy = [
        "datasets/preprocessed_splits",  # Preprocessed data
        "datasets/original_harmonized"   # Original data
    ]
    
    # Copy each directory structure
    for dir_path in dirs_to_copy:
        source_dir = os.path.join(original_base_path, dir_path)
        
        # Skip if source doesn't exist
        if not os.path.exists(source_dir):
            logger.warning(f"Source directory not found: {source_dir}")
            continue
            
        dest_dir = os.path.join(scratch_data_dir, dir_path)
        
        # Create the destination directory structure
        os.makedirs(dest_dir, exist_ok=True)
        
        # Copy data for each task and dataset
        for task in config.tasks:
            task_dir = os.path.join(source_dir, task)
            
            if not os.path.exists(task_dir):
                logger.warning(f"Task directory not found: {task_dir}")
                continue
                
            # Create task directory in destination
            os.makedirs(os.path.join(dest_dir, task), exist_ok=True)
            
            for dataset in config.datasets:
                dataset_dir = os.path.join(task_dir, dataset)
                
                if not os.path.exists(dataset_dir):
                    logger.warning(f"Dataset directory not found: {dataset_dir}")
                    continue
                    
                # Create dataset directory in destination
                dataset_dest_dir = os.path.join(dest_dir, task, dataset)
                os.makedirs(dataset_dest_dir, exist_ok=True)
                
                try:
                    # For preprocessed_splits, copy each configuration subdirectory
                    if "preprocessed_splits" in dir_path:
                        for config_dir in os.listdir(dataset_dir):
                            source_config_path = os.path.join(dataset_dir, config_dir)
                            
                            # Skip if not a directory
                            if not os.path.isdir(source_config_path):
                                continue
                                
                            dest_config_path = os.path.join(dataset_dest_dir, config_dir)
                            os.makedirs(dest_config_path, exist_ok=True)
                            
                            # Copy all files in this configuration directory
                            for item in tqdm(os.listdir(source_config_path), 
                                            desc=f"Copying {task}/{dataset}/{config_dir}"):
                                src_item = os.path.join(source_config_path, item)
                                dst_item = os.path.join(dest_config_path, item)
                                
                                if os.path.isdir(src_item):
                                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src_item, dst_item)
                    
                    # For original_harmonized, copy the parquet files directly
                    else:
                        for item in tqdm(os.listdir(dataset_dir), 
                                        desc=f"Copying original data {task}/{dataset}"):
                            src_item = os.path.join(dataset_dir, item)
                            dst_item = os.path.join(dataset_dest_dir, item)
                            
                            if os.path.isdir(src_item):
                                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                            else:
                                shutil.copy2(src_item, dst_item)
                    
                    data_copied = True
                    
                except Exception as e:
                    logger.error(f"Error copying dataset '{task}/{dataset}': {str(e)}")
    
    # Update base_path in config to point to scratch
    if data_copied:
        # Check if scratch_data_dir is already in the path to avoid duplication
        if scratch_data_dir not in config.base_path:
            config.base_path = scratch_data_dir
            logger.info(f"Updated base_path from {original_base_path} to {scratch_data_dir}")
            logger.info(f"Original base path saved for permanent storage: {config.original_base_path}")
        else:
            logger.warning(f"base_path already contains scratch directory: {config.base_path}")
    
    return config, data_copied