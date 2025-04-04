import os
import shutil
import logging
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
        config: Updated config with modified dataset paths
    """
    if not is_on_slurm():
        logger.info("Not running on Slurm, using original data paths.")
        return config
    
    if not config.general.get('use_scratch', False):
        logger.info("Scratch usage disabled in config, using original data paths.")
        return config
    
    # Get scratch directory
    scratch_dir = get_local_scratch_dir()
    if not scratch_dir:
        logger.warning("Scratch directory not found. Using original data paths.")
        return config
    
    # Create a directory for this job in scratch
    job_id = os.getenv("SLURM_JOB_ID")
    scratch_data_dir = os.path.join(scratch_dir, f"job_{job_id}_data")
    os.makedirs(scratch_data_dir, exist_ok=True)
    
    logger.info(f"Using scratch directory: {scratch_data_dir}")
    
    # Build the base path to the original datasets
    base_path = config.base_path
    
    # Copy the data for each task and dataset combination
    for task in config.tasks:
        for dataset in config.datasets:
            source_path = os.path.join(
                base_path, 
                "datasets", 
                "preprocessed_splits", 
                task, 
                dataset,
                "train_val_test_standardized"
            )

            # TODO: "train_val_test_standardized" should not be hardcoded
            
            if not os.path.exists(source_path):
                logger.warning(f"Dataset path not found: {source_path}")
                continue
                
            # Create corresponding structure in scratch
            dest_path = os.path.join(
                scratch_data_dir,
                "datasets",
                "preprocessed_splits",
                task,
                dataset,
                "train_val_test_standardized"
            )
            
            os.makedirs(dest_path, exist_ok=True)
            logger.info(f"Copying dataset '{task}/{dataset}' to scratch: {dest_path}")
            
            try:
                # List and copy all files in the directory
                for item in tqdm(os.listdir(source_path), desc=f"Copying {task}/{dataset}"):
                    src_item = os.path.join(source_path, item)
                    dst_item = os.path.join(dest_path, item)
                    
                    if os.path.isdir(src_item):
                        shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_item, dst_item)
                        
                logger.info(f"Successfully copied dataset '{task}/{dataset}' to scratch")
                
            except Exception as e:
                logger.error(f"Error copying dataset '{task}/{dataset}' to scratch: {str(e)}")
    
    # Update base_path in config to point to scratch
    scratch_base_path = os.path.join(scratch_data_dir)
    logger.info(f"Updated base_path from {config.base_path} to {scratch_base_path}")
    config.base_path = scratch_base_path
    
    return config