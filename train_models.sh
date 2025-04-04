#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --output=output/slurm_output/delete/training_%j_%N_%x.output

#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=480000
#SBATCH --partition=compute

# Uncomment these lines if you need GPU resources
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# Create timestamp variables
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
JOB_ID=$SLURM_JOB_ID

# Create output directory
mkdir -p output/slurm_output

# Redirect output to timestamped files
exec > output/slurm_output/training_${TIMESTAMP}_${JOB_ID}.output
exec 2> output/slurm_output/training_${TIMESTAMP}_${JOB_ID}.error

# Change to main directory
cd $SLURM_SUBMIT_DIR

# Activate virtual environment from the venv folder in your project
source .venv/bin/activate

# Print job and node information
echo "======================================================================="
echo "Job Information:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  Started at: $(date)"
echo "======================================================================="
echo "Environment:"
echo "  TMPDIR: $TMPDIR"  # Slurm scratch directory
echo "  SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "======================================================================="

# Ensure scratch directory exists and has space
if [ -n "$TMPDIR" ]; then
    mkdir -p $TMPDIR/job_${SLURM_JOB_ID}_data
    echo "Scratch directory created at: $TMPDIR/job_${SLURM_JOB_ID}_data"
    echo "Available space in scratch directory:"
    df -h $TMPDIR
else
    echo "WARNING: TMPDIR not set. Scratch space might not be available."
fi

# Run the training script with the existing config
echo "Starting training on $(hostname) at $(date)"
python train_models.py --config config_train.yaml

# Clean up scratch space (optional - uncomment if you want to clean up after job)
if [ -n "$TMPDIR" ] && [ -d "$TMPDIR/job_${SLURM_JOB_ID}_data" ]; then
    echo "Cleaning up scratch directory: $TMPDIR/job_${SLURM_JOB_ID}_data"
    rm -rf $TMPDIR/job_${SLURM_JOB_ID}_data
fi

echo "Job completed at $(date)"