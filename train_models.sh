#!/bin/bash
#SBATCH --job-name=train_models
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=60000
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

# Run the training script with the existing config
echo "Starting training on $(hostname) at $(date)"
python train_models.py --config config_train.yaml
echo "Job completed at $(date)"