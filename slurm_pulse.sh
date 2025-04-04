#!/bin/bash
#SBATCH --job-name=icu_prediction
#SBATCH --output=icu_prediction_%j.out
#SBATCH --error=icu_prediction_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=compute

# Uncomment these lines if you need GPU resources
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1

# Email notifications (optional)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

# Navigate to project directory
# Change this to the location of your repository
cd $SLURM_SUBMIT_DIR/icu_prediction_framework

# Activate virtual environment
source $SLURM_SUBMIT_DIR/venv/bin/activate

# Run the training script with the existing config
echo "Starting training on $(hostname) at $(date)"
python train_models.py --config config_train.yaml

echo "Job completed at $(date)"