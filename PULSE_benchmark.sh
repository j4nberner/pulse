#!/bin/bash
#SBATCH --job-name=PULSE_benchmark
#SBATCH --output=output/slurm_output/delete/training_%j_%N_%x.output
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:35000m
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@example.com

##SBATCH --partition=compute

export PYTHONHASHSEED=42
module load python/3.11.6
module load cuda
nvidia-smi

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

# Load python module, create/activate venv, and install requirements
# python -m venv .venv
source .venv/bin/activate
# pip install -r requirements.txt

# Print job and node information
echo "======================================================================="
echo "Job Information:"
echo " Job ID: $SLURM_JOB_ID"
echo " Job Name: $SLURM_JOB_NAME"
echo " Node: $(hostname)"
echo " Started at: $(date)"
echo "======================================================================="

# Extract and print all Slurm settings
echo "Slurm Settings:"
echo " Number of Tasks: $SLURM_NTASKS"
echo " CPUs per Task: $SLURM_CPUS_PER_TASK"
echo " Total CPUs: $SLURM_CPUS_ON_NODE"
echo " Memory per CPU: $(grep -oP '(?<=#SBATCH --mem-per-cpu=)[0-9]+' $0) MB"
echo " Time Limit: $(grep -oP '(?<=#SBATCH --time=)[0-9:]+' $0)"
echo " Partition: $SLURM_JOB_PARTITION"

# Check if GPU is enabled and print GPU information
if [ -n "$SLURM_GPUS" ]; then
    echo " GPU Configuration: $SLURM_GPUS GPU(s) allocated"
    if command -v nvidia-smi &> /dev/null; then
        echo " GPU Information:"
        nvidia-smi --query-gpu=name --format=csv,noheader
    else
        echo " GPU Information: nvidia-smi command not available"
    fi
else
    echo " GPU Configuration: No GPU allocated"
fi

echo "======================================================================="

echo "Environment:"
echo " TMPDIR: $TMPDIR" # Slurm scratch directory
echo " SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
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

# Start memory monitoring in the background
echo "Starting memory monitoring..."
while true; do
    echo "===== Memory Usage at $(date) =====" >> output/slurm_output/memory_${TIMESTAMP}_${JOB_ID}.log
    free -h >> output/slurm_output/memory_${TIMESTAMP}_${JOB_ID}.log

    # Attempt to log GPU memory usage
    if ! nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> output/slurm_output/memory_${TIMESTAMP}_${JOB_ID}.log; then
        echo "WARNING: Failed to log GPU memory usage at $(date)" >> output/slurm_output/memory_${TIMESTAMP}_${JOB_ID}.log
    else
        echo "GPU memory usage logged successfully at $(date)" >> output/slurm_output/memory_${TIMESTAMP}_${JOB_ID}.log
    fi
    
    echo "===================================" >> output/slurm_output/memory_${TIMESTAMP}_${JOB_ID}.log
    sleep 60  # Log every 60 seconds
done &
MEMORY_MONITOR_PID=$!

# Run the training script with the existing config
echo "Starting training on $(hostname) at $(date)"
python PULSE_benchmark.py --config configs/config_benchmark.yaml

# Stop memory monitoring
echo "Stopping memory monitoring..."
kill $MEMORY_MONITOR_PID

# Clean up scratch space at the end of the job
if [ -n "$TMPDIR" ] && [ -d "$TMPDIR/job_${SLURM_JOB_ID}_data" ]; then
  echo "Cleaning up scratch directory: $TMPDIR/job_${SLURM_JOB_ID}_data"
  rm -rf "$TMPDIR/job_${SLURM_JOB_ID}_data"
fi

echo "Job completed at $(date)"
echo "Total runtime: $(($(date +%s) - $(date -d "$(grep 'Started at:' output/slurm_output/training_${TIMESTAMP}_${JOB_ID}.output | cut -d':' -f2- | xargs)" +%s))) seconds"