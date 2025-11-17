# config.sh - Configuration file for gradient analysis
# Source this file or modify variables as needed

# Base directories - MODIFY THESE FOR YOUR SETUP
export BASE_DATA_DIR="./data"
export MODEL_DIR="./models" 
export OUTPUT_DIR="./outputs"
export SCRIPT_NAME="gradient_computation.py"

# SLURM configuration
export SLURM_JOB_NAME="gradient_computation"
export SLURM_PARTITION="gpu"  # Change to your cluster's GPU partition
export SLURM_GPUS="1"
export SLURM_CPUS="4"
export SLURM_MEMORY="64G"
export SLURM_TIME="24:00:00"
export SLURM_QOS="normal"

# Analysis parameters
export DEFAULT_SPARSITY_RATIO="0.5"
export DEFAULT_NSAMPLES="128"
export DEFAULT_SEED="42"

# Email notifications (optional - remove if not needed)
# export SLURM_MAIL_USER="your.email@university.edu"
# export SLURM_MAIL_TYPE="ALL"