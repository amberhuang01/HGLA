#!/bin/bash
#SBATCH --job-name=download_models
#SBATCH --output=download_models_%j.out
#SBATCH --error=download_models_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --qos=normal

# Set your HF token here or use environment variable
HF_TOKEN="${HF_TOKEN:-your_token_here}"

# Download all models
python model_downloader.py --hf_token "$HF_TOKEN" --output_dir "./models"