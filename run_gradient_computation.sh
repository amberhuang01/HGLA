#!/bin/bash
#SBATCH --job-name=run_gradient_computation
#SBATCH --output=logs/run_gradient_computation_%j.out
#SBATCH --error=logs/run_gradient_computation_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --qos=normal

# Create logs directory if it doesn't exist
mkdir -p logs

# Configuration variables - MODIFY THESE FOR YOUR SETUP
BASE_DATA_DIR="./data"
MODEL_DIR="./models"
OUTPUT_DIR="./outputs"
SCRIPT_NAME="gradient_computation.py"

# Define models to analyze
MODELS=(
    "meta-llama/Meta-Llama-3-8B"
    "google/gemma-2b-it"
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# Define political datasets
POLITICAL_DATASETS=(
    "democrat_only.pkl"
    "republican_only.pkl"
    "random_calibration_df.pkl"
    "unbiased_calibration.pkl"
    "unbias_equal_representation.pkl"
    "bias_equal_representation.pkl"
    "biased_unbiased_combined.pkl"
)

# Define review datasets
REVIEW_DATASETS=(
    "negative_only.pkl"
    "positive_only.pkl"
    "random_df.pkl"
    "balanced_dev.pkl"
    "unbias_equal_representation.pkl"
    "bias_equal_representation.pkl"
    "biased_unbiased_combined.pkl"
)

# Function to extract model name from path
get_model_name() {
    local model_path=$1
    echo "$(basename "$model_path")"
}

# Function to run gradient analysis
run_gradient_analysis() {
    local model=$1
    local dataset_path=$2
    local is_political=$3
    local sparsity_ratio=${4:-0.5}  # Default sparsity ratio
    
    local political_flag=""
    if [ "$is_political" = true ]; then
        political_flag="--political"
    fi
    
    echo "Running gradient analysis for model: $model, dataset: $dataset_path"
    
    python "$SCRIPT_NAME" \
        --sparsity_ratio "$sparsity_ratio" \
        --model_path "$model" \
        --input_data_path "$dataset_path" \
        --output_dir "$OUTPUT_DIR" \
        --cache_dir "$MODEL_DIR/cache" \
        $political_flag
    
    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed: $model with $(basename "$dataset_path")"
    else
        echo "✗ Failed: $model with $(basename "$dataset_path")"
        return 1
    fi
}

# Function to construct dataset path
get_dataset_path() {
    local base_dir=$1
    local model_name=$2
    local dataset=$3
    local is_output_variant=$4
    
    if [ "$is_output_variant" = true ]; then
        echo "$base_dir/${model_name}_output_calibration_data/$dataset"
    else
        echo "$base_dir/$dataset"
    fi
}

# Function to check if dataset is an output variant
is_output_dataset() {
    local dataset=$1
    [[ "$dataset" == *"equal_representation"* ]] && return 0 || return 1
}

# Main execution
main() {
    echo "Starting gradient analysis batch job..."
    echo "Base data directory: $BASE_DATA_DIR"
    echo "Output directory: $OUTPUT_DIR"
    echo "Models: ${MODELS[*]}"
    echo "================================"
    
    # Create necessary directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "$MODEL_DIR/cache"
    
    local total_jobs=0
    local successful_jobs=0
    local failed_jobs=0
    
    # Loop through each model
    for model in "${MODELS[@]}"; do
        model_name=$(get_model_name "$model")
        echo "Processing model: $model_name"
        
        # Process political datasets
        echo "  Processing political datasets..."
        for dataset in "${POLITICAL_DATASETS[@]}"; do
            if is_output_dataset "$dataset"; then
                dataset_path=$(get_dataset_path "$BASE_DATA_DIR/political_input_data" "$model_name" "$dataset" true)
            else
                dataset_path=$(get_dataset_path "$BASE_DATA_DIR/political_input_data" "$model_name" "$dataset" false)
            fi
            
            # Check if dataset file exists
            if [ ! -f "$dataset_path" ]; then
                echo "    Warning: Dataset not found - $dataset_path"
                continue
            fi
            
            total_jobs=$((total_jobs + 1))
            if run_gradient_analysis "$model" "$dataset_path" true; then
                successful_jobs=$((successful_jobs + 1))
            else
                failed_jobs=$((failed_jobs + 1))
            fi
        done
        
        # Process review datasets
        echo "  Processing review datasets..."
        for dataset in "${REVIEW_DATASETS[@]}"; do
            if is_output_dataset "$dataset"; then
                dataset_path=$(get_dataset_path "$BASE_DATA_DIR/review_input_data" "$model_name" "$dataset" true)
            else
                dataset_path=$(get_dataset_path "$BASE_DATA_DIR/review_input_data" "$model_name" "$dataset" false)
            fi
            
            # Check if dataset file exists
            if [ ! -f "$dataset_path" ]; then
                echo "    Warning: Dataset not found - $dataset_path"
                continue
            fi
            
            total_jobs=$((total_jobs + 1))
            if run_gradient_analysis "$model" "$dataset_path" false; then
                successful_jobs=$((successful_jobs + 1))
            else
                failed_jobs=$((failed_jobs + 1))
            fi
        done
        
        echo "  Completed processing for $model_name"
        echo "--------------------------------"
    done
    
    # Print summary
    echo "================================"
    echo "Batch job completed!"
    echo "Total jobs: $total_jobs"
    echo "Successful: $successful_jobs"
    echo "Failed: $failed_jobs"
    echo "Success rate: $(( successful_jobs * 100 / total_jobs ))%"
    
    if [ $failed_jobs -gt 0 ]; then
        echo "Some jobs failed. Check the error logs for details."
        exit 1
    else
        echo "All jobs completed successfully!"
        exit 0
    fi
}

# Run main function
main "$@"