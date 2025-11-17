#!/bin/bash
#SBATCH --job-name=run_prune_hgla
#SBATCH --output=logs/run_prune_hgla_%j.out
#SBATCH --error=logs/run_prune_hgla_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --qos=normal

# Create logs directory
mkdir -p logs

# Configuration - MODIFY THESE FOR YOUR SETUP
BASE_DATA_DIR="./data"
BASE_GRADIENT_DIR="./outputs/gradients"
BASE_OUTPUT_DIR="./outputs/pruning_results"
MODEL_DIR="./models"
SCRIPT_NAME="prune_hgla.py"

# Define sparsity ratios to test
SPARSITY_RATIOS=(0.1 0.2 0.3 0.4 0.5)

# Define models to test
MODELS=(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    "meta-llama/Meta-Llama-3-8B"
    "google/gemma-2b-it"
)

# Define datasets and their categories
POLITICAL_DATASETS=(
    "democrat_only"
    "republican_only" 
    "unbiased_calibration"
    "biased_unbiased_combined"
    "bias_equal_representation"
    "unbias_equal_representation"
)

REVIEW_DATASETS=(
    "positive_only"
    "negative_only"
    "unbiased_calibration"
    "biased_unbiased_combined"
    "bias_equal_representation"
    "unbias_equal_representation"
)

# Function to extract clean model name for paths
get_model_name() {
    local model_path=$1
    echo "$(basename "$model_path")"
}

# Function to determine if dataset is output variant
is_output_dataset() {
    local dataset=$1
    [[ "$dataset" == *"equal_representation"* ]] && return 0 || return 1
}

# Function to construct gradient path
get_gradient_path() {
    local model_name=$1
    local dataset_type=$2  # "political" or "review"
    local dataset_name=$3
    echo "$BASE_GRADIENT_DIR/$model_name/$dataset_type/gradients_aggregate_norm_l2_model_128_42_512_${dataset_name}.pth"
}

# Function to construct input data path
get_input_data_path() {
    local model_name=$1
    local dataset_type=$2  # "political" or "review"  
    local dataset_name=$3
    local is_output=$4
    
    local base_path="$BASE_DATA_DIR/${dataset_type}_input_data"
    
    if [ "$is_output" = true ]; then
        echo "$base_path/${model_name}_output_calibration_data/${dataset_name}.pkl"
    else
        echo "$base_path/${dataset_name}.pkl"
    fi
}

# Function to construct output paths
get_output_paths() {
    local model_name=$1
    local dataset_type=$2
    local dataset_name=$3
    local method=$4  # "hgla" or "gradient"
    
    local perf_path="$BASE_OUTPUT_DIR/$model_name/$method/$dataset_type/$dataset_name/performance/"
    local fair_path="$BASE_OUTPUT_DIR/$model_name/$method/$dataset_type/$dataset_name/fairness/"
    
    echo "$perf_path" "$fair_path"
}

# Function to run pruning experiment
run_pruning_experiment() {
    local sparsity_ratio=$1
    local model_path=$2
    local dataset_type=$3  # "political" or "review"
    local dataset_name=$4
    local method=$5  # "hgla" or "gradient"
    
    local model_name=$(get_model_name "$model_path")
    local is_output=false
    
    # Check if this is an output dataset variant
    if is_output_dataset "$dataset_name"; then
        is_output=true
    fi
    
    # Construct paths
    local gradient_path=$(get_gradient_path "$model_name" "$dataset_type" "$dataset_name")
    local input_path=$(get_input_data_path "$model_name" "$dataset_type" "$dataset_name" "$is_output")
    local output_paths=($(get_output_paths "$model_name" "$dataset_type" "$dataset_name" "$method"))
    local perf_path=${output_paths[0]}
    local fair_path=${output_paths[1]}
    
    # Set political flag
    local political_flag=""
    if [ "$dataset_type" = "political" ]; then
        political_flag="--political"
    fi
    
    # Check if required files exist
    if [ ! -f "$gradient_path" ]; then
        echo "Warning: Gradient file not found - $gradient_path"
        return 1
    fi
    
    if [ ! -f "$input_path" ]; then
        echo "Warning: Input data file not found - $input_path"
        return 1
    fi
    
    echo "Running experiment: $model_name, $dataset_type/$dataset_name, sparsity=$sparsity_ratio, method=$method"
    
    # Run the pruning experiment
    python "$SCRIPT_NAME" \
        --sparsity_ratio "$sparsity_ratio" \
        --gradient_path "$gradient_path" \
        --model_path "$model_path" \
        --input_data_path "$input_path" \
        --cache_dir "$MODEL_DIR/cache" \
        --performance_output_path "$perf_path" \
        --fairness_output_path "$fair_path" \
        --prune_method "$method" \
        $political_flag
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: $model_name, $dataset_type/$dataset_name, sparsity=$sparsity_ratio"
        return 0
    else
        echo "✗ Failed: $model_name, $dataset_type/$dataset_name, sparsity=$sparsity_ratio"
        return 1
    fi
}

# Main execution function
main() {
    echo "Starting pruning experiments..."
    echo "Sparsity ratios: ${SPARSITY_RATIOS[*]}"
    echo "Models: ${MODELS[*]}"
    echo "Base output directory: $BASE_OUTPUT_DIR"
    echo "================================"
    
    # Create necessary directories
    mkdir -p "$BASE_OUTPUT_DIR"
    mkdir -p "$MODEL_DIR/cache"
    
    local total_experiments=0
    local successful_experiments=0
    local failed_experiments=0
    
    # Loop through all combinations
    for sparsity_ratio in "${SPARSITY_RATIOS[@]}"; do
        echo "Processing sparsity ratio: $sparsity_ratio"
        echo "----------------------------------------"
        
        for model in "${MODELS[@]}"; do
            model_name=$(get_model_name "$model")
            echo "  Model: $model_name"
            
            # Process political datasets
            echo "    Political datasets:"
            for dataset in "${POLITICAL_DATASETS[@]}"; do
                total_experiments=$((total_experiments + 1))
                if run_pruning_experiment "$sparsity_ratio" "$model" "political" "$dataset" "hgla"; then
                    successful_experiments=$((successful_experiments + 1))
                else
                    failed_experiments=$((failed_experiments + 1))
                fi
            done
            
            # Process review datasets  
            echo "    Review datasets:"
            for dataset in "${REVIEW_DATASETS[@]}"; do
                total_experiments=$((total_experiments + 1))
                if run_pruning_experiment "$sparsity_ratio" "$model" "review" "$dataset" "hgla"; then
                    successful_experiments=$((successful_experiments + 1))
                else
                    failed_experiments=$((failed_experiments + 1))
                fi
            done
            
            echo "  Completed $model_name"
        done
        
        echo "Completed sparsity ratio: $sparsity_ratio"
        echo ""
    done
    
    # Print final summary
    echo "================================"
    echo "Experiment Summary:"
    echo "Total experiments: $total_experiments"
    echo "Successful: $successful_experiments"
    echo "Failed: $failed_experiments"
    echo "Success rate: $(( successful_experiments * 100 / total_experiments ))%"
    
    if [ $failed_experiments -gt 0 ]; then
        echo ""
        echo "Some experiments failed. Check the logs for details."
        exit 1
    else
        echo ""
        echo "All experiments completed successfully!"
        exit 0
    fi
}

# Run main function
main "$@"