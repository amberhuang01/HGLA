# HGLA Pruning

This repository contains the implementation for HGLA pruning of large language models, focusing on analysing the impact of different calibration data on model performance and fairness.

## Overview

The codebase implements two main pruning methods:
- **Gradient-based pruning**: Uses gradient information to identify important weights
- **HGLA (High Gradient Low Activation)**: Combines gradient and activation statistics for more sophisticated pruning decisions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Accelerate 0.20+
- CUDA-capable GPU(s)
- Sufficient disk space (50GB+ recommended)

Install dependencies:
```bash
pip install torch transformers accelerate datasets pandas numpy tqdm
```

## Quick Start

### 1. Download Models
```bash
# Set your Hugging Face token
export HF_TOKEN="your_token_here"

# Download models
sbatch run_model_downloader.sh
```

### 2. Compute Gradients
```bash
# Run gradient computation for all models and datasets
sbatch run_gradient_computation.sh
```

### 3. Perform Pruning
```bash
# Run pruning experiments
sbatch run_prune_hgla.sh
```

## Usage

### Model Download
Download the required models (TinyLlama, Llama-3-8B, Gemma-2B):

```bash
python model_downloader.py --hf_token "your_token" --output_dir "./models"
```

### Gradient Computation
Compute gradients for calibration data:

```bash
python gradient_computation.py \
    --sparsity_ratio 0.5 \
    --model_path "meta-llama/Meta-Llama-3-8B" \
    --input_data_path "./data/political_input_data/democrat_only.pkl" \
    --output_dir "./outputs" \
    --political
```

### Model Pruning
Prune models using computed gradients:

```bash
python prune_hgla.py \
    --sparsity_ratio 0.5 \
    --gradient_path "./outputs/gradients/model/l2_gradients.pth" \
    --model_path "meta-llama/Meta-Llama-3-8B" \
    --input_data_path "./data/calibration_data.pkl" \
    --prune_method hgla
```

## Configuration

### Key Parameters
- `--sparsity_ratio`: Target sparsity level (0.1-0.9)
- `--prune_method`: Pruning method (`gradient` or `hgla`)
- `--political`: Use political dataset formatting
- `--gradient_inv`: Use inverse gradient weighting

### Supported Models
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `meta-llama/Meta-Llama-3-8B` 
- `google/gemma-2b-it`

### Dataset Types
- **Political**: Tweet data about US elections 
- **Review**: Product/business review data 

## Data Structure

### Input Data Format
Data should be provided as pickle files (pandas DataFrames) with the following structure:

**Required columns:**
```python
df = pd.DataFrame({
    'input': List[str],  # Input texts for processing (e.g., tweet collections)
    # Additional columns as needed for analysis
})
```

**Optional columns for analysis:**
```python
df = pd.DataFrame({
    'input': List[str],           # Input texts separated by " || "
    'label': List[List[str]],     # Labels for each text segment (e.g., ['Pro-Republican', 'Pro-Democrat'])
    'proportion': List[Dict],     # Overall proportions (e.g., {'Pro-Republican': 0.53, 'Pro-Democrat': 0.47})
    'input_proportion': List[Dict] # Numeric proportions (e.g., {0: 0.53, 1: 0.47})
})
```

**Example data format:**
```python
# Political tweet data
input_text = "Tweet 1 || Tweet 2 || Tweet 3..."
labels = ['Pro-Republican', 'Pro-Democrat', 'Pro-Republican']
proportions = {'Pro-Republican': 0.667, 'Pro-Democrat': 0.333}

# Review data  
input_text = "Review 1 || Review 2 || Review 3..."
labels = ['Positive', 'Negative', 'Positive']
```


### Expected Directory Structure
```
project/
├── gradient_computation.py
├── pruning_script.py  
├── model_downloader.py
├── run_*.sh
├── data/
│   ├── political_input_data/
│   └── review_input_data/
├── models/
├── outputs/
└── logs/
```

## Output Files

### Gradient Files
- `gradients_aggregate_norm_l1_*.pth`: L1 gradient statistics
- `gradients_aggregate_norm_l2_*.pth`: L2 gradient statistics

### Pruning Results
Results are organized by:
- Model type
- Pruning method
- Dataset type and variant
- Performance vs fairness metrics

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{huang-etal-2025-less,
    title = "Less Is More? Examining Fairness in Pruned Large Language Models for Summarising Opinions",
    author = "Huang, Nannan  and
      Fayek, Haytham M.  and
      Zhang, Xiuzhen",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.909/",
    doi = "10.18653/v1/2025.emnlp-main.909",
    pages = "18005--18029",
    ISBN = "979-8-89176-332-6",
    abstract = "Model compression through post-training pruning offers a way to reduce model size and computational requirements without significantly impacting model performance. However, the effect of pruning on the fairness of LLM-generated summaries remains unexplored, particularly for opinion summarisation where biased outputs could influence public views. In this paper, we present a comprehensive empirical analysis of opinion summarisation, examining three state-of-the-art pruning methods and various calibration sets across three open-source LLMs using four fairness metrics. Our systematic analysis reveals that pruning methods have larger impact on fairness than calibration sets. Building on these insights, we propose High Gradient Low Activation (HGLA) pruning, which identifies and removes parameters that are redundant for input processing but influential in output generation. Our experiments demonstrate that HGLA can better maintain or even improve fairness compared to existing methods, showing promise across models and tasks where traditional methods have limitations. Our human evaluation shows HGLA-generated outputs are fairer than existing state-of-the-art pruning methods."
}
```


