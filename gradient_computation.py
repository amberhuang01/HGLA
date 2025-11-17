import numpy as np
import pandas as pd
import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version
from transformers import AdamW
from datasets import load_dataset, Dataset
import torch.nn as nn
from tqdm import tqdm
import argparse
import os
import time
import torch.nn.functional as F
import gc
import csv
from pathlib import Path

# Configuration
SEQLEN = 512
DEFAULT_NSAMPLES = 128
DEFAULT_SEED = 42
DEFAULT_SCALE = 100  # Following alpha = 100 in the paper
DEFAULT_LR = 0.01
DEFAULT_EPS = 0.01

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class TokenizerWrapper:
    """Wrapper for tokenized input IDs."""
    def __init__(self, input_ids):
        self.input_ids = input_ids


def get_custom_data(traindata, nsamples, seed, seqlen, tokenizer):
    """
    Generate samples from training set for calibration.
    
    Args:
        traindata: Training dataset
        nsamples (int): Number of samples to generate
        seed (int): Random seed
        seqlen (int): Sequence length
        tokenizer: Tokenizer instance
        
    Returns:
        list: List of (input, target) tuples for training
    """
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['input'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_llm(model_path, cache_dir="./model_cache"):
    """
    Load language model from pretrained checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        cache_dir (str): Directory to cache model weights
        
    Returns:
        model: Loaded language model
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    print("GPU allocation for model layers:")
    print(model.hf_device_map)
    model.seqlen = SEQLEN
    return model


class GradientComputation:
    """
    Class for computing and storing gradients during model training.
    """
    def __init__(self, model, scale=DEFAULT_SCALE):
        self.model = model
        self.gradients_l1 = dict()
        self.gradients_l2 = dict()
        self.nsample = 0
        self.scale = scale
        self.device = torch.device("cpu")
        self.gradients_init()

    def gradients_init(self):
        """Initialize gradient storage dictionaries."""
        layers = self.model.model.layers
        for i in tqdm(range(len(layers)), desc="Initializing gradient storage"):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                self.gradients_l1[indexed_name] = torch.zeros_like(
                    subset[name].weight, dtype=torch.float16, device=self.device
                )
                self.gradients_l2[indexed_name] = torch.zeros_like(
                    subset[name].weight, dtype=torch.float32, device=self.device
                )

    def update_gradient(self, model, nsample):
        """
        Update gradient statistics with current sample.
        
        Args:
            model: Model instance
            nsample (int): Current sample number
        """
        assert nsample - self.nsample == 1, "Sample number must be incremented by 1"
        
        layers = model.model.layers
        for i in tqdm(range(len(layers)), desc=f"Updating gradients for sample {nsample}"):
            layer = layers[i]
            subset = find_layers(layer)
            for name in subset:
                indexed_name = f"{name}_layer_{i}"
                if subset[name].weight.grad is None:
                    print(f"Warning: {name} has None gradient")
                    continue
                    
                assert subset[name].weight.requires_grad, f"requires_grad must be True for {name}"
                
                grad = subset[name].weight.grad.detach().clone().to(dtype=torch.float32)
                
                # Check for all-zero gradients
                if torch.all(torch.abs(grad) == 0):
                    print(f"Warning: All gradients are zero for {name}")
                    continue
                
                assert self.gradients_l1[indexed_name].shape == grad.shape, "Shape mismatch"
                
                # Update L1 and L2 gradient norms
                scaled_grad = grad * self.scale
                self.gradients_l1[indexed_name] += torch.abs(scaled_grad).to(
                    device=self.device, dtype=torch.float16
                )
                self.gradients_l2[indexed_name] += torch.abs(scaled_grad**2).to(
                    device=self.device
                )
        
        self.nsample = nsample


def load_calibration_data(input_data_path, political=False, nsamples=DEFAULT_NSAMPLES, seed=DEFAULT_SEED):
    """
    Load and preprocess calibration data.
    
    Args:
        input_data_path (str): Path to input data file
        political (bool): Whether to use political dataset formatting
        nsamples (int): Number of samples to use
        seed (int): Random seed
        
    Returns:
        tuple: (dataset, calibration_type)
    """
    print("Loading calibration data...")
    
    if political:
        df = pd.read_pickle(input_data_path)
        
        if "random" in input_data_path or "output" in input_data_path:
            df = df[['input']]
        elif "democrat" in input_data_path or "republican" in input_data_path:
            df["input"] = df["input"].apply(lambda x: " || ".join(x))
            df['input'] = ("Tweets about US Presidential Election. Each tweet is separated by || : " + 
                          df['input'] + 
                          "\nPlease write a short text containing the salient information, i.e. a summary. The summary of the tweets is:")
        else:
            df['input'] = ("Tweets about US Presidential Election. Each tweet is separated by || : " + 
                          df['input'] + 
                          "\nPlease write a short text containing the salient information, i.e. a summary. The summary of the tweets is:")
        
        dataset = Dataset.from_pandas(df)
        calibration_type = "political"
        
    else:
        df = pd.read_pickle(input_data_path)
        
        if "output" not in input_data_path:
            df['input'] = ("Reviews about a product or business. Each review is separated by || : " + 
                          df['input'] + 
                          "\nPlease write a short text containing the salient information, i.e. a summary. The summary of the reviews is:")
        
        # Randomize and limit dataset
        randomized_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        randomized_df = randomized_df.iloc[:200, :]
        dataset = Dataset.from_pandas(randomized_df)
        calibration_type = "review"
    
    print("Dataset loading complete")
    return dataset, calibration_type


def create_output_filename(base_dir, model_name, calibration_type, gradient_type, 
                          nsamples, seed, seqlen, input_filename, is_output=False):
    """
    Create standardized output filename.
    
    Args:
        base_dir (str): Base directory for outputs
        model_name (str): Name of the model
        calibration_type (str): Type of calibration data
        gradient_type (str): Type of gradient (l1 or l2)
        nsamples (int): Number of samples
        seed (int): Random seed
        seqlen (int): Sequence length
        input_filename (str): Input data filename
        is_output (bool): Whether this is an output variant
        
    Returns:
        str: Complete output filename path
    """
    # Create output directory structure
    output_dir = Path(base_dir) / "gradients" / model_name / calibration_type
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract base name from input filename
    base_name = Path(input_filename).stem
    
    # Create filename
    filename = f"gradients_aggregate_norm_{gradient_type}_model_{nsamples}_{seed}_{seqlen}_{base_name}"
    if is_output:
        filename += "_output"
    filename += ".pth"
    
    return output_dir / filename


def main():
    """Main execution function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Gradient-based Language Model Analysis")
    parser.add_argument("--sparsity_ratio", type=float, required=True,
                       help="Sparsity ratio for analysis")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input_data_path", type=str, required=True,
                       help="Path to input calibration data")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for saving gradients")
    parser.add_argument("--cache_dir", type=str, default="./model_cache",
                       help="Directory to cache model weights")
    parser.add_argument("--political", action="store_true",
                       help="Use political dataset formatting")
    parser.add_argument("--nsamples", type=int, default=DEFAULT_NSAMPLES,
                       help="Number of calibration samples")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="Random seed for reproducibility")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                       help="Gradient scaling factor")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_name = Path(args.model_path).name
    model = get_llm(args.model_path, args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load calibration data
    dataset, calibration_type = load_calibration_data(
        args.input_data_path, args.political, args.nsamples, args.seed
    )
    
    # Create data loader
    dataloader = get_custom_data(dataset, args.nsamples, args.seed, SEQLEN, tokenizer)
    
    # Set up optimizer and gradient computation
    optimizer = AdamW(model.parameters(), lr=DEFAULT_LR, eps=DEFAULT_EPS)
    optimizer.zero_grad()
    
    grad_computer = GradientComputation(model, args.scale)
    
    # Training loop for gradient computation
    model.train()
    nsample = 0
    
    print("Starting gradient computation...")
    for input_ids, labels in tqdm(dataloader, desc="Processing samples"):
        nsample += 1
        print(f"Computing gradients for sample: {nsample}")
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"Loss: {loss.item():.4f}")
        
        loss.backward()
        grad_computer.update_gradient(model, nsample)
        optimizer.zero_grad()
    
    print("Gradient computation complete!")
    
    # Save gradients
    print("Saving gradient files...")
    is_output = "output" in args.input_data_path
    input_filename = Path(args.input_data_path).name
    
    # Save L2 gradients
    l2_filename = create_output_filename(
        args.output_dir, model_name, calibration_type, "l2",
        args.nsamples, args.seed, SEQLEN, input_filename, is_output
    )
    torch.save(grad_computer.gradients_l2, l2_filename)
    print(f"L2 gradients saved to: {l2_filename}")
    
    # Save L1 gradients
    l1_filename = create_output_filename(
        args.output_dir, model_name, calibration_type, "l1",
        args.nsamples, args.seed, SEQLEN, input_filename, is_output
    )
    torch.save(grad_computer.gradients_l1, l1_filename)
    print(f"L1 gradients saved to: {l1_filename}")
    
    print("All operations completed successfully!")


if __name__ == "__main__":
    main()