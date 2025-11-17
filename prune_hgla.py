import math
import time
import sys
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import argparse
import gc
from datasets import Dataset
import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

# Configuration
SEQLEN = 512
DEFAULT_NSAMPLES = 128
DEFAULT_SEED = 42
BATCH_SIZE = 1

# Set up device and precision settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize accelerator
accelerator = Accelerator()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_llm(model_path, cache_dir="./model_cache"):
    """
    Load language model from pretrained checkpoint.
    
    Args:
        model_path (str): Path to model checkpoint
        cache_dir (str): Directory to cache model weights
        
    Returns:
        model: Loaded language model
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    model_args = {
        "torch_dtype": torch.float16,
        "cache_dir": cache_dir,
        "low_cpu_mem_usage": True,
        "device_map": "auto"
    }

    # Special configuration for Gemma models
    if "gemma-7b-it" in model_path:
        model_args["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_args)

    print("GPU allocation for model layers:")
    print(model.hf_device_map)

    model.seqlen = SEQLEN
    return model


def prepare_calibration_input(model, nsamples, dataloader, device, seqlen):
    """
    Prepare calibration inputs for pruning by intercepting layer inputs.
    
    Args:
        model: Language model
        nsamples (int): Number of samples
        dataloader: Data loader
        device: Device to use
        seqlen (int): Sequence length
        
    Returns:
        tuple: (inputs, outputs, attention_mask, position_ids)
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
            
    layers[0] = Catcher(layers[0])
    
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
            
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


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


class WrappedGPT:
    """
    Wrapper class for GPT layers to collect activation statistics.
    """
    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]
        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0
        self.layer_id = layer_id
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        """Add batch statistics for activation scaling."""
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2 / self.nsamples


def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    """
    Return pruning mask for given alpha threshold.
    
    Args:
        alpha (float): Threshold parameter
        sort_res: Sorted results
        W_metric: Weight metrics
        tmp_metric: Temporary metrics
        sum_before: Sum before normalization
        
    Returns:
        tuple: (mask, current_sparsity)
    """
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1, 1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True) - 1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask == True).sum() / W_mask.numel()
    return W_mask, cur_sparsity


def prune_gradient(args, model, device=device, prune_n=0, prune_m=0):
    """
    Prune model weights using gradient information.
    
    Args:
        args: Command line arguments
        model: Language model to prune
        device: Device to use
        prune_n (int): N for n:m structured pruning
        prune_m (int): M for n:m structured pruning
    """
    print("Starting gradient-based pruning...")
    
    layers = model.model.layers
    
    # Load gradients
    print(f"Loading gradients from: {args.gradient_path}")
    gradients = torch.load(args.gradient_path, map_location='cpu')

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            indexed_name = f"{name}_layer_{i}"
            print(f"Pruning layer {i}, component {name}")
            
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            
            if indexed_name not in gradients:
                print(f"Warning: Gradient not found for {indexed_name}, skipping...")
                continue
            
            gradient = gradients[indexed_name].to(device=W_metric.device)
            
            if not args.gradient_inv:
                W_metric = W_metric.to(dtype=torch.float32) * torch.abs(gradient).to(dtype=torch.float32)
            else:
                small_value = torch.tensor(1e-8, dtype=gradient.dtype, device=gradient.device)
                gradient_inv = 1 / (torch.abs(gradient) + small_value)
                W_metric = W_metric.to(dtype=torch.float32) * gradient_inv.to(dtype=torch.float32)
            
            W_mask = torch.zeros_like(W, dtype=torch.bool)
            
            if prune_n != 0:
                # Structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                # Unstructured pruning
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                W_mask.scatter_(1, indices, True)

            W[W_mask] = 0
            
    print("Gradient-based pruning completed!")


def prune_hgla(args, model, tokenizer, dataloader, device=device, use_variant=True, 
               seed=DEFAULT_SEED, nsamples=DEFAULT_NSAMPLES, prune_n=0, prune_m=0):
    """
    Prune model using HGLA (Hessian-Gradient Layer-wise Adaptive) method.
    
    Args:
        args: Command line arguments
        model: Language model to prune
        tokenizer: Tokenizer
        dataloader: Data loader for calibration
        device: Device to use
        use_variant (bool): Whether to use WANDA variant
        seed (int): Random seed
        nsamples (int): Number of samples
        prune_n (int): N for n:m structured pruning
        prune_m (int): M for n:m structured pruning
    """
    print("Starting HGLA pruning...")
    
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # Load gradients
    print(f"Loading gradients from: {args.gradient_path}")
    gradients = torch.load(args.gradient_path, map_location='cpu')

    # Prepare calibration inputs
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(
            model, nsamples, dataloader, device, SEQLEN
        )

    layers = model.model.layers
    for i, layer in enumerate(layers):
        print(f"Processing layer {i}/{len(layers)}")
        subset = find_layers(layer)

        # Handle multi-GPU setups
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = map(
                lambda x: x.to(dev), (inps, outs, attention_mask, position_ids)
            )

        wrapped_layers = {
            name: WrappedGPT(subset[name], layer_id=i, layer_name=name) 
            for name in subset
        }

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        # Register forward hooks
        handles = [subset[name].register_forward_hook(add_batch(name)) for name in wrapped_layers]

        # Forward pass to collect statistics
        with torch.no_grad():
            for j in range(nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        # Remove hooks
        for h in handles:
            h.remove()

        # Prune weights
        for name in subset:
            indexed_name = f"{name}_layer_{i}"
            print(f"  Pruning {name}")

            if indexed_name not in gradients:
                print(f"  Warning: Gradient not found for {indexed_name}, skipping...")
                continue

            W = subset[name].weight.data
            W_metric = torch.abs(W) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            # Scale gradients to match activation scale
            gradient = gradients[indexed_name].to(device=W_metric.device)
            gradient_scale = W_metric.abs().mean() / gradient.abs().mean()
            scaled_gradient = gradient * gradient_scale

            if not args.gradient_inv:
                W_metric += torch.abs(W) * torch.abs(scaled_gradient)
            else:
                small_value = torch.tensor(1e-8, dtype=scaled_gradient.dtype, device=scaled_gradient.device)
                gradient_inv = 1 / (torch.abs(scaled_gradient) + small_value)
                W_metric *= gradient_inv

            # Handle inf and NaN values
            if torch.isinf(W_metric).any() or torch.isnan(W_metric).any():
                print(f"  Warning: Inf or NaN detected in W_metric for {indexed_name}")
                max_value = torch.finfo(W_metric.dtype).max
                W_metric = torch.where(
                    torch.isinf(W_metric) | torch.isnan(W_metric),
                    torch.tensor(max_value, dtype=W_metric.dtype, device=W_metric.device),
                    W_metric
                )

            W_mask = torch.zeros_like(W, dtype=torch.bool)

            if prune_n != 0:
                # Structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii:(ii + prune_m)].float()
                        W_mask.scatter_(1, ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if use_variant:
                    # WANDA variant with adaptive threshold
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    
                    while abs(cur_sparsity - args.sparsity_ratio) > 0.001 and (alpha_hist[1] - alpha_hist[0] >= 0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    
                    print(f"    Alpha: {alpha:.4f}, Sparsity: {cur_sparsity:.6f}")
                else:
                    # Standard unstructured pruning
                    indices = sort_res[1][:, :int(W_metric.shape[1] * args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)
                    
            W[W_mask] = 0

        # Update inputs for next layer
        inps = outs

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    print("HGLA pruning completed!")


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


def load_pruning_data(input_data_path, political=False, seed=DEFAULT_SEED):
    """
    Load and preprocess data for pruning calibration.
    
    Args:
        input_data_path (str): Path to input data file
        political (bool): Whether to use political dataset formatting
        seed (int): Random seed
        
    Returns:
        Dataset: Processed dataset
    """
    print("Loading pruning calibration data...")
    
    df = pd.read_pickle(input_data_path)
    
    if political:
        if "random" in input_data_path or "output" in input_data_path:
            df = df[['input']]
        elif "combined" in input_data_path:
            df['input'] = ("Tweets about US Presidential Election. Each tweet is separated by || : " + 
                          df['input'] + 
                          "\nPlease write a short text containing the salient information, i.e. a summary. The summary of the tweets is:")
        else:
            df["input"] = df["input"].apply(lambda x: " || ".join(x))
            df['input'] = ("Tweets about US Presidential Election. Each tweet is separated by || : " + 
                          df['input'] + 
                          "\nPlease write a short text containing the salient information, i.e. a summary. The summary of the tweets is:")
    else:
        if "output" not in input_data_path:
            df['input'] = ("Reviews about a product or business. Each review is separated by || : " + 
                          df['input'] + 
                          "\nPlease write a short text containing the salient information, i.e. a summary. The summary of the reviews is:")
        # Randomize for review data
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    dataset = Dataset.from_pandas(df)
    print("Data loading completed!")
    return dataset


def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Model Pruning and Evaluation")
    
    # Model and data arguments
    parser.add_argument("--sparsity_ratio", type=float, required=True,
                       help="Target sparsity ratio for pruning")
    parser.add_argument("--gradient_path", type=str, required=True,
                       help="Path to gradient file")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--input_data_path", type=str, required=True,
                       help="Path to calibration data")
    parser.add_argument("--cache_dir", type=str, default="./model_cache",
                       help="Directory to cache model weights")
    
    # Pruning method and options
    parser.add_argument("--prune_method", type=str, choices=["gradient", "hgla"], required=True,
                       help="Pruning method to use")
    parser.add_argument("--gradient_inv", action="store_true",
                       help="Use inverse of gradient for pruning")
    parser.add_argument("--political", action="store_true",
                       help="Use political dataset formatting")
    
    # Output paths
    parser.add_argument("--performance_output_path", type=str, 
                       default="./outputs/performance/",
                       help="Output path for performance evaluation results")
    parser.add_argument("--fairness_output_path", type=str,
                       default="./outputs/fairness/",
                       help="Output path for fairness evaluation results")
    
    # Optional parameters
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="Random seed for reproducibility")
    parser.add_argument("--nsamples", type=int, default=DEFAULT_NSAMPLES,
                       help="Number of calibration samples")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    print(f"Starting pruning with method: {args.prune_method}")
    print(f"Target sparsity: {args.sparsity_ratio}")
    print(f"Using gradients from: {args.gradient_path}")
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = get_llm(args.model_path, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, truncation=True)
    
    # Perform pruning
    if args.prune_method == "gradient":
        prune_gradient(args, model=model, device=device)
    elif args.prune_method == "hgla":
        # Load calibration data for HGLA
        dataset = load_pruning_data(args.input_data_path, args.political, args.seed)
        dataloader = get_custom_data(dataset, args.nsamples, args.seed, SEQLEN, tokenizer)
        prune_hgla(args, model, tokenizer, dataloader, device=device, 
                  seed=args.seed, nsamples=args.nsamples)
    
    print("Pruning completed successfully!")
    print("Model is ready for evaluation.")
    

if __name__ == "__main__":
    main()