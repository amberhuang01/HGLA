import torch
import os
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default models to download
DEFAULT_MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "meta-llama/Meta-Llama-3-8B", 
    "google/gemma-2b-it"
]

# Model-specific configurations
MODEL_CONFIGS = {
    "meta-llama/Meta-Llama-3-8B": {
        "requires_token": True,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    },
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {
        "requires_token": False,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    },
    "google/gemma-2b-it": {
        "requires_token": True,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    }
}


def get_model_config(model_name: str) -> Dict:
    """Get configuration for a specific model."""
    return MODEL_CONFIGS.get(model_name, {
        "requires_token": False,
        "torch_dtype": torch.float16,
        "device_map": "auto"
    })


def validate_token(hf_token: str, model_name: str) -> bool:
    """Validate HF token for models that require authentication."""
    config = get_model_config(model_name)
    
    if config.get("requires_token", False):
        if not hf_token or hf_token == "Your HF Token here":
            logger.error(f"Model {model_name} requires a valid Hugging Face token")
            return False
    
    return True


def create_model_directory(base_path: str, model_name: str) -> Path:
    """Create directory for model storage."""
    model_dir = Path(base_path) / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def download_tokenizer(model_name: str, save_path: Path, hf_token: Optional[str] = None) -> bool:
    """Download and save tokenizer."""
    try:
        logger.info(f"Downloading tokenizer for {model_name}...")
        
        tokenizer_args = {
            "trust_remote_code": True,
            "cache_dir": str(save_path / "cache")
        }
        
        if hf_token:
            tokenizer_args["token"] = hf_token
            
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token for {model_name}")
        
        tokenizer.save_pretrained(save_path)
        logger.info(f"Tokenizer saved to: {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download tokenizer for {model_name}: {e}")
        return False


def download_model(model_name: str, save_path: Path, hf_token: Optional[str] = None) -> bool:
    """Download and save model."""
    try:
        logger.info(f"Downloading model {model_name}...")
        config = get_model_config(model_name)
        
        model_args = {
            "torch_dtype": config.get("torch_dtype", torch.float16),
            "device_map": config.get("device_map", "auto"),
            "trust_remote_code": True,
            "cache_dir": str(save_path / "cache"),
            "low_cpu_mem_usage": True
        }
        
        if hf_token:
            model_args["token"] = hf_token
            
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
        
        # Save model with safetensors format and appropriate sharding
        model.save_pretrained(
            save_path,
            safe_serialization=True,
            max_shard_size="5GB"
        )
        
        logger.info(f"Model saved to: {save_path}")
        
        # Clean up memory
        del model
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {e}")
        return False


def save_download_info(save_path: Path, model_name: str, success: bool):
    """Save download information for tracking."""
    info_file = save_path / "download_info.json"
    info = {
        "model_name": model_name,
        "download_success": success,
        "torch_version": torch.__version__,
        "saved_path": str(save_path)
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)


def test_model_loading(model_path: Path, model_name: str) -> bool:
    """Test if downloaded model can be loaded successfully."""
    try:
        logger.info(f"Testing model loading for {model_name}...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        
        # Load model (just for testing, use CPU to save GPU memory)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True
        )
        
        logger.info(f"Successfully loaded {model_name} from {model_path}")
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load {model_name} from {model_path}: {e}")
        return False


def download_single_model(model_name: str, base_path: str, hf_token: Optional[str] = None, 
                         test_loading: bool = True) -> bool:
    """Download a single model and tokenizer."""
    logger.info(f"Starting download of {model_name}")
    
    # Validate token if required
    if not validate_token(hf_token, model_name):
        return False
    
    # Create model directory
    model_dir = create_model_directory(base_path, model_name)
    
    # Check if model already exists
    if (model_dir / "config.json").exists():
        logger.info(f"Model {model_name} already exists at {model_dir}")
        if test_loading:
            return test_model_loading(model_dir, model_name)
        return True
    
    # Download tokenizer
    tokenizer_success = download_tokenizer(model_name, model_dir, hf_token)
    if not tokenizer_success:
        return False
    
    # Download model
    model_success = download_model(model_name, model_dir, hf_token)
    
    # Save download info
    save_download_info(model_dir, model_name, model_success)
    
    # Test loading if requested
    if model_success and test_loading:
        model_success = test_model_loading(model_dir, model_name)
    
    if model_success:
        logger.info(f"Successfully completed download of {model_name}")
    else:
        logger.error(f"Failed to download {model_name}")
    
    return model_success


def download_models(models: List[str], base_path: str, hf_token: Optional[str] = None,
                   test_loading: bool = True) -> Dict[str, bool]:
    """Download multiple models."""
    results = {}
    successful = 0
    
    logger.info(f"Starting batch download of {len(models)} models")
    logger.info(f"Download path: {base_path}")
    logger.info(f"Models: {models}")
    
    # Create base directory
    Path(base_path).mkdir(parents=True, exist_ok=True)
    
    for model_name in models:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing model {successful + 1}/{len(models)}: {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            success = download_single_model(model_name, base_path, hf_token, test_loading)
            results[model_name] = success
            
            if success:
                successful += 1
                logger.info(f"✓ Successfully downloaded {model_name}")
            else:
                logger.error(f"✗ Failed to download {model_name}")
                
        except Exception as e:
            logger.error(f"✗ Unexpected error downloading {model_name}: {e}")
            results[model_name] = False
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Total models: {len(models)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {len(models) - successful}")
    logger.info(f"Success rate: {successful/len(models)*100:.1f}%")
    
    for model_name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {model_name}")
    
    return results


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Download Hugging Face models for research")
    
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                       help="List of model names to download")
    parser.add_argument("--output_dir", type=str, default="./models",
                       help="Directory to save downloaded models")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face authentication token")
    parser.add_argument("--skip_test", action="store_true",
                       help="Skip testing model loading after download")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Download single model (overrides --models)")
    
    args = parser.parse_args()
    
    # Handle single model download
    if args.model_name:
        models = [args.model_name]
    else:
        models = args.models
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("CUDA not available - using CPU (will be slower)")
    
    # Validate token for models that require it
    token_required_models = [m for m in models if get_model_config(m).get("requires_token", False)]
    if token_required_models and not args.hf_token:
        logger.error("The following models require a Hugging Face token:")
        for model in token_required_models:
            logger.error(f"  - {model}")
        logger.error("Please provide --hf_token argument or set HF_TOKEN environment variable")
        return
    
    # Use environment token if not provided
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    # Download models
    results = download_models(
        models=models,
        base_path=args.output_dir,
        hf_token=hf_token,
        test_loading=not args.skip_test
    )
    
    # Exit with error code if any downloads failed
    if not all(results.values()):
        exit(1)


if __name__ == "__main__":
    main()