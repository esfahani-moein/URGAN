"""
This file loads VGG19 model and saves its pretrained weights for training purposes.
We directly use PyTorch 2.6's built-in SafeTensors support.

VGG19 and VGG19_BN both serve as feature extractors in perceptual loss, 
but they have architectural differences that impact performance, training dynamics, and feature quality.

VGG19 Layer Sequence:
Conv2D → ReLU → Conv2D → ReLU → MaxPool → ...

VGG19_BN Layer Sequence:
Conv2D → BatchNorm2D → ReLU → Conv2D → BatchNorm2D → ReLU → MaxPool → ...
"""

import torch
import torchvision.models as models
import os, sys
import json
from typing import Dict, List, Any, Optional
from safetensors.torch import save_file, load_file 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import config_model_gan as config


def validate_model(model: torch.nn.Module) -> bool:
    """Validate model by performing a forward pass with test input."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Move model to evaluation mode
    model.eval()
    
    # Create a dummy input (ImageNet standard size)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Run forward pass
    with torch.no_grad():
        try:
            output = model(dummy_input)
            # Check output dimensions (should be [1, 1000] for ImageNet classification)
            if output.shape == torch.Size([1, 1000]):
                print(f"✓ Model validation successful. Output shape: {output.shape}")
                
                # Additional check: verify output contains reasonable probabilities
                if torch.isnan(output).any():
                    print("✗ Warning: Model output contains NaN values!")
                    return False
                    
                # Check that output sums to approximately 1 (softmax output)
                softmax_output = torch.nn.functional.softmax(output, dim=1)
                if abs(softmax_output.sum().item() - 1.0) > 0.01:
                    print(f"✗ Warning: Softmax output sum ({softmax_output.sum().item():.5f}) doesn't match expected 1.0")
                    return False
                
                return True
            else:
                print(f"✗ Unexpected output shape: {output.shape}, expected [1, 1000]")
                return False
        except Exception as e:
            print(f"✗ Model validation failed with error: {e}")
            return False


def save_model_with_metadata(
    state_dict: Dict[str, torch.Tensor],
    output_path: str,
    metadata: Dict[str, str]
) -> None:
    """Save model with metadata in SafeTensors format.
    
    Args:
        state_dict: Model state dict
        output_path: Path to save the model
        metadata: Metadata to include with the model
    """
    # Method 1: Using torch.save directly with new PyTorch 2.6 SafeTensors support
    # Note: PyTorch's direct safetensors support has limited metadata capabilities
    if output_path.endswith(".safetensors"):
        # Still using safetensors library for full metadata support
        save_file(state_dict, output_path, metadata=metadata)
    else:
        # For .pth files, use standard PyTorch save
        torch.save(state_dict, output_path)


def convert_pytorch_to_safetensors(
    model_name: str = "vgg19", 
    batch_norm: bool = False,
    pretrained_dir: str = None
) -> str:    
    """Convert VGG19/VGG19_BN PyTorch models to SafeTensors format
    
    Args:
        model_name: Base model name ('vgg19')
        batch_norm: Whether to use batch norm version
        pretrained_dir: Directory to save models (defaults to config.current_dir)
        
    Returns:
        Path to saved SafeTensors file
    """
    
    # Determine full model name and output path
    full_model_name = f"{model_name}_bn" if batch_norm else model_name
    output_safetensors_path = os.path.join(pretrained_dir, f"{full_model_name}_imagenet.safetensors")
    output_pth_path = os.path.join(pretrained_dir, f"{full_model_name}_imagenet.pth")
    
    # Check if files already exist
    if os.path.exists(output_safetensors_path) and os.path.exists(output_pth_path):
        print(f"Files already exist at: {output_safetensors_path}")
        print("Skipping download. Delete files if you want to re-download.")
        return output_safetensors_path

    # Track memory usage if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        starting_memory = torch.cuda.memory_allocated()

    # Load model architecture with modern weights API
    if batch_norm:
        model = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
    else:
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    
    print(f"Loaded {full_model_name} model")
    
    # Report memory usage
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        model_memory = current_memory - starting_memory
        print(f"Model memory usage: {model_memory / 1e6:.2f} MB")
        print(f"Peak memory usage: {peak_memory / 1e6:.2f} MB")

    # Validate model by running a forward pass
    if validate_model(model):
        print("Model validated successfully!")
    else:
        print("Model validation failed! The downloaded weights might be corrupted.")
        sys.exit(1)  # Exit if validation fails

    # Extract tensor data from state_dict
    state_dict = model.state_dict()
    
    # Prepare metadata (layer information, for better diagnostics)
    metadata: Dict[str, str] = {}
    layer_info: List[Dict[str, Any]] = []
    
    # Analyze model structure for metadata
    for name, param in state_dict.items():
        shape_str = "×".join([str(dim) for dim in param.shape])
        layer_info.append({
            "name": name,
            "shape": shape_str,
            "dtype": str(param.dtype),
            "size_bytes": param.numel() * param.element_size()
        })
    
    # Add model info metadata
    metadata["model_name"] = full_model_name
    metadata["framework"] = "pytorch"
    metadata["pytorch_version"] = torch.__version__
    metadata["num_params"] = str(sum(p.numel() for p in model.parameters()))
    metadata["has_batch_norm"] = "true" if batch_norm else "false"
    metadata["layer_info"] = json.dumps(layer_info)
    
    # Add feature nodes metadata 
    # The node names differ between VGG19 and VGG19_BN due to BatchNorm layers
    if batch_norm:
        # Each conv+bn+relu counts as 3 layers instead of 2 in non-BN version
        feature_nodes = ["features.2", "features.9", "features.22", "features.35", "features.48"]
    else:
        feature_nodes = ["features.2", "features.7", "features.16", "features.25", "features.34"]
        
    metadata["feature_nodes"] = json.dumps(feature_nodes)
    
    # Save in PyTorch format first
    torch.save(state_dict, output_pth_path)
    print(f"Saved PyTorch weights to {output_pth_path}")
    
    # Save in SafeTensors format with metadata
    save_model_with_metadata(state_dict, output_safetensors_path, metadata)
    
    print(f"Successfully converted {full_model_name} to SafeTensors format")
    print(f"Saved to {output_safetensors_path}")
    
    return output_safetensors_path


def inspect_model(model_path: str) -> None:
    """Inspect a saved model file and print its metadata.
    
    Args:
        model_path: Path to the saved model file
    """
    print(f"\nInspecting model: {model_path}")
    
    if model_path.endswith(".safetensors"):
        # Load metadata from safetensors file
        metadata = load_file(model_path, metadata_only=True)
        print("SafeTensors Metadata:")
        for key, value in metadata.items():
            # For large values like layer_info, just show length
            if len(str(value)) > 100:
                print(f"  {key}: [content length: {len(str(value))} chars]")
            else:
                print(f"  {key}: {value}")
    
    elif model_path.endswith(".pth"):
        # Load PyTorch file
        state_dict = torch.load(model_path, map_location="cpu")
        print("PyTorch State Dict:")
        print(f"  Number of tensors: {len(state_dict)}")
        
        # Sample a few tensors
        sample_keys = list(state_dict.keys())[:3]
        for key in sample_keys:
            tensor = state_dict[key]
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    else:
        print(f"Unsupported file format: {model_path}")


def main() -> Dict[str, str]:
    """Main function to download and save VGG19 models.
    
    Args:
        output_dir: Directory to save models (defaults to config.current_dir)
        
    Returns:
        Dict of model types and their file paths
    """
    
    output_dir = config.current_dir
    pretrained_dir = os.path.join(output_dir, "checkpoints/pretrained_models")
    os.makedirs(pretrained_dir, exist_ok=True)

    print(f"Starting VGG19 model preparation...")
    print(f"Output directory: {pretrained_dir}")
    
    model_paths = {}
    
    # Download standard VGG19
    print("\n=== Processing VGG19 ===")
    vgg19_path = convert_pytorch_to_safetensors("vgg19", batch_norm=False, pretrained_dir=pretrained_dir)
    model_paths["vgg19"] = vgg19_path
    
    # Download VGG19 with BatchNorm
    print("\n=== Processing VGG19_BN ===")
    vgg19bn_path = convert_pytorch_to_safetensors("vgg19", batch_norm=True, pretrained_dir=pretrained_dir)
    model_paths["vgg19_bn"] = vgg19bn_path

    # Optional: inspect the saved models
    inspect_model(vgg19_path)
    inspect_model(vgg19bn_path)

    print(f"\nConversion complete. Files saved at:")
    for model_type, path in model_paths.items():
        print(f"{model_type}: {path}")
        
    return model_paths


if __name__ == "__main__":
    
    model_paths = main()
    
   