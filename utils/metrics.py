"""
Metrics calculation utilities
"""

import os
import time
import torch
import torch.nn as nn


def count_parameters(model):
    """
    Count total number of parameters
    
    Args:
        model: Neural network model
    
    Returns:
        total_params: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_nonzero_parameters(model):
    """
    Count number of non-zero parameters
    
    Args:
        model: Neural network model
    
    Returns:
        nonzero_params: Number of non-zero parameters
    """
    return sum((p != 0).sum().item() for p in model.parameters())


def calculate_sparsity(model):
    """
    Calculate model sparsity
    
    Args:
        model: Neural network model
    
    Returns:
        sparsity: Sparsity ratio (0.0 = dense, 1.0 = fully sparse)
    """
    total = count_parameters(model)
    nonzero = count_nonzero_parameters(model)
    
    if total == 0:
        return 0.0
    
    return 1.0 - (nonzero / total)


def measure_model_size(model, filepath='temp_model.pth'):
    """
    Measure model size in MB
    
    Args:
        model: Neural network model
        filepath: Temporary file path to save model
    
    Returns:
        size_mb: Model size in megabytes
    """
    torch.save(model.state_dict(), filepath)
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    
    if os.path.exists(filepath):
        os.remove(filepath)
    
    return size_mb


def measure_inference_latency(model, input_shape=(1, 3, 32, 32), 
                               num_runs=100, device='cuda', warmup_runs=10):
    """
    Measure inference latency in milliseconds per image
    
    Args:
        model: Neural network model
        input_shape: Shape of input tensor
        num_runs: Number of inference runs for measurement
        device: Device to run inference on
        warmup_runs: Number of warm-up runs
    
    Returns:
        latency_ms: Average latency in milliseconds per image
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warm-up runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Synchronize if using CUDA
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    end_time = time.time()
    
    # Calculate average latency in milliseconds
    latency_ms = (end_time - start_time) / num_runs * 1000
    
    return latency_ms


def get_model_info(model, device='cuda', verbose=True):
    """
    Get comprehensive model information
    
    Args:
        model: Neural network model
        device: Device model is on
        verbose: Whether to print information
    
    Returns:
        info: Dictionary containing model information
    """
    total_params = count_parameters(model)
    nonzero_params = count_nonzero_parameters(model)
    sparsity = calculate_sparsity(model)
    model_size = measure_model_size(model)
    
    # Try to measure latency if device is available
    try:
        latency = measure_inference_latency(model, device=device)
    except:
        latency = None
    
    info = {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity': sparsity,
        'model_size_mb': model_size,
        'inference_latency_ms': latency
    }
    
    if verbose:
        print("=" * 50)
        print("Model Information")
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print(f"Non-zero parameters: {nonzero_params:,}")
        print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}%)")
        print(f"Model size: {model_size:.2f} MB")
        if latency is not None:
            print(f"Inference latency: {latency:.2f} ms/image")
        print("=" * 50)
    
    return info


def print_layer_sparsity(model):
    """
    Print sparsity for each layer
    
    Args:
        model: Neural network model
    """
    print("\nLayer-wise Sparsity:")
    print("-" * 60)
    print(f"{'Layer Name':<40} {'Sparsity':<10} {'Shape'}")
    print("-" * 60)
    
    for name, param in model.named_parameters():
        if param.requires_grad and len(param.shape) > 1:  # Only conv and linear layers
            total = param.numel()
            nonzero = (param != 0).sum().item()
            sparsity = 1.0 - (nonzero / total)
            print(f"{name:<40} {sparsity:>7.4f}    {list(param.shape)}")
    
    print("-" * 60)
