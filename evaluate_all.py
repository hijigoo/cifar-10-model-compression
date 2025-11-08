"""
Evaluate all trained models and collect results
"""

import os
import json
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models import ResNet18
from utils import (
    load_checkpoint,
    evaluate_accuracy,
    count_parameters,
    count_nonzero_parameters,
    calculate_sparsity,
    measure_model_size,
    measure_inference_latency
)


def get_test_loader(batch_size=128):
    """Get CIFAR-10 test loader"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return test_loader


def evaluate_model(checkpoint_path, device):
    """
    Evaluate a single model checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to evaluate on
    
    Returns:
        results: Dictionary of evaluation results
    """
    # Load model
    model = ResNet18(num_classes=10).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model)
    model.eval()
    
    # Get test loader
    test_loader = get_test_loader()
    
    # Evaluate accuracy
    accuracy = evaluate_accuracy(model, test_loader, device)
    
    # Calculate metrics
    total_params = count_parameters(model)
    nonzero_params = count_nonzero_parameters(model)
    sparsity = calculate_sparsity(model)
    model_size = measure_model_size(model)
    
    # Measure latency
    try:
        latency = measure_inference_latency(model, device=device)
    except:
        latency = None
    
    # Extract info from checkpoint
    method = checkpoint.get('method', 'dense')
    target_sparsity = checkpoint.get('sparsity', 0.0)
    seed = checkpoint.get('seed', 0)
    
    results = {
        'checkpoint': os.path.basename(checkpoint_path),
        'method': method,
        'target_sparsity': target_sparsity,
        'actual_sparsity': sparsity,
        'seed': seed,
        'test_accuracy': accuracy,
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'params_millions': total_params / 1e6,
        'nonzero_params_millions': nonzero_params / 1e6,
        'model_size_mb': model_size,
        'inference_latency_ms': latency
    }
    
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 80)
    
    # Find all checkpoints
    checkpoint_dir = 'experiments/checkpoints'
    checkpoint_patterns = [
        os.path.join(checkpoint_dir, 'dense_model_*_best.pth'),
        os.path.join(checkpoint_dir, 'pruned_*_best.pth')
    ]
    
    all_checkpoints = []
    for pattern in checkpoint_patterns:
        all_checkpoints.extend(glob.glob(pattern))
    
    if not all_checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        print("Please train models first.")
        return
    
    print(f"Found {len(all_checkpoints)} checkpoints")
    print("-" * 80)
    
    # Evaluate all models
    all_results = []
    for i, checkpoint_path in enumerate(sorted(all_checkpoints), 1):
        print(f"\n[{i}/{len(all_checkpoints)}] Evaluating: {os.path.basename(checkpoint_path)}")
        
        try:
            results = evaluate_model(checkpoint_path, device)
            all_results.append(results)
            
            # Print results
            print(f"  Method: {results['method']}")
            print(f"  Sparsity: {results['actual_sparsity']:.4f}")
            print(f"  Accuracy: {results['test_accuracy']:.2f}%")
            print(f"  Params: {results['nonzero_params_millions']:.2f}M / {results['params_millions']:.2f}M")
            print(f"  Size: {results['model_size_mb']:.2f} MB")
            if results['inference_latency_ms']:
                print(f"  Latency: {results['inference_latency_ms']:.2f} ms/image")
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    # Save results
    results_dir = 'experiments/results'
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, 'all_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("\n" + "=" * 80)
    print(f"Evaluation completed!")
    print(f"Results saved to: {results_file}")
    print(f"Total models evaluated: {len(all_results)}")
    print("=" * 80)
    
    # Print summary table
    print("\nSummary Table:")
    print("-" * 100)
    print(f"{'Model':<20} {'Sparsity':<10} {'Method':<15} {'Acc %':<10} {'Params(M)':<12} {'Size(MB)':<10} {'Lat(ms)':<10}")
    print("-" * 100)
    
    for result in sorted(all_results, key=lambda x: (x['method'], x['actual_sparsity'])):
        model_name = "ResNet18"
        method = result['method'] if result['method'] != 'dense' else 'N/A'
        
        print(f"{model_name:<20} "
              f"{result['actual_sparsity']:>8.2f}  "
              f"{method:<15} "
              f"{result['test_accuracy']:>7.2f}  "
              f"{result['nonzero_params_millions']:>10.2f}  "
              f"{result['model_size_mb']:>8.2f}  "
              f"{result['inference_latency_ms']:>8.2f}" if result['inference_latency_ms'] else "    N/A")
    
    print("-" * 100)


if __name__ == '__main__':
    main()
