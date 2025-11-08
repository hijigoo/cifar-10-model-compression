"""
Magnitude-based pruning implementation
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import copy


def magnitude_prune_global(model, sparsity):
    """
    Apply global magnitude-based pruning
    
    This method ranks all weights globally by their absolute values
    and prunes the smallest ones until the target sparsity is reached.
    
    Args:
        model: Neural network model
        sparsity: Target sparsity (0.0 to 1.0)
    
    Returns:
        pruned_model: Model with pruning masks applied
    """
    # Make a copy to avoid modifying the original
    pruned_model = copy.deepcopy(model)
    
    # Collect all parameters to prune
    parameters_to_prune = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    return pruned_model


def magnitude_prune_layerwise(model, sparsity):
    """
    Apply layer-wise magnitude-based pruning
    
    Each layer is pruned independently with the same sparsity ratio.
    
    Args:
        model: Neural network model
        sparsity: Target sparsity per layer (0.0 to 1.0)
    
    Returns:
        pruned_model: Model with pruning masks applied
    """
    pruned_model = copy.deepcopy(model)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prune.l1_unstructured(module, name='weight', amount=sparsity)
            prune.remove(module, 'weight')
    
    return pruned_model


def get_pruning_mask(model):
    """
    Get pruning masks from a pruned model
    
    Args:
        model: Pruned model
    
    Returns:
        masks: Dictionary of pruning masks
    """
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if hasattr(module, 'weight_mask'):
                masks[name] = module.weight_mask.clone()
    return masks


def apply_pruning_mask(model, masks):
    """
    Apply pruning masks to a model
    
    Args:
        model: Model to apply masks to
        masks: Dictionary of pruning masks
    """
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data *= masks[name]


if __name__ == '__main__':
    # Test magnitude pruning
    from models import ResNet18
    
    print("Testing Magnitude Pruning...")
    model = ResNet18()
    
    # Count original parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {total:,}")
    
    # Test different sparsity levels
    for sparsity in [0.3, 0.5, 0.9]:
        pruned = magnitude_prune_global(model, sparsity)
        nonzero = sum((p != 0).sum().item() for p in pruned.parameters())
        actual_sparsity = 1.0 - (nonzero / total)
        print(f"Target sparsity: {sparsity:.2f}, "
              f"Actual sparsity: {actual_sparsity:.2f}, "
              f"Nonzero params: {nonzero:,}")
