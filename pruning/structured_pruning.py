"""
Structured pruning implementation
"""

import torch
import torch.nn as nn
import copy


def structured_prune_filters(model, sparsity):
    """
    Apply structured pruning by removing entire filters/channels
    
    Filters are ranked by their L1 norm and the least important ones are removed.
    This actually reduces the model size and can lead to real speedup.
    
    Args:
        model: Neural network model
        sparsity: Target sparsity (0.0 to 1.0)
    
    Returns:
        pruned_model: Model with filters pruned
    """
    pruned_model = copy.deepcopy(model)
    
    # Collect importance scores for all conv layers
    conv_layers = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Calculate L1 norm for each filter (output channel)
            importance = torch.sum(
                torch.abs(module.weight.data), 
                dim=(1, 2, 3)
            )
            conv_layers.append({
                'name': name,
                'module': module,
                'importance': importance,
                'num_filters': module.out_channels
            })
    
    # Prune each layer
    for layer_info in conv_layers:
        module = layer_info['module']
        importance = layer_info['importance']
        num_filters = layer_info['num_filters']
        
        # Calculate number of filters to prune
        num_prune = int(sparsity * num_filters)
        
        if num_prune > 0 and num_prune < num_filters:
            # Get indices of filters to keep
            _, indices = torch.sort(importance, descending=True)
            keep_indices = indices[:num_filters - num_prune]
            
            # Zero out pruned filters
            mask = torch.zeros(num_filters, dtype=torch.bool)
            mask[keep_indices] = True
            
            # Apply mask to weights
            module.weight.data[~mask] = 0
    
    return pruned_model


def structured_prune_channels(model, sparsity):
    """
    Apply structured pruning by removing entire channels
    
    Similar to filter pruning but focuses on input channels.
    
    Args:
        model: Neural network model
        sparsity: Target sparsity (0.0 to 1.0)
    
    Returns:
        pruned_model: Model with channels pruned
    """
    pruned_model = copy.deepcopy(model)
    
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels > 1:
            # Calculate importance for each input channel
            importance = torch.sum(
                torch.abs(module.weight.data),
                dim=(0, 2, 3)  # Sum over output channels and spatial dimensions
            )
            
            num_channels = module.in_channels
            num_prune = int(sparsity * num_channels)
            
            if num_prune > 0 and num_prune < num_channels:
                # Get indices of channels to keep
                _, indices = torch.sort(importance, descending=True)
                keep_indices = indices[:num_channels - num_prune]
                
                # Zero out pruned channels
                mask = torch.zeros(num_channels, dtype=torch.bool)
                mask[keep_indices] = True
                
                # Apply mask to weights
                module.weight.data[:, ~mask, :, :] = 0
    
    return pruned_model


def structured_prune_combined(model, sparsity):
    """
    Apply combined structured pruning (both filters and channels)
    
    Args:
        model: Neural network model
        sparsity: Target sparsity (0.0 to 1.0)
    
    Returns:
        pruned_model: Model with structured pruning applied
    """
    # Apply filter pruning
    pruned_model = structured_prune_filters(model, sparsity * 0.5)
    # Apply channel pruning
    pruned_model = structured_prune_channels(pruned_model, sparsity * 0.5)
    
    return pruned_model


if __name__ == '__main__':
    # Test structured pruning
    from models import ResNet18
    from utils import calculate_sparsity
    
    print("Testing Structured Pruning...")
    model = ResNet18()
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {total:,}")
    
    # Test different sparsity levels
    for sparsity in [0.3, 0.5, 0.9]:
        pruned = structured_prune_filters(model, sparsity)
        actual_sparsity = calculate_sparsity(pruned)
        nonzero = sum((p != 0).sum().item() for p in pruned.parameters())
        print(f"Target sparsity: {sparsity:.2f}, "
              f"Actual sparsity: {actual_sparsity:.2f}, "
              f"Nonzero params: {nonzero:,}")
