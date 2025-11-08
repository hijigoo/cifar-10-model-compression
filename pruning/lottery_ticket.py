"""
Lottery Ticket Hypothesis pruning implementation
Based on "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
"""

import torch
import torch.nn as nn
import copy


class LotteryTicketPruner:
    """
    Implements Lottery Ticket Hypothesis pruning
    
    The key idea is to:
    1. Train the network
    2. Prune the smallest magnitude weights
    3. Reset remaining weights to their initial values
    4. Retrain the pruned network
    """
    
    def __init__(self, model, initial_state_dict=None):
        """
        Initialize Lottery Ticket Pruner
        
        Args:
            model: Neural network model
            initial_state_dict: Initial weights (if None, use current weights)
        """
        self.model = model
        if initial_state_dict is None:
            self.initial_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.initial_state_dict = copy.deepcopy(initial_state_dict)
        
        self.masks = {}
        self._initialize_masks()
    
    def _initialize_masks(self):
        """Initialize all masks to 1 (no pruning)"""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() > 1:  # Only conv and linear layers
                self.masks[name] = torch.ones_like(param.data)
    
    def prune_by_magnitude(self, sparsity):
        """
        Prune model by magnitude and update masks
        
        Args:
            sparsity: Target sparsity (0.0 to 1.0)
        """
        # Collect all weights that are currently active (not already pruned)
        weights = []
        for name, param in self.model.named_parameters():
            if name in self.masks:
                active_weights = param.data[self.masks[name] == 1]
                weights.append(active_weights.view(-1))
        
        # Concatenate all weights
        all_weights = torch.cat(weights)
        
        # Calculate threshold for pruning
        num_prune = int(sparsity * all_weights.numel())
        if num_prune > 0:
            threshold = torch.sort(torch.abs(all_weights))[0][num_prune]
            
            # Update masks
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    # Keep weights above threshold and already masked weights
                    self.masks[name] = (torch.abs(param.data) >= threshold).float()
    
    def apply_masks(self):
        """Apply current masks to model weights"""
        for name, param in self.model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
    
    def reset_to_initial(self, keep_masks=True):
        """
        Reset weights to initial values
        
        Args:
            keep_masks: If True, maintain pruning masks
        """
        # Load initial weights
        self.model.load_state_dict(self.initial_state_dict)
        
        # Apply masks if keeping them
        if keep_masks:
            self.apply_masks()
    
    def get_sparsity(self):
        """Calculate current sparsity"""
        total = 0
        zeros = 0
        for name in self.masks:
            total += self.masks[name].numel()
            zeros += (self.masks[name] == 0).sum().item()
        
        return zeros / total if total > 0 else 0.0
    
    def get_pruned_model(self):
        """
        Get a copy of the model with current pruning applied
        
        Returns:
            pruned_model: Model with pruning applied
        """
        pruned_model = copy.deepcopy(self.model)
        
        # Apply masks to the copied model
        for name, param in pruned_model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]
        
        return pruned_model


def lottery_ticket_prune(model, initial_state_dict, sparsity):
    """
    Apply one-shot lottery ticket pruning
    
    Args:
        model: Trained model
        initial_state_dict: Initial weights before training
        sparsity: Target sparsity
    
    Returns:
        pruned_model: Model reset to initial weights with pruning mask
    """
    pruner = LotteryTicketPruner(model, initial_state_dict)
    
    # Prune based on current weights
    pruner.prune_by_magnitude(sparsity)
    
    # Reset to initial weights (with masks applied)
    pruner.reset_to_initial(keep_masks=True)
    
    return pruner.get_pruned_model()


def iterative_magnitude_pruning(model, initial_state_dict, target_sparsity, 
                                num_iterations=5, train_fn=None):
    """
    Iterative Magnitude Pruning (IMP) for Lottery Ticket Hypothesis
    
    Args:
        model: Initial model
        initial_state_dict: Initial weights
        target_sparsity: Final target sparsity
        num_iterations: Number of pruning iterations
        train_fn: Function to train model (must be provided)
    
    Returns:
        pruned_model: Final pruned model
        history: Training history
    """
    if train_fn is None:
        raise ValueError("train_fn must be provided for iterative pruning")
    
    pruner = LotteryTicketPruner(model, initial_state_dict)
    history = []
    
    # Calculate sparsity per iteration
    sparsity_per_iter = target_sparsity ** (1.0 / num_iterations)
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Train model
        trained_model, train_info = train_fn(pruner.model)
        
        # Update pruner's model
        pruner.model = trained_model
        
        # Prune by magnitude
        current_sparsity = 1.0 - (1.0 - sparsity_per_iter) ** (iteration + 1)
        pruner.prune_by_magnitude(current_sparsity)
        
        # Reset to initial weights
        pruner.reset_to_initial(keep_masks=True)
        
        # Record history
        history.append({
            'iteration': iteration + 1,
            'sparsity': pruner.get_sparsity(),
            **train_info
        })
        
        print(f"Current sparsity: {pruner.get_sparsity():.4f}")
    
    return pruner.get_pruned_model(), history


if __name__ == '__main__':
    # Test lottery ticket pruning
    from models import ResNet18
    from utils import calculate_sparsity
    
    print("Testing Lottery Ticket Pruning...")
    model = ResNet18()
    
    # Save initial state
    initial_state = copy.deepcopy(model.state_dict())
    
    # Simulate training by adding some noise
    for param in model.parameters():
        param.data += torch.randn_like(param.data) * 0.01
    
    total = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {total:,}")
    
    # Test different sparsity levels
    for sparsity in [0.3, 0.5, 0.9]:
        pruned = lottery_ticket_prune(model, initial_state, sparsity)
        actual_sparsity = calculate_sparsity(pruned)
        nonzero = sum((p != 0).sum().item() for p in pruned.parameters())
        print(f"Target sparsity: {sparsity:.2f}, "
              f"Actual sparsity: {actual_sparsity:.2f}, "
              f"Nonzero params: {nonzero:,}")
