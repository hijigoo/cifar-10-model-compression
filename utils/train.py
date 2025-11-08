"""
Training utilities for model training
"""

import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Train model for one epoch
    
    Args:
        model: Neural network model
        loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        avg_loss = running_loss / (batch_idx + 1)
        acc = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{avg_loss:.3f}',
            'acc': f'{acc:.2f}%'
        })
    
    return running_loss / len(loader), 100.0 * correct / total


def validate(model, loader, criterion, device, epoch=None):
    """
    Validate model
    
    Args:
        model: Neural network model
        loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number (optional)
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    desc = f'Epoch {epoch} [Val]' if epoch is not None else 'Validation'
    pbar = tqdm(loader, desc=desc)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{avg_loss:.3f}',
                'acc': f'{acc:.2f}%'
            })
    
    return running_loss / len(loader), 100.0 * correct / total


def save_checkpoint(model, optimizer, epoch, accuracy, filepath, **kwargs):
    """
    Save model checkpoint
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        accuracy: Model accuracy
        filepath: Path to save checkpoint
        **kwargs: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        **kwargs
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer=None):
    """
    Load model checkpoint
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
    
    Returns:
        checkpoint: Checkpoint dictionary
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint
