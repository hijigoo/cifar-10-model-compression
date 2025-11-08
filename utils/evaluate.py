"""
Evaluation utilities
"""

import torch
from tqdm import tqdm


def evaluate_accuracy(model, loader, device):
    """
    Evaluate model accuracy on given dataset
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Device to evaluate on
    
    Returns:
        accuracy: Top-1 accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_per_class_accuracy(model, loader, device, num_classes=10):
    """
    Evaluate per-class accuracy
    
    Args:
        model: Neural network model
        loader: Data loader
        device: Device to evaluate on
        num_classes: Number of classes
    
    Returns:
        class_correct: Correct predictions per class
        class_total: Total samples per class
        class_accuracy: Accuracy per class
    """
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc='Per-class eval'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            c = predicted.eq(targets)
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    class_accuracy = [
        100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        for i in range(num_classes)
    ]
    
    return class_correct, class_total, class_accuracy
