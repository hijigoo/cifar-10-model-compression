"""
Train pruned models on CIFAR-10
"""

import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import copy

from models import ResNet18
from utils import train_epoch, validate, save_checkpoint, load_checkpoint, get_model_info
from pruning import (
    magnitude_prune_global,
    structured_prune_filters,
    lottery_ticket_prune
)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """Get CIFAR-10 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, test_loader


def get_optimizer_and_scheduler(model, args, finetune=True):
    """Get optimizer and scheduler for training or fine-tuning"""
    lr = args.finetune_lr if finetune else args.lr
    epochs = args.finetune_epochs if finetune else args.epochs
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )
    
    return optimizer, scheduler


def apply_pruning(model, method, sparsity, initial_state_dict=None):
    """
    Apply pruning method to model
    
    Args:
        model: Model to prune
        method: Pruning method ('magnitude', 'structured', 'lottery_ticket')
        sparsity: Target sparsity
        initial_state_dict: Initial state dict (for lottery ticket)
    
    Returns:
        pruned_model: Pruned model
    """
    if method == 'magnitude':
        return magnitude_prune_global(model, sparsity)
    elif method == 'structured':
        return structured_prune_filters(model, sparsity)
    elif method == 'lottery_ticket':
        if initial_state_dict is None:
            raise ValueError("lottery_ticket requires initial_state_dict")
        return lottery_ticket_prune(model, initial_state_dict, sparsity)
    else:
        raise ValueError(f"Unknown pruning method: {method}")


def train_pruned_model(args):
    """Train pruned model"""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Load dense model checkpoint
    dense_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'dense_model_seed{args.seed}_best.pth'
    )
    
    if not os.path.exists(dense_checkpoint_path):
        raise FileNotFoundError(
            f"Dense model checkpoint not found: {dense_checkpoint_path}\n"
            f"Please train dense model first using train_dense.py"
        )
    
    print(f"Loading dense model from: {dense_checkpoint_path}")
    model = ResNet18(num_classes=10).to(device)
    checkpoint = load_checkpoint(dense_checkpoint_path, model)
    
    print(f"Dense model accuracy: {checkpoint['accuracy']:.2f}%")
    
    # Save initial state for lottery ticket
    initial_state_dict = None
    if args.method == 'lottery_ticket':
        # For lottery ticket, we need the initial random initialization
        # In practice, you might want to save this during dense training
        # For now, we'll use a workaround
        init_model = ResNet18(num_classes=10)
        set_seed(args.seed)  # Reset to get same initialization
        initial_state_dict = copy.deepcopy(init_model.state_dict())
    
    # Apply pruning
    print(f"\nApplying {args.method} pruning with sparsity {args.sparsity}...")
    pruned_model = apply_pruning(
        model, args.method, args.sparsity, initial_state_dict
    ).to(device)
    
    print("\nPruned model info:")
    get_model_info(pruned_model, device=device)
    
    # Fine-tune pruned model
    print(f"\nFine-tuning for {args.finetune_epochs} epochs...")
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(pruned_model, args, finetune=True)
    
    best_acc = 0.0
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(1, args.finetune_epochs + 1):
        train_loss, train_acc = train_epoch(
            pruned_model, train_loader, criterion, optimizer, device, epoch
        )
        
        val_loss, val_acc = validate(
            pruned_model, test_loader, criterion, device, epoch
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch}/{args.finetune_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"LR: {current_lr:.6f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'pruned_{args.method}_s{args.sparsity:.2f}_seed{args.seed}_best.pth'
            )
            save_checkpoint(
                pruned_model, optimizer, epoch, val_acc, checkpoint_path,
                seed=args.seed,
                method=args.method,
                sparsity=args.sparsity,
                train_acc=train_acc
            )
            print(f"✓ Best pruned model saved: {best_acc:.2f}%")
        
        print("-" * 80)
    
    print("=" * 80)
    print(f"Fine-tuning completed!")
    print(f"Method: {args.method}, Sparsity: {args.sparsity:.2f}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train Pruned CIFAR-10 Model')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
    
    # Pruning parameters
    parser.add_argument('--method', type=str, required=True,
                       choices=['magnitude', 'structured', 'lottery_ticket'],
                       help='Pruning method')
    parser.add_argument('--sparsity', type=float, required=True,
                       help='Target sparsity (0.0 to 1.0)')
    
    # Fine-tuning parameters
    parser.add_argument('--finetune-epochs', type=int, default=100,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--finetune-lr', type=float, default=0.01,
                       help='Fine-tuning learning rate')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    
    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Validate sparsity
    if args.sparsity < 0.0 or args.sparsity > 1.0:
        raise ValueError("Sparsity must be between 0.0 and 1.0")
    
    train_pruned_model(args)


if __name__ == '__main__':
    main()
