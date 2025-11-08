"""
Train dense baseline model on CIFAR-10
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

from models import ResNet18
from utils import train_epoch, validate, save_checkpoint, get_model_info


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Get CIFAR-10 data loaders with data augmentation
    
    Args:
        batch_size: Batch size
        num_workers: Number of data loading workers
    
    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # No augmentation for test
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Download and load datasets
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


def get_optimizer_and_scheduler(model, args):
    """
    Get optimizer and learning rate scheduler
    
    Args:
        model: Model to optimize
        args: Arguments containing hyperparameters
    
    Returns:
        optimizer: Optimizer
        scheduler: Learning rate scheduler
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    return optimizer, scheduler


def train_dense_model(args):
    """
    Train dense baseline model
    
    Args:
        args: Training arguments
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"Creating ResNet18 model (seed={args.seed})...")
    model = ResNet18(num_classes=10).to(device)
    
    # Print model info
    get_model_info(model, device=device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_acc = 0.0
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch summary
        print(f"Epoch {epoch}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f'dense_model_seed{args.seed}_best.pth'
            )
            save_checkpoint(
                model, optimizer, epoch, val_acc, checkpoint_path,
                seed=args.seed, train_acc=train_acc, val_loss=val_loss
            )
            print(f"✓ Best model saved with accuracy: {best_acc:.2f}%")
        
        print("-" * 80)
    
    # Save final model
    final_checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'dense_model_seed{args.seed}_final.pth'
    )
    save_checkpoint(
        model, optimizer, args.epochs, val_acc, final_checkpoint_path,
        seed=args.seed, best_acc=best_acc
    )
    
    print("=" * 80)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Final validation accuracy: {val_acc:.2f}%")
    print(f"Best model saved to: {checkpoint_path}")
    print(f"Final model saved to: {final_checkpoint_path}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train Dense CIFAR-10 Model')
    
    # Training parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--num-workers', type=int, default=2, help='Number of data loading workers')
    
    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Train model
    train_dense_model(args)


if __name__ == '__main__':
    main()
