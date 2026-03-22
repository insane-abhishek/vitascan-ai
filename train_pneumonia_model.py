"""
============================================
Antigravity - Pneumonia Model Training Script
============================================
Fine-tunes a ResNet18 (pretrained on ImageNet) for 
binary classification: Normal vs Pneumonia chest X-rays.

Dataset: Chest X-Ray Images (Pneumonia) from Kaggle
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Usage:
  python train_pneumonia_model.py --data_dir ./data/chest_xray --epochs 10

The trained model is saved to: models/pneumonia_resnet18.pth
============================================
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def get_transforms():
    """
    Define training and validation transforms.
    Training includes data augmentation to improve generalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def create_model(num_classes=2):
    """
    Create a ResNet18 model pretrained on ImageNet, with the
    final fully connected layer replaced for binary classification.
    
    Args:
        num_classes: Number of output classes (2: Normal, Pneumonia)
        
    Returns:
        nn.Module: Modified ResNet18 model
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze early layers (optional — train only later layers)
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze layer4 and FC for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Replace the final FC layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_features, num_classes),
    )
    
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch and return average loss + accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model and return loss, accuracy, and predictions."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot and save training curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss")
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, train_accs, "b-o", label="Train Accuracy")
    ax2.plot(epochs, val_accs, "r-o", label="Val Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training plots saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Model")
    parser.add_argument("--data_dir", type=str, default="./data/chest_xray",
                        help="Path to chest X-ray dataset (with train/val/test subdirs)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--output", type=str, default="./models/pneumonia_resnet18.pth",
                        help="Output path for trained model")
    args = parser.parse_args()
    
    # ── Device setup ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ── Load datasets ──
    train_transform, val_transform = get_transforms()
    
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    test_dir = os.path.join(args.data_dir, "test")
    
    print(f"Loading training data from: {train_dir}")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {train_dataset.classes}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)
    
    # ── Create model ──
    model = create_model(num_classes=2)
    model = model.to(device)
    print(f"\nModel created: ResNet18 (pretrained)")
    
    # ── Loss and optimizer ──
    # Use weighted loss to handle class imbalance
    class_counts = np.bincount([label for _, label in train_dataset])
    weights = 1.0 / class_counts.astype(np.float32)
    weights = weights / weights.sum()
    class_weights = torch.FloatTensor(weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # ── Training loop ──
    print(f"\n{'='*50}")
    print(f"Starting training for {args.epochs} epochs...")
    print(f"{'='*50}\n")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}]")
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"  ✓ Best model saved (val_acc = {val_acc:.4f})")
        
        scheduler.step()
        print()
    
    # ── Test evaluation ──
    print(f"\n{'='*50}")
    print("Evaluating on test set...")
    print(f"{'='*50}")
    
    # Load best model
    model.load_state_dict(torch.load(args.output, weights_only=True))
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=train_dataset.classes))
    
    # ── Save training plots ──
    plot_training_history(
        train_losses, val_losses, train_accs, val_accs,
        save_path="./models/training_history.png"
    )
    
    print(f"\n{'='*50}")
    print(f"Training complete! Model saved to: {args.output}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
