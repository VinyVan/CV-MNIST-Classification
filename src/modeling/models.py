"""
Model definitions for CV Classification
Supports custom CNN and transfer learning models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Any, Optional


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST (28x28 grayscale images)"""
    
    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        # After two maxpool operations: 28 -> 14 -> 7
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class DeeperCNN(nn.Module):
    """Deeper CNN for more complex tasks"""
    
    def __init__(self, num_classes: int = 10, num_channels: int = 1):
        super(DeeperCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_resnet18(num_classes: int = 10, pretrained: bool = True, num_channels: int = 3):
    """Create ResNet18 model"""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Modify first conv layer for different input channels
    if num_channels != 3:
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify final FC layer for num_classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def create_resnet34(num_classes: int = 10, pretrained: bool = True, num_channels: int = 3):
    """Create ResNet34 model"""
    model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
    
    if num_channels != 3:
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def create_efficientnet_b0(num_classes: int = 10, pretrained: bool = True, num_channels: int = 3):
    """Create EfficientNet-B0 model"""
    try:
        model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        
        if num_channels != 3:
            # Modify first conv layer
            old_conv = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                num_channels, old_conv.out_channels, 
                kernel_size=old_conv.kernel_size, 
                stride=old_conv.stride, 
                padding=old_conv.padding, 
                bias=False
            )
        
        # Modify classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    except Exception as e:
        print(f"EfficientNet not available: {e}")
        return create_resnet18(num_classes, pretrained, num_channels)


def get_model(model_name: str, config, pretrained: bool = True):
    """Factory function to create models"""
    num_classes = config.num_classes
    num_channels = config.num_channels
    
    if model_name == 'cnn':
        return SimpleCNN(num_classes=num_classes, num_channels=num_channels)
    elif model_name == 'cnn_deep':
        return DeeperCNN(num_classes=num_classes, num_channels=num_channels)
    elif model_name == 'resnet18':
        return create_resnet18(num_classes, pretrained, num_channels)
    elif model_name == 'resnet34':
        return create_resnet34(num_classes, pretrained, num_channels)
    elif model_name == 'efficientnet_b0':
        return create_efficientnet_b0(num_classes, pretrained, num_channels)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_size: tuple = (1, 28, 28)):
    """Print a summary of the model architecture"""
    total_params = count_parameters(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model Architecture:\n{model}")
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print("="*60)


def freeze_backbone(model: nn.Module, unfreeze_layers: Optional[int] = None):
    """Freeze backbone layers for transfer learning"""
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze classifier layers
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    elif hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # Optionally unfreeze last N layers
    if unfreeze_layers:
        children = list(model.children())
        for child in children[-unfreeze_layers:]:
            for param in child.parameters():
                param.requires_grad = True
