"""
Configuration for CV Classification Pipeline (MNIST Example)
All parameters are centralized here for reproducibility and easy experimentation
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch


@dataclass
class Config:
    # Reproducibility
    seed: int = 42
    
    # Task settings
    task_type: str = "multiclass_classification"  # MNIST: 10 digits (0-9)
    num_classes: int = 10
    target_column: str = "label"
    id_column: str = "image_id"
    
    # Data settings
    image_size: Tuple[int, int] = (28, 28)  # MNIST standard size
    num_channels: int = 1  # Grayscale for MNIST
    
    # Cross-validation
    n_folds: int = 5
    cv_strategy: str = "stratified_kfold"
    
    # Model configuration
    model_name: str = "cnn"  # Options: cnn, resnet18, resnet34, efficientnet_b0
    available_models: List[str] = field(default_factory=lambda: [
        "cnn", "resnet18", "resnet34", "efficientnet_b0"
    ])
    
    # Training parameters
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    
    # Optimizer and scheduler
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "plateau"  # plateau, step, cosine
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Data augmentation
    use_augmentation: bool = True
    augmentation_params: dict = None
    
    # Normalization (ImageNet stats for transfer learning, MNIST stats for from scratch)
    normalize_mean: Tuple[float, ...] = (0.1307,)  # MNIST mean
    normalize_std: Tuple[float, ...] = (0.3081,)   # MNIST std
    
    # Evaluation metric
    eval_metric: str = "accuracy"
    
    # Output paths
    model_output_path: str = "outputs/models"
    oof_output_path: str = "outputs/oof"
    submission_output_path: str = "outputs/submissions"
    log_output_path: str = "outputs/logs"
    
    # Debug mode
    debug_mode: bool = False
    debug_sample_size: int = 1000  # Number of samples to use in debug mode
    
    # Data paths
    train_data_path: str = "data/raw/train.csv"  # Or folder for image files
    test_data_path: str = "data/raw/test.csv"
    sample_submission_path: str = "data/raw/sample_submission.csv"
    
    # Training control
    save_best_only: bool = True
    validate_every: int = 1  # Validate every N epochs
    
    def __post_init__(self):
        # Initialize augmentation parameters
        if self.augmentation_params is None:
            self.augmentation_params = {
                "rotation": 10,      # degrees
                "translation": 0.1,  # fraction of image size
                "scale": (0.9, 1.1),
                "shear": 5,          # degrees
            }
        
        # Update normalization for transfer learning models
        if self.model_name in ["resnet18", "resnet34", "efficientnet_b0"]:
            # ImageNet normalization
            self.normalize_mean = (0.485, 0.456, 0.406)
            self.normalize_std = (0.229, 0.224, 0.225)
            self.num_channels = 3  # RGB
