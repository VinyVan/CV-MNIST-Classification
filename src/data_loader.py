"""
Data loading and preprocessing for CV Classification
Supports both CSV format (MNIST-style) and image folder format
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Callable
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold, KFold
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MNISTDataset(Dataset):
    """Dataset for MNIST-style CSV data (label + pixel values)"""
    
    def __init__(
        self, 
        data: pd.DataFrame,
        transform: Optional[Callable] = None,
        is_train: bool = True,
        image_size: Tuple[int, int] = (28, 28)
    ):
        self.is_train = is_train
        self.transform = transform
        self.image_size = image_size
        
        if is_train:
            self.labels = data.iloc[:, 0].values
            self.images = data.iloc[:, 1:].values.reshape(-1, 1, 28, 28)
        else:
            self.labels = None
            self.images = data.values.reshape(-1, 1, 28, 28)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].astype(np.float32) / 255.0
        
        # Convert to PIL Image for transforms
        image = Image.fromarray((image[0] * 255).astype(np.uint8), mode='L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_train:
            label = self.labels[idx]
            return image, label
        return image, idx


class ImageFolderDataset(Dataset):
    """Dataset for image folder structure (class folders)"""
    
    def __init__(
        self,
        image_paths: list,
        labels: Optional[list] = None,
        transform: Optional[Callable] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB' if self.transform else 'L')
        
        if self.transform:
            image = self.transform(image)
        
        if self.labels is not None:
            return image, self.labels[idx]
        return image, idx


def get_train_transforms(config, is_train: bool = True):
    """Get image transforms for training/validation"""
    transforms_list = []
    
    # Convert to tensor first
    transforms_list.append(transforms.ToTensor())
    
    if is_train and config.use_augmentation:
        aug_params = config.augmentation_params
        transforms_list.extend([
            transforms.RandomRotation(aug_params.get("rotation", 10)),
            transforms.RandomAffine(
                degrees=0,
                translate=(aug_params.get("translation", 0.1),) * 2,
                scale=aug_params.get("scale", (0.9, 1.1)),
                shear=aug_params.get("shear", 5)
            ),
        ])
    
    # Normalize
    transforms_list.append(
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std
        )
    )
    
    return transforms.Compose(transforms_list)


def create_data_loaders(
    train_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    config = None,
    is_folder_structure: bool = False
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create PyTorch DataLoaders for train/val/test"""
    
    if is_folder_structure:
        raise NotImplementedError("Folder structure not yet implemented. Use CSV format.")
    
    # Get transforms
    train_transform = get_train_transforms(config, is_train=True)
    val_transform = get_train_transforms(config, is_train=False)
    
    # Create datasets
    train_dataset = MNISTDataset(train_df, transform=train_transform, is_train=True)
    
    val_dataset = None
    if val_df is not None:
        val_dataset = MNISTDataset(val_df, transform=val_transform, is_train=True)
    
    test_dataset = None
    if test_df is not None:
        test_dataset = MNISTDataset(test_df, transform=val_transform, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True if config.device == "cuda" else False
        )
    
    return train_loader, val_loader, test_loader


def create_cv_splits(
    df: pd.DataFrame,
    config,
    n_splits: Optional[int] = None
):
    """Create cross-validation splits for image classification"""
    n_splits = n_splits or config.n_folds
    
    labels = df.iloc[:, 0].values if len(df.columns) > 1 else None
    
    if config.cv_strategy == "stratified_kfold" and labels is not None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        splits = list(cv.split(np.zeros(len(df)), labels))
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=config.seed)
        splits = list(cv.split(df))
    
    return splits


def load_mnist_data(data_dir: Path) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load MNIST data from CSV files"""
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    
    train_df = None
    test_df = None
    
    if train_path.exists():
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)
        logger.info(f"Train shape: {train_df.shape}")
    else:
        logger.warning(f"Training file not found: {train_path}")
    
    if test_path.exists():
        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)
        logger.info(f"Test shape: {test_df.shape}")
    else:
        logger.warning(f"Test file not found: {test_path}")
    
    return train_df, test_df


def download_mnist_kaggle(output_dir: Path):
    """Download MNIST dataset from Kaggle"""
    try:
        import kaggle
        logger.info("Downloading MNIST dataset from Kaggle...")
        kaggle.competition_download_files('digit-recognizer', path=output_dir, quiet=False)
        logger.info(f"Downloaded to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to download MNIST: {e}")
        logger.info("Please download manually from: https://www.kaggle.com/c/digit-recognizer/data")
