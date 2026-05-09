"""
Utility functions for CV Classification Pipeline
"""

import random
import numpy as np
import torch
import logging
from pathlib import Path
from config.paths import Paths


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_name: str = "experiment", level=logging.INFO):
    """Setup logging configuration to file and console"""
    Paths.ensure_directories()
    log_file = Paths.LOGS_DIR / f"{log_name}.log"
    
    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def get_device(device_name: str = "auto") -> str:
    """Get the appropriate device for training"""
    if device_name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_name


def count_model_params(model) -> int:
    """Count trainable parameters in a PyTorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model, input_size=(1, 28, 28)):
    """Print a summary of the model architecture"""
    total_params = count_model_params(model)
    
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(f"Model Architecture:\n{model}")
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print("="*60)


def save_submission(predictions: np.ndarray, ids: np.ndarray, output_path: Path):
    """Save predictions to submission CSV file"""
    import pandas as pd
    
    submission_df = pd.DataFrame({
        'ImageId': ids,
        'Label': predictions
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission_df


def log_config(config, logger):
    """Log all configuration parameters"""
    logger.info("="*60)
    logger.info("CONFIGURATION")
    logger.info("="*60)
    for key, value in vars(config).items():
        logger.info(f"{key}: {value}")
    logger.info("="*60)
