"""
Main entry point for CV Classification Pipeline (MNIST Example)
Run via: python main.py --running_mode debug --model cnn
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

import torch
import numpy as np
import pandas as pd

from config.config import Config
from config.paths import Paths
from src.data_loader import (
    load_mnist_data, 
    create_data_loaders, 
    create_cv_splits,
    download_mnist_kaggle
)
from src.modeling.models import get_model, print_model_summary, count_parameters
from src.training.trainer import CVTrainer
from src.utils import set_seed, setup_logging, get_device, save_submission


def create_fold_data_loaders(train_df, val_indices, test_df, config):
    """Create data loaders for a specific fold"""
    from src.data_loader import create_data_loaders
    
    # Split train/val
    train_fold = train_df.iloc[val_indices[0]].reset_index(drop=True)
    val_fold = train_df.iloc[val_indices[1]].reset_index(drop=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_fold, val_fold, test_df, config
    )
    
    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser(description="CV Classification Pipeline")
    
    # Running mode
    parser.add_argument("--running_mode", type=str, default="debug",
                       choices=["debug", "train", "predict"],
                       help="Pipeline running mode")
    
    # Model configuration
    parser.add_argument("--model", type=str, default="cnn",
                       choices=["cnn", "cnn_deep", "resnet18", "resnet34", "efficientnet_b0"],
                       help="Model architecture to use")
    
    # Training parameters
    parser.add_argument("--fold", type=int, default=5,
                       help="Number of cross-validation folds")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    # Cross-validation
    parser.add_argument("--cv_strategy", type=str, default="stratified_kfold",
                       choices=["stratified_kfold", "kfold"],
                       help="Cross-validation strategy")
    
    # Test predictions
    parser.add_argument("--use_test", action="store_true", default=False,
                       help="Whether to generate test predictions")
    parser.add_argument("--submit", action="store_true", default=False,
                       help="Generate submission file")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Initialize config
    config = Config()
    config.n_folds = args.fold
    config.cv_strategy = args.cv_strategy
    config.model_name = args.model
    config.device = get_device(args.device)
    config.use_test = args.use_test
    
    # Override config with command line args
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    # Set seed
    set_seed(config.seed)
    
    # Setup logging
    log_name = f"run_{args.running_mode}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = setup_logging(log_name)
    
    # Print header
    print("=" * 60)
    print("CV CLASSIFICATION PIPELINE - MNIST")
    print("=" * 60)
    print(f"Mode: {args.running_mode}")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Folds: {config.n_folds}")
    print(f"CV Strategy: {config.cv_strategy}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Use Test: {config.use_test}")
    print("=" * 60)
    
    logger.info(f"Starting pipeline: mode={args.running_mode}, model={config.model_name}")
    
    # Ensure directories
    Paths.ensure_directories()
    
    # Check for CUDA
    if config.device == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\n1. LOADING DATA")
    print("-" * 40)
    
    # Load data
    train_df, test_df = load_mnist_data(Paths.RAW_DATA_DIR)
    
    # Debug mode: use smaller sample
    if args.running_mode == "debug":
        print(f"[DEBUG MODE] Using {config.debug_sample_size} samples")
        train_df = train_df.head(config.debug_sample_size).reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    if test_df is not None:
        print(f"Test samples: {len(test_df)}")
    else:
        print("No test data found")
    
    # Log label distribution
    labels = train_df.iloc[:, 0].values
    unique, counts = np.unique(labels, return_counts=True)
    print(f"Label distribution: {dict(zip(unique, counts))}")
    
    print("\n2. MODEL CONFIGURATION")
    print("-" * 40)
    
    # Create model
    model = get_model(config.model_name, config)
    print_model_summary(model)
    logger.info(f"Model: {config.model_name}, Parameters: {count_parameters(model):,}")
    
    print("\n3. CROSS-VALIDATION TRAINING")
    print("-" * 40)
    
    # Create CV splits
    cv_splits = create_cv_splits(train_df, config)
    
    # Storage for OOF predictions
    oof_predictions = np.zeros(len(train_df))
    fold_scores = []
    trained_models = []
    
    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\nFold {fold_idx + 1}/{config.n_folds}")
        print("-" * 40)
        
        # Create fold data loaders
        train_loader, val_loader, _ = create_data_loaders(
            train_df.iloc[train_idx].reset_index(drop=True),
            train_df.iloc[val_idx].reset_index(drop=True),
            None,
            config
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Create fresh model for this fold
        fold_model = get_model(config.model_name, config)
        
        # Create trainer
        trainer = CVTrainer(fold_model, config)
        
        # Train
        history = trainer.fit(train_loader, val_loader, fold_idx)
        fold_scores.append(history['best_val_acc'])
        
        # Store OOF predictions
        _, _, fold_preds = trainer.validate(val_loader)
        oof_predictions[val_idx] = fold_preds
        
        # Save fold model
        trainer.save_model(Paths.MODELS_DIR, fold=fold_idx)
        trained_models.append(trainer)
        
        logger.info(f"Fold {fold_idx+1} complete - Best Val Acc: {history['best_val_acc']:.2f}%")
    
    # Calculate CV score
    cv_accuracy = 100. * accuracy_score(labels, oof_predictions)
    print(f"\n{'='*60}")
    print(f"Cross-Validation Results")
    print(f"{'='*60}")
    print(f"Fold scores: {[f'{s:.2f}%' for s in fold_scores]}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.2f}% (+/- {np.std(fold_scores):.2f}%)")
    print(f"OOF Accuracy: {cv_accuracy:.2f}%")
    print(f"{'='*60}")
    
    logger.info(f"CV complete - Mean Acc: {np.mean(fold_scores):.2f}%, OOF Acc: {cv_accuracy:.2f}%")
    
    # Save OOF predictions
    if args.running_mode != "debug":
        oof_df = pd.DataFrame({
            'ID': range(len(train_df)),
            'oof_prediction': oof_predictions.astype(int),
            'label': labels
        })
        oof_df.to_csv(Paths.OOF_DIR / 'oof_predictions.csv', index=False)
        logger.info(f"OOF predictions saved to {Paths.OOF_DIR / 'oof_predictions.csv'}")
    
    # Test predictions
    if config.use_test and test_df is not None:
        print("\n4. GENERATING TEST PREDICTIONS")
        print("-" * 40)
        
        # Create test data loader
        _, _, test_loader = create_data_loaders(
            train_df.head(1), None, test_df, config
        )
        
        # Ensemble predictions from all folds
        all_predictions = []
        all_probabilities = []
        
        for fold_idx, trainer in enumerate(trained_models):
            print(f"Predicting with fold {fold_idx + 1} model...")
            predictions, probabilities = trainer.predict(test_loader)
            all_predictions.append(predictions)
            all_probabilities.append(probabilities)
        
        # Average predictions
        avg_probabilities = np.mean(all_probabilities, axis=0)
        final_predictions = np.argmax(avg_probabilities, axis=1)
        
        print(f"Generated predictions for {len(final_predictions)} test samples")
        
        # Generate submission
        if args.submit:
            print("\n5. GENERATING SUBMISSION")
            print("-" * 40)
            
            submission_path = Paths.SUBMISSIONS_DIR / f"submission_{config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_submission(final_predictions, np.arange(1, len(final_predictions) + 1), submission_path)
            
            # Log prediction distribution
            unique, counts = np.unique(final_predictions, return_counts=True)
            logger.info(f"Submission distribution: {dict(zip(unique, counts))}")
            logger.info(f"Submission saved to {submission_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Mean CV Accuracy: {np.mean(fold_scores):.2f}%")
    print(f"OOF Accuracy: {cv_accuracy:.2f}%")
    if config.use_test:
        print(f"Test predictions generated: {len(final_predictions) if 'final_predictions' in locals() else 0}")
    print("=" * 60)
    
    logger.info("Pipeline complete successfully")


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    main()
