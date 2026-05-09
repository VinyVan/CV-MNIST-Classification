"""
Training module for CV Classification
Handles training loops, validation, cross-validation, and early stopping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class CVTrainer:
    """Trainer class for Computer Vision classification tasks"""
    
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.device = config.device
        
        # Training tracking
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.oof_predictions = None
        self.fold_models = []
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        
        # Early stopping
        self.early_stop_counter = 0
        self.best_model_state = None
    
    def _create_optimizer(self):
        """Create optimizer based on config"""
        params = self.model.parameters()
        lr = self.config.learning_rate
        weight_decay = self.config.weight_decay
        
        if self.config.optimizer == 'adam':
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == 'adamw':
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif self.config.optimizer == 'sgd':
            return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    
    def _create_scheduler(self, optimizer):
        """Create learning rate scheduler"""
        if self.config.scheduler == 'plateau':
            return ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.config.scheduler_factor,
                patience=self.config.scheduler_patience
            )
        elif self.config.scheduler == 'step':
            return StepLR(optimizer, step_size=5, gamma=0.5)
        elif self.config.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        return None
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader) -> Tuple[float, float, np.ndarray]:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation", leave=False)
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, np.array(all_preds)
    
    def fit(self, train_loader, val_loader, fold: int = 0) -> Dict[str, List[float]]:
        """Train model with early stopping"""
        logger.info(f"\nTraining Fold {fold + 1}/{self.config.n_folds}")
        logger.info(f"Model: {self.config.model_name}, Device: {self.device}")
        logger.info(f"Epochs: {self.config.num_epochs}, LR: {self.config.learning_rate}")
        
        # Create scheduler
        self.scheduler = self._create_scheduler(self.optimizer)
        
        # Reset tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.best_model_state = None
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, _ = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            epoch_time = time.time() - epoch_start
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.early_stop_counter = 0
                logger.info(f"  -> New best model! Val Acc: {val_acc:.2f}%")
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if self.early_stop_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        total_time = time.time() - start_time
        logger.info(f"Fold {fold+1} complete. Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
        logger.info(f"Total training time: {total_time/60:.1f} minutes")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }
    
    def cross_validate(self, data_loader_splits: List[Tuple]) -> Tuple[float, np.ndarray]:
        """Run cross-validation"""
        logger.info(f"\nStarting {self.config.n_folds}-Fold Cross-Validation")
        
        fold_scores = []
        oof_preds = None
        
        for fold, (train_loader, val_loader) in enumerate(data_loader_splits):
            # Reset model for each fold
            from ..modeling.models import get_model
            self.model = get_model(self.config.model_name, self.config).to(self.device)
            self.optimizer = self._create_optimizer()
            
            # Train
            history = self.fit(train_loader, val_loader, fold)
            fold_scores.append(history['best_val_acc'])
            
            # Get OOF predictions
            _, _, val_preds = self.validate(val_loader)
            if oof_preds is None:
                oof_preds = val_preds
            
            # Save fold model
            if self.config.save_best_only:
                self.fold_models.append(self.best_model_state)
        
        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)
        
        logger.info(f"\nCross-Validation Results:")
        logger.info(f"Fold scores: {[f'{s:.2f}%' for s in fold_scores]}")
        logger.info(f"Mean CV Accuracy: {mean_acc:.2f}% (+/- {std_acc:.2f}%)")
        
        self.oof_predictions = oof_preds
        return mean_acc, oof_preds
    
    def predict(self, test_loader) -> np.ndarray:
        """Make predictions on test set"""
        self.model.eval()
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Predicting", leave=False)
            for images, _ in pbar:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def save_model(self, output_dir: Path, fold: Optional[int] = None):
        """Save model state"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if fold is not None:
            model_path = output_dir / f"model_fold_{fold}.pth"
        else:
            model_path = output_dir / "best_model.pth"
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.config,
        }, model_path)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Path):
        """Load model state"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        logger.info(f"Model loaded from {model_path}")
    
    def save_oof(self, output_dir: Path, ids: pd.Series):
        """Save out-of-fold predictions"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        oof_df = pd.DataFrame({
            'ID': ids,
            'oof_prediction': self.oof_predictions
        })
        oof_df.to_csv(output_dir / 'oof_predictions.csv', index=False)
        logger.info(f"OOF predictions saved to {output_dir / 'oof_predictions.csv'}")
