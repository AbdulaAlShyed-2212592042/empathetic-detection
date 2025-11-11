"""
MIMAMO Net Training with Optuna Hyperparameter Optimization
===========================================================

Clean version without Unicode characters for compatibility.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from datetime import datetime
import warnings

# Optuna for hyperparameter optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import MIMAMO components
from video_MIMAMO_Net import (
    MIMAMONet, 
    VideoSequentialDataset, 
    train_epoch, 
    validate_epoch
)

warnings.filterwarnings('ignore')

class OptunaTrainingConfig:
    """Configuration for Optuna optimization"""
    
    def __init__(self):
        # Data paths
        self.train_json = 'json/mapped_train_data_video_aligned.json'
        self.val_json = 'json/mapped_val_data_video_aligned.json'
        self.test_json = 'json/mapped_test_data_video_aligned.json'
        self.video_dir = 'data/train_video/video_v5_0'
        
        # Training parameters
        self.num_classes = 7
        self.max_dialogue_length = 10
        self.num_epochs = 10  # Reduced for faster optimization
        self.patience = 4
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Result directories
        self.results_dir = 'optuna_mimamo_results'
        self.checkpoint_dir = 'optuna_mimamo_checkpoints'
        
        # Create directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_data_loaders(config, batch_size=8):
    """Create data loaders"""
    
    print(f"Creating datasets...")
    
    train_dataset = VideoSequentialDataset(
        config.train_json,
        config.video_dir,
        max_dialogue_length=config.max_dialogue_length
    )
    
    val_dataset = VideoSequentialDataset(
        config.val_json,
        config.video_dir,
        max_dialogue_length=config.max_dialogue_length
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset

def objective(trial, config):
    """Optuna objective function"""
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 6, 8])
    embed_dim = trial.suggest_categorical('embed_dim', [256, 384])
    num_heads = trial.suggest_categorical('num_heads', [4, 6, 8])
    num_blocks = trial.suggest_int('num_mimamo_blocks', 2, 4)
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    
    print(f"\\nTrial {trial.number}:")
    print(f"  lr={learning_rate:.6f}, batch={batch_size}, embed={embed_dim}")
    print(f"  heads={num_heads}, blocks={num_blocks}, dropout={dropout:.3f}")
    
    try:
        # Create data loaders
        train_loader, val_loader, train_dataset = create_data_loaders(config, batch_size)
        
        # Ensure embed_dim divisible by num_heads
        if embed_dim % num_heads != 0:
            embed_dim = (embed_dim // num_heads) * num_heads
        
        # Create model
        model = MIMAMONet(
            dataset=train_dataset,
            num_classes=config.num_classes,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_mimamo_blocks=num_blocks,
            dropout=dropout
        ).to(config.device)
        
        # Training setup
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = GradScaler()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            # Training
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, None, criterion, 
                config.device, scaler, epoch_num=epoch + 1
            )
            
            # Validation
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)
            
            print(f"  Epoch {epoch+1}: Train={train_acc:.3f}, Val={val_acc:.3f}")
            
            # Report and check pruning
            trial.report(val_acc, epoch)
            if trial.should_prune():
                print(f"  Trial {trial.number} pruned at epoch {epoch+1}")
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        print(f"  Trial {trial.number} completed: {best_val_acc:.4f}")
        return best_val_acc
        
    except Exception as e:
        print(f"  Trial {trial.number} failed: {e}")
        return 0.0

def run_optimization(config, n_trials=20):
    """Run Optuna optimization"""
    
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not available")
        return None
    
    print("\\nSTARTING OPTUNA OPTIMIZATION")
    print("=" * 50)
    
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2)
    )
    
    print(f"Running {n_trials} trials...")
    
    try:
        study.optimize(lambda trial: objective(trial, config), n_trials=n_trials)
    except KeyboardInterrupt:
        print("\\nOptimization interrupted")
    
    # Results
    print("\\nOPTIMIZATION RESULTS")
    print("=" * 30)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{config.results_dir}/optuna_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    return study

def train_final_model(config, best_params):
    """Train final model with best parameters"""
    
    print("\\nTRAINING FINAL MODEL")
    print("=" * 30)
    
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Create data loaders
    train_loader, val_loader, train_dataset = create_data_loaders(
        config, best_params['batch_size']
    )
    
    # Create model
    embed_dim = best_params['embed_dim']
    num_heads = best_params['num_heads']
    if embed_dim % num_heads != 0:
        embed_dim = (embed_dim // num_heads) * num_heads
    
    model = MIMAMONet(
        dataset=train_dataset,
        num_classes=config.num_classes,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_mimamo_blocks=best_params['num_mimamo_blocks'],
        dropout=best_params['dropout']
    ).to(config.device)
    
    # Training setup
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=best_params['learning_rate'], 
        weight_decay=1e-4
    )
    scaler = GradScaler()
    
    # Extended training
    best_val_acc = 0.0
    extended_epochs = 20
    
    print(f"Training for {extended_epochs} epochs...")
    
    for epoch in range(extended_epochs):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, None, criterion, 
            config.device, scaler, epoch_num=epoch + 1
        )
        
        # Validation
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.device)
        
        print(f"Epoch {epoch+1:2d}: Train={train_acc:.4f}, Val={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{config.checkpoint_dir}/best_model_{timestamp}_acc{val_acc:.4f}.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc,
                'best_params': best_params
            }, model_path)
            
            print(f"  -> Best model saved: {model_path}")
    
    print(f"\\nFinal training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

def main():
    """Main function"""
    
    print("MIMAMO NET OPTUNA OPTIMIZATION")
    print("=" * 40)
    
    # Check Optuna
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not available. Install with: pip install optuna")
        return
    
    print("SUCCESS: Optuna available")
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize config
    config = OptunaTrainingConfig()
    
    # Get user input
    try:
        n_trials = int(input("\\nNumber of trials (default=15): ") or "15")
    except:
        n_trials = 15
    
    # Run optimization
    study = run_optimization(config, n_trials)
    
    if study is None:
        print("Optimization failed!")
        return
    
    # Ask about final training
    try:
        train_final = input("\\nTrain final model? (y/n): ").strip().lower()
        if train_final == 'y':
            train_final_model(config, study.best_params)
    except:
        print("Skipping final training")
    
    print("\\nOptimization completed!")
    print(f"Results in: {config.results_dir}/")
    print(f"Checkpoints in: {config.checkpoint_dir}/")

if __name__ == "__main__":
    main()