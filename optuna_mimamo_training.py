"""
MIMAMO Net Training with Optuna Hyperparameter Optimization
===========================================================

This script implements automatic hyperparameter optimization for MIMAMO Net
using Optuna framework for efficient search of optimal training configurations.

Author: Your Name
Date: November 2025
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import argparse

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
    """Configuration class for Optuna optimization"""
    
    def __init__(self, quick_mode=False):
        # Data paths
        self.train_json = 'json/mapped_train_data_video_aligned.json'
        self.val_json = 'json/mapped_val_data_video_aligned.json'
        self.test_json = 'json/mapped_test_data_video_aligned.json'
        self.video_dir = 'data/train_video/video_v5_0'
        
        # Fixed parameters
        self.num_classes = 7
        self.max_dialogue_length = 10
        self.num_epochs = 1 if quick_mode else 15  # Quick mode for testing
        self.patience = 5
        
        # Device settings
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Result directories
        self.optuna_results_dir = 'optuna_mimamo_results'
        self.checkpoint_dir = 'optuna_mimamo_checkpoints'
        self.plots_dir = 'optuna_mimamo_plots'
        
        # Create directories
        os.makedirs(self.optuna_results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_data_loaders(config, batch_size=8):
    """Create train, validation, and test data loaders"""
    
    print(f"📁 Creating datasets...")
    
    # Create datasets
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
    
    test_dataset = VideoSequentialDataset(
        config.test_json,
        config.video_dir,
        max_dialogue_length=config.max_dialogue_length
    )
    
    print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset

def objective(trial, config):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [4, 6, 8, 12]),
        'embed_dim': trial.suggest_categorical('embed_dim', [256, 384, 512]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 6, 8, 12]),
        'num_mimamo_blocks': trial.suggest_int('num_mimamo_blocks', 2, 6),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 512]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'focal_alpha': trial.suggest_float('focal_alpha', 0.5, 2.0),
        'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
        'warmup_ratio': trial.suggest_float('warmup_ratio', 0.1, 0.3)
    }
    
    print(f"\\n🔬 Trial {trial.number}: Testing hyperparameters:")
    for key, value in params.items():
        print(f"   {key}: {value}")
    
    try:
        # Create data loaders with suggested batch size
        train_loader, val_loader, _, train_dataset = create_data_loaders(config, params['batch_size'])
        
        # Ensure embed_dim is divisible by num_heads
        if params['embed_dim'] % params['num_heads'] != 0:
            params['embed_dim'] = (params['embed_dim'] // params['num_heads']) * params['num_heads']
        
        # Create model with suggested parameters
        model = MIMAMONet(
            dataset=train_dataset,
            num_classes=config.num_classes,
            embed_dim=params['embed_dim'],
            num_heads=params['num_heads'],
            num_mimamo_blocks=params['num_mimamo_blocks'],
            dropout=params['dropout'],
            hidden_size=params['hidden_size']
        ).to(config.device)
        
        # Loss function with suggested focal loss parameters
        criterion = FocalLoss(
            alpha=params['focal_alpha'],
            gamma=params['focal_gamma'],
            num_classes=config.num_classes
        )
        
        # Optimizer with suggested parameters
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Learning rate scheduler
        total_steps = len(train_loader) * config.num_epochs
        warmup_steps = int(total_steps * params['warmup_ratio'])
        
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Mixed precision scaler
        scaler = GradScaler()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            print(f"\\n📈 Epoch {epoch + 1}/{config.num_epochs}")
            
            # Training
            train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, 
                config.device, scaler, epoch_num=epoch + 1
            )
            
            # Validation
            val_loss, val_acc, val_precision, val_recall, val_f1, val_predictions, val_labels = validate_epoch(
                model, val_loader, criterion, config.device
            )
            
            print(f"   Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"   Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
            # Report to Optuna
            trial.report(val_acc, epoch)
            
            # Check for pruning
            if trial.should_prune():
                print(f"   ✂️ Trial {trial.number} pruned at epoch {epoch + 1}")
                raise optuna.exceptions.TrialPruned()
            
            # Early stopping and best model tracking
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model for this trial
                trial_checkpoint_path = f"{config.checkpoint_dir}/trial_{trial.number}_best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'hyperparameters': params
                }, trial_checkpoint_path)
                
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print(f"   ⏹️ Early stopping triggered at epoch {epoch + 1}")
                    break
        
        print(f"   🏁 Trial {trial.number} completed - Best Val Acc: {best_val_acc:.4f}")
        return best_val_acc
    
    except Exception as e:
        print(f"   ❌ Trial {trial.number} failed: {e}")
        return 0.0  # Return worst possible score for failed trials

def run_optuna_optimization(config, n_trials=50, timeout=7200):
    """Run Optuna hyperparameter optimization"""
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not available. Please install: pip install optuna")
        return None
    
    print("\\n" + "="*80)
    print("🔍 STARTING MIMAMO NET OPTUNA OPTIMIZATION")
    print("="*80)
    
    # Create study
    study_name = f"mimamo_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage = f"sqlite:///{config.optuna_results_dir}/{study_name}.db"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    print(f"📊 Study created: {study_name}")
    print(f"🎯 Target: Maximize validation accuracy")
    print(f"⏱️ Trials: {n_trials}, Timeout: {timeout}s")
    
    # Optimize
    try:
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        print("\\n⏸️ Optimization interrupted by user")
    
    # Results analysis
    print("\\n" + "="*80)
    print("📈 OPTIMIZATION RESULTS")
    print("="*80)
    
    print(f"🏆 Best trial: {study.best_trial.number}")
    print(f"🎯 Best value: {study.best_value:.4f}")
    print(f"⚙️ Best parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    # Save results
    results_data = {
        'study_name': study_name,
        'best_trial': study.best_trial.number,
        'best_value': study.best_value,
        'best_params': study.best_params,
        'n_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
    }
    
    results_path = f"{config.optuna_results_dir}/optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    
    # Create visualizations
    create_optimization_plots(study, config)
    
    return study

def create_optimization_plots(study, config):
    """Create optimization visualization plots"""
    
    print("\\n📊 Creating optimization plots...")
    
    try:
        # Import optuna plotting functions
        import optuna.visualization.matplotlib as vis
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Optimization history
        fig = vis.plot_optimization_history(study)
        plt.title('MIMAMO Net Hyperparameter Optimization History')
        plt.tight_layout()
        plt.savefig(f"{config.plots_dir}/optimization_history_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Parameter importances
        try:
            fig = vis.plot_param_importances(study)
            plt.title('MIMAMO Net Hyperparameter Importances')
            plt.tight_layout()
            plt.savefig(f"{config.plots_dir}/param_importances_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except:
            print("   ⚠️ Could not create parameter importance plot (need more trials)")
        
        # 3. Parallel coordinate plot
        try:
            fig = vis.plot_parallel_coordinate(study)
            plt.title('MIMAMO Net Hyperparameter Parallel Coordinates')
            plt.tight_layout()
            plt.savefig(f"{config.plots_dir}/parallel_coordinates_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()
        except:
            print("   ⚠️ Could not create parallel coordinates plot")
        
        # 4. Custom accuracy distribution plot
        plt.figure(figsize=(10, 6))
        values = [trial.value for trial in study.trials if trial.value is not None]
        plt.hist(values, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Validation Accuracy')
        plt.ylabel('Number of Trials')
        plt.title('MIMAMO Net Validation Accuracy Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{config.plots_dir}/accuracy_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Plots saved to: {config.plots_dir}/")
        
    except Exception as e:
        print(f"   ⚠️ Could not create all plots: {e}")

def train_best_model(config, best_params, extended_epochs=30):
    """Train the final model with best hyperparameters for extended epochs"""
    
    print("\\n" + "="*80)
    print("🚀 TRAINING FINAL MODEL WITH BEST PARAMETERS")
    print("="*80)
    
    # Print best parameters
    print("🏆 Best hyperparameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Create data loaders
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(config, best_params['batch_size'])
    
    # Ensure embed_dim is divisible by num_heads
    if best_params['embed_dim'] % best_params['num_heads'] != 0:
        best_params['embed_dim'] = (best_params['embed_dim'] // best_params['num_heads']) * best_params['num_heads']
    
    # Create final model
    model = MIMAMONet(
        dataset=train_dataset,
        num_classes=config.num_classes,
        embed_dim=best_params['embed_dim'],
        num_heads=best_params['num_heads'],
        num_mimamo_blocks=best_params['num_mimamo_blocks'],
        dropout=best_params['dropout'],
        hidden_size=best_params['hidden_size']
    ).to(config.device)
    
    # Loss function and optimizer with best parameters
    criterion = FocalLoss(
        alpha=best_params['focal_alpha'],
        gamma=best_params['focal_gamma'],
        num_classes=config.num_classes
    )
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * extended_epochs
    warmup_steps = int(total_steps * best_params['warmup_ratio'])
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = None
    patience_counter = 0
    patience = 8  # Increased for longer training
    
    print(f"🎯 Training for {extended_epochs} epochs with patience {patience}")
    
    for epoch in range(extended_epochs):
        print(f"\\n📈 Epoch {epoch + 1}/{extended_epochs}")
        
        # Training
        train_loss, train_acc, train_precision, train_recall, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, 
            config.device, scaler, epoch_num=epoch + 1
        )
        
        # Validation
        val_loss, val_acc, val_precision, val_recall, val_f1, val_predictions, val_labels = validate_epoch(
            model, val_loader, criterion, config.device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"   📊 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"   📊 Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = f"{config.checkpoint_dir}/best_optuna_mimamo_model_{timestamp}_acc{val_acc:.4f}.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'hyperparameters': best_params,
                'history': history
            }, best_model_path)
            
            print(f"   ✅ New best model saved! Accuracy: {val_acc:.4f}")
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   ⏹️ Early stopping triggered after {patience} epochs without improvement")
                break
    
    print(f"\\n🎉 Final training completed!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
    print(f"💾 Best model saved to: {best_model_path}")
    
    # Test the final model
    if best_model_path and os.path.exists(best_model_path):
        print("\\n🧪 Testing final model...")
        
        # Load best model
        checkpoint = torch.load(best_model_path, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test evaluation
        test_loss, test_acc, test_precision, test_recall, test_f1, test_predictions, test_labels = validate_epoch(model, test_loader, criterion, config.device)
        print(f"📊 Test Results - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Save test results
        test_results = {
            'test_accuracy': test_acc,
            'test_loss': test_loss,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'best_val_accuracy': best_val_acc,
            'best_hyperparameters': best_params,
            'model_path': best_model_path
        }
        
        test_results_path = f"{config.optuna_results_dir}/final_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"📁 Test results saved to: {test_results_path}")
    
    return best_model_path, history

def main():
    """Main function for Optuna MIMAMO training"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MIMAMO Net Optuna Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=30, help='Number of Optuna trials (default: 30)')
    parser.add_argument('--timeout', type=int, default=7200, help='Timeout in seconds (default: 7200)')
    parser.add_argument('--quick', action='store_true', help='Quick mode: 1 trial, 1 epoch, 60s timeout')
    parser.add_argument('--no-final-train', action='store_true', help='Skip final model training')
    args = parser.parse_args()
    
    # Override settings for quick mode
    if args.quick:
        n_trials = 1
        timeout = 60
        quick_mode = True
        print("🚀 QUICK MODE: 1 trial, 1 epoch, 60s timeout")
    else:
        n_trials = args.trials
        timeout = args.timeout
        quick_mode = False
    
    print("🎭 MIMAMO NET OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Check Optuna availability
    if not OPTUNA_AVAILABLE:
        print("ERROR: Optuna not available. Please install: pip install optuna")
        return
    else:
        print("SUCCESS: Optuna available for hyperparameter optimization")
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"GPU Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize configuration
    config = OptunaTrainingConfig(quick_mode=quick_mode)
    
    # Show settings
    print(f"\n[SETTINGS] Configuration:")
    print(f"   Trials: {n_trials}")
    print(f"   Timeout: {timeout}s")
    print(f"   Epochs per trial: {config.num_epochs}")
    print(f"   Quick mode: {quick_mode}")
    
    # Run optimization
    study = run_optuna_optimization(config, n_trials=n_trials, timeout=timeout)
    
    if study is None:
        print("❌ Optimization failed!")
        return
    
    # Ask user if they want to train final model (unless disabled)
    if not args.no_final_train and not quick_mode:
        train_final = input("\n[TRAIN] Train final model with best parameters? (y/n): ").strip().lower()
        
        if train_final == 'y':
            extended_epochs = int(input("Number of epochs for final training (default=30): ") or "30")
            best_model_path, history = train_best_model(config, study.best_params, extended_epochs)
            
            # Create final training plots
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history['train_loss'], label='Train Loss')
            plt.plot(history['val_loss'], label='Val Loss')
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 2)
            plt.plot(history['train_acc'], label='Train Accuracy')
            plt.plot(history['val_acc'], label='Val Accuracy')
            plt.title('Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 3, 3)
            plt.plot([acc - loss for acc, loss in zip(history['val_acc'], history['val_loss'])], 
                    label='Val Acc - Val Loss')
            plt.title('Performance Metric')
            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{config.plots_dir}/final_training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\\n📈 Training history plot saved to: {config.plots_dir}/")
    elif quick_mode:
        print("\\n⚡ Quick mode: Skipping final training")
    
    print("\\n" + "="*60)
    print("🎉 OPTUNA MIMAMO OPTIMIZATION COMPLETED!")
    print("="*60)
    print(f"📊 Results directory: {config.optuna_results_dir}/")
    print(f"💾 Checkpoints directory: {config.checkpoint_dir}/")
    print(f"📈 Plots directory: {config.plots_dir}/")
    
    if quick_mode:
        print("\\n✅ Quick test completed successfully!")
        print("   Script is working properly. You can now run full optimization with:")
        print("   python optuna_mimamo_training.py --trials 30 --timeout 7200")

if __name__ == "__main__":
    main()