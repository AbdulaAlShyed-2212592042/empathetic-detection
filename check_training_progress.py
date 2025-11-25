"""
Check OwlViT Training Progress
Monitors epoch number and training status
"""

import os
import glob
import torch
from datetime import datetime

def check_training_progress():
    """Check current training epoch and progress"""
    
    checkpoint_dir = 'owlvit_multimodal_checkpoints'
    
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints found. Training may not have started yet.")
        return
    
    # Get all checkpoint files
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    
    if not checkpoints:
        print("No checkpoint files found.")
        return
    
    # Sort by modification time
    checkpoints.sort(key=os.path.getmtime)
    
    print("="*80)
    print("OWLVIT MULTIMODAL TRANSFORMER - TRAINING PROGRESS")
    print("="*80)
    
    # Get the latest checkpoint
    latest_checkpoint = checkpoints[-1]
    checkpoint_name = os.path.basename(latest_checkpoint)
    
    # Extract accuracy from filename
    acc_str = checkpoint_name.split('acc')[1].replace('.pth', '')
    val_acc = float(acc_str)
    
    # Get modification time
    mod_time = os.path.getmtime(latest_checkpoint)
    mod_datetime = datetime.fromtimestamp(mod_time)
    
    print(f"\n📊 LATEST CHECKPOINT:")
    print(f"  File: {checkpoint_name}")
    print(f"  Validation Accuracy: {val_acc*100:.2f}%")
    print(f"  Last updated: {mod_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load checkpoint to get epoch info
    try:
        checkpoint_data = torch.load(latest_checkpoint, map_location='cpu')
        
        if 'epoch' in checkpoint_data:
            current_epoch = checkpoint_data['epoch'] + 1  # +1 because it's 0-indexed
            total_epochs = 30  # From config
            remaining_epochs = total_epochs - current_epoch
            
            print(f"\n📈 EPOCH PROGRESS:")
            print(f"  Current Epoch: {current_epoch}/{total_epochs}")
            print(f"  Completed: {current_epoch} epochs")
            print(f"  Remaining: {remaining_epochs} epochs")
            print(f"  Progress: {(current_epoch/total_epochs)*100:.1f}%")
            
            # Progress bar
            bar_length = 50
            filled = int((current_epoch / total_epochs) * bar_length)
            bar = "█" * filled + "░" * (bar_length - filled)
            print(f"  [{bar}] {current_epoch}/{total_epochs}")
            
        else:
            print(f"\n⚠️  Epoch information not found in checkpoint")
        
        if 'config' in checkpoint_data:
            config = checkpoint_data['config']
            print(f"\n⚙️  CONFIGURATION:")
            print(f"  Batch size: {config.get('batch_size', 'N/A')}")
            print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
            print(f"  Gradient accumulation: {config.get('gradient_accumulation_steps', 'N/A')}")
            print(f"  Effective batch size: {config.get('batch_size', 1) * config.get('gradient_accumulation_steps', 1)}")
    
    except Exception as e:
        print(f"\n⚠️  Could not load checkpoint details: {e}")
    
    # Show all checkpoints
    print(f"\n📁 ALL CHECKPOINTS ({len(checkpoints)} saved):")
    for i, ckpt in enumerate(checkpoints, 1):
        name = os.path.basename(ckpt)
        acc = float(name.split('acc')[1].replace('.pth', ''))
        mod_time = datetime.fromtimestamp(os.path.getmtime(ckpt))
        
        # Mark the latest one
        marker = "← LATEST" if ckpt == latest_checkpoint else ""
        print(f"  {i:2d}. {name} ({mod_time.strftime('%H:%M:%S')}) {marker}")
    
    # Calculate improvement
    if len(checkpoints) > 1:
        first_acc = float(os.path.basename(checkpoints[0]).split('acc')[1].replace('.pth', ''))
        improvement = (val_acc - first_acc) * 100
        print(f"\n📈 IMPROVEMENT:")
        print(f"  First checkpoint: {first_acc*100:.2f}%")
        print(f"  Best checkpoint: {val_acc*100:.2f}%")
        print(f"  Total improvement: +{improvement:.2f}%")
    
    # Estimate time remaining
    if len(checkpoints) > 1 and 'epoch' in checkpoint_data:
        first_time = os.path.getmtime(checkpoints[0])
        latest_time = os.path.getmtime(checkpoints[-1])
        
        time_elapsed = latest_time - first_time
        epochs_done = current_epoch
        
        if epochs_done > 0:
            time_per_epoch = time_elapsed / epochs_done
            time_remaining = time_per_epoch * remaining_epochs
            
            hours = int(time_remaining // 3600)
            minutes = int((time_remaining % 3600) // 60)
            
            print(f"\n⏱️  TIME ESTIMATE:")
            print(f"  Time elapsed: {time_elapsed/3600:.1f} hours")
            print(f"  Avg per epoch: {time_per_epoch/3600:.2f} hours")
            print(f"  Est. remaining: {hours}h {minutes}m")
    
    print(f"\n" + "="*80)


if __name__ == '__main__':
    check_training_progress()
