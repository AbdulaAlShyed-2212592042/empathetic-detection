"""
Generate Training Results JSON from OwlViT Checkpoints

This script extracts training progress information from checkpoint files and
generates a comprehensive JSON report with accuracy progression, training timeline,
and model performance metrics.
"""

import os
import json
import torch
from datetime import datetime
from pathlib import Path
import re


def extract_checkpoint_info(checkpoint_dir: str = "owlvit_multimodal_checkpoints"):
    """
    Extract information from all checkpoint files.
    
    Args:
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Dictionary with training results
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint directory '{checkpoint_dir}' not found")
        return None
    
    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_path.glob("best_owlvit_multimodal_acc*.pth"))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoint_dir}'")
        return None
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    checkpoints_data = []
    
    for checkpoint_file in checkpoint_files:
        # Extract accuracy from filename
        match = re.search(r'acc([\d]+\.[\d]+)', checkpoint_file.name)
        if not match:
            continue
            
        accuracy = float(match.group(1))
        
        # Get file metadata
        stat = checkpoint_file.stat()
        timestamp = datetime.fromtimestamp(stat.st_mtime)
        size_mb = stat.st_size / (1024 * 1024)
        
        # Try to load checkpoint to get epoch info
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            epoch = checkpoint.get('epoch', None)
            train_loss = checkpoint.get('train_loss', None)
            val_loss = checkpoint.get('val_loss', None)
            
            # Get optimizer state info
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            
            checkpoint_info = {
                'filename': checkpoint_file.name,
                'accuracy': accuracy,
                'epoch': epoch,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'train_loss': float(train_loss) if train_loss is not None else None,
                'val_loss': float(val_loss) if val_loss is not None else None,
                'size_mb': round(size_mb, 2),
                'has_optimizer_state': optimizer_state is not None
            }
            
            # Clear checkpoint from memory
            del checkpoint
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Warning: Could not load {checkpoint_file.name}: {e}")
            checkpoint_info = {
                'filename': checkpoint_file.name,
                'accuracy': accuracy,
                'epoch': None,
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'train_loss': None,
                'val_loss': None,
                'size_mb': round(size_mb, 2),
                'has_optimizer_state': None
            }
        
        checkpoints_data.append(checkpoint_info)
        print(f"  Processed: {checkpoint_file.name} - Acc: {accuracy:.4f} - Epoch: {checkpoint_info['epoch']}")
    
    # Sort by accuracy
    checkpoints_data.sort(key=lambda x: x['accuracy'])
    
    # Calculate statistics
    accuracies = [cp['accuracy'] for cp in checkpoints_data]
    epochs = [cp['epoch'] for cp in checkpoints_data if cp['epoch'] is not None]
    
    # Get best checkpoint
    best_checkpoint = checkpoints_data[-1] if checkpoints_data else None
    
    # Calculate training duration
    if len(checkpoints_data) >= 2:
        start_time = datetime.strptime(checkpoints_data[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
        end_time = datetime.strptime(checkpoints_data[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
        duration_hours = (end_time - start_time).total_seconds() / 3600
    else:
        duration_hours = None
    
    # Create comprehensive results
    results = {
        'model_name': 'OwlViT Multimodal Transformer',
        'task': 'Empathetic Emotion Recognition',
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        'training_summary': {
            'total_checkpoints': len(checkpoints_data),
            'training_duration_hours': round(duration_hours, 2) if duration_hours else None,
            'first_checkpoint_date': checkpoints_data[0]['timestamp'] if checkpoints_data else None,
            'last_checkpoint_date': checkpoints_data[-1]['timestamp'] if checkpoints_data else None,
            'epochs_completed': max(epochs) if epochs else None,
        },
        
        'performance_metrics': {
            'best_accuracy': max(accuracies) if accuracies else None,
            'worst_accuracy': min(accuracies) if accuracies else None,
            'average_accuracy': round(sum(accuracies) / len(accuracies), 4) if accuracies else None,
            'accuracy_improvement': round(max(accuracies) - min(accuracies), 4) if accuracies else None,
            'final_accuracy': accuracies[-1] if accuracies else None,
        },
        
        'best_checkpoint': {
            'filename': best_checkpoint['filename'] if best_checkpoint else None,
            'accuracy': best_checkpoint['accuracy'] if best_checkpoint else None,
            'epoch': best_checkpoint['epoch'] if best_checkpoint else None,
            'timestamp': best_checkpoint['timestamp'] if best_checkpoint else None,
            'train_loss': best_checkpoint['train_loss'] if best_checkpoint else None,
            'val_loss': best_checkpoint['val_loss'] if best_checkpoint else None,
        } if best_checkpoint else None,
        
        'accuracy_progression': [
            {
                'checkpoint_number': i + 1,
                'accuracy': cp['accuracy'],
                'epoch': cp['epoch'],
                'timestamp': cp['timestamp']
            }
            for i, cp in enumerate(checkpoints_data)
        ],
        
        'all_checkpoints': checkpoints_data
    }
    
    return results


def main():
    """Main function to generate training results JSON."""
    print("=" * 60)
    print("OwlViT Multimodal Training Results Generator")
    print("=" * 60)
    print()
    
    # Generate results
    results = extract_checkpoint_info()
    
    if results is None:
        print("\nFailed to generate results")
        return
    
    # Save to JSON
    output_file = "owlvit_training_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"Training Results Summary")
    print("=" * 60)
    print(f"Total Checkpoints: {results['training_summary']['total_checkpoints']}")
    print(f"Training Duration: {results['training_summary']['training_duration_hours']:.2f} hours" 
          if results['training_summary']['training_duration_hours'] else "Duration: N/A")
    print(f"Epochs Completed: {results['training_summary']['epochs_completed']}")
    print(f"\nBest Accuracy: {results['performance_metrics']['best_accuracy']:.4f}")
    print(f"Average Accuracy: {results['performance_metrics']['average_accuracy']:.4f}")
    print(f"Accuracy Improvement: {results['performance_metrics']['accuracy_improvement']:.4f}")
    print(f"\nBest Checkpoint: {results['best_checkpoint']['filename']}")
    print(f"Best Checkpoint Epoch: {results['best_checkpoint']['epoch']}")
    print(f"Best Checkpoint Date: {results['best_checkpoint']['timestamp']}")
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
