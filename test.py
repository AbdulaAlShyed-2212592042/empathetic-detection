#!/usr/bin/env python3
"""
Test script for evaluating the best trained model on the test dataset.
Generates comprehensive evaluation metrics and visualizations.
"""

import torch
import torch.nn as nn
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import BertTokenizer, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import os
from datetime import datetime
import pandas as pd

# Import the same components from train.py
from train import (
    MultimodalSequentialDataset, 
    MultimodalLSTMModel,
    FocalLoss,
    compute_metrics
)

# Emotion labels for 7 basic emotion classes
EMOTION_LABELS = [
    'happy', 'surprised', 'angry', 'fear', 'sad', 'disgusted', 'contempt'
]

def load_best_model(checkpoint_path, device, test_dataset):
    """Load the best trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model with correct parameters using the provided dataset
    model = MultimodalLSTMModel(
        dataset=test_dataset,
        num_classes=7,
        hidden_size=256,
        num_layers=2,
        dropout_rate=0.2,
        use_wav2vec=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully! Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    return model, checkpoint

def create_training_vocab_dataset(tokenizer, wav2vec_extractor):
    """Create dataset with the same vocabulary as used during training."""
    print("Loading training vocabulary...")
    
    # Load training dataset to get the exact vocabulary used during training
    train_dataset = MultimodalSequentialDataset(
        data_path='json/mapped_train_data.json',
        audio_dir='cached_audio_spec',
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_extractor,
        max_length=384,
        max_dialogue_length=10
    )
    
    print(f"Training vocabularies: {len(train_dataset.event_scenario_vocab)} scenarios, "
          f"{len(train_dataset.emotion_cause_vocab)} causes, {len(train_dataset.goal_response_vocab)} goals, "
          f"{len(train_dataset.topic_vocab)} topics")
    
    return train_dataset

def create_test_dataloader(tokenizer, wav2vec_extractor, training_dataset, batch_size=8):
    """Create test data loader using the test data but with training vocabularies."""
    print("Loading test dataset with training vocabularies...")
    
    # Create test dataset
    test_dataset = MultimodalSequentialDataset(
        data_path='json/mapped_test_data.json',
        audio_dir='cached_audio_spec',
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_extractor,
        max_length=384,
        max_dialogue_length=10
    )
    
    # Override the vocabularies with the training ones
    test_dataset.event_scenario_vocab = training_dataset.event_scenario_vocab
    test_dataset.emotion_cause_vocab = training_dataset.emotion_cause_vocab
    test_dataset.goal_response_vocab = training_dataset.goal_response_vocab
    test_dataset.topic_vocab = training_dataset.topic_vocab
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    return test_loader, test_dataset

def evaluate_model(model, test_loader, device):
    """Evaluate model on test set and return predictions and labels."""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    total_loss = 0.0
    criterion = FocalLoss(alpha=1.0, gamma=2.0, num_classes=7)
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Move batch to device - using correct key names
            input_ids = batch['context_input_ids'].to(device)
            attention_mask = batch['context_attention_mask'].to(device)
            dialogue_input_ids = batch['dialogue_input_ids'].to(device)
            dialogue_attention_mask = batch['dialogue_attention_mask'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)  # Note: 'label' not 'labels'
            
            # Handle audio features
            audio_features = batch['dialogue_audio'].to(device)
            dialogue_roles = batch['dialogue_roles'].to(device)
            sequence_length = batch['sequence_length'].to(device)
            
            # Forward pass - using correct parameter names
            outputs = model(
                context_input_ids=input_ids,
                context_attention_mask=attention_mask,
                dialogue_input_ids=dialogue_input_ids,
                dialogue_attention_mask=dialogue_attention_mask,
                dialogue_audio=audio_features,
                dialogue_roles=dialogue_roles,
                metadata=metadata,
                sequence_length=sequence_length
            )
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    return all_predictions, all_labels, all_probabilities, avg_loss

def calculate_metrics(predictions, labels):
    """Calculate comprehensive evaluation metrics."""
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Classification report
    class_report = classification_report(
        labels, predictions, 
        target_names=EMOTION_LABELS, 
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support_per_class': support_per_class.tolist(),
        'classification_report': class_report
    }

def plot_confusion_matrix(labels, predictions, save_path):
    """Create and save confusion matrix visualization."""
    cm = confusion_matrix(labels, predictions)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Test Set Evaluation', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved: {save_path}")
    return cm

def plot_per_class_metrics(metrics, save_path):
    """Create and save per-class metrics visualization."""
    emotions = EMOTION_LABELS
    precision = metrics['precision_per_class']
    recall = metrics['recall_per_class']
    f1 = metrics['f1_per_class']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    x_pos = np.arange(len(emotions))
    
    # Precision
    axes[0].bar(x_pos, precision, color='skyblue', alpha=0.7)
    axes[0].set_title('Precision per Emotion Class', fontweight='bold')
    axes[0].set_ylabel('Precision')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(emotions, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Recall
    axes[1].bar(x_pos, recall, color='lightcoral', alpha=0.7)
    axes[1].set_title('Recall per Emotion Class', fontweight='bold')
    axes[1].set_ylabel('Recall')
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(emotions, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    # F1-Score
    axes[2].bar(x_pos, f1, color='lightgreen', alpha=0.7)
    axes[2].set_title('F1-Score per Emotion Class', fontweight='bold')
    axes[2].set_ylabel('F1-Score')
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(emotions, rotation=45, ha='right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class metrics plot saved: {save_path}")

def save_detailed_results(metrics, cm, probabilities, predictions, labels, save_path):
    """Save comprehensive test results to JSON."""
    
    # Convert confusion matrix to list for JSON serialization
    cm_list = cm.tolist()
    
    # Create detailed results dictionary
    results = {
        'test_evaluation': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(labels),
            'model_architecture': 'BERT-base + Wav2Vec2-base + Metadata',
            'overall_metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            },
            'per_class_metrics': {
                'emotion_labels': EMOTION_LABELS,
                'precision_per_class': metrics['precision_per_class'],
                'recall_per_class': metrics['recall_per_class'],
                'f1_per_class': metrics['f1_per_class'],
                'support_per_class': metrics['support_per_class']
            },
            'confusion_matrix': {
                'matrix': cm_list,
                'labels': EMOTION_LABELS
            },
            'classification_report': metrics['classification_report']
        }
    }
    
    # Save to JSON
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Detailed results saved: {save_path}")
    
    return results

def main():
    """Main testing function."""
    print("🧪 Starting Model Testing...")
    print("=" * 50)
    
    # Create results directory
    os.makedirs('result_1', exist_ok=True)
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer and feature extractor
    print("Initializing tokenizer and feature extractor...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    
    # Find the best model checkpoint
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    # Look for the best model file (try new naming convention first)
    best_model_path = os.path.join(checkpoint_dir, 'best_7class_model.pth')
    if not os.path.exists(best_model_path):
        # Fallback to old naming convention
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if not os.path.exists(best_model_path):
            print(f"❌ Best model not found: {best_model_path}")
            # Try to find any 7-class checkpoint file
            checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                              if f.endswith('.pth') and ('7class' in f or 'best' in f)]
            if not checkpoint_files:
                # Fallback to any checkpoint file
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoint_files:
                # Use the most recent file
                checkpoint_files.sort(reverse=True)
                best_model_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                print(f"📁 Using checkpoint: {checkpoint_files[0]}")
            else:
                print(f"❌ No checkpoint files found in {checkpoint_dir}")
                return
    else:
        print(f"📁 Found 7-class model: best_7class_model.pth")
    
    try:
        # Create training vocabulary dataset for model initialization
        training_dataset = create_training_vocab_dataset(tokenizer, wav2vec_extractor)
        
        # Load model with the training vocabulary dataset
        model, checkpoint_info = load_best_model(best_model_path, device, training_dataset)
        
        # Create test dataloader with the same vocabulary
        test_loader, test_dataset = create_test_dataloader(tokenizer, wav2vec_extractor, training_dataset, batch_size=8)
        
        # Evaluate model
        predictions, labels, probabilities, test_loss = evaluate_model(model, test_loader, device)
        
        # Calculate metrics
        print("\n📊 Calculating metrics...")
        metrics = calculate_metrics(predictions, labels)
        
        # Print summary results
        print("\n🎯 TEST RESULTS SUMMARY:")
        print("=" * 30)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        # Create visualizations
        print("\n📈 Creating visualizations...")
        
        # Confusion matrix
        cm_path = f'result_1/confusion_matrix_{timestamp}.png'
        cm = plot_confusion_matrix(labels, predictions, cm_path)
        
        # Per-class metrics
        metrics_path = f'result_1/per_class_metrics_{timestamp}.png'
        plot_per_class_metrics(metrics, metrics_path)
        
        # Save detailed results
        results_path = f'result_1/test_results_{timestamp}.json'
        detailed_results = save_detailed_results(metrics, cm, probabilities, predictions, labels, results_path)
        
        print(f"\n✅ Testing completed successfully!")
        print(f"📁 Results saved in 'result_1/' directory:")
        print(f"   • Test metrics: {results_path}")
        print(f"   • Confusion matrix: {cm_path}")
        print(f"   • Per-class metrics: {metrics_path}")
        
        # Show top-5 and bottom-5 performing classes
        print(f"\n🏆 TOP-5 PERFORMING EMOTIONS (F1-Score):")
        f1_scores = np.array(metrics['f1_per_class'])
        top_indices = np.argsort(f1_scores)[-5:][::-1]
        for i, idx in enumerate(top_indices, 1):
            print(f"   {i}. {EMOTION_LABELS[idx]}: {f1_scores[idx]:.4f}")
        
        print(f"\n📉 BOTTOM-5 PERFORMING EMOTIONS (F1-Score):")
        bottom_indices = np.argsort(f1_scores)[:5]
        for i, idx in enumerate(bottom_indices, 1):
            print(f"   {i}. {EMOTION_LABELS[idx]}: {f1_scores[idx]:.4f}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
