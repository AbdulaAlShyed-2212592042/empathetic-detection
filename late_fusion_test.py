import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import sys
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import copy
import time
import librosa
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Mixed precision training support
from torch.cuda.amp import autocast, GradScaler

# Import model architectures
from video_training import VideoOnlyModel, VideoTransform, VideoSequentialDataset, EnhancedTimeSformerEncoder

# Add path to audio train and test folder
import importlib.util
import os

# Load train_audio_text_metadata module manually
audio_module_path = os.path.join(os.path.dirname(__file__), 'audio train and test', 'train_audio_text_metadata.py')
spec = importlib.util.spec_from_file_location("train_audio_text_metadata", audio_module_path)
train_audio_text_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_audio_text_metadata)

# Import the classes we need
MultimodalLSTMModel = train_audio_text_metadata.MultimodalLSTMModel
MultimodalSequentialDataset = train_audio_text_metadata.MultimodalSequentialDataset

class LateFusionDataset(Dataset):
    """Combined dataset for late fusion training - optimized for testing"""
    
    def __init__(self, json_file, video_dir, audio_dir, tokenizer, wav2vec_feature_extractor=None, 
                 video_transform=None, max_length=512, max_dialogue_length=10, sample_rate=16000):
        
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.video_dir = video_dir
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.wav2vec_feature_extractor = wav2vec_feature_extractor
        self.video_transform = video_transform or VideoTransform()
        self.max_length = max_length
        self.max_dialogue_length = max_dialogue_length
        self.sample_rate = sample_rate
        
        # Emotion mapping (7 classes)
        self.emotion_mapping = {
            'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        
        # ED emotion projection for audio-text model
        self.ed_emotion_projection = {
            'conflicted': 'anxious', 'annoyed': 'angry', 'devastated': 'sad',
            'excited': 'joyful', 'nostalgic': 'content', 'proud': 'confident',
            'joyful': 'joy', 'prepared': 'confident', 'confident': 'confident',
            'hopeful': 'hopeful', 'content': 'content', 'faithful': 'hopeful',
            'furious': 'angry', 'jealous': 'angry', 'terrified': 'afraid',
            'guilty': 'sad', 'sad': 'sadness', 'lonely': 'sad', 'impressed': 'surprised',
            'grateful': 'joyful', 'angry': 'anger', 'caring': 'caring',
            'trusting': 'trusting', 'disgusted': 'disgust', 'anticipating': 'anticipating',
            'anxious': 'afraid', 'surprised': 'surprise', 'embarrassed': 'embarrassed',
            'apprehensive': 'afraid', 'disappointed': 'sad', 'afraid': 'fear',
            'sentimental': 'content', 'ashamed': 'sad', 'happy': 'joy'
        }
        
        # Create vocabularies (combine from both datasets)
        self._build_vocabularies()
        
        # Role mapping
        self.role_to_id = {'speaker': 0, 'listener': 1, 'padding': 2}
        
    def _build_vocabularies(self):
        """Build vocabularies from data"""
        event_scenarios = set()
        emotion_causes = set()
        goal_responses = set()
        topics = set()
        
        for item in self.data:
            chain = item['turn'].get('chain_of_empathy', {})
            event_scenarios.add(chain.get('event_scenario', 'unknown'))
            emotion_causes.add(chain.get('emotion_cause', 'unknown'))
            goal_responses.add(chain.get('goal_to_response', 'unknown'))
            topics.add(item.get('topic', 'unknown'))
        
        # Create vocab mappings
        self.event_scenario_vocab = {v: i for i, v in enumerate(sorted(event_scenarios))}
        self.emotion_cause_vocab = {v: i for i, v in enumerate(sorted(emotion_causes))}
        self.goal_response_vocab = {v: i for i, v in enumerate(sorted(goal_responses))}
        self.topic_vocab = {v: i for i, v in enumerate(sorted(topics))}
        
    def load_audio(self, audio_path):
        """Load and preprocess audio"""
        try:
            if not os.path.exists(audio_path):
                return np.zeros(16000, dtype=np.float32)  # Silent audio
            
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
            
            # Ensure minimum length
            if len(audio) < self.sample_rate:
                audio = np.pad(audio, (0, self.sample_rate - len(audio)))
            
            return audio.astype(np.float32)
        except:
            return np.zeros(16000, dtype=np.float32)
    
    def _map_to_basic_emotion(self, emotion):
        """Map emotion to 7 basic classes"""
        if emotion in self.emotion_mapping:
            return self.emotion_mapping[emotion]
        
        # Try ED emotion projection for audio model
        mapped = self.ed_emotion_projection.get(emotion, emotion)
        
        # Final mapping to basic emotions
        emotion_map = {
            'joyful': 1, 'joy': 1, 'happy': 1, 'confident': 1, 'content': 1, 'hopeful': 1, 'grateful': 1,
            'sad': 2, 'sadness': 2, 'lonely': 2, 'guilty': 2, 'disappointed': 2, 'ashamed': 2, 'sentimental': 2,
            'angry': 3, 'anger': 3, 'furious': 3, 'jealous': 3, 'annoyed': 3,
            'afraid': 4, 'fear': 4, 'terrified': 4, 'anxious': 4, 'apprehensive': 4,
            'disgusted': 5, 'disgust': 5,
            'surprised': 6, 'surprise': 6, 'impressed': 6,
            'caring': 0, 'trusting': 0, 'anticipating': 0, 'embarrassed': 0, 'prepared': 0
        }
        
        return emotion_map.get(mapped, 0)  # Default to neutral
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        turn = item['turn']
        
        # Get emotion label
        chain_of_empathy = turn.get('chain_of_empathy', {})
        speaker_emotion = chain_of_empathy.get('speaker_emotion', 'neutral')
        label = self._map_to_basic_emotion(speaker_emotion)
        
        # Extract dialogue sequence
        dialogue = turn.get('dialogue', [])
        
        # Process for video model
        dialogue_videos = []
        dialogue_roles_video = []
        
        # Process for audio-text model
        dialogue_texts = []
        dialogue_audio_features = []
        dialogue_roles_audio = []
        dialogue_indices = []
        
        for i, utt in enumerate(dialogue[:self.max_dialogue_length]):
            # Common processing
            role = utt.get('role', 'speaker')
            role_id = self.role_to_id.get(role, 0)
            
            # Video processing with error handling
            video_name = utt.get('video_name', None)
            if video_name:
                try:
                    video_path = os.path.join(self.video_dir, video_name)
                    if os.path.exists(video_path):
                        video_tensor = self.video_transform(video_path)
                    else:
                        video_tensor = torch.zeros(3, 8, 224, 224)
                except Exception as e:
                    video_tensor = torch.zeros(3, 8, 224, 224)
            else:
                video_tensor = torch.zeros(3, 8, 224, 224)
            
            dialogue_videos.append(video_tensor)
            dialogue_roles_video.append(role_id)
            
            # Audio-text processing
            text = utt.get('text', '')
            dialogue_texts.append(text)
            dialogue_roles_audio.append(role_id)
            dialogue_indices.append(utt.get('index', 0))
            
            # Audio features - use full audio length for testing
            audio_name = utt.get('audio_name', None)
            if audio_name:
                audio_path = os.path.join(self.audio_dir, audio_name)
                audio = self.load_audio(audio_path)
                # Use full raw audio for Wav2Vec2 (10 seconds)
                audio_features = audio  # Full audio length
            else:
                # Use silent audio segment for missing audio
                audio_features = np.zeros(160000, dtype=np.float32)  # 10 seconds of silence
            
            dialogue_audio_features.append(audio_features)
        
        # Pad sequences for video
        while len(dialogue_videos) < self.max_dialogue_length:
            dialogue_videos.append(torch.zeros(3, 8, 224, 224))
            dialogue_roles_video.append(2)  # padding role
        
        # Pad sequences for audio-text
        while len(dialogue_texts) < self.max_dialogue_length:
            dialogue_texts.append('[EMPTY]')
            dialogue_audio_features.append(np.zeros(160000, dtype=np.float32))  # Full audio size
            dialogue_roles_audio.append(2)  # padding role
            dialogue_indices.append(0)
        
        # Convert to tensors
        dialogue_video = torch.stack(dialogue_videos)
        dialogue_roles_video = torch.tensor(dialogue_roles_video, dtype=torch.long)
        
        # Pad audio features to same length (full audio length)
        max_audio_length = 160000  # Full 10 seconds audio length
        padded_audio_features = []
        for af in dialogue_audio_features:
            if isinstance(af, np.ndarray):
                if len(af) > max_audio_length:
                    # Truncate if too long
                    af = af[:max_audio_length]
                elif len(af) < max_audio_length:
                    # Pad if too short
                    af = np.pad(af, (0, max_audio_length - len(af)), 'constant', constant_values=0)
                padded_audio_features.append(af)
            else:
                # Create zero tensor of correct size
                padded_audio_features.append(np.zeros(max_audio_length, dtype=np.float32))
        
        dialogue_audio = torch.stack([torch.from_numpy(af) for af in padded_audio_features])
        dialogue_roles_audio = torch.tensor(dialogue_roles_audio, dtype=torch.long)
        dialogue_indices = torch.tensor(dialogue_indices, dtype=torch.long)
        
        sequence_length = torch.tensor(min(len(dialogue), self.max_dialogue_length), dtype=torch.long)
        
        # Tokenize text for audio-text model
        if dialogue_texts:
            context_text = " ".join([t for t in dialogue_texts if t != '[EMPTY]'])
            context_encoding = self.tokenizer(
                context_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            context_input_ids = context_encoding['input_ids'].squeeze(0)
            context_attention_mask = context_encoding['attention_mask'].squeeze(0)
            
            dialogue_encoding = self.tokenizer(
                dialogue_texts,
                max_length=128,
                padding='max_length', 
                truncation=True,
                return_tensors='pt'
            )
            dialogue_input_ids = dialogue_encoding['input_ids']
            dialogue_attention_mask = dialogue_encoding['attention_mask']
        else:
            context_input_ids = torch.zeros(self.max_length, dtype=torch.long)
            context_attention_mask = torch.zeros(self.max_length, dtype=torch.long)
            dialogue_input_ids = torch.zeros(self.max_dialogue_length, 128, dtype=torch.long)
            dialogue_attention_mask = torch.zeros(self.max_dialogue_length, 128, dtype=torch.long)
        
        # Process metadata
        speaker_info = item.get('speaker_profile', {})
        listener_info = item.get('listener_profile', {})
        
        # Safe metadata extraction
        def safe_get_age(age_str):
            age_map = {'young': 0, 'middle': 1, 'old': 2, 'unknown': 3}
            return age_map.get(age_str, 3)
        
        def safe_get_gender(gender_str):
            return 0 if gender_str == 'male' else 1
        
        def safe_get_timbre(timbre_str):
            timbre_map = {'low': 0, 'mid': 1, 'high': 2}
            return timbre_map.get(timbre_str, 1)
        
        speaker_age = safe_get_age(speaker_info.get('age', 'young'))
        speaker_gender = safe_get_gender(speaker_info.get('gender', 'male'))
        speaker_timbre = safe_get_timbre(speaker_info.get('timbre', 'mid'))
        speaker_id = min(speaker_info.get('ID', 0), 99)
        
        listener_age = safe_get_age(listener_info.get('age', 'young'))
        listener_gender = safe_get_gender(listener_info.get('gender', 'male'))
        listener_timbre = safe_get_timbre(listener_info.get('timbre', 'mid'))
        listener_id = min(listener_info.get('ID', 0), 99)
        
        event_scenario_id = self.event_scenario_vocab.get(chain_of_empathy.get('event_scenario', 'unknown'), 0)
        emotion_cause_id = self.emotion_cause_vocab.get(chain_of_empathy.get('emotion_cause', 'unknown'), 0)
        goal_response_id = self.goal_response_vocab.get(chain_of_empathy.get('goal_to_response', 'unknown'), 0)
        topic_id = self.topic_vocab.get(item.get('topic', 'unknown'), 0)
        
        metadata = torch.tensor([
            speaker_age, speaker_gender, speaker_timbre, speaker_id,
            listener_age, listener_gender, listener_timbre, listener_id,
            event_scenario_id, emotion_cause_id, goal_response_id, topic_id
        ], dtype=torch.long)
        
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            # Video data
            'dialogue_video': dialogue_video,
            'dialogue_roles_video': dialogue_roles_video,
            
            # Audio-text data
            'context_input_ids': context_input_ids,
            'context_attention_mask': context_attention_mask,
            'dialogue_input_ids': dialogue_input_ids,
            'dialogue_attention_mask': dialogue_attention_mask,
            'dialogue_audio': dialogue_audio,
            'dialogue_roles_audio': dialogue_roles_audio,
            'dialogue_indices': dialogue_indices,
            
            # Common data
            'metadata': metadata,
            'sequence_length': sequence_length,
            'label': label
        }

class LateFusionModel(nn.Module):
    """Simplified Late fusion model for RTX 3060 12GB"""
    
    def __init__(self, video_model, audio_text_model, num_classes=7, video_weight=0.7):
        super(LateFusionModel, self).__init__()
        
        self.video_model = video_model
        self.audio_text_model = audio_text_model
        
        # Freeze pretrained models to save memory
        for param in self.video_model.parameters():
            param.requires_grad = False
        for param in self.audio_text_model.parameters():
            param.requires_grad = False
        
        # Simple learnable fusion weight (only trainable parameter)
        self.fusion_weight = nn.Parameter(torch.tensor(video_weight, dtype=torch.float32))
        
    def forward(self, video_data, audio_text_data, return_features=False):
        # Extract logits from pretrained models (no gradients)
        with torch.no_grad():
            # Video logits
            video_logits = self.video_model(
                video_data['dialogue_video'],
                video_data['dialogue_roles'],
                video_data['metadata'],
                video_data['sequence_length']
            )
            
            # Audio-text logits  
            audio_text_logits = self.audio_text_model(
                audio_text_data['context_input_ids'],
                audio_text_data['context_attention_mask'],
                audio_text_data['dialogue_input_ids'],
                audio_text_data['dialogue_attention_mask'],
                audio_text_data['dialogue_audio'],
                audio_text_data['dialogue_roles'],
                audio_text_data['metadata'],
                audio_text_data['sequence_length']
            )
        
        # Learnable weighted fusion at logit level (memory efficient)
        video_weight = torch.sigmoid(self.fusion_weight)
        audio_text_weight = 1.0 - video_weight
        
        # Weighted fusion of logits
        main_logits = video_weight * video_logits + audio_text_weight * audio_text_logits
        
        if return_features:
            fusion_weights = torch.stack([video_weight, audio_text_weight], dim=0).unsqueeze(0).repeat(video_logits.shape[0], 1)
            return {
                'main_logits': main_logits,
                'video_logits': video_logits,
                'audio_text_logits': audio_text_logits,
                'fusion_weights': fusion_weights
            }
        
        return main_logits, video_logits, audio_text_logits

def load_pretrained_models(video_checkpoint_path, audio_text_checkpoint_path):
    """Load pretrained video and audio-text models"""
    
    print("Loading pretrained models for testing...")
    
    # Create a full dataset with original training data to get correct vocabulary sizes for video model
    print("Loading original training vocabulary for video model...")
    full_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    full_video_transform = VideoTransform()
    
    print("Creating full training dataset for video model vocab...")
    full_train_dataset = LateFusionDataset(
        json_file='json/mapped_train_data_video_aligned.json',
        video_dir='data/train_video/video_v5_0',
        audio_dir='data/train_audio/audio_v5_0',
        tokenizer=full_tokenizer,
        video_transform=full_video_transform,
        max_dialogue_length=10
    )
    
    # Load video model with correct vocabulary sizes
    print("Initializing video model...")
    video_model = VideoOnlyModel(full_train_dataset, num_classes=7)
    print("Loading video model checkpoint...")
    video_checkpoint = torch.load(video_checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in video_checkpoint:
        video_model.load_state_dict(video_checkpoint['model_state_dict'])
        print(f"Video model loaded - Validation Accuracy: {video_checkpoint.get('val_accuracy', 'N/A'):.4f}")
    else:
        video_model.load_state_dict(video_checkpoint)
    
    # Create a dummy dataset with original training data to get correct vocabulary sizes
    print("Loading original training vocabulary for audio-text model...")
    original_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    original_wav2vec = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    
    # Use the original training data to get correct vocab sizes
    print("Creating original dataset for audio-text model vocab...")
    original_dataset = MultimodalSequentialDataset(
        data_path='json/mapped_train_data.json',
        audio_dir='data/train_audio',
        tokenizer=original_tokenizer,
        wav2vec_feature_extractor=original_wav2vec,
        max_length=384,
        max_dialogue_length=10
    )
    
    # Load audio-text model with original vocabulary sizes
    print("Initializing audio-text model...")
    audio_text_model = MultimodalLSTMModel(original_dataset, num_classes=7, use_wav2vec=True)
    print("Loading audio-text model checkpoint...")
    audio_text_checkpoint = torch.load(audio_text_checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in audio_text_checkpoint:
        audio_text_model.load_state_dict(audio_text_checkpoint['model_state_dict'])
        print(f"Audio-text model loaded - Validation Accuracy: {audio_text_checkpoint.get('val_accuracy', 'N/A'):.4f}")
    else:
        audio_text_model.load_state_dict(audio_text_checkpoint)
    
    print("✅ Both pretrained models loaded successfully!")
    return video_model, audio_text_model

def test_late_fusion_model(fusion_model, test_loader, device, results_dir='late_fusion_results'):
    """Test the late fusion model on test dataset with memory optimization"""
    
    print(f"\n🧪 Starting comprehensive test evaluation...")
    fusion_model.eval()
    
    all_predictions = []
    all_labels = []
    all_video_predictions = []
    all_audio_text_predictions = []
    all_fusion_weights = []
    
    # Memory optimization for testing
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Clear cache every few batches to prevent memory buildup (more frequent with full audio)
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            try:
                # Prepare data
                video_data = {
                    'dialogue_video': batch['dialogue_video'].to(device, non_blocking=True),
                    'dialogue_roles': batch['dialogue_roles_video'].to(device, non_blocking=True),
                    'metadata': batch['metadata'].to(device, non_blocking=True),
                    'sequence_length': batch['sequence_length'].to(device, non_blocking=True)
                }
                
                audio_text_data = {
                    'context_input_ids': batch['context_input_ids'].to(device, non_blocking=True),
                    'context_attention_mask': batch['context_attention_mask'].to(device, non_blocking=True),
                    'dialogue_input_ids': batch['dialogue_input_ids'].to(device, non_blocking=True),
                    'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device, non_blocking=True),
                    'dialogue_audio': batch['dialogue_audio'].to(device, non_blocking=True),
                    'dialogue_roles': batch['dialogue_roles_audio'].to(device, non_blocking=True),
                    'metadata': batch['metadata'].to(device, non_blocking=True),
                    'sequence_length': batch['sequence_length'].to(device, non_blocking=True)
                }
                
                labels = batch['label'].to(device, non_blocking=True)
                
                # Forward pass with feature extraction and mixed precision
                with autocast():
                    outputs = fusion_model(video_data, audio_text_data, return_features=True)
                
                # Get predictions
                main_predictions = torch.max(outputs['main_logits'], 1)[1]
                video_predictions = torch.max(outputs['video_logits'], 1)[1]
                audio_text_predictions = torch.max(outputs['audio_text_logits'], 1)[1]
                
                # Store results
                all_predictions.extend(main_predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_video_predictions.extend(video_predictions.cpu().numpy())
                all_audio_text_predictions.extend(audio_text_predictions.cpu().numpy())
                all_fusion_weights.extend(outputs['fusion_weights'].cpu().numpy())
                
                # Clear batch data to free memory
                del video_data, audio_text_data, labels, outputs
                del main_predictions, video_predictions, audio_text_predictions
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ Memory error at batch {batch_idx}, skipping...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
    
    # Calculate metrics
    labels = np.array(all_labels)
    main_preds = np.array(all_predictions)
    video_preds = np.array(all_video_predictions)
    audio_text_preds = np.array(all_audio_text_predictions)
    fusion_weights = np.array(all_fusion_weights)
    
    # Main fusion model metrics
    main_accuracy = accuracy_score(labels, main_preds)
    main_precision, main_recall, main_f1, _ = precision_recall_fscore_support(
        labels, main_preds, average='weighted', zero_division=0
    )
    
    # Component model metrics
    video_accuracy = accuracy_score(labels, video_preds)
    audio_text_accuracy = accuracy_score(labels, audio_text_preds)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels, main_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, main_preds)
    
    emotion_classes = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
    
    # Print test results
    print(f"\n📊 LATE FUSION TEST RESULTS:")
    print("=" * 50)
    print(f"🎯 Late Fusion Accuracy: {main_accuracy:.4f} ({main_accuracy*100:.2f}%)")
    print(f"📈 Late Fusion F1-Score: {main_f1:.4f}")
    print(f"📈 Video-Only Accuracy:  {video_accuracy:.4f} ({video_accuracy*100:.2f}%)")
    print(f"📈 Audio-Text Accuracy:  {audio_text_accuracy:.4f} ({audio_text_accuracy*100:.2f}%)")
    
    improvement_acc = (main_accuracy - max(video_accuracy, audio_text_accuracy)) * 100
    print(f"\n🎯 Fusion Improvement: +{improvement_acc:.2f}% accuracy")
    
    print(f"\n🏋️ Fusion Weight Analysis:")
    print(f"   Mean Video Weight:     {np.mean(fusion_weights[:, 0]):.3f} ± {np.std(fusion_weights[:, 0]):.3f}")
    print(f"   Mean Audio-Text Weight: {np.mean(fusion_weights[:, 1]):.3f} ± {np.std(fusion_weights[:, 1]):.3f}")
    
    # Save test results
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.title('Late Fusion Model - Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/late_fusion_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = ['Late Fusion', 'Video-Only', 'Audio-Text']
    accuracies = [main_accuracy, video_accuracy, audio_text_accuracy]
    f1_scores = [main_f1, 0, 0]  # Only have F1 for main model
    
    bars1 = ax1.bar(models, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Per-class performance
    x = np.arange(len(emotion_classes))
    ax2.bar(x, per_class_f1, alpha=0.7, color='skyblue')
    ax2.set_title('Per-Class F1-Score (Late Fusion)')
    ax2.set_xlabel('Emotion Classes')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotion_classes, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/late_fusion_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed JSON results
    test_results = {
        'test_info': {
            'timestamp': timestamp,
            'test_samples': len(labels),
            'emotion_classes': emotion_classes
        },
        'performance_metrics': {
            'late_fusion': {
                'accuracy': float(main_accuracy),
                'precision': float(main_precision),
                'recall': float(main_recall),
                'f1_score': float(main_f1)
            },
            'video_only': {'accuracy': float(video_accuracy)},
            'audio_text': {'accuracy': float(audio_text_accuracy)}
        },
        'per_class_metrics': {
            emotion_classes[i]: {
                'precision': float(per_class_precision[i]) if i < len(per_class_precision) else 0,
                'recall': float(per_class_recall[i]) if i < len(per_class_recall) else 0,
                'f1_score': float(per_class_f1[i]) if i < len(per_class_f1) else 0,
                'support': int(support[i]) if i < len(support) else 0
            } for i in range(len(emotion_classes))
        },
        'fusion_analysis': {
            'mean_video_weight': float(np.mean(fusion_weights[:, 0])),
            'mean_audio_text_weight': float(np.mean(fusion_weights[:, 1])),
            'video_weight_std': float(np.std(fusion_weights[:, 0])),
            'audio_text_weight_std': float(np.std(fusion_weights[:, 1]))
        },
        'confusion_matrix': cm.tolist()
    }
    
    with open(f'{results_dir}/late_fusion_test_results_{timestamp}.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Test results saved to {results_dir}/")
    print(f"   📊 Confusion Matrix: late_fusion_confusion_matrix_{timestamp}.png")
    print(f"   📈 Performance Charts: late_fusion_performance_{timestamp}.png")
    print(f"   📄 Detailed Results: late_fusion_test_results_{timestamp}.json")
    
    return test_results

def main():
    """Main testing function"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # Create results directory
    os.makedirs('late_fusion_results', exist_ok=True)
    
    # Paths
    test_json = 'json/mapped_test_data_video_aligned.json'
    video_dir = 'data/train_video/video_v5_0'
    audio_dir = 'data/train_audio/audio_v5_0'
    
    # Pretrained model paths
    video_checkpoint_path = 'checkpoints_2/best_video_model_20251019_161926_acc0.7281.pth'
    audio_text_checkpoint_path = 'checkpoints/best_7class_model.pth'
    
    # Find the latest fusion model checkpoint
    import glob
    fusion_checkpoints = glob.glob('late_fusion_checkpoint/best_late_fusion_model_*.pth')
    if not fusion_checkpoints:
        print("❌ No late fusion checkpoint found! Please train the model first.")
        return
    
    # Use the most recent checkpoint
    latest_checkpoint = max(fusion_checkpoints, key=os.path.getctime)
    print(f"📂 Using latest fusion checkpoint: {latest_checkpoint}")
    
    # Initialize tokenizer and feature extractor
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    video_transform = VideoTransform()
    
    # Load pretrained models
    video_model, audio_text_model = load_pretrained_models(
        video_checkpoint_path, audio_text_checkpoint_path
    )
    
    # Create late fusion model
    fusion_model = LateFusionModel(
        video_model=video_model,
        audio_text_model=audio_text_model,
        num_classes=7,
        video_weight=0.7
    ).to(device)
    
    # Load the trained fusion model
    print(f"📂 Loading trained fusion model from: {latest_checkpoint}")
    checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Fusion model loaded - Training Accuracy: {checkpoint.get('val_accuracy', 'N/A'):.2f}%")
    
    # Create test dataset with memory optimization
    print("Creating test dataset with memory optimization...")
    test_dataset = LateFusionDataset(
        json_file=test_json,
        video_dir=video_dir,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        video_transform=video_transform,
        max_dialogue_length=10
    )
    
    # Create optimized test dataloader for memory efficiency
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,  # Further reduced batch size to handle full audio length
        shuffle=False,
        num_workers=0,  # Single worker to prevent memory issues
        pin_memory=False,  # Disable pin memory to save VRAM
        drop_last=False,
        prefetch_factor=None  # Disable prefetching to save memory
    )
    
    print(f"📋 Test samples: {len(test_dataset)}")
    print(f"📊 Test batches: {len(test_loader)}")
    
    # Clear GPU cache before testing
    torch.cuda.empty_cache()
    print(f"🔧 GPU Memory before testing: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # Run comprehensive test evaluation
    print("\n" + "="*60)
    print("🧪 STARTING MEMORY-OPTIMIZED TEST EVALUATION")
    print("="*60)
    
    test_results = test_late_fusion_model(
        fusion_model=fusion_model,
        test_loader=test_loader,
        device=device,
        results_dir='late_fusion_results'
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("🎉 LATE FUSION TEST COMPLETE!")
    print("="*60)
    print(f"📊 Final Test Accuracy: {test_results['performance_metrics']['late_fusion']['accuracy']:.4f} ({test_results['performance_metrics']['late_fusion']['accuracy']*100:.2f}%)")
    print(f"📁 All results saved in: late_fusion_results/")
    print("="*60)

if __name__ == "__main__":
    main()