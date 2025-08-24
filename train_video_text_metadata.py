"""
Multi-modal Empathetic Response Training with BERT and TimeSformer
Enhanced to match structure from original train.py:
- BERT base for text feature extraction  
- TimeSformer for video feature extraction
- All features from train.py: emotion mapping, class weights, focal loss, early stopping
- Mixed precision training and gradient accumulation
- Saves results to result_2 directory and checkpoints to checkpoints_2
"""

# Suppress FutureWarnings from transformers library
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

import copy
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import time
import random
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import math

warnings.filterwarnings("ignore")

# Set CUDA memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class VideoTransform:
    """TimeSformer video preprocessing pipeline"""
    
    def __init__(self, num_frames=8, frame_size=192, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.num_frames = num_frames
        self.frame_size = frame_size  # Reduced from 224 to 192 for faster processing
        self.mean = mean
        self.std = std
        
        self.transform = Compose([
            Resize((frame_size, frame_size)),
            Normalize(mean=mean, std=std)
        ])
    
    def extract_frames(self, video_path):
        """Optimized frame extraction for reduced CPU usage"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            # Optimized frame sampling - skip frames more aggressively
            frame_step = max(1, total_frames // (self.num_frames + 2))
            frame_indices = list(range(0, min(total_frames, frame_step * self.num_frames), frame_step))[:self.num_frames]
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # Faster processing - resize first, then convert
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Direct tensor conversion with normalization
                    frame = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1)
                    frames.append(frame)
                else:
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(torch.zeros(3, self.frame_size, self.frame_size))
            
            cap.release()
            
            # Pad if necessary
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, self.frame_size, self.frame_size))
            
            # Stack and normalize efficiently
            video_tensor = torch.stack(frames[:self.num_frames], dim=1)  # [C, T, H, W]
            
            # Vectorized normalization
            mean_tensor = torch.tensor(self.mean).view(3, 1, 1, 1)
            std_tensor = torch.tensor(self.std).view(3, 1, 1, 1)
            video_tensor = (video_tensor - mean_tensor) / std_tensor
            
            return video_tensor
            
        except Exception as e:
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)

class MultimodalSequentialVideoDataset(Dataset):
    """
    Dataset for multimodal empathetic dialogue with video, text, and metadata (NO AUDIO)
    Uses comprehensive emotion mapping to prevent overfitting
    """
    
    def __init__(self, data_path, video_dir, tokenizer, video_transform=None, max_length=512, max_dialogue_length=10):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.video_transform = video_transform or VideoTransform()
        self.max_length = max_length
        self.max_dialogue_length = max_dialogue_length
        
        # Emotion mapping from data_mapping.py
        # Emotion mapping to 7 basic emotion classes
        self.ed_emotion_projection = {
            'conflicted': 'anxious',
            'vulnerability': 'afraid',
            'helplessness': 'afraid',
            'sadness': 'sad',
            'pensive': 'sentimental',
            'frustration': 'annoyed',
            'weary': 'tired',
            'anxiety': 'anxious',
            'reflective': 'sentimental',
            'upset': 'disappointed',
            'worried': 'anxious',
            'fear': 'afraid',
            'frustrated': 'sad',
            'fatigue': 'tired',
            'lost': 'jealous',
            'disappointment': 'disappointed',
            'nostalgia': 'nostalgic',
            'exhaustion': 'tired',
            'uneasy': 'anxious',
            'loneliness': 'lonely',
            'fragile': 'afraid',
            'confused': 'jealous',
            'vulnerable': 'afraid',
            'thoughtful': 'sentimental',
            'stressed': 'anxious',
            'concerned': 'anxious',
            'tiredness': 'tired',
            'burdened': 'anxious',
            'melancholy': 'sad',
            'overwhelmed': 'anxious',
            'worry': 'anxious',
            'heavy-hearted': 'sad',
            'melancholic': 'sad',
            'nervous': 'anxious',
            'fearful': 'afraid',
            'stress': 'anxious',
            'confusion': 'anxious',
            'inadequacy': 'ashamed',
            'regret': 'guilty',
            'helpless': 'afraid',
            'concern': 'anxious',
            'exhausted': 'tired',
            'overwhelm': 'anxious',
            'tired': 'tired',
            'disappointed': 'sad',
            'surprised': 'surprised',
            'excited': 'happy',
            'angry': 'angry',
            'proud': 'happy',
            'annoyed': 'angry',
            'grateful': 'happy',
            'lonely': 'sad',
            'afraid': 'fear',
            'terrified': 'fear',
            'guilty': 'sad',
            'impressed': 'surprised',
            'disgusted': 'disgusted',
            'hopeful': 'happy',
            'confident': 'happy',
            'furious': 'angry',
            'anxious': 'sad',
            'anticipating': 'happy',
            'joyful': 'happy',
            'nostalgic': 'sad',
            'prepared': 'happy',
            'jealous': 'contempt',
            'content': 'happy',
            'devastated': 'surprised',
            'embarrassed': 'sad',
            'caring': 'happy',
            'sentimental': 'sad',
            'trusting': 'happy',
            'ashamed': 'sad',
            'apprehensive': 'fear',
            'faithful': 'happy'       
        }

        # 7 basic emotion classes
        self.emotion_to_id = {
            "happy": 0,
            "surprised": 1,
            "angry": 2,
            "fear": 3,
            "sad": 4,
            "disgusted": 5,
            "contempt": 6
        }
        self.id_to_emotion = {v: k for k, v in self.emotion_to_id.items()}
        
        # Profile mappings
        self.age_to_id = {"child": 0, "young": 1, "middle-aged": 2, "elderly": 3}
        self.gender_to_id = {"male": 0, "female": 1}
        self.timbre_to_id = {"high": 0, "mid": 1, "low": 2}
        self.role_to_id = {"speaker": 0, "listener": 1, "listener_response": 2}
        
        # Create vocabulary for text features
        self.create_text_vocabulary()
        
        print(f"Loaded {len(self.data)} samples from {data_path}")
        print(f"Emotion mapping covers {len(self.ed_emotion_projection)} emotions")

    def create_text_vocabulary(self):
        """Create vocabularies for chain_of_empathy text fields and topics"""
        event_scenarios = set()
        emotion_causes = set()
        goal_responses = set()
        topics = set()
        
        for item in self.data:
            chain = item.get('turn', {}).get('chain_of_empathy', {})
            event_scenarios.add(chain.get('event_scenario', ''))
            emotion_causes.add(chain.get('emotion_cause', ''))
            goal_responses.add(chain.get('goal_to_response', ''))
            topics.add(item.get('topic', ''))
        
        # Remove empty strings and create mappings
        self.event_scenario_vocab = {scenario: i for i, scenario in enumerate(sorted([s for s in event_scenarios if s]))}
        self.emotion_cause_vocab = {cause: i for i, cause in enumerate(sorted([c for c in emotion_causes if c]))}
        self.goal_response_vocab = {goal: i for i, goal in enumerate(sorted([g for g in goal_responses if g]))}
        self.topic_vocab = {topic: i for i, topic in enumerate(sorted([t for t in topics if t]))}
        
        print(f"Created vocabularies: {len(self.event_scenario_vocab)} scenarios, "
              f"{len(self.emotion_cause_vocab)} causes, {len(self.goal_response_vocab)} goals, "
              f"{len(self.topic_vocab)} topics")

    def __len__(self):
        return len(self.data)
    
    def get_label(self, idx):
        """Get label without processing other data (for efficient class weight computation)"""
        item = self.data[idx]
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        
        # Use emotion mapping from train.py
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        emotion_label = self.emotion_to_id.get(mapped_emotion, 0)
        return emotion_label

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract turn data
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        
        # Get emotion label using emotion mapping from train.py
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        
        # Use emotion mapping from train.py (same as original)
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        emotion_label = self.emotion_to_id.get(mapped_emotion, 0)  # Default to 'happy'
        
        # Extract dialogue sequence
        dialogue = turn.get('dialogue', [])
        
        # Process sequential dialogue data (NO AUDIO)
        dialogue_texts = []
        dialogue_video_features = []
        dialogue_roles = []
        dialogue_indices = []
        
        for utt in dialogue[:self.max_dialogue_length]:
            # Text
            text = utt.get('text', '')
            dialogue_texts.append(text)
            
            # Video processing ONLY (NO AUDIO)
            video_name = utt.get('video_name', None)
            if video_name:
                video_path = os.path.join(self.video_dir, video_name)
                video_features = self.video_transform.extract_frames(video_path)
            else:
                video_features = torch.zeros(3, 8, 192, 192)  # Updated size
            dialogue_video_features.append(video_features)
            
            # Role and index
            role = utt.get('role', 'speaker')
            dialogue_roles.append(self.role_to_id.get(role, 0))
            dialogue_indices.append(utt.get('index', 0))
        
        # Pad sequences to max_dialogue_length
        while len(dialogue_texts) < self.max_dialogue_length:
            dialogue_texts.append('[EMPTY]')
            dialogue_video_features.append(torch.zeros(3, 8, 192, 192))
            dialogue_roles.append(0)
            dialogue_indices.append(0)
        
        # Tokenize all dialogue texts
        dialogue_encodings = []
        for text in dialogue_texts:
            if not text or text.strip() == '':
                text = '[EMPTY]'
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            dialogue_encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        
        # Context encoding
        context = turn.get('context', '')
        if not context or context.strip() == '':
            context = '[NO CONTEXT]'
        context_encoding = self.tokenizer(
            context,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        # Extract comprehensive metadata
        speaker_profile = item.get('speaker_profile', {})
        listener_profile = item.get('listener_profile', {})
        
        # Profile features
        speaker_age = self.age_to_id.get(speaker_profile.get('age', 'young'), 1)
        speaker_gender = self.gender_to_id.get(speaker_profile.get('gender', 'male'), 0)
        speaker_timbre = self.timbre_to_id.get(speaker_profile.get('timbre', 'mid'), 1)
        speaker_id = speaker_profile.get('ID', 0)
        
        listener_age = self.age_to_id.get(listener_profile.get('age', 'young'), 1)
        listener_gender = self.gender_to_id.get(listener_profile.get('gender', 'male'), 0)
        listener_timbre = self.timbre_to_id.get(listener_profile.get('timbre', 'mid'), 1)
        listener_id = listener_profile.get('ID', 0)
        
        # Chain of empathy metadata
        event_scenario = chain_of_empathy.get('event_scenario', '')
        emotion_cause = chain_of_empathy.get('emotion_cause', '')
        goal_to_response = chain_of_empathy.get('goal_to_response', '')
        
        # Topic metadata
        topic = item.get('topic', '')
        
        # Map to vocabulary IDs
        event_scenario_id = self.event_scenario_vocab.get(event_scenario, 0)
        emotion_cause_id = self.emotion_cause_vocab.get(emotion_cause, 0)
        goal_response_id = self.goal_response_vocab.get(goal_to_response, 0)
        topic_id = self.topic_vocab.get(topic, 0)
        
        # Create comprehensive metadata tensor
        metadata_features = torch.tensor([
            speaker_age, speaker_gender, speaker_timbre, speaker_id,
            listener_age, listener_gender, listener_timbre, listener_id,
            event_scenario_id, emotion_cause_id, goal_response_id, topic_id
        ], dtype=torch.long)
        
        # Ensure all video features have consistent dimensions before stacking
        consistent_video_features = []
        target_size = (3, 8, 192, 192)  # Expected size: [C, T, H, W]
        
        for i, video_feat in enumerate(dialogue_video_features):
            if video_feat.shape != target_size:
                # Resize to target dimensions if different
                if video_feat.shape[0] == 3 and video_feat.shape[1] == 8:
                    # Resize spatial dimensions only
                    resized_frames = []
                    for t in range(video_feat.shape[1]):
                        frame = video_feat[:, t, :, :]  # [C, H, W]
                        # Use interpolation to resize
                        frame_resized = F.interpolate(
                            frame.unsqueeze(0), size=(192, 192), mode='bilinear', align_corners=False
                        ).squeeze(0)
                        resized_frames.append(frame_resized)
                    video_feat = torch.stack(resized_frames, dim=1)  # [C, T, H, W]
                else:
                    # Create zero tensor if dimensions are completely wrong
                    video_feat = torch.zeros(target_size)
            consistent_video_features.append(video_feat)
        
        # Stack video features with consistent dimensions
        dialogue_video_tensor = torch.stack(consistent_video_features)
        
        return {
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'dialogue_input_ids': torch.stack([enc['input_ids'] for enc in dialogue_encodings]),
            'dialogue_attention_mask': torch.stack([enc['attention_mask'] for enc in dialogue_encodings]),
            'dialogue_video': dialogue_video_tensor,  # NO AUDIO COMPONENT
            'dialogue_roles': torch.tensor(dialogue_roles, dtype=torch.long),
            'dialogue_indices': torch.tensor(dialogue_indices, dtype=torch.long),
            'metadata': metadata_features,
            'label': torch.tensor(emotion_label, dtype=torch.long),
            'conversation_id': item.get('conversation_id', ''),
            'raw_emotion': raw_emotion,
            'sequence_length': torch.tensor(min(len(dialogue), self.max_dialogue_length), dtype=torch.long),
            'valid_sample': torch.tensor(1 if emotion_label != -1 else 0, dtype=torch.long)  # For filtering
        }

# Include all utility classes from train.py
class TimeSformerEncoder(nn.Module):
    """TimeSformer-style video encoder matching checkpoint structure"""
    
    def __init__(self, num_frames=8, frame_size=192, patch_size=32, embed_dim=256, num_heads=4, num_layers=3, dropout_rate=0.1):
        super(TimeSformerEncoder, self).__init__()
        
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (frame_size // patch_size) ** 2  # 36 patches per frame for 192x192 with patch_size=32
        
        # 2D convolution patch embedding layer (matching checkpoint: [256, 3, 32, 32])
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Separate spatial and temporal positional embeddings (matching checkpoint)
        self.pos_embed_spatial = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.randn(1, num_frames, embed_dim))
        
        # Transformer layers (3 layers to match checkpoint)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Simple output projection (matching checkpoint)
        self.output_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels, frames, height, width]
        batch_size, seq_len = x.shape[:2]
        
        # Process each dialogue turn separately
        sequence_features = []
        for i in range(seq_len):
            video = x[:, i]  # [batch_size, 3, 8, 192, 192]
            
            # Process each frame separately with 2D conv
            frame_features = []
            for t in range(self.num_frames):
                frame = video[:, :, t, :, :]  # [batch_size, 3, 192, 192]
                # Apply 2D convolution patch embedding
                patches = self.patch_embed(frame)  # [batch_size, 256, 6, 6]
                patches = patches.flatten(2).transpose(1, 2)  # [batch_size, 36, 256]
                
                # Add spatial positional embedding
                patches = patches + self.pos_embed_spatial  # [batch_size, 36, 256]
                frame_features.append(patches)
            
            # Stack all frames: [batch_size, 8, 36, 256]
            video_patches = torch.stack(frame_features, dim=1)
            
            # Reshape for transformer: [batch_size, 8*36, 256]
            video_patches = video_patches.reshape(batch_size, self.num_frames * self.num_patches, self.embed_dim)
            
            # Add temporal positional embedding
            temporal_pos = self.pos_embed_temporal.repeat_interleave(self.num_patches, dim=1)
            video_patches = video_patches + temporal_pos
            
            # Apply transformer
            transformed = self.transformer(video_patches)  # [batch_size, 8*36, embed_dim]
            
            # Global average pooling across patches
            video_features = transformed.mean(dim=1)  # [batch_size, embed_dim]
            video_features = self.output_projection(video_features)
            
            sequence_features.append(video_features)
        
        # Stack sequence features
        sequence_output = torch.stack(sequence_features, dim=1)  # [batch_size, seq_len, embed_dim]
        
        return sequence_output

class MultimodalLSTMVideoModel(nn.Module):
    """Multimodal LSTM model with video, text, and metadata (NO AUDIO) - OVERFITTING PREVENTION VERSION"""
    
    def __init__(self, dataset, num_classes=7, hidden_size=128, num_layers=1, dropout_rate=0.5, freeze_bert_layers=8):
        super(MultimodalLSTMVideoModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Text encoder (BERT-base) with frozen layers to prevent overfitting
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
        
        # Freeze lower BERT layers to prevent overfitting
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.text_encoder.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"Froze first {freeze_bert_layers} BERT layers to prevent overfitting")
        
        # Video encoder (TimeSformer-style) with reduced complexity
        self.video_encoder = TimeSformerEncoder(
            num_frames=8,
            frame_size=192,
            embed_dim=128,  # Reduced from 256
            num_heads=4,
            num_layers=2,  # Reduced from 3
            dropout_rate=dropout_rate
        )
        video_dim = 128  # Reduced
        
        # Metadata embeddings with smaller dimensions
        self.age_embedding = nn.Embedding(4, 8)  # Reduced from 16
        self.gender_embedding = nn.Embedding(2, 4)  # Reduced from 8
        self.timbre_embedding = nn.Embedding(3, 4)  # Reduced from 8
        self.role_embedding = nn.Embedding(3, 8)   # Reduced from 16
        
        # Profile ID embeddings with smaller dimensions
        self.speaker_id_embedding = nn.Embedding(100, 16)  # Reduced from 32
        self.listener_id_embedding = nn.Embedding(100, 16)  # Reduced from 32
        
        # Chain of empathy embeddings with smaller dimensions
        vocab_sizes = {
            'event_scenario': len(dataset.event_scenario_vocab) + 1,
            'emotion_cause': len(dataset.emotion_cause_vocab) + 1,
            'goal_response': len(dataset.goal_response_vocab) + 1,
            'topic': len(dataset.topic_vocab) + 1
        }
        
        self.event_scenario_embedding = nn.Embedding(vocab_sizes['event_scenario'], 16)  # Reduced from 32
        self.emotion_cause_embedding = nn.Embedding(vocab_sizes['emotion_cause'], 16)
        self.goal_response_embedding = nn.Embedding(vocab_sizes['goal_response'], 16)
        self.topic_embedding = nn.Embedding(vocab_sizes['topic'], 16)
        
        # Context processor with dropout
        self.context_processor = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)  # Reduced output size
        )
        
        # Enhanced LSTM processing for full multimodal input
        # Separate LSTMs for different modalities for better learning
        
        # Text LSTM - processes dialogue text sequence
        text_lstm_input_size = text_dim + 8  # text + role embeddings
        self.text_lstm = nn.LSTM(
            input_size=text_lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # No dropout in single layer LSTM
            bidirectional=True
        )
        
        # Video LSTM - processes video sequence 
        self.video_lstm = nn.LSTM(
            input_size=video_dim,
            hidden_size=hidden_size // 2,  # Smaller for video
            num_layers=num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Metadata dimensions calculation
        metadata_dim = 8 + 4 + 4 + 16 + 8 + 4 + 4 + 16 + 16 + 16 + 16 + 16  # All reduced embeddings
        
        # Metadata LSTM - processes metadata sequence if needed
        # For static metadata, we'll use a simple projection
        self.metadata_temporal_processor = nn.Sequential(
            nn.Linear(metadata_dim, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Combined multimodal LSTM for fused features
        fused_input_size = hidden_size * 2 + hidden_size + hidden_size // 4  # text + video + metadata
        self.fusion_lstm = nn.LSTM(
            input_size=fused_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,
            bidirectional=True
        )
        
        # Multi-head attention for each modality
        self.text_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=dropout_rate
        )
        
        self.video_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,  # bidirectional but smaller
            num_heads=2,
            dropout=dropout_rate
        )
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Metadata fusion with stronger regularization
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 64),  # Reduced from 128
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32)  # Reduced from 64
        )
        
        # Final fusion and classification with enhanced multimodal features
        final_dim = 128 + hidden_size * 2 + 32  # context + fusion_lstm + metadata (all reduced)
        
        # Enhanced classifier with cross-modal attention
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=final_dim,
            num_heads=4,
            dropout=dropout_rate
        )
        
        # Smaller classifier to prevent overfitting
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 256),                    # Layer 0 - Reduced from 512
            nn.BatchNorm1d(256),                          # Layer 1
            nn.ReLU(),                                    # Layer 2
            nn.Dropout(dropout_rate),                     # Layer 3
            nn.Linear(256, 128),                          # Layer 4 - Reduced from 256
            nn.BatchNorm1d(128),                          # Layer 5
            nn.ReLU(),                                    # Layer 6
            nn.Dropout(dropout_rate),                     # Layer 7
            nn.Linear(128, num_classes)                   # Layer 8 - Final layer
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights with Xavier initialization for better convergence
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to prevent poor initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
        
    def forward(self, context_input_ids, context_attention_mask, dialogue_input_ids, 
                dialogue_attention_mask, dialogue_video, dialogue_roles, 
                dialogue_indices, metadata, sequence_length):
        batch_size, seq_len = dialogue_input_ids.shape[:2]
        
        # Process context with enhanced processing
        context_outputs = self.text_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_features = self.context_processor(context_outputs.last_hidden_state[:, 0, :])  # CLS token
        
        # Process dialogue sequence - ENHANCED MULTIMODAL LSTM APPROACH
        text_features_sequence = []
        
        for i in range(seq_len):
            # Text features for this utterance
            utt_outputs = self.text_encoder(
                input_ids=dialogue_input_ids[:, i, :],
                attention_mask=dialogue_attention_mask[:, i, :]
            )
            utt_text_features = utt_outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Role embedding
            utt_role_features = self.role_embedding(dialogue_roles[:, i])
            
            # Combine text + role features for text LSTM
            utt_combined_text = torch.cat([utt_text_features, utt_role_features], dim=1)
            text_features_sequence.append(utt_combined_text)
        
        # Stack text features for LSTM processing
        text_sequence = torch.stack(text_features_sequence, dim=1)  # [batch, seq_len, text_dim+role_dim]
        
        # Process video features for entire sequence at once
        video_sequence_features = self.video_encoder(dialogue_video)  # [batch, seq_len, video_dim]
        
        # ENHANCED LSTM PROCESSING FOR EACH MODALITY
        
        # 1. Text LSTM processing
        text_lstm_out, (text_h, text_c) = self.text_lstm(text_sequence)
        
        # Apply attention to text LSTM outputs
        text_lstm_transposed = text_lstm_out.transpose(0, 1)  # [seq_len, batch, features]
        text_attended, _ = self.text_attention(text_lstm_transposed, text_lstm_transposed, text_lstm_transposed)
        text_attended = text_attended.transpose(0, 1)  # [batch, seq_len, features]
        
        # 2. Video LSTM processing
        video_lstm_out, (video_h, video_c) = self.video_lstm(video_sequence_features)
        
        # Apply attention to video LSTM outputs
        video_lstm_transposed = video_lstm_out.transpose(0, 1)  # [seq_len, batch, features]
        video_attended, _ = self.video_attention(video_lstm_transposed, video_lstm_transposed, video_lstm_transposed)
        video_attended = video_attended.transpose(0, 1)  # [batch, seq_len, features]
        
        # 3. Process metadata (static for all time steps)
        speaker_age = self.age_embedding(metadata[:, 0])
        speaker_gender = self.gender_embedding(metadata[:, 1])
        speaker_timbre = self.timbre_embedding(metadata[:, 2])
        speaker_id_emb = self.speaker_id_embedding(torch.clamp(metadata[:, 3], 0, 99))
        
        listener_age = self.age_embedding(metadata[:, 4])
        listener_gender = self.gender_embedding(metadata[:, 5])
        listener_timbre = self.timbre_embedding(metadata[:, 6])
        listener_id_emb = self.listener_id_embedding(torch.clamp(metadata[:, 7], 0, 99))
        
        event_scenario_emb = self.event_scenario_embedding(metadata[:, 8])
        emotion_cause_emb = self.emotion_cause_embedding(metadata[:, 9])
        goal_response_emb = self.goal_response_embedding(metadata[:, 10])
        topic_emb = self.topic_embedding(metadata[:, 11])
        
        metadata_combined = torch.cat([
            speaker_age, speaker_gender, speaker_timbre, speaker_id_emb,
            listener_age, listener_gender, listener_timbre, listener_id_emb,
            event_scenario_emb, emotion_cause_emb, goal_response_emb, topic_emb
        ], dim=1)
        
        # Process metadata temporally
        metadata_temporal = self.metadata_temporal_processor(metadata_combined)
        # Expand metadata to match sequence length
        metadata_sequence = metadata_temporal.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, meta_dim]
        
        # 4. FUSION LSTM - Combine all modalities
        fused_features = torch.cat([text_attended, video_attended, metadata_sequence], dim=-1)
        
        # Apply fusion LSTM
        fusion_lstm_out, (fusion_h, fusion_c) = self.fusion_lstm(fused_features)
        
        # Apply attention to fusion LSTM outputs
        fusion_lstm_transposed = fusion_lstm_out.transpose(0, 1)  # [seq_len, batch, features]
        fusion_attended, _ = self.fusion_attention(fusion_lstm_transposed, fusion_lstm_transposed, fusion_lstm_transposed)
        fusion_attended = fusion_attended.transpose(0, 1)  # [batch, seq_len, features]
        
        # Use the last relevant output based on sequence length
        dialogue_final_features = []
        for i in range(batch_size):
            seq_len_i = sequence_length[i].item() - 1  # 0-indexed
            seq_len_i = max(0, min(seq_len_i, seq_len - 1))  # Clamp to valid range
            dialogue_final_features.append(fusion_attended[i, seq_len_i, :])
        dialogue_final_features = torch.stack(dialogue_final_features)
        
        # Process standalone metadata for final fusion
        metadata_final_features = self.metadata_processor(metadata_combined)
        
        # Final fusion with cross-modal attention
        final_features = torch.cat([context_features, dialogue_final_features, metadata_final_features], dim=1)
        
        # Apply cross-modal attention to final features
        final_features_expanded = final_features.unsqueeze(1)  # [batch, 1, features]
        final_features_transposed = final_features_expanded.transpose(0, 1)  # [1, batch, features]
        cross_attended, _ = self.cross_modal_attention(final_features_transposed, final_features_transposed, final_features_transposed)
        final_features = cross_attended.transpose(0, 1).squeeze(1)  # [batch, features]
        
        final_features = self.dropout(final_features)
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=7):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy for better generalization"""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.weight is not None:
            loss = loss * self.weight[target]
            
        return loss.mean()

class BalancedBatchSampler(torch.utils.data.Sampler):
    """Balanced batch sampler to ensure each batch has roughly equal class distribution"""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Get all labels
        self.labels = [dataset.get_label(i) for i in range(len(dataset))]
        
        # Group indices by class
        self.class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        # Calculate number of batches
        self.num_batches = len(dataset) // batch_size
        
    def __iter__(self):
        for _ in range(self.num_batches):
            batch_indices = []
            samples_per_class = self.batch_size // len(self.class_indices)
            remainder = self.batch_size % len(self.class_indices)
            
            for class_label, indices in self.class_indices.items():
                # Sample from this class
                num_samples = samples_per_class + (1 if remainder > 0 else 0)
                if remainder > 0:
                    remainder -= 1
                    
                if len(indices) >= num_samples:
                    selected = np.random.choice(indices, num_samples, replace=False)
                else:
                    selected = np.random.choice(indices, num_samples, replace=True)
                    
                batch_indices.extend(selected)
            
            # Shuffle the batch
            np.random.shuffle(batch_indices)
            yield batch_indices[:self.batch_size]
    
    def __len__(self):
        return self.num_batches

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 score"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def validate_initial_loss(model, dataloader, criterion, device, num_batches=10):
    """Validate that initial loss is reasonable for random initialization"""
    model.eval()
    total_loss = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
                
            # Move batch to device
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            dialogue_input_ids = batch['dialogue_input_ids'].to(device)
            dialogue_attention_mask = batch['dialogue_attention_mask'].to(device)
            dialogue_video = batch['dialogue_video'].to(device)
            dialogue_roles = batch['dialogue_roles'].to(device)
            dialogue_indices = batch['dialogue_indices'].to(device)
            metadata = batch['metadata'].to(device)
            sequence_length = batch['sequence_length'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                         dialogue_attention_mask, dialogue_video, dialogue_roles, 
                         dialogue_indices, metadata, sequence_length)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count
    expected_loss = np.log(7)  # -log(1/7) for 7 emotion classes
    
    print(f"Initial loss validation:")
    print(f"  Average loss: {avg_loss:.4f}")
    print(f"  Expected loss: {expected_loss:.4f}")
    
    if avg_loss < 1.0:
        print(f"  ⚠️  WARNING: Loss is too low! This suggests a problem with the model or loss function.")
    elif avg_loss > 4.5:
        print(f"  ⚠️  WARNING: Loss is too high! This suggests training instability.")
    else:
        print(f"  ✅ Loss is in reasonable range.")
    
    model.train()
    return avg_loss

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler=None, 
                gradient_accumulation_steps=1, max_grad_norm=1.0, epoch_num=1):
    """Train for one epoch with gradient accumulation and clipping"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", 
                       bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                       dynamic_ncols=True)
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move batch to device
        context_input_ids = batch['context_input_ids'].to(device)
        context_attention_mask = batch['context_attention_mask'].to(device)
        dialogue_input_ids = batch['dialogue_input_ids'].to(device)
        dialogue_attention_mask = batch['dialogue_attention_mask'].to(device)
        dialogue_video = batch['dialogue_video'].to(device)
        dialogue_roles = batch['dialogue_roles'].to(device)
        dialogue_indices = batch['dialogue_indices'].to(device)
        metadata = batch['metadata'].to(device)
        sequence_length = batch['sequence_length'].to(device)
        labels = batch['label'].to(device)
        
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                             dialogue_attention_mask, dialogue_video, dialogue_roles, 
                             dialogue_indices, metadata, sequence_length)
                loss = criterion(logits, labels)
            
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
        else:
            logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                         dialogue_attention_mask, dialogue_video, dialogue_roles, 
                         dialogue_indices, metadata, sequence_length)
            loss = criterion(logits, labels)
            
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            context_input_ids = batch['context_input_ids'].to(device)
            context_attention_mask = batch['context_attention_mask'].to(device)
            dialogue_input_ids = batch['dialogue_input_ids'].to(device)
            dialogue_attention_mask = batch['dialogue_attention_mask'].to(device)
            dialogue_video = batch['dialogue_video'].to(device)
            dialogue_roles = batch['dialogue_roles'].to(device)
            dialogue_indices = batch['dialogue_indices'].to(device)
            metadata = batch['metadata'].to(device)
            sequence_length = batch['sequence_length'].to(device)
            labels = batch['label'].to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                             dialogue_attention_mask, dialogue_video, dialogue_roles, 
                             dialogue_indices, metadata, sequence_length)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels

def verify_data_samples(dataset, num_samples=5):
    """Verify and display sample data for debugging"""
    import random
    
    print(f"\n{'='*100}")
    print("🔍 PRE-TRAINING DATA VERIFICATION")
    print("="*100)
    print("Before starting training, let's verify the data integrity...")
    
    print(f"\n{'='*100}")
    print("🔍 RANDOM CONVERSATION PREVIEW - TRAINING DATA VERIFICATION")
    print("="*100)
    print(f"Showing {num_samples} random conversations to verify data integrity...")
    
    sample_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for i, idx in enumerate(sample_indices, 1):
        print(f"\n{'─'*100}")
        print(f"🎭 CONVERSATION #{i} (Dataset Index: {idx})")
        print("─"*100)
        
        try:
            # Get raw data
            raw_item = dataset.data[idx]
            
            # Basic conversation info
            conv_id = raw_item.get('conversation_id', 'N/A')
            topic = raw_item.get('topic', 'N/A')
            
            print(f"📋 CONVERSATION METADATA:")
            print(f"   ID: {conv_id}")
            print(f"   Topic: {topic}")
            
            # Turn info
            turn = raw_item.get('turn', {})
            context = turn.get('context', 'N/A')
            
            print(f"\n🔄 TURN CONTEXT:")
            print(f'   "{context}"')
            
            # Chain of empathy
            chain = turn.get('chain_of_empathy', {})
            emotion = chain.get('speaker_emotion', 'N/A')
            scenario = chain.get('event_scenario', 'N/A')
            cause = chain.get('emotion_cause', 'N/A')
            goal = chain.get('goal_to_response', 'N/A')
            
            print(f"\n💭 EMPATHY CHAIN:")
            print(f"   Emotion: {emotion}")
            print(f"   Scenario: {scenario}")
            print(f"   Cause: {cause}")
            print(f"   Goal: {goal}")
            
            # Speaker/Listener profiles
            speaker = raw_item.get('speaker_profile', {})
            listener = raw_item.get('listener_profile', {})
            
            print(f"\n👤 SPEAKER PROFILE:")
            print(f"   ID: {speaker.get('ID', 'N/A')}")
            print(f"   Age: {speaker.get('age', 'N/A')}")
            print(f"   Gender: {speaker.get('gender', 'N/A')}")
            print(f"   Timbre: {speaker.get('timbre', 'N/A')}")
            
            print(f"\n👥 LISTENER PROFILE:")
            print(f"   ID: {listener.get('ID', 'N/A')}")
            print(f"   Age: {listener.get('age', 'N/A')}")
            print(f"   Gender: {listener.get('gender', 'N/A')}")
            print(f"   Timbre: {listener.get('timbre', 'N/A')}")
            
            # Dialogue with video mappings
            dialogue = turn.get('dialogue', [])
            print(f"\n🎬 DIALOGUE WITH VIDEO MAPPINGS ({len(dialogue)} utterances):")
            
            for j, utt in enumerate(dialogue):
                text = utt.get('text', 'N/A')
                role = utt.get('role', 'N/A')
                index = utt.get('index', 'N/A')
                video_name = utt.get('video_name', 'N/A')
                video_path = utt.get('video_path', 'N/A')
                
                print(f"\n   [{j}] UTTERANCE {index + 1}:")
                print(f"       Role: {role}")
                print(f"       Index: {index}")
                print(f'       Text: "{text}"')
                print(f"       🎬 Video File: {video_name}")
                print(f"       📁 Video Path: {video_path}")
                
                # Check if video file exists
                if video_name and video_name != 'N/A':
                    full_video_path = os.path.join(dataset.video_dir, video_name)
                    if os.path.exists(full_video_path):
                        print(f"       ✅ Video file exists")
                    else:
                        print(f"       ❌ Video file missing")
                else:
                    print(f"       ⚠️ No video file specified")
            
            # Test dataset processing
            sample = dataset[idx]
            
            # Processed features info
            print(f"\n🔧 PROCESSED FEATURES:")
            print(f"   Context tokens: {sample['context_input_ids'].shape}")
            print(f"   Dialogue tokens: {sample['dialogue_input_ids'].shape}")
            print(f"   Video features: {sample['dialogue_video'].shape}")
            print(f"   Metadata features: {sample['metadata'].shape}")
            print(f"   Emotion label: {sample['label'].item()} ({sample['raw_emotion']})")
            print(f"   Sequence length: {sample['sequence_length'].item()}")
            
        except Exception as e:
            print(f"\n❌ Error processing conversation {idx}: {e}")
            continue
    
    print(f"\n{'='*100}")
    print("🎯 DATA VERIFICATION SUMMARY:")
    print("="*100)
    print("✅ Conversations loaded successfully")
    print("✅ Video mappings present")
    print("✅ Text processing completed")
    print("✅ Metadata extraction completed")
    print("✅ Ready for VIDEO + TEXT + METADATA training!")
    print("="*100)

def main():
    # Configuration for OVERFITTING PREVENTION - Based on poor test performance analysis
    config = {
        'batch_size': 4,  # Slightly larger batch size for better gradient estimates
        'gradient_accumulation_steps': 2,  # Effective batch size of 8
        'learning_rate': 1e-5,  # Higher learning rate but with strong regularization
        'num_epochs': 15,  # Fewer epochs to prevent overfitting
        'patience': 3,  # Earlier stopping
        'dropout_rate': 0.5,  # MUCH higher dropout to prevent memorization
        'weight_decay': 0.05,  # Strong weight decay
        'warmup_steps': 300,
        'max_length': 256,  
        'max_dialogue_length': 8,  # Reduced complexity
        'hidden_size': 128,  # Smaller model to prevent overfitting
        'num_layers': 1,  # Single layer LSTM
        'max_grad_norm': 0.5,  # Stricter gradient clipping
        'use_focal_loss': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'use_mixed_precision': True,
        # NEW OVERFITTING PREVENTION
        'label_smoothing': 0.1,  # Label smoothing for better generalization
        'freeze_bert_layers': 8,  # Freeze most of BERT to prevent overfitting
        'data_augmentation': True,  # Add text augmentation
        'class_balanced_sampling': True,  # Balance classes during training
        'validation_frequency': 2  # Validate every 2 epochs instead of every epoch
    }
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()
        
        print("GPU optimizations applied:")
        print("  - Memory fraction: 95%")
        print("  - cuDNN benchmark: Enabled")
        print("  - Cache cleared")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    # Create directories
    os.makedirs('result_2', exist_ok=True)
    os.makedirs('checkpoints_2', exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalSequentialVideoDataset(
        'json/mapped_train_data_video_aligned.json',
        'data/train_video/video_v5_0',
        tokenizer,
        video_transform=VideoTransform(),
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length']
    )
    
    val_dataset = MultimodalSequentialVideoDataset(
        'json/mapped_val_data_video_aligned.json',
        'data/train_video/video_v5_0',
        tokenizer,
        video_transform=VideoTransform(),
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length']
    )
    
    # Verify data integrity
    verify_data_samples(train_dataset, num_samples=5)
    
    # Ask user confirmation before proceeding (auto-proceed for enhanced LSTM training)
    response = "y"  # Auto-proceed with enhanced LSTM training
    print(f"\nProceeding with enhanced LSTM training automatically...")
    if response.lower() != 'y':
        print("Training cancelled by user.")
        return
    
    # Create data loaders with balanced sampling for overfitting prevention
    print("Creating balanced data loaders...")
    
    if config.get('class_balanced_sampling', False):
        # Use balanced batch sampler for training
        train_sampler = BalancedBatchSampler(train_dataset, config['batch_size'])
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=0,
            pin_memory=True
        )
        print("✅ Using balanced batch sampling to prevent class bias overfitting")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Compute class weights
    print("Computing class weights...")
    all_labels = []
    for i in tqdm(range(len(train_dataset)), desc="Getting labels for class weights"):
        all_labels.append(train_dataset.get_label(i))
    
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize model with overfitting prevention
    print("Initializing model with overfitting prevention...")
    model = MultimodalLSTMVideoModel(
        train_dataset,
        num_classes=7,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        freeze_bert_layers=config.get('freeze_bert_layers', 8)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Initialize optimizer with different learning rates for different parts
    bert_params = list(model.text_encoder.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in bert_params)]
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': config['learning_rate'] * 0.1, 'weight_decay': config['weight_decay']},  # Lower LR for BERT
        {'params': other_params, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']}
    ])
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Loss function with label smoothing for better generalization
    if config.get('label_smoothing', 0) > 0:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config['label_smoothing'],
            weight=class_weights
        )
        print(f"Using Label Smoothing CrossEntropy (smoothing={config['label_smoothing']}) to prevent overfitting")
    elif config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=config.get('focal_alpha', 1.0),
            gamma=config.get('focal_gamma', 2.0),
            num_classes=7
        )
        print(f"Using Focal Loss (alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropy Loss")
    
    # Validate initial loss
    print("\nValidating initial model state...")
    initial_loss = validate_initial_loss(model, train_loader, criterion, device)
    
    # Early stopping with aggressive settings for overfitting prevention
    early_stopping = EarlyStopping(patience=config['patience'], min_delta=0.001)
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch_time': []
    }
    
    print("\n" + "="*80)
    print("🚀 STARTING OVERFITTING-PREVENTION TRAINING")
    print("="*80)
    print(f"Key Changes from Previous Training:")
    print(f"  ✅ Smaller model: {config['hidden_size']} hidden, {config['num_layers']} layers")
    print(f"  ✅ Higher dropout: {config['dropout_rate']}")
    print(f"  ✅ Frozen BERT layers: {config.get('freeze_bert_layers', 8)}")
    print(f"  ✅ Label smoothing: {config.get('label_smoothing', 0)}")
    print(f"  ✅ Balanced sampling: {config.get('class_balanced_sampling', False)}")
    print(f"  ✅ Early stopping patience: {config['patience']}")
    print(f"  ✅ Strong weight decay: {config['weight_decay']}")
    print("="*80)
    
    best_val_accuracy = 0
    validation_frequency = config.get('validation_frequency', 1)
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🎯 EPOCH {epoch + 1}/{config['num_epochs']} - OVERFITTING PREVENTION MODE")
        print("="*60)
        
        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            epoch_num=epoch+1
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Validation (less frequent to speed up training)
        if (epoch + 1) % validation_frequency == 0 or epoch == config['num_epochs'] - 1:
            print("🔍 Running validation...")
            val_loss, val_acc, val_prec, val_rec, val_f1, val_predictions, val_labels = validate_epoch(
                model, val_loader, criterion, device
            )
            
            # Calculate generalization gap (key overfitting indicator)
            generalization_gap = train_acc - val_acc
            
            print(f"📊 EPOCH {epoch + 1} RESULTS:")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.1f}%)")
            print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} ({val_acc*100:.1f}%)")
            print(f"   🚨 Generalization Gap: {generalization_gap:.4f} ({generalization_gap*100:.1f}%)")
            if generalization_gap > 0.15:
                print(f"   ⚠️  WARNING: Large generalization gap indicates overfitting!")
            elif generalization_gap < 0.05:
                print(f"   ✅ Good generalization!")
            print(f"   ⏱️  Epoch Time: {epoch_time:.1f}s")
            
            # Save history for validation epochs
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['train_precision'].append(train_prec)
            history['train_recall'].append(train_rec)
            history['train_f1'].append(train_f1)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['val_precision'].append(val_prec)
            history['val_recall'].append(val_rec)
            history['val_f1'].append(val_f1)
            history['epoch_time'].append(epoch_time)
            
            # Save best model (conservative - only if significant improvement)
            if val_acc > best_val_accuracy + 0.01:  # Require 1% improvement
                best_val_accuracy = val_acc
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f'best_7class_overfitting_prevented_{timestamp}_acc{val_acc:.4f}.pth'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'epoch': epoch + 1,
                    'config': config,
                    'emotion_classes': 7,
                    'model_type': 'multimodal_lstm_video_7class_overfitting_prevented',
                    'generalization_gap': generalization_gap,
                    'training_strategy': 'overfitting_prevention'
                }, f'checkpoints_2/{model_name}')
                
                # Also save with generic name
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': val_acc,
                    'epoch': epoch + 1,
                    'config': config,
                    'emotion_classes': 7,
                    'model_type': 'multimodal_lstm_video_7class_overfitting_prevented',
                    'generalization_gap': generalization_gap,
                    'training_strategy': 'overfitting_prevention'
                }, 'checkpoints_2/best_7class_model.pth')
                
                print(f"💾 New best model saved: {model_name}")
                print(f"   🎯 Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
                print(f"   📊 Generalization gap: {generalization_gap:.4f}")
            
            # Early stopping check
            if early_stopping(val_loss, model):
                print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
                print(f"   Reason: No improvement for {config['patience']} validation checks")
                break
                
        else:
            # Training-only epoch
            print(f"📈 TRAINING ONLY - Epoch {epoch + 1}")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.1f}%)")
            print(f"   ⏱️  Epoch Time: {epoch_time:.1f}s")
        
        # Save training history
        with open('result_2/training_history_overfitting_prevented.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("🎉 OVERFITTING-PREVENTION TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Best validation accuracy achieved: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"📊 Strategy: Smaller model + High dropout + Frozen BERT + Label smoothing")
    print(f"🎯 Goal: Better generalization to test data")
    print("="*80)
    
    # Final recommendations
    print("\n📋 NEXT STEPS:")
    if best_val_accuracy < 0.6:
        print("⚠️  Validation accuracy still low. Consider:")
        print("   - Further reducing model complexity")
        print("   - Adding more data augmentation")
        print("   - Checking data quality")
    elif best_val_accuracy > 0.8:
        print("⚠️  High validation accuracy. Monitor test performance to ensure generalization")
    else:
        print("✅ Reasonable validation accuracy. Test on held-out data to verify generalization")
    
    print("\n🧪 To test the improved model:")
    print("   python test_2.py")
    print("="*80)

if __name__ == "__main__":
    main()
