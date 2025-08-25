"""
Test Video Text Metadata Model
Evaluates the trained multimodal model on test data with video, text, and metadata features
Saves comprehensive test results to result_2 directory
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import copy

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")
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
        self.frame_size = frame_size
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
                print(f"Warning: Could not open video {video_path}")
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                print(f"Warning: No frames in video {video_path}")
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
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (self.frame_size, self.frame_size))
                    frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                    frames.append(frame_tensor)
            
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
            print(f"Error processing video {video_path}: {e}")
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
        
        # Emotion mapping from training script
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
        
        print(f"Loaded {len(self.data)} test samples from {data_path}")
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
                if os.path.exists(video_path):
                    video_features = self.video_transform.extract_frames(video_path)
                else:
                    print(f"Warning: Video not found: {video_path}")
                    video_features = torch.zeros(3, 8, 192, 192)
            else:
                video_features = torch.zeros(3, 8, 192, 192)
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
        
        # Map to vocabulary IDs with proper bounds checking
        event_scenario_id = min(self.event_scenario_vocab.get(event_scenario, 0), 
                               getattr(self, 'max_event_scenario_id', len(self.event_scenario_vocab)))
        emotion_cause_id = min(self.emotion_cause_vocab.get(emotion_cause, 0),
                              getattr(self, 'max_emotion_cause_id', len(self.emotion_cause_vocab)))
        goal_response_id = min(self.goal_response_vocab.get(goal_to_response, 0),
                              getattr(self, 'max_goal_response_id', len(self.goal_response_vocab)))
        topic_id = min(self.topic_vocab.get(topic, 0),
                      getattr(self, 'max_topic_id', len(self.topic_vocab)))
        
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
                video_feat = torch.zeros(*target_size)
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

# Import model classes from training script
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
                frame_patches = self.patch_embed(frame)  # [batch_size, embed_dim, 6, 6]
                frame_patches = frame_patches.flatten(2).transpose(1, 2)  # [batch_size, 36, embed_dim]
                frame_patches = frame_patches + self.pos_embed_spatial
                frame_features.append(frame_patches)
            
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

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")

def plot_per_class_metrics(precision, recall, f1, class_names, save_path):
    """Plot and save per-class metrics"""
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Emotion Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics - Test Set')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Per-class metrics plot saved to: {save_path}")

def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test set and return comprehensive metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_raw_emotions = []
    all_conversation_ids = []
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
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
            
            # Forward pass
            with torch.cuda.amp.autocast():
                logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                             dialogue_attention_mask, dialogue_video, dialogue_roles, 
                             dialogue_indices, metadata, sequence_length)
            
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_raw_emotions.extend(batch['raw_emotion'])
            all_conversation_ids.extend(batch['conversation_id'])
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Overall metrics
    overall_precision = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )[0]
    overall_recall = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )[1]
    overall_f1 = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )[2]
    
    # Classification report
    report = classification_report(
        all_labels, all_predictions, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1': overall_f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'raw_emotions': all_raw_emotions,
        'conversation_ids': all_conversation_ids,
        'classification_report': report
    }

def save_test_results(results, class_names, timestamp, save_dir):
    """Save comprehensive test results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results
    test_results = {
        'timestamp': timestamp,
        'overall_metrics': {
            'accuracy': float(results['accuracy']),
            'precision': float(results['overall_precision']),
            'recall': float(results['overall_recall']),
            'f1_score': float(results['overall_f1'])
        },
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(results['precision'][i]),
                'recall': float(results['recall'][i]),
                'f1_score': float(results['f1'][i]),
                'support': int(results['support'][i])
            }
            for i in range(len(class_names))
        },
        'classification_report': results['classification_report'],
        'detailed_predictions': [
            {
                'conversation_id': results['conversation_ids'][i],
                'true_label': int(results['labels'][i]),
                'predicted_label': int(results['predictions'][i]),
                'true_emotion': class_names[results['labels'][i]],
                'predicted_emotion': class_names[results['predictions'][i]],
                'raw_emotion': results['raw_emotions'][i],
                'correct': bool(results['labels'][i] == results['predictions'][i])
            }
            for i in range(len(results['labels']))
        ]
    }
    
    # Save to JSON
    results_path = os.path.join(save_dir, f'test_results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"Test results saved to: {results_path}")
    
    # Save confusion matrix plot
    cm_path = os.path.join(save_dir, f'confusion_matrix_{timestamp}.png')
    plot_confusion_matrix(results['labels'], results['predictions'], class_names, cm_path)
    
    # Save per-class metrics plot
    metrics_path = os.path.join(save_dir, f'per_class_metrics_{timestamp}.png')
    plot_per_class_metrics(results['precision'], results['recall'], results['f1'], class_names, metrics_path)
    
    # Save summary text
    summary_path = os.path.join(save_dir, f'test_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("VIDEO TEXT METADATA MODEL - TEST EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Test Samples: {len(results['labels'])}\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['overall_precision']:.4f}\n")
        f.write(f"Recall:    {results['overall_recall']:.4f}\n")
        f.write(f"F1-Score:  {results['overall_f1']:.4f}\n\n")
        
        f.write("PER-CLASS METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}\n")
        f.write("-"*60 + "\n")
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name:<12} {results['precision'][i]:<10.4f} "
                   f"{results['recall'][i]:<10.4f} {results['f1'][i]:<10.4f} {results['support'][i]:<8}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write("="*80 + "\n")
        f.write(classification_report(results['labels'], results['predictions'], 
                                    target_names=class_names, zero_division=0))
    
    print(f"Test summary saved to: {summary_path}")
    
    return results_path, cm_path, metrics_path, summary_path

def main():
    # Configuration
    config = {
        'batch_size': 8,
        'max_length': 256,
        'max_dialogue_length': 8,
        'hidden_size': 128,
        'num_layers': 1,
        'dropout_rate': 0.5,
        'freeze_bert_layers': 8
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
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    # Create results directory
    os.makedirs('result_2', exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = MultimodalSequentialVideoDataset(
        'json/mapped_test_data_video_aligned.json',
        'data/train_video/video_v5_0',
        tokenizer,
        video_transform=VideoTransform(),
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length']
    )
    
    print(f"Test dataset loaded: {len(test_dataset)} samples")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalLSTMVideoModel(
        test_dataset,
        num_classes=7,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        freeze_bert_layers=config['freeze_bert_layers']
    ).to(device)
    
    # Load trained model (VIDEO+TEXT+METADATA model) - Use latest overfitting-prevented checkpoint
    checkpoint_path = 'checkpoints_2/best_improved_model_20250825_093314_acc0.9591.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading trained VIDEO+TEXT+METADATA model from: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # Check if checkpoint is a dictionary with model_state_dict key
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                print("✅ Checkpoint loaded successfully!")
                print(f"   📊 Training info: Epoch {checkpoint.get('epoch', 'N/A')}, "
                      f"Val Acc: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
            else:
                # Try loading as direct state_dict
                model_state_dict = checkpoint
                print("✅ Checkpoint loaded as direct state_dict!")
                
            # Get vocabulary sizes from the checkpoint to match training
            event_scenario_size = model_state_dict['event_scenario_embedding.weight'].shape[0]
            emotion_cause_size = model_state_dict['emotion_cause_embedding.weight'].shape[0]
            goal_response_size = model_state_dict['goal_response_embedding.weight'].shape[0]
            topic_size = model_state_dict['topic_embedding.weight'].shape[0]
            
            print(f"📊 Checkpoint vocabulary sizes:")
            print(f"   Event scenarios: {event_scenario_size}")
            print(f"   Emotion causes: {emotion_cause_size}")
            print(f"   Goal responses: {goal_response_size}")
            print(f"   Topics: {topic_size}")
            
            # Update model embeddings to match checkpoint sizes
            model.event_scenario_embedding = nn.Embedding(event_scenario_size, 16).to(device)
            model.emotion_cause_embedding = nn.Embedding(emotion_cause_size, 16).to(device)
            model.goal_response_embedding = nn.Embedding(goal_response_size, 16).to(device)
            model.topic_embedding = nn.Embedding(topic_size, 16).to(device)
            
            # Now load the state dict
            model.load_state_dict(model_state_dict)
            
            # Update test dataset vocabulary limits to match training
            test_dataset.max_event_scenario_id = event_scenario_size - 1
            test_dataset.max_emotion_cause_id = emotion_cause_size - 1
            test_dataset.max_goal_response_id = goal_response_size - 1
            test_dataset.max_topic_id = topic_size - 1
            
            print("✅ Video+Text+Metadata model loaded successfully with matched vocabulary sizes!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("The checkpoint might be from a different model architecture.")
            return
    else:
        print(f"❌ ERROR: Video+Text+Metadata model checkpoint not found at: {checkpoint_path}")
        print("\n📝 NOTE: The checkpoint in 'checkpoints_1/' is from a different model (likely audio-based)")
        print("   and is NOT compatible with this video+text+metadata model.")
        print("\n🚀 To generate the correct checkpoint:")
        print("   1. Run: python train_video_text_metadata.py")
        print("   2. Wait for training to complete")
        print("   3. Then run: python test_video.py")
        return
    
    # Define emotion class names
    class_names = ["happy", "surprised", "angry", "fear", "sad", "disgusted", "contempt"]
    
    print("\n" + "="*80)
    print("🧪 STARTING TEST EVALUATION")
    print("="*80)
    print(f"Model: Multimodal LSTM (Video + Text + Metadata)")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Emotion classes: {class_names}")
    print(f"Device: {device}")
    print("="*80)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    print("\nSaving test results...")
    results_path, cm_path, metrics_path, summary_path = save_test_results(
        results, class_names, timestamp, 'result_2'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 TEST EVALUATION COMPLETED!")
    print("="*80)
    print(f"Overall Test Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Overall Precision:     {results['overall_precision']:.4f}")
    print(f"Overall Recall:        {results['overall_recall']:.4f}")
    print(f"Overall F1-Score:      {results['overall_f1']:.4f}")
    print("\nPer-Class Performance:")
    print("-"*50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<12}: Prec={results['precision'][i]:.3f}, "
              f"Rec={results['recall'][i]:.3f}, F1={results['f1'][i]:.3f}, "
              f"Support={results['support'][i]}")
    
    print("\n📁 Results saved to:")
    print(f"  📊 Test metrics: {results_path}")
    print(f"  📈 Confusion matrix: {cm_path}")
    print(f"  📉 Per-class metrics: {metrics_path}")
    print(f"  📋 Summary report: {summary_path}")
    print("="*80)
    
    # Additional analysis
    print("\n📊 ADDITIONAL ANALYSIS:")
    print("-"*50)
    
    # Best and worst performing classes
    best_f1_idx = np.argmax(results['f1'])
    worst_f1_idx = np.argmin(results['f1'])
    
    print(f"Best performing class:  {class_names[best_f1_idx]} (F1: {results['f1'][best_f1_idx]:.3f})")
    print(f"Worst performing class: {class_names[worst_f1_idx]} (F1: {results['f1'][worst_f1_idx]:.3f})")
    
    # Correct vs incorrect predictions
    correct_predictions = sum(1 for i in range(len(results['labels'])) 
                            if results['labels'][i] == results['predictions'][i])
    total_predictions = len(results['labels'])
    
    print(f"Correct predictions: {correct_predictions}/{total_predictions}")
    print(f"Incorrect predictions: {total_predictions - correct_predictions}/{total_predictions}")
    
    print("\n✅ Test evaluation completed successfully!")

if __name__ == "__main__":
    main() 
