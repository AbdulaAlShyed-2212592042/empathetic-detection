"""
Test Improved Video Text Metadata Model
Evaluates the improved multimodal model on test data with enhanced architecture
Specifically designed for testing the improved model checkpoint
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
    """Enhanced TimeSformer video preprocessing pipeline"""
    
    def __init__(self, num_frames=8, frame_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.num_frames = num_frames
        self.frame_size = frame_size  # Enhanced to 224 for better performance
        self.mean = mean
        self.std = std
        
        self.transform = Compose([
            Resize((frame_size, frame_size)),
            Normalize(mean=mean, std=std)
        ])
    
    def extract_frames(self, video_path):
        """Optimized frame extraction for enhanced model"""
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
            
            # Enhanced frame sampling for better temporal coverage
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
    """Enhanced dataset for improved model testing"""
    
    def __init__(self, data_path, video_dir, tokenizer, video_transform=None, max_length=512, max_dialogue_length=10):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.video_transform = video_transform or VideoTransform()
        self.max_length = max_length
        self.max_dialogue_length = max_dialogue_length
        
        # Emotion mapping (same as training)
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
        """Get label without processing other data"""
        item = self.data[idx]
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        emotion_label = self.emotion_to_id.get(mapped_emotion, 0)
        return emotion_label

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract turn data
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        
        # Get emotion label using emotion mapping
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        emotion_label = self.emotion_to_id.get(mapped_emotion, 0)
        
        # Extract dialogue sequence
        dialogue = turn.get('dialogue', [])
        
        # Process sequential dialogue data
        dialogue_texts = []
        dialogue_video_features = []
        dialogue_roles = []
        dialogue_indices = []
        
        for utt in dialogue[:self.max_dialogue_length]:
            # Text
            text = utt.get('text', '')
            dialogue_texts.append(text)
            
            # Video processing
            video_name = utt.get('video_name', None)
            if video_name:
                video_path = os.path.join(self.video_dir, video_name)
                if os.path.exists(video_path):
                    video_features = self.video_transform.extract_frames(video_path)
                else:
                    print(f"Warning: Video not found: {video_path}")
                    video_features = torch.zeros(3, 8, 224, 224)  # Enhanced size
            else:
                video_features = torch.zeros(3, 8, 224, 224)  # Enhanced size
            dialogue_video_features.append(video_features)
            
            # Role and index
            role = utt.get('role', 'speaker')
            dialogue_roles.append(self.role_to_id.get(role, 0))
            dialogue_indices.append(utt.get('index', 0))
        
        # Pad sequences to max_dialogue_length
        while len(dialogue_texts) < self.max_dialogue_length:
            dialogue_texts.append('[EMPTY]')
            dialogue_video_features.append(torch.zeros(3, 8, 224, 224))  # Enhanced size
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
        
        # Extract metadata
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
        
        # Map to vocabulary IDs with bounds checking
        event_scenario_id = min(self.event_scenario_vocab.get(event_scenario, 0), 
                               getattr(self, 'max_event_scenario_id', len(self.event_scenario_vocab)))
        emotion_cause_id = min(self.emotion_cause_vocab.get(emotion_cause, 0),
                              getattr(self, 'max_emotion_cause_id', len(self.emotion_cause_vocab)))
        goal_response_id = min(self.goal_response_vocab.get(goal_to_response, 0),
                              getattr(self, 'max_goal_response_id', len(self.goal_response_vocab)))
        topic_id = min(self.topic_vocab.get(topic, 0),
                      getattr(self, 'max_topic_id', len(self.topic_vocab)))
        
        # Create metadata tensor
        metadata_features = torch.tensor([
            speaker_age, speaker_gender, speaker_timbre, speaker_id,
            listener_age, listener_gender, listener_timbre, listener_id,
            event_scenario_id, emotion_cause_id, goal_response_id, topic_id
        ], dtype=torch.long)
        
        # Ensure consistent video dimensions
        consistent_video_features = []
        target_size = (3, 8, 224, 224)  # Enhanced size
        
        for i, video_feat in enumerate(dialogue_video_features):
            if video_feat.shape != target_size:
                video_feat = torch.zeros(*target_size)
            consistent_video_features.append(video_feat)
        
        # Stack video features
        dialogue_video_tensor = torch.stack(consistent_video_features)
        
        return {
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'dialogue_input_ids': torch.stack([enc['input_ids'] for enc in dialogue_encodings]),
            'dialogue_attention_mask': torch.stack([enc['attention_mask'] for enc in dialogue_encodings]),
            'dialogue_video': dialogue_video_tensor,
            'dialogue_roles': torch.tensor(dialogue_roles, dtype=torch.long),
            'dialogue_indices': torch.tensor(dialogue_indices, dtype=torch.long),
            'metadata': metadata_features,
            'label': torch.tensor(emotion_label, dtype=torch.long),
            'conversation_id': item.get('conversation_id', ''),
            'raw_emotion': raw_emotion,
            'sequence_length': torch.tensor(min(len(dialogue), self.max_dialogue_length), dtype=torch.long),
            'valid_sample': torch.tensor(1 if emotion_label != -1 else 0, dtype=torch.long)
        }

# Enhanced TimeSformer Encoder (matching improved model exactly)
class EnhancedTimeSformerEncoder(nn.Module):
    """Enhanced TimeSformer-style video encoder with better architecture"""
    
    def __init__(self, num_frames=8, frame_size=224, patch_size=32, embed_dim=384, num_heads=6, num_layers=4, dropout_rate=0.1):
        super(EnhancedTimeSformerEncoder, self).__init__()
        
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (frame_size // patch_size) ** 2
        
        # Enhanced patch embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Enhanced positional embeddings (matching training script)
        self.pos_embed_spatial = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.pos_embed_temporal = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)
        
        # Enhanced transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced output projection (matching training script)
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization (matching training script)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        
        sequence_features = []
        for i in range(seq_len):
            video = x[:, i]  # [batch_size, 3, 8, 224, 224]
            
            # Process frames
            frame_features = []
            for t in range(self.num_frames):
                frame = video[:, :, t, :, :]
                frame_patches = self.patch_embed(frame)  # [batch_size, embed_dim, 7, 7]
                frame_patches = frame_patches.flatten(2).transpose(1, 2)  # [batch_size, 49, embed_dim]
                frame_patches = frame_patches + self.pos_embed_spatial
                frame_features.append(frame_patches)
            
            video_patches = torch.stack(frame_features, dim=1)  # [batch_size, 8, 49, embed_dim]
            video_patches = video_patches.reshape(batch_size, self.num_frames * self.num_patches, self.embed_dim)
            
            # Add temporal positional embedding
            temporal_pos = self.pos_embed_temporal.repeat_interleave(self.num_patches, dim=1)
            video_patches = video_patches + temporal_pos
            
            # Apply transformer
            transformed = self.transformer(video_patches)
            
            # Enhanced pooling and layer norm (matching training script)
            video_features = transformed.mean(dim=1)
            video_features = self.layer_norm(video_features)
            video_features = self.output_projection(video_features)
            
            sequence_features.append(video_features)
        
        sequence_output = torch.stack(sequence_features, dim=1)
        return sequence_output

# Enhanced Multimodal LSTM Model (matching improved architecture exactly)
class ImprovedMultimodalLSTMVideoModel(nn.Module):
    """Improved multimodal LSTM model with enhanced architecture"""
    
    def __init__(self, dataset, num_classes=7, hidden_size=256, num_layers=2, dropout_rate=0.3, freeze_bert_layers=6):
        super(ImprovedMultimodalLSTMVideoModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Enhanced text encoder
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size
        
        # Freeze fewer BERT layers for better performance
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.text_encoder.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"Froze first {freeze_bert_layers} BERT layers")
        
        # Enhanced video encoder (matching training script exactly)
        self.video_encoder = EnhancedTimeSformerEncoder(
            num_frames=8,
            frame_size=224,
            embed_dim=384,
            num_heads=6,
            num_layers=4,
            dropout_rate=dropout_rate
        )
        video_dim = 384
        
        # Enhanced metadata embeddings (matching training script exactly)
        self.age_embedding = nn.Embedding(4, 16)
        self.gender_embedding = nn.Embedding(2, 8)
        self.timbre_embedding = nn.Embedding(3, 8)
        self.role_embedding = nn.Embedding(3, 16)
        
        self.speaker_id_embedding = nn.Embedding(100, 32)
        self.listener_id_embedding = nn.Embedding(100, 32)
        
        # Chain of empathy embeddings (matching training script exactly)
        vocab_sizes = {
            'event_scenario': len(dataset.event_scenario_vocab) + 1,
            'emotion_cause': len(dataset.emotion_cause_vocab) + 1,
            'goal_response': len(dataset.goal_response_vocab) + 1,
            'topic': len(dataset.topic_vocab) + 1
        }
        
        self.event_scenario_embedding = nn.Embedding(vocab_sizes['event_scenario'], 32)
        self.emotion_cause_embedding = nn.Embedding(vocab_sizes['emotion_cause'], 32)
        self.goal_response_embedding = nn.Embedding(vocab_sizes['goal_response'], 32)
        self.topic_embedding = nn.Embedding(vocab_sizes['topic'], 32)
        
        # Enhanced context processor (matching training script exactly)
        self.context_processor = nn.Sequential(
            nn.Linear(text_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128)
        )
        
        # Enhanced LSTM layers (matching training script exactly)
        text_lstm_input_size = text_dim + 16  # text + role embeddings
        self.text_lstm = nn.LSTM(
            input_size=text_lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.video_lstm = nn.LSTM(
            input_size=video_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Metadata processing (matching training script exactly)
        metadata_dim = 16 + 8 + 8 + 32 + 16 + 8 + 8 + 32 + 32 + 32 + 32 + 32  # 256
        
        self.metadata_temporal_processor = nn.Sequential(
            nn.Linear(metadata_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Enhanced fusion LSTM (matching training script exactly)
        fused_input_size = hidden_size * 2 + hidden_size * 2 + hidden_size  # text + video + metadata
        self.fusion_lstm = nn.LSTM(
            input_size=fused_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Enhanced attention mechanisms (matching training script exactly)
        self.text_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.video_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Enhanced metadata processor (matching training script exactly)
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Enhanced final classifier (matching training script exactly)
        final_dim = 128 + hidden_size * 2 + 64  # context + fusion_lstm + metadata
        
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=final_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Enhanced classifier with better architecture (matching training script exactly)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self._init_weights()
        
    def _init_weights(self):
        """Enhanced weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        
    def forward(self, context_input_ids, context_attention_mask, dialogue_input_ids, 
                dialogue_attention_mask, dialogue_video, dialogue_roles, 
                dialogue_indices, metadata, sequence_length):
        batch_size, seq_len = dialogue_input_ids.shape[:2]
        
        # Enhanced context processing
        context_outputs = self.text_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_features = self.context_processor(context_outputs.last_hidden_state[:, 0, :])
        
        # Enhanced dialogue processing
        text_features_sequence = []
        
        for i in range(seq_len):
            utt_outputs = self.text_encoder(
                input_ids=dialogue_input_ids[:, i, :],
                attention_mask=dialogue_attention_mask[:, i, :]
            )
            utt_text_features = utt_outputs.last_hidden_state[:, 0, :]
            utt_role_features = self.role_embedding(dialogue_roles[:, i])
            utt_combined_text = torch.cat([utt_text_features, utt_role_features], dim=1)
            text_features_sequence.append(utt_combined_text)
        
        text_sequence = torch.stack(text_features_sequence, dim=1)
        video_sequence_features = self.video_encoder(dialogue_video)
        
        # Enhanced LSTM processing
        text_lstm_out, _ = self.text_lstm(text_sequence)
        video_lstm_out, _ = self.video_lstm(video_sequence_features)
        
        # Enhanced attention (using batch_first=True)
        text_attended, _ = self.text_attention(text_lstm_out, text_lstm_out, text_lstm_out)
        video_attended, _ = self.video_attention(video_lstm_out, video_lstm_out, video_lstm_out)
        
        # Enhanced metadata processing
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
        
        metadata_temporal = self.metadata_temporal_processor(metadata_combined)
        metadata_sequence = metadata_temporal.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Enhanced fusion
        fused_features = torch.cat([text_attended, video_attended, metadata_sequence], dim=-1)
        fusion_lstm_out, _ = self.fusion_lstm(fused_features)
        
        fusion_attended, _ = self.fusion_attention(fusion_lstm_out, fusion_lstm_out, fusion_lstm_out)
        
        # Enhanced final features
        dialogue_final_features = []
        for i in range(batch_size):
            seq_len_i = sequence_length[i].item() - 1
            seq_len_i = max(0, min(seq_len_i, seq_len - 1))
            dialogue_final_features.append(fusion_attended[i, seq_len_i, :])
        dialogue_final_features = torch.stack(dialogue_final_features)
        
        metadata_final_features = self.metadata_processor(metadata_combined)
        
        final_features = torch.cat([context_features, dialogue_final_features, metadata_final_features], dim=1)
        
        # Enhanced cross-modal attention (using batch_first=True)
        final_features_expanded = final_features.unsqueeze(1)
        cross_attended, _ = self.cross_modal_attention(final_features_expanded, final_features_expanded, final_features_expanded)
        final_features = cross_attended.squeeze(1)
        
        final_features = self.dropout(final_features)
        logits = self.classifier(final_features)
        
        return logits

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Improved Model Test Set')
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
    ax.set_title('Per-Class Performance Metrics - Improved Model Test Set')
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
    """Evaluate improved model on test set"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_raw_emotions = []
    all_conversation_ids = []
    
    print("Evaluating improved model on test set...")
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
    """Save comprehensive test results for improved model"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save main results
    test_results = {
        'timestamp': timestamp,
        'model_type': 'improved_multimodal_lstm',
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
    results_path = os.path.join(save_dir, f'improved_test_results_{timestamp}.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    print(f"Improved test results saved to: {results_path}")
    
    # Save plots
    cm_path = os.path.join(save_dir, f'improved_confusion_matrix_{timestamp}.png')
    plot_confusion_matrix(results['labels'], results['predictions'], class_names, cm_path)
    
    metrics_path = os.path.join(save_dir, f'improved_per_class_metrics_{timestamp}.png')
    plot_per_class_metrics(results['precision'], results['recall'], results['f1'], class_names, metrics_path)
    
    # Save summary
    summary_path = os.path.join(save_dir, f'improved_test_summary_{timestamp}.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("IMPROVED VIDEO TEXT METADATA MODEL - TEST EVALUATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Test Date: {timestamp}\n")
        f.write(f"Model: Improved Multimodal LSTM\n")
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
    
    print(f"Improved test summary saved to: {summary_path}")
    
    return results_path, cm_path, metrics_path, summary_path

def main():
    # Enhanced configuration for improved model testing
    config = {
        'batch_size': 6,
        'max_length': 256,
        'max_dialogue_length': 10,
        'hidden_size': 256,
        'num_layers': 2,
        'dropout_rate': 0.3,
        'freeze_bert_layers': 6
    }
    
    # Device configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
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
    print("Loading test dataset for improved model...")
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
    
    # Initialize improved model
    print("Initializing improved model...")
    model = ImprovedMultimodalLSTMVideoModel(
        test_dataset,
        num_classes=7,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        freeze_bert_layers=config['freeze_bert_layers']
    ).to(device)
    
    # Load improved model checkpoint
    checkpoint_path = 'checkpoints_2/best_improved_model_20250825_093314_acc0.9591.pth'
    
    if os.path.exists(checkpoint_path):
        print(f"Loading IMPROVED VIDEO+TEXT+METADATA model from: {checkpoint_path}")
        try:
            # Load with weights_only=False for compatibility with newer checkpoint format
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state_dict = checkpoint['model_state_dict']
                print("✅ Improved checkpoint loaded successfully!")
                print(f"   📊 Training info: Epoch {checkpoint.get('epoch', 'N/A')}, "
                      f"Val Acc: {checkpoint.get('val_accuracy', 'N/A'):.4f}")
            else:
                model_state_dict = checkpoint
                print("✅ Improved checkpoint loaded as direct state_dict!")
                
            # Handle vocabulary size matching for improved model
            event_scenario_size = model_state_dict['event_scenario_embedding.weight'].shape[0]
            emotion_cause_size = model_state_dict['emotion_cause_embedding.weight'].shape[0]
            goal_response_size = model_state_dict['goal_response_embedding.weight'].shape[0]
            topic_size = model_state_dict['topic_embedding.weight'].shape[0]
            
            print(f"📊 Improved model vocabulary sizes:")
            print(f"   Event scenarios: {event_scenario_size}")
            print(f"   Emotion causes: {emotion_cause_size}")
            print(f"   Goal responses: {goal_response_size}")
            print(f"   Topics: {topic_size}")
            
            # Update model embeddings to match checkpoint
            model.event_scenario_embedding = nn.Embedding(event_scenario_size, 32).to(device)
            model.emotion_cause_embedding = nn.Embedding(emotion_cause_size, 32).to(device)
            model.goal_response_embedding = nn.Embedding(goal_response_size, 32).to(device)
            model.topic_embedding = nn.Embedding(topic_size, 32).to(device)
            
            # Load state dict
            model.load_state_dict(model_state_dict)
            
            # Update dataset vocabulary limits
            test_dataset.max_event_scenario_id = event_scenario_size - 1
            test_dataset.max_emotion_cause_id = emotion_cause_size - 1
            test_dataset.max_goal_response_id = goal_response_size - 1
            test_dataset.max_topic_id = topic_size - 1
            
            print("✅ Improved Video+Text+Metadata model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading improved model: {e}")
            return
    else:
        print(f"❌ ERROR: Improved model checkpoint not found at: {checkpoint_path}")
        print("Please run the improved training script first.")
        return
    
    # Define emotion class names
    class_names = ["happy", "surprised", "angry", "fear", "sad", "disgusted", "contempt"]
    
    print("\n" + "="*80)
    print("🧪 TESTING IMPROVED MODEL")
    print("="*80)
    print(f"Model: Improved Multimodal LSTM (Video + Text + Metadata)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Emotion classes: {class_names}")
    print(f"Device: {device}")
    print("="*80)
    
    # Evaluate improved model
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    print("\nSaving improved model test results...")
    results_path, cm_path, metrics_path, summary_path = save_test_results(
        results, class_names, timestamp, 'result_2'
    )
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 IMPROVED MODEL TEST EVALUATION COMPLETED!")
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
    
    # Performance comparison note
    print("\n🔍 PERFORMANCE ANALYSIS:")
    print("-"*50)
    if results['accuracy'] > 0.8:
        print("✅ Excellent performance! The improved model shows significant enhancement.")
    elif results['accuracy'] > 0.6:
        print("✅ Good performance! Notable improvement over baseline model.")
    elif results['accuracy'] > 0.4:
        print("⚠️  Moderate performance. Some improvement but room for further enhancement.")
    else:
        print("⚠️  Performance needs improvement. Consider further architectural changes.")
    
    # Best and worst performing classes
    best_f1_idx = np.argmax(results['f1'])
    worst_f1_idx = np.argmin(results['f1'])
    
    print(f"Best performing class:  {class_names[best_f1_idx]} (F1: {results['f1'][best_f1_idx]:.3f})")
    print(f"Worst performing class: {class_names[worst_f1_idx]} (F1: {results['f1'][worst_f1_idx]:.3f})")
    
    print("\n✅ Improved model test evaluation completed successfully!")

if __name__ == "__main__":
    main()
