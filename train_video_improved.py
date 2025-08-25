"""
Improved Multi-modal Empathetic Response Training
Enhanced architecture to address low performance issues while maintaining generalization
Key improvements:
- Larger but well-regularized model
- Better fusion strategies
- Enhanced attention mechanisms
- Improved loss functions
- Better data preprocessing
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import warnings
import random
import time
import copy
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
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize

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
        self.frame_size = frame_size  # Increased back to 224 for better features
        self.mean = mean
        self.std = std
        
        self.transform = Compose([
            Resize((frame_size, frame_size)),
            Normalize(mean=mean, std=std)
        ])
    
    def extract_frames(self, video_path):
        """Enhanced frame extraction with better quality"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            # Better frame sampling strategy
            if total_frames <= self.num_frames:
                frame_indices = list(range(total_frames))
            else:
                # Use uniform sampling across the entire video
                frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
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
            
            # Stack and normalize
            video_tensor = torch.stack(frames[:self.num_frames], dim=1)  # [C, T, H, W]
            
            # Vectorized normalization
            mean_tensor = torch.tensor(self.mean).view(3, 1, 1, 1)
            std_tensor = torch.tensor(self.std).view(3, 1, 1, 1)
            video_tensor = (video_tensor - mean_tensor) / std_tensor
            
            return video_tensor
            
        except Exception as e:
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)

class MultimodalSequentialVideoDataset(Dataset):
    """Enhanced dataset with better preprocessing and augmentation"""
    
    def __init__(self, data_path, video_dir, tokenizer, video_transform=None, max_length=512, max_dialogue_length=10, augment=False):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.video_transform = video_transform or VideoTransform()
        self.max_length = max_length
        self.max_dialogue_length = max_dialogue_length
        self.augment = augment
        
        # Enhanced emotion mapping with better coverage
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
        """Get label without processing other data"""
        item = self.data[idx]
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        emotion_label = self.emotion_to_id.get(mapped_emotion, 0)
        return emotion_label

    def augment_text(self, text):
        """Simple text augmentation for training"""
        if not self.augment or random.random() > 0.3:
            return text
        
        # Simple augmentations
        augmentations = [
            lambda x: x.lower(),  # lowercase
            lambda x: x.capitalize(),  # capitalize
            lambda x: x.replace('.', ' .'),  # add spaces around periods
            lambda x: x.replace('!', ' !'),  # add spaces around exclamations
        ]
        
        aug_func = random.choice(augmentations)
        return aug_func(text)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract turn data
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        
        # Get emotion label
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
            # Text with augmentation
            text = utt.get('text', '')
            text = self.augment_text(text)
            dialogue_texts.append(text)
            
            # Video processing
            video_name = utt.get('video_name', None)
            if video_name:
                video_path = os.path.join(self.video_dir, video_name)
                if os.path.exists(video_path):
                    video_features = self.video_transform.extract_frames(video_path)
                else:
                    video_features = torch.zeros(3, 8, 224, 224)  # Updated size
            else:
                video_features = torch.zeros(3, 8, 224, 224)
            dialogue_video_features.append(video_features)
            
            # Role and index
            role = utt.get('role', 'speaker')
            dialogue_roles.append(self.role_to_id.get(role, 0))
            dialogue_indices.append(utt.get('index', 0))
        
        # Pad sequences to max_dialogue_length
        while len(dialogue_texts) < self.max_dialogue_length:
            dialogue_texts.append('[EMPTY]')
            dialogue_video_features.append(torch.zeros(3, 8, 224, 224))
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
        
        # Context encoding with augmentation
        context = turn.get('context', '')
        if not context or context.strip() == '':
            context = '[NO CONTEXT]'
        context = self.augment_text(context)
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
        
        # Ensure all video features have consistent dimensions
        consistent_video_features = []
        target_size = (3, 8, 224, 224)  # Updated size
        
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
        
        # Patch embedding with better initialization
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Enhanced positional embeddings
        self.pos_embed_spatial = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.pos_embed_temporal = nn.Parameter(torch.randn(1, num_frames, embed_dim) * 0.02)
        
        # Enhanced transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation='gelu',  # Better activation
            batch_first=True,
            norm_first=True  # Pre-norm for better training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Enhanced output projection with residual
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Better weight initialization"""
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
            
            # Process each frame with improved approach
            frame_features = []
            for t in range(self.num_frames):
                frame = video[:, :, t, :, :]  # [batch_size, 3, 224, 224]
                frame_patches = self.patch_embed(frame)  # [batch_size, embed_dim, 7, 7]
                frame_patches = frame_patches.flatten(2).transpose(1, 2)  # [batch_size, 49, embed_dim]
                frame_patches = frame_patches + self.pos_embed_spatial
                frame_features.append(frame_patches)
            
            # Stack and process
            video_patches = torch.stack(frame_features, dim=1)  # [batch_size, 8, 49, embed_dim]
            video_patches = video_patches.reshape(batch_size, self.num_frames * self.num_patches, self.embed_dim)
            
            # Add temporal positional embedding
            temporal_pos = self.pos_embed_temporal.repeat_interleave(self.num_patches, dim=1)
            video_patches = video_patches + temporal_pos
            
            # Apply layer norm before transformer
            video_patches = self.layer_norm(video_patches)
            
            # Apply transformer
            transformed = self.transformer(video_patches)
            
            # Enhanced pooling strategy
            video_features = transformed.mean(dim=1)  # Global average pooling
            
            # Apply output projection with residual connection
            video_features_proj = self.output_projection(video_features)
            video_features = video_features + video_features_proj  # Residual connection
            
            sequence_features.append(video_features)
        
        sequence_output = torch.stack(sequence_features, dim=1)
        return sequence_output

class ImprovedMultimodalLSTMVideoModel(nn.Module):
    """Improved multimodal LSTM model with better architecture and capacity"""
    
    def __init__(self, dataset, num_classes=7, hidden_size=256, num_layers=2, dropout_rate=0.3, freeze_bert_layers=6):
        super(ImprovedMultimodalLSTMVideoModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Text encoder with partial freezing
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size
        
        # Freeze fewer BERT layers for better capacity
        if freeze_bert_layers > 0:
            for i, layer in enumerate(self.text_encoder.encoder.layer):
                if i < freeze_bert_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            print(f"Froze first {freeze_bert_layers} BERT layers")
        
        # Enhanced video encoder
        self.video_encoder = EnhancedTimeSformerEncoder(
            num_frames=8,
            frame_size=224,
            embed_dim=384,  # Increased capacity
            num_heads=6,
            num_layers=4,  # More layers
            dropout_rate=dropout_rate
        )
        video_dim = 384
        
        # Enhanced metadata embeddings
        self.age_embedding = nn.Embedding(4, 16)
        self.gender_embedding = nn.Embedding(2, 8)
        self.timbre_embedding = nn.Embedding(3, 8)
        self.role_embedding = nn.Embedding(3, 16)
        
        # Enhanced profile ID embeddings
        self.speaker_id_embedding = nn.Embedding(100, 32)
        self.listener_id_embedding = nn.Embedding(100, 32)
        
        # Enhanced chain of empathy embeddings
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
        
        # Enhanced context processor
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
        
        # Enhanced LSTM modules
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
        
        # Metadata processing
        metadata_dim = 16 + 8 + 8 + 32 + 16 + 8 + 8 + 32 + 32 + 32 + 32 + 32
        
        self.metadata_temporal_processor = nn.Sequential(
            nn.Linear(metadata_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Enhanced fusion LSTM
        fused_input_size = hidden_size * 2 + hidden_size * 2 + hidden_size  # text + video + metadata
        self.fusion_lstm = nn.LSTM(
            input_size=fused_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Enhanced attention mechanisms
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
        
        # Enhanced metadata processor
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
        
        # Enhanced final classifier
        final_dim = 128 + hidden_size * 2 + 64  # context + fusion_lstm + metadata
        
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=final_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Enhanced classifier with better architecture
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
        
        # Initialize weights
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
                        nn.init.constant_(param.data, 0)
        
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
            # Text features
            utt_outputs = self.text_encoder(
                input_ids=dialogue_input_ids[:, i, :],
                attention_mask=dialogue_attention_mask[:, i, :]
            )
            utt_text_features = utt_outputs.last_hidden_state[:, 0, :]
            
            # Role embedding
            utt_role_features = self.role_embedding(dialogue_roles[:, i])
            
            # Combine features
            utt_combined_text = torch.cat([utt_text_features, utt_role_features], dim=1)
            text_features_sequence.append(utt_combined_text)
        
        text_sequence = torch.stack(text_features_sequence, dim=1)
        
        # Enhanced video processing
        video_sequence_features = self.video_encoder(dialogue_video)
        
        # Enhanced LSTM processing
        text_lstm_out, _ = self.text_lstm(text_sequence)
        video_lstm_out, _ = self.video_lstm(video_sequence_features)
        
        # Enhanced attention
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
        
        # Get final features
        dialogue_final_features = []
        for i in range(batch_size):
            seq_len_i = sequence_length[i].item() - 1
            seq_len_i = max(0, min(seq_len_i, seq_len - 1))
            dialogue_final_features.append(fusion_attended[i, seq_len_i, :])
        dialogue_final_features = torch.stack(dialogue_final_features)
        
        # Final metadata processing
        metadata_final_features = self.metadata_processor(metadata_combined)
        
        # Enhanced final fusion
        final_features = torch.cat([context_features, dialogue_final_features, metadata_final_features], dim=1)
        
        # Cross-modal attention
        final_features_expanded = final_features.unsqueeze(1)
        cross_attended, _ = self.cross_modal_attention(final_features_expanded, final_features_expanded, final_features_expanded)
        final_features = cross_attended.squeeze(1)
        
        final_features = self.dropout(final_features)
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits

# Include utility classes from previous version
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
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

def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1 score"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler=None, 
                gradient_accumulation_steps=1, max_grad_norm=1.0, epoch_num=1):
    """Enhanced training loop"""
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
                scheduler.step()
                optimizer.zero_grad()
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
                scheduler.step()
                optimizer.zero_grad()
        
        total_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1

def validate_epoch(model, val_loader, criterion, device):
    """Enhanced validation loop"""
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

def main():
    # Enhanced configuration for better performance
    config = {
        'batch_size': 6,  # Increased batch size
        'gradient_accumulation_steps': 2,  # Effective batch size of 12
        'learning_rate': 2e-5,  # Slightly higher learning rate
        'num_epochs': 20,  # More epochs
        'patience': 5,  # More patience
        'dropout_rate': 0.3,  # Reduced dropout for better capacity
        'weight_decay': 0.01,  # Reduced weight decay
        'warmup_steps': 500,
        'max_length': 256,  
        'max_dialogue_length': 10,  # Increased context
        'hidden_size': 256,  # Larger model
        'num_layers': 2,  # More layers
        'max_grad_norm': 1.0,
        'use_focal_loss': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'use_mixed_precision': True,
        'freeze_bert_layers': 6,  # Freeze fewer layers
        'data_augmentation': True,
        'validation_frequency': 1  # Validate every epoch
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
    
    # Create directories
    os.makedirs('result_2', exist_ok=True)
    os.makedirs('checkpoints_2', exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load datasets with augmentation
    print("Loading enhanced datasets...")
    train_dataset = MultimodalSequentialVideoDataset(
        'json/mapped_train_data_video_aligned.json',
        'data/train_video/video_v5_0',
        tokenizer,
        video_transform=VideoTransform(),
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length'],
        augment=config.get('data_augmentation', False)
    )
    
    val_dataset = MultimodalSequentialVideoDataset(
        'json/mapped_val_data_video_aligned.json',
        'data/train_video/video_v5_0',
        tokenizer,
        video_transform=VideoTransform(),
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length'],
        augment=False  # No augmentation for validation
    )
    
    # Create data loaders
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
    for i in tqdm(range(len(train_dataset)), desc="Getting labels"):
        all_labels.append(train_dataset.get_label(i))
    
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize improved model
    print("Initializing improved model...")
    model = ImprovedMultimodalLSTMVideoModel(
        train_dataset,
        num_classes=7,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        freeze_bert_layers=config.get('freeze_bert_layers', 6)
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize training components
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Enhanced optimizer with different learning rates
    bert_params = list(model.text_encoder.parameters())
    other_params = [p for p in model.parameters() if not any(p is bp for bp in bert_params)]
    
    optimizer = torch.optim.AdamW([
        {'params': bert_params, 'lr': config['learning_rate'] * 0.1, 'weight_decay': config['weight_decay']},
        {'params': other_params, 'lr': config['learning_rate'], 'weight_decay': config['weight_decay']}
    ])
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Loss function
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=config.get('focal_alpha', 1.0),
            gamma=config.get('focal_gamma', 2.0),
            num_classes=7
        )
        print("Using Focal Loss for class imbalance")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropy Loss")
    
    # Early stopping
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
    print("🚀 STARTING IMPROVED TRAINING")
    print("="*80)
    print(f"Improvements:")
    print(f"  ✅ Larger model: {config['hidden_size']} hidden, {config['num_layers']} layers")
    print(f"  ✅ Better architecture: Enhanced TimeSformer, improved fusion")
    print(f"  ✅ Higher resolution: 224x224 frames")
    print(f"  ✅ Data augmentation: {config.get('data_augmentation', False)}")
    print(f"  ✅ Better learning rate: {config['learning_rate']}")
    print(f"  ✅ More capacity: {trainable_params:,} trainable parameters")
    print("="*80)
    
    best_val_accuracy = 0
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🎯 EPOCH {epoch + 1}/{config['num_epochs']} - IMPROVED TRAINING")
        print("="*60)
        
        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            epoch_num=epoch+1
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_predictions, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Calculate generalization gap
        generalization_gap = abs(train_acc - val_acc)
        
        print(f"📊 EPOCH {epoch + 1} RESULTS:")
        print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc:.4f} ({train_acc*100:.1f}%)")
        print(f"   Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f} ({val_acc*100:.1f}%)")
        print(f"   📈 Generalization Gap: {generalization_gap:.4f} ({generalization_gap*100:.1f}%)")
        print(f"   ⏱️  Epoch Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f'checkpoints_2/best_improved_model_{timestamp}_acc{val_acc:.4f}.pth'
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_accuracy': val_acc,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config,
                'history': history
            }, checkpoint_path)
            
            print(f"💾 New best model saved: {checkpoint_path}")
            print(f"   🎯 Validation accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"   📊 Generalization gap: {generalization_gap:.4f}")
        
        # Save training history
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
        
        with open('result_2/training_history_improved.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping check
        if early_stopping(val_loss, model):
            print(f"🛑 Early stopping triggered after {epoch + 1} epochs")
            break
        
        # Check for generalization issues
        if generalization_gap > 0.2:
            print(f"   ⚠️  WARNING: Large generalization gap ({generalization_gap:.3f})")
        elif generalization_gap < 0.05:
            print(f"   ✅ Good generalization gap ({generalization_gap:.3f})")
    
    print("\n" + "="*80)
    print("🎉 IMPROVED TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Best validation accuracy achieved: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"📊 Improvements over previous model:")
    print(f"   - Previous: ~17.67% validation accuracy")
    print(f"   - Current: {best_val_accuracy*100:.2f}% validation accuracy")
    print(f"   - Improvement: {(best_val_accuracy - 0.1767)*100:.2f} percentage points")
    print("="*80)
    
    print("\n🧪 To test the improved model:")
    print("   python test_video.py")
    print("="*80)

if __name__ == "__main__":
    main()
