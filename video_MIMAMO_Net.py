import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm
import copy
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class VideoTransform:
    """Enhanced video transformation for better frame extraction"""
    
    def __init__(self, target_fps=2, num_frames=8, frame_size=224):
        self.target_fps = target_fps
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, video_path):
        """Extract frames from video with enhanced preprocessing"""
        try:
            if not os.path.exists(video_path):
                # Create zero tensor if video doesn't exist
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            # Calculate frame indices for uniform sampling
            duration = total_frames / fps
            target_duration = min(duration, self.num_frames / self.target_fps)
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.transform(frame)
                    frames.append(frame_tensor)
                else:
                    # Use last valid frame or zeros if no valid frames
                    if frames:
                        frames.append(frames[-1])
                    else:
                        frames.append(torch.zeros(3, self.frame_size, self.frame_size))
            
            cap.release()
            
            # Stack frames: [C, T, H, W]
            video_tensor = torch.stack(frames, dim=1)
            return video_tensor
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)

class VideoSequentialDataset(Dataset):
    """Simplified dataset for video-only training with metadata"""
    
    def __init__(self, json_path, video_dir, video_transform=None, max_dialogue_length=10, augment=False):
        self.video_dir = video_dir
        self.video_transform = video_transform or VideoTransform()
        self.max_dialogue_length = max_dialogue_length
        self.augment = augment
        
        # Load data
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Emotion mapping (7-class)
        self.emotion_mapping = {
            'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        
        # Create vocabularies for metadata
        self._create_vocabularies()
        
        print(f"Dataset loaded: {len(self.data)} samples")
        print(f"Emotion classes: {list(self.emotion_mapping.keys())}")
    
    def _map_to_basic_emotion(self, emotion_str):
        """Map complex emotions to 7 basic emotions"""
        emotion_str = emotion_str.lower()
        
        # Mapping from complex emotions to basic 7 emotions
        emotion_map = {
            # Joy/Happiness
            'joy': 'joy', 'happy': 'joy', 'excited': 'joy', 'content': 'joy', 
            'grateful': 'joy', 'impressed': 'joy', 'proud': 'joy', 'hopeful': 'joy',
            
            # Sadness
            'sad': 'sadness', 'devastated': 'sadness', 'disappointed': 'sadness', 
            'lonely': 'sadness', 'sentimental': 'sadness', 'nostalgic': 'sadness',
            
            # Anger
            'angry': 'anger', 'annoyed': 'anger', 'furious': 'anger', 'irritated': 'anger',
            
            # Fear
            'afraid': 'fear', 'terrified': 'fear', 'anxious': 'fear', 
            'apprehensive': 'fear', 'worried': 'fear', 'nervous': 'fear',
            
            # Disgust
            'disgusted': 'disgust', 'ashamed': 'disgust', 'embarrassed': 'disgust',
            
            # Surprise
            'surprised': 'surprise', 'amazed': 'surprise', 'shocked': 'surprise',
            
            # Neutral (default)
            'neutral': 'neutral', 'calm': 'neutral', 'faithful': 'neutral', 'jealous': 'neutral'
        }
        
        basic_emotion = emotion_map.get(emotion_str, 'neutral')
        return self.emotion_mapping.get(basic_emotion, 0)
    
    def _create_vocabularies(self):
        """Create vocabularies for categorical metadata features"""
        event_scenarios = set()
        emotion_causes = set()
        goal_responses = set()
        topics = set()
        
        for item in self.data:
            chain_of_empathy = item['turn'].get('chain_of_empathy', {})
            event_scenarios.add(chain_of_empathy.get('event_scenario', 'unknown'))
            emotion_causes.add(chain_of_empathy.get('emotion_cause', 'unknown'))
            goal_responses.add(chain_of_empathy.get('goal_to_response', 'unknown'))
            topics.add(item.get('topic', 'unknown'))
        
        self.event_scenario_vocab = {v: i for i, v in enumerate(sorted(event_scenarios))}
        self.emotion_cause_vocab = {v: i for i, v in enumerate(sorted(emotion_causes))}
        self.goal_response_vocab = {v: i for i, v in enumerate(sorted(goal_responses))}
        self.topic_vocab = {v: i for i, v in enumerate(sorted(topics))}
        
        print(f"Vocabularies created:")
        print(f"  Event scenarios: {len(self.event_scenario_vocab)}")
        print(f"  Emotion causes: {len(self.emotion_cause_vocab)}")
        print(f"  Goal responses: {len(self.goal_response_vocab)}")
        print(f"  Topics: {len(self.topic_vocab)}")
    
    def __len__(self):
        return len(self.data)
    
    def get_label(self, idx):
        """Get label for computing class weights"""
        item = self.data[idx]
        chain_of_empathy = item['turn'].get('chain_of_empathy', {})
        speaker_emotion = chain_of_empathy.get('speaker_emotion', 'neutral')
        return self._map_to_basic_emotion(speaker_emotion)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get dialogue sequence from turn
        dialogue = item['turn']['dialogue'][:self.max_dialogue_length]
        
        # Process videos
        dialogue_videos = []
        dialogue_roles = []
        
        for i in range(self.max_dialogue_length):
            if i < len(dialogue):
                utt = dialogue[i]
                video_filename = utt.get('video_name', '')
                video_path = os.path.join(self.video_dir, video_filename)
                video_tensor = self.video_transform(video_path)
                
                # Role encoding: 0=speaker, 1=listener, 2=other
                role = 0 if utt.get('role') == 'speaker' else 1
                
            else:
                # Padding
                video_tensor = torch.zeros(3, 8, 224, 224)  # [C, T, H, W]
                role = 2  # padding role
            
            dialogue_videos.append(video_tensor)
            dialogue_roles.append(role)
        
        dialogue_video = torch.stack(dialogue_videos)  # [seq_len, C, T, H, W]
        dialogue_roles = torch.tensor(dialogue_roles, dtype=torch.long)
        sequence_length = torch.tensor(len(dialogue), dtype=torch.long)
        
        # Process metadata
        speaker_info = item.get('speaker_profile', {})
        listener_info = item.get('listener_profile', {})
        chain_of_empathy = item['turn'].get('chain_of_empathy', {})
        
        # Categorical features with bounds checking
        def safe_get_age(age_str):
            age_map = {'young': 0, 'middle': 1, 'old': 2, 'unknown': 3}
            return age_map.get(age_str, 3)
        
        def safe_get_gender(gender_str):
            return 0 if gender_str == 'male' else 1
        
        def safe_get_timbre(timbre_str):
            timbre_map = {'low': 0, 'mid': 1, 'high': 2}
            return timbre_map.get(timbre_str, 1)
        
        # Extract metadata features
        speaker_age = safe_get_age(speaker_info.get('age', 'young'))
        speaker_gender = safe_get_gender(speaker_info.get('gender', 'male'))
        speaker_timbre = safe_get_timbre(speaker_info.get('timbre', 'mid'))
        speaker_id = min(speaker_info.get('ID', 0), 99)  # Clamp to valid range
        
        listener_age = safe_get_age(listener_info.get('age', 'young'))
        listener_gender = safe_get_gender(listener_info.get('gender', 'male'))
        listener_timbre = safe_get_timbre(listener_info.get('timbre', 'mid'))
        listener_id = min(listener_info.get('ID', 0), 99)  # Clamp to valid range
        
        event_scenario_id = self.event_scenario_vocab.get(chain_of_empathy.get('event_scenario', 'unknown'), 0)
        emotion_cause_id = self.emotion_cause_vocab.get(chain_of_empathy.get('emotion_cause', 'unknown'), 0)
        goal_response_id = self.goal_response_vocab.get(chain_of_empathy.get('goal_to_response', 'unknown'), 0)
        topic_id = self.topic_vocab.get(item.get('topic', 'unknown'), 0)
        
        metadata = torch.tensor([
            speaker_age, speaker_gender, speaker_timbre, speaker_id,
            listener_age, listener_gender, listener_timbre, listener_id,
            event_scenario_id, emotion_cause_id, goal_response_id, topic_id
        ], dtype=torch.long)
        
        # Get emotion label and map to 7 basic emotions
        speaker_emotion = chain_of_empathy.get('speaker_emotion', 'neutral')
        label = self._map_to_basic_emotion(speaker_emotion)
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            'dialogue_video': dialogue_video,
            'dialogue_roles': dialogue_roles,
            'metadata': metadata,
            'sequence_length': sequence_length,
            'label': label
        }

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for MIMAMO Net"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.out_proj(out)
        
        return out

class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing different modalities"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, query, key, value):
        B, N_q, C = query.shape
        B, N_kv, C = key.shape
        
        # Generate Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Cross-attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        
        return out

class ModalitySpecificEncoder(nn.Module):
    """Modality-specific encoder for video features"""
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super(ModalitySpecificEncoder, self).__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)

class MIMAMOBlock(nn.Module):
    """MIMAMO (Modality-Invariant Multi-Modal Attention) Block"""
    
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MIMAMOBlock, self).__init__()
        
        # Self-attention for each modality
        self.video_self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.metadata_self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        # Cross-modal attention
        self.video_to_metadata_attn = CrossModalAttention(embed_dim, num_heads, dropout)
        self.metadata_to_video_attn = CrossModalAttention(embed_dim, num_heads, dropout)
        
        # Fusion mechanisms
        self.video_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.metadata_fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.video_norm1 = nn.LayerNorm(embed_dim)
        self.video_norm2 = nn.LayerNorm(embed_dim)
        self.metadata_norm1 = nn.LayerNorm(embed_dim)
        self.metadata_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, video_features, metadata_features):
        # Self-attention within each modality
        video_self = self.video_self_attn(video_features)
        metadata_self = self.metadata_self_attn(metadata_features)
        
        # Residual connection and normalization
        video_features = self.video_norm1(video_features + video_self)
        metadata_features = self.metadata_norm1(metadata_features + metadata_self)
        
        # Cross-modal attention
        video_cross = self.video_to_metadata_attn(video_features, metadata_features, metadata_features)
        metadata_cross = self.metadata_to_video_attn(metadata_features, video_features, video_features)
        
        # Fusion
        video_fused = self.video_fusion(torch.cat([video_features, video_cross], dim=-1))
        metadata_fused = self.metadata_fusion(torch.cat([metadata_features, metadata_cross], dim=-1))
        
        # Final residual connection and normalization
        video_output = self.video_norm2(video_features + video_fused)
        metadata_output = self.metadata_norm2(metadata_features + metadata_fused)
        
        return video_output, metadata_output

class VideoFeatureExtractor(nn.Module):
    """Enhanced video feature extractor using 3D CNN + Transformer"""
    
    def __init__(self, embed_dim=384, num_frames=8, frame_size=224, patch_size=32):
        super(VideoFeatureExtractor, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.num_patches = (frame_size // patch_size) ** 2
        
        # 3D CNN for initial feature extraction
        self.conv3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 7, 7))
        )
        
        # Patch embedding for transformer
        self.patch_embed = nn.Linear(256 * 7 * 7, embed_dim)
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, C, T, H, W]
        batch_size, seq_len = x.shape[:2]
        
        sequence_features = []
        for i in range(seq_len):
            video = x[:, i]  # [batch_size, C, T, H, W]
            
            # 3D CNN feature extraction
            conv_features = self.conv3d(video)  # [batch_size, 256, 1, 7, 7]
            conv_features = conv_features.view(batch_size, -1)  # [batch_size, 256*7*7]
            
            # Patch embedding
            patch_features = self.patch_embed(conv_features)  # [batch_size, embed_dim]
            patch_features = patch_features.unsqueeze(1)  # [batch_size, 1, embed_dim]
            
            # Add positional embedding
            patch_features = patch_features + self.pos_embed
            
            # Transformer processing
            transformed = self.transformer(patch_features)  # [batch_size, 1, embed_dim]
            video_features = transformed.squeeze(1)  # [batch_size, embed_dim]
            
            sequence_features.append(video_features)
        
        # Stack sequence features
        sequence_output = torch.stack(sequence_features, dim=1)  # [batch_size, seq_len, embed_dim]
        return sequence_output

class MIMAMONet(nn.Module):
    """MIMAMO Net: Modality-Invariant Multi-Modal Attention Network for Emotion Recognition"""
    
    def __init__(self, dataset, num_classes=7, embed_dim=384, num_heads=8, num_mimamo_blocks=3, 
                 dropout=0.1, hidden_size=256):
        super(MIMAMONet, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Video feature extractor
        self.video_extractor = VideoFeatureExtractor(embed_dim=embed_dim)
        
        # Metadata embeddings
        self.age_embedding = nn.Embedding(4, 32)  # 4 age groups
        self.gender_embedding = nn.Embedding(2, 16)  # 2 genders
        self.timbre_embedding = nn.Embedding(3, 16)  # 3 timbre levels
        self.role_embedding = nn.Embedding(3, 32)  # speaker/listener/padding
        self.speaker_id_embedding = nn.Embedding(100, 64)  # speaker IDs
        self.listener_id_embedding = nn.Embedding(100, 64)  # listener IDs
        
        # Contextual metadata embeddings
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
        
        # Metadata projection to match embed_dim
        metadata_dim = 32 + 16 + 16 + 64 + 32 + 16 + 16 + 64 + 32 + 32 + 32 + 32  # Total: 384
        self.metadata_proj = ModalitySpecificEncoder(metadata_dim, embed_dim, num_layers=2, dropout=dropout)
        
        # Role-aware video processing
        self.role_video_fusion = nn.Sequential(
            nn.Linear(embed_dim + 32, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # MIMAMO blocks for cross-modal fusion
        self.mimamo_blocks = nn.ModuleList([
            MIMAMOBlock(embed_dim, num_heads, dropout)
            for _ in range(num_mimamo_blocks)
        ])
        
        # Temporal modeling with bidirectional LSTM
        self.temporal_lstm = nn.LSTM(
            input_size=embed_dim * 2,  # video + metadata concatenated
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        
        # Global attention mechanism
        self.global_attention = MultiHeadSelfAttention(hidden_size * 2, num_heads=8, dropout=dropout)
        
        # Final classifier with multiple stages
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
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
    
    def forward(self, dialogue_video, dialogue_roles, metadata, sequence_length):
        batch_size, seq_len = dialogue_video.shape[:2]
        
        # Extract video features
        video_features = self.video_extractor(dialogue_video)  # [batch_size, seq_len, embed_dim]
        
        # Role-aware video processing
        role_enhanced_video = []
        for i in range(seq_len):
            role_emb = self.role_embedding(dialogue_roles[:, i])  # [batch_size, 32]
            video_feat = video_features[:, i]  # [batch_size, embed_dim]
            combined = torch.cat([video_feat, role_emb], dim=1)
            fused = self.role_video_fusion(combined)
            role_enhanced_video.append(fused)
        
        role_enhanced_video = torch.stack(role_enhanced_video, dim=1)  # [batch_size, seq_len, embed_dim]
        
        # Process metadata
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
        
        # Combine all metadata features
        metadata_combined = torch.cat([
            speaker_age, speaker_gender, speaker_timbre, speaker_id_emb,
            listener_age, listener_gender, listener_timbre, listener_id_emb,
            event_scenario_emb, emotion_cause_emb, goal_response_emb, topic_emb
        ], dim=1)  # [batch_size, metadata_dim]
        
        # Project metadata to embed_dim
        metadata_features = self.metadata_proj(metadata_combined)  # [batch_size, embed_dim]
        metadata_sequence = metadata_features.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, embed_dim]
        
        # Apply MIMAMO blocks for cross-modal fusion
        video_fused = role_enhanced_video
        metadata_fused = metadata_sequence
        
        for mimamo_block in self.mimamo_blocks:
            video_fused, metadata_fused = mimamo_block(video_fused, metadata_fused)
        
        # Concatenate fused features for temporal modeling
        fused_features = torch.cat([video_fused, metadata_fused], dim=-1)  # [batch_size, seq_len, embed_dim*2]
        
        # Temporal modeling with LSTM
        lstm_out, _ = self.temporal_lstm(fused_features)  # [batch_size, seq_len, hidden_size*2]
        
        # Global attention
        attended_features = self.global_attention(lstm_out)  # [batch_size, seq_len, hidden_size*2]
        
        # Extract final features based on actual sequence length
        final_features = []
        for i in range(batch_size):
            seq_len_i = sequence_length[i].item() - 1
            seq_len_i = max(0, min(seq_len_i, seq_len - 1))
            final_features.append(attended_features[i, seq_len_i, :])
        final_features = torch.stack(final_features)  # [batch_size, hidden_size*2]
        
        # Apply dropout
        final_features = self.dropout(final_features)
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits

class EarlyStopping:
    """Early stopping utility"""
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

def collate_fn(batch):
    """Custom collate function to handle variable sequence lengths"""
    dialogue_video = torch.stack([item['dialogue_video'] for item in batch])
    dialogue_roles = torch.stack([item['dialogue_roles'] for item in batch])
    metadata = torch.stack([item['metadata'] for item in batch])
    sequence_length = torch.stack([item['sequence_length'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'dialogue_video': dialogue_video,
        'dialogue_roles': dialogue_roles,
        'metadata': metadata,
        'sequence_length': sequence_length,
        'label': labels
    }

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler=None, 
                gradient_accumulation_steps=1, max_grad_norm=1.0, epoch_num=1):
    """Training loop for one epoch"""
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
        dialogue_video = batch['dialogue_video'].to(device)
        dialogue_roles = batch['dialogue_roles'].to(device)
        metadata = batch['metadata'].to(device)
        sequence_length = batch['sequence_length'].to(device)
        labels = batch['label'].to(device)
        
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(dialogue_video, dialogue_roles, metadata, sequence_length)
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
            logits = model(dialogue_video, dialogue_roles, metadata, sequence_length)
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
    """Validation loop for one epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            dialogue_video = batch['dialogue_video'].to(device)
            dialogue_roles = batch['dialogue_roles'].to(device)
            metadata = batch['metadata'].to(device)
            sequence_length = batch['sequence_length'].to(device)
            labels = batch['label'].to(device)
            
            with torch.cuda.amp.autocast():
                logits = model(dialogue_video, dialogue_roles, metadata, sequence_length)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels

def main():
    """Main training function"""
    
    # Configuration for MIMAMO Net
    config = {
        'batch_size': 4,  # Reduced due to larger model
        'gradient_accumulation_steps': 4,  # Increased to compensate
        'learning_rate': 1e-5,  # Lower learning rate for stable training
        'num_epochs': 25,
        'patience': 6,
        'dropout_rate': 0.2,
        'weight_decay': 0.01,
        'warmup_ratio': 0.15,
        'max_dialogue_length': 10,
        'hidden_size': 256,
        'max_grad_norm': 1.0,
        'use_focal_loss': True,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'use_mixed_precision': True,
        'data_augmentation': True,
        'embed_dim': 384,
        'num_heads': 8,
        'num_mimamo_blocks': 3,
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
    os.makedirs('mimamo_net_result', exist_ok=True)
    os.makedirs('checkpoint_3', exist_ok=True)
    
    # Load datasets
    print("Loading video datasets for MIMAMO Net...")
    train_dataset = VideoSequentialDataset(
        r'C:\Users\sslue\empathetic-detection\json\mapped_train_data_video_aligned.json',
        r'C:\Users\sslue\empathetic-detection\data\train_video\video_v5_0',
        video_transform=VideoTransform(),
        max_dialogue_length=config['max_dialogue_length'],
        augment=config['data_augmentation']
    )
    
    val_dataset = VideoSequentialDataset(
        r'C:\Users\sslue\empathetic-detection\json\mapped_val_data_video_aligned.json',
        r'C:\Users\sslue\empathetic-detection\data\train_video\video_v5_0',
        video_transform=VideoTransform(),
        max_dialogue_length=config['max_dialogue_length'],
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Compute class weights
    print("Computing class weights...")
    all_labels = []
    for i in tqdm(range(len(train_dataset)), desc="Getting labels"):
        all_labels.append(train_dataset.get_label(i))
    
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {class_weights}")
    
    # Initialize MIMAMO Net model
    print("Initializing MIMAMO Net model...")
    model = MIMAMONet(
        train_dataset,
        num_classes=7,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_mimamo_blocks=config['num_mimamo_blocks'],
        dropout=config['dropout_rate'],
        hidden_size=config['hidden_size']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize training components
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Loss function
    if config['use_focal_loss']:
        criterion = FocalLoss(
            alpha=config['focal_alpha'],
            gamma=config['focal_gamma'],
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
    print("🚀 STARTING MIMAMO NET TRAINING")
    print("="*80)
    print(f"Model features:")
    print(f"  ✅ MIMAMO Architecture: Modality-Invariant Multi-Modal Attention")
    print(f"  ✅ 3D CNN + Transformer Video Encoder: {config['embed_dim']}D")
    print(f"  ✅ Cross-Modal Attention Blocks: {config['num_mimamo_blocks']} blocks")
    print(f"  ✅ Multi-Head Attention: {config['num_heads']} heads")
    print(f"  ✅ Bidirectional LSTM: {config['hidden_size']} hidden")
    print(f"  ✅ Parameters: {trainable_params:,} trainable")
    print(f"  ✅ Mixed precision: {config['use_mixed_precision']}")
    print("="*80)
    
    best_val_accuracy = 0
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"🎯 EPOCH {epoch + 1}/{config['num_epochs']} - MIMAMO NET")
        print("="*60)
        
        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
            gradient_accumulation_steps=config['gradient_accumulation_steps'],
            max_grad_norm=config['max_grad_norm'],
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
            checkpoint_path = f'checkpoint_3/best_mimamo_model_{timestamp}_acc{val_acc:.4f}.pth'
            
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
        
        with open('mimamo_net_result/mimamo_training_history.json', 'w') as f:
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
    print("🎉 MIMAMO NET TRAINING COMPLETED!")
    print("="*80)
    print(f"🏆 Best validation accuracy achieved: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
    print(f"📊 Model architecture:")
    print(f"   - MIMAMO: Modality-Invariant Multi-Modal Attention Network")
    print(f"   - 3D CNN + Transformer video processing")
    print(f"   - Cross-modal attention with {config['num_mimamo_blocks']} MIMAMO blocks")
    print(f"   - Total parameters: {trainable_params:,}")
    print("="*80)
    
    # Plot confusion matrix
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(val_labels, val_predictions)
        emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_labels, yticklabels=emotion_labels)
        plt.title(f'Confusion Matrix - MIMAMO Net (Acc: {best_val_accuracy:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('mimamo_net_result/confusion_matrix_mimamo.png', dpi=300, bbox_inches='tight')
        print("📊 Confusion matrix saved to mimamo_net_result/confusion_matrix_mimamo.png")
    except Exception as e:
        print(f"⚠️  Could not save confusion matrix: {e}")
    
    # Save final training summary
    summary = {
        'model_name': 'MIMAMO Net',
        'description': 'Modality-Invariant Multi-Modal Attention Network for Emotion Recognition',
        'best_validation_accuracy': best_val_accuracy,
        'total_epochs': len(history['epoch']),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'config': config,
        'architecture_features': [
            '3D CNN + Transformer video encoder',
            f'{config["num_mimamo_blocks"]} MIMAMO cross-modal attention blocks',
            f'{config["num_heads"]}-head multi-head attention',
            'Bidirectional LSTM for temporal modeling',
            'Role-aware video processing',
            'Comprehensive metadata integration'
        ]
    }
    
    with open('mimamo_net_result/training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("📋 Training summary saved to mimamo_net_result/training_summary.json")

if __name__ == "__main__":
    main()