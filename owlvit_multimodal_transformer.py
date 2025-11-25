"""
OwlViT-Based Unified Multimodal Transformer for English Emotion/Empathy Recognition

A powerful end-to-end multimodal architecture that uses OwlViT as the unified backbone
for vision-language understanding, extended to handle audio and metadata in a single
transformer architecture.

Key Features:
- OwlViT vision-language alignment as the core backbone
- Wav2Vec2 audio features projected to OwlViT's embedding space
- Metadata encoding through lightweight feed-forward layers
- Single unified transformer with global cross-modal attention
- End-to-end training on CMU-MOSEI, MELD, IEMOCAP datasets
- Compact architecture with strong multimodal fusion

Architecture:
1. Video: OwlViT vision encoder extracts frame embeddings
2. Text: OwlViT text encoder extracts token embeddings
3. Audio: Wav2Vec2 → projection layer → same embedding dimension
4. Metadata: Feed-forward embedding layer
5. All modalities concatenated → Unified Transformer → Classification
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    OwlViTProcessor, 
    OwlViTModel,
    Wav2Vec2Model,
    Wav2Vec2Processor,
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import librosa
from PIL import Image
import cv2
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class AudioProjection(nn.Module):
    """Projects Wav2Vec2 audio features to OwlViT embedding dimension."""
    
    def __init__(self, wav2vec_dim: int = 768, owlvit_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(wav2vec_dim, owlvit_dim * 2),
            nn.LayerNorm(owlvit_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(owlvit_dim * 2, owlvit_dim),
            nn.LayerNorm(owlvit_dim)
        )
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: (batch, seq_len, wav2vec_dim)
        Returns:
            projected_features: (batch, seq_len, owlvit_dim)
        """
        return self.projection(audio_features)


class MetadataEmbedding(nn.Module):
    """Lightweight feed-forward embedding for metadata features."""
    
    def __init__(self, metadata_dim: int, owlvit_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(metadata_dim, owlvit_dim),
            nn.LayerNorm(owlvit_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(owlvit_dim, owlvit_dim),
            nn.LayerNorm(owlvit_dim)
        )
        
    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            metadata: (batch, metadata_dim)
        Returns:
            embedded_metadata: (batch, 1, owlvit_dim)
        """
        # Add sequence dimension for concatenation with other modalities
        return self.embedding(metadata).unsqueeze(1)


class UnifiedMultimodalTransformer(nn.Module):
    """
    Single transformer stack for global cross-modal attention across all modalities.
    Enables speech cues, facial expressions, textual meaning, and metadata to 
    influence each other in one unified architecture.
    """
    
    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Modality type embeddings (learnable position for each modality)
        self.modality_embeddings = nn.Parameter(torch.randn(4, hidden_dim))  # video, text, audio, metadata
        
        # Unified transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        video_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_embeds: torch.Tensor,
        metadata_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Global cross-modal attention across all modalities.
        
        Args:
            video_embeds: (batch, video_seq_len, hidden_dim)
            text_embeds: (batch, text_seq_len, hidden_dim)
            audio_embeds: (batch, audio_seq_len, hidden_dim)
            metadata_embeds: (batch, 1, hidden_dim)
            attention_mask: (batch, total_seq_len) optional
            
        Returns:
            Dictionary with fused representations and modality-specific outputs
        """
        batch_size = video_embeds.size(0)
        
        # Add modality type embeddings
        video_seq_len = video_embeds.size(1)
        text_seq_len = text_embeds.size(1)
        audio_seq_len = audio_embeds.size(1)
        metadata_seq_len = metadata_embeds.size(1)
        
        video_embeds = video_embeds + self.modality_embeddings[0].unsqueeze(0).unsqueeze(0)
        text_embeds = text_embeds + self.modality_embeddings[1].unsqueeze(0).unsqueeze(0)
        audio_embeds = audio_embeds + self.modality_embeddings[2].unsqueeze(0).unsqueeze(0)
        metadata_embeds = metadata_embeds + self.modality_embeddings[3].unsqueeze(0).unsqueeze(0)
        
        # Concatenate all modality embeddings
        # Shape: (batch, video_seq + text_seq + audio_seq + 1, hidden_dim)
        multimodal_embeds = torch.cat([
            video_embeds,
            text_embeds,
            audio_embeds,
            metadata_embeds
        ], dim=1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            total_seq_len = multimodal_embeds.size(1)
            attention_mask = torch.ones(batch_size, total_seq_len, device=multimodal_embeds.device)
        
        # Convert attention mask to transformer format (0 = attend, -inf = ignore)
        # PyTorch transformer expects inverted mask
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Apply unified transformer with global cross-modal attention
        fused_features = self.transformer(multimodal_embeds, src_key_padding_mask=None)
        fused_features = self.norm(fused_features)
        
        # Extract modality-specific outputs for analysis
        video_start = 0
        text_start = video_seq_len
        audio_start = text_start + text_seq_len
        metadata_start = audio_start + audio_seq_len
        
        return {
            'fused_features': fused_features,  # Full multimodal sequence
            'video_features': fused_features[:, video_start:text_start, :],
            'text_features': fused_features[:, text_start:audio_start, :],
            'audio_features': fused_features[:, audio_start:metadata_start, :],
            'metadata_features': fused_features[:, metadata_start:, :],
            'global_pooled': fused_features.mean(dim=1)  # Global representation
        }


class OwlViTMultimodalEmotionModel(nn.Module):
    """
    End-to-end multimodal transformer for emotion/empathy recognition.
    Uses OwlViT as unified backbone, extended to audio and metadata.
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        owlvit_model_name: str = "google/owlvit-base-patch32",
        wav2vec_model_name: str = "facebook/wav2vec2-base-960h",
        metadata_dim: int = 10,
        hidden_dim: int = 512,  # Changed from 768 to 512 to match OwlViT
        num_transformer_layers: int = 6,
        num_heads: int = 8,  # Changed from 12 to 8 for 512-dim
        ff_dim: int = 2048,  # Changed from 3072 to 2048
        dropout: float = 0.1,
        freeze_owlvit: bool = False,
        freeze_wav2vec: bool = True
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # OwlViT backbone for vision-language alignment
        print(f"Loading OwlViT model: {owlvit_model_name}")
        self.owlvit = OwlViTModel.from_pretrained(owlvit_model_name)
        self.owlvit_processor = OwlViTProcessor.from_pretrained(owlvit_model_name)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.owlvit, 'gradient_checkpointing_enable'):
            self.owlvit.gradient_checkpointing_enable()
            print("OwlViT gradient checkpointing enabled")
        
        if freeze_owlvit:
            for param in self.owlvit.parameters():
                param.requires_grad = False
            print("OwlViT frozen - using as feature extractor only")
        
        # Wav2Vec2 for audio features
        print(f"Loading Wav2Vec2 model: {wav2vec_model_name}")
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        
        # Enable gradient checkpointing to save memory
        if hasattr(self.wav2vec, 'gradient_checkpointing_enable'):
            self.wav2vec.gradient_checkpointing_enable()
            print("Wav2Vec2 gradient checkpointing enabled")
        
        if freeze_wav2vec:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
            print("Wav2Vec2 frozen - using as feature extractor only")
        
        # Audio projection to OwlViT dimension
        wav2vec_dim = self.wav2vec.config.hidden_size
        self.audio_projection = AudioProjection(wav2vec_dim, hidden_dim, dropout)
        
        # Vision projection (OwlViT vision outputs 768, need to project to hidden_dim)
        owlvit_vision_dim = self.owlvit.config.vision_config.hidden_size  # 768
        if owlvit_vision_dim != hidden_dim:
            self.vision_projection = nn.Linear(owlvit_vision_dim, hidden_dim)
        else:
            self.vision_projection = nn.Identity()
        
        # Text projection (OwlViT text outputs 512, might need projection)
        owlvit_text_dim = self.owlvit.config.text_config.hidden_size  # 512
        if owlvit_text_dim != hidden_dim:
            self.text_projection = nn.Linear(owlvit_text_dim, hidden_dim)
        else:
            self.text_projection = nn.Identity()
        
        # Metadata embedding
        self.metadata_embedding = MetadataEmbedding(metadata_dim, hidden_dim, dropout)
        
        # Unified multimodal transformer
        self.multimodal_transformer = UnifiedMultimodalTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize classification head weights."""
        for module in [self.classifier]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        video_frames: torch.Tensor,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        audio_waveform: torch.Tensor,
        metadata: torch.Tensor,
        verbose: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through unified multimodal transformer.
        
        Args:
            video_frames: (batch, num_frames, C, H, W) - Video frames
            text_input_ids: (batch, text_seq_len) - Tokenized text
            text_attention_mask: (batch, text_seq_len) - Text attention mask
            audio_waveform: (batch, audio_length) - Raw audio waveform
            metadata: (batch, metadata_dim) - Contextual metadata
            verbose: Print detailed feature extraction info
            
        Returns:
            Dictionary with logits and intermediate features
        """
        batch_size = video_frames.size(0)
        num_frames = video_frames.size(1)
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"FORWARD PASS - FEATURE EXTRACTION")
            print(f"{'='*80}")
            print(f"Input shapes:")
            print(f"  Video: {video_frames.shape}")
            print(f"  Text IDs: {text_input_ids.shape}")
            print(f"  Audio: {audio_waveform.shape}")
            print(f"  Metadata: {metadata.shape}")
        
        # 1. Extract video features using OwlViT vision encoder
        # Flatten batch and frames: (batch * num_frames, C, H, W)
        video_flat = video_frames.view(batch_size * num_frames, *video_frames.shape[2:])
        
        with torch.set_grad_enabled(not self.training or self.owlvit.training):
            vision_outputs = self.owlvit.vision_model(pixel_values=video_flat)
            video_embeds_flat = vision_outputs.last_hidden_state  # (batch*frames, seq_len, 768)
        
        # Project to hidden_dim
        video_embeds_flat = self.vision_projection(video_embeds_flat)  # (batch*frames, seq_len, hidden_dim)
        
        # Reshape back: (batch, num_frames * seq_len, hidden)
        seq_len_per_frame = video_embeds_flat.size(1)
        video_embeds = video_embeds_flat.view(batch_size, num_frames * seq_len_per_frame, -1)
        
        if verbose:
            print(f"\n📹 VIDEO FEATURES:")
            print(f"  OwlViT vision output: {vision_outputs.last_hidden_state.shape}")
            print(f"  After projection: {video_embeds.shape}")
            print(f"  Range: [{video_embeds.min():.3f}, {video_embeds.max():.3f}]")
        
        # 2. Extract text features using OwlViT text encoder
        with torch.set_grad_enabled(not self.training or self.owlvit.training):
            text_outputs = self.owlvit.text_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
            text_embeds = text_outputs.last_hidden_state  # (batch, text_seq_len, 512)
        
        # Project to hidden_dim
        text_embeds = self.text_projection(text_embeds)  # (batch, text_seq_len, hidden_dim)
        
        if verbose:
            print(f"\n📝 TEXT FEATURES:")
            print(f"  OwlViT text output: {text_outputs.last_hidden_state.shape}")
            print(f"  After projection: {text_embeds.shape}")
            print(f"  Range: [{text_embeds.min():.3f}, {text_embeds.max():.3f}]")
        
        # 3. Extract audio features using Wav2Vec2
        with torch.set_grad_enabled(not self.training or self.wav2vec.training):
            audio_outputs = self.wav2vec(audio_waveform)
            audio_embeds_raw = audio_outputs.last_hidden_state  # (batch, audio_seq_len, wav2vec_dim)
        
        # Project audio to OwlViT dimension
        audio_embeds = self.audio_projection(audio_embeds_raw)
        
        if verbose:
            print(f"\n🔊 AUDIO FEATURES:")
            print(f"  Wav2Vec2 output: {audio_embeds_raw.shape}")
            print(f"  After projection: {audio_embeds.shape}")
            print(f"  Range: [{audio_embeds.min():.3f}, {audio_embeds.max():.3f}]")
        
        # 4. Embed metadata
        metadata_embeds = self.metadata_embedding(metadata)
        
        if verbose:
            print(f"\n🏷️  METADATA FEATURES:")
            print(f"  After embedding: {metadata_embeds.shape}")
            print(f"  Range: [{metadata_embeds.min():.3f}, {metadata_embeds.max():.3f}]")
        
        # 5. Unified multimodal transformer with global cross-modal attention
        multimodal_outputs = self.multimodal_transformer(
            video_embeds=video_embeds,
            text_embeds=text_embeds,
            audio_embeds=audio_embeds,
            metadata_embeds=metadata_embeds
        )
        
        if verbose:
            print(f"\n🔄 MULTIMODAL FUSION:")
            print(f"  Fused features: {multimodal_outputs['fused_features'].shape}")
            print(f"  Global pooled: {multimodal_outputs['global_pooled'].shape}")
            print(f"  Range: [{multimodal_outputs['global_pooled'].min():.3f}, {multimodal_outputs['global_pooled'].max():.3f}]")
        
        # 6. Classification from global pooled representation
        global_features = multimodal_outputs['global_pooled']
        logits = self.classifier(global_features)
        
        if verbose:
            print(f"\n🎯 CLASSIFICATION:")
            print(f"  Logits: {logits.shape}")
            print(f"  Predictions: {logits.argmax(dim=-1).cpu().tolist()}")
            print(f"  Logit values: {logits[0].cpu().tolist()}")
            print(f"{'='*80}\n")
        
        return {
            'logits': logits,
            'global_features': global_features,
            'fused_features': multimodal_outputs['fused_features'],
            'video_features': multimodal_outputs['video_features'],
            'text_features': multimodal_outputs['text_features'],
            'audio_features': multimodal_outputs['audio_features'],
            'metadata_features': multimodal_outputs['metadata_features']
        }


class MultimodalEmotionDataset(Dataset):
    """
    Dataset for English emotion/empathy recognition.
    Supports CMU-MOSEI, MELD, IEMOCAP formats and nested dialogue structure.
    """
    
    def __init__(
        self,
        data_json_path: str,
        video_root: str,
        audio_root: str,
        owlvit_processor: OwlViTProcessor,
        wav2vec_processor: Wav2Vec2Processor,
        num_frames: int = 8,
        audio_sample_rate: int = 16000,
        audio_max_length: int = 16000 * 5,  # 5 seconds
        max_text_length: int = 77,
        metadata_keys: List[str] = None
    ):
        super().__init__()
        
        self.video_root = video_root
        self.audio_root = audio_root
        self.owlvit_processor = owlvit_processor
        self.wav2vec_processor = wav2vec_processor
        self.num_frames = num_frames
        self.audio_sample_rate = audio_sample_rate
        self.audio_max_length = audio_max_length
        self.max_text_length = max_text_length
        self.metadata_keys = metadata_keys or ['gender', 'age', 'context']
        
        # 7 basic emotions mapping
        self.emotion_mapping = {
            'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3,
            'fear': 4, 'disgust': 5, 'surprise': 6
        }
        
        # Load dataset and flatten dialogue structure
        with open(data_json_path, 'r') as f:
            raw_data = json.load(f)
        
        # Flatten nested dialogue structure - extract speaker utterances only
        self.data = []
        for item in raw_data:
            dialogue = item['turn']['dialogue']
            for utt in dialogue:
                if utt.get('role') == 'speaker':  # Focus on speaker emotion
                    sample = {
                        'conversation_id': item['conversation_id'],
                        'turn_id': item['turn']['turn_id'],
                        'text': utt['text'],
                        'audio_name': utt.get('audio_name', ''),
                        'video_name': utt.get('video_name', ''),
                        'speaker_profile': item.get('speaker_profile', {}),
                        'chain_of_empathy': item['turn'].get('chain_of_empathy', {}),
                        'topic': item.get('topic', 'unknown')
                    }
                    self.data.append(sample)
        
        print(f"Loaded {len(self.data)} speaker utterances from {data_json_path}")
    
    def _map_to_basic_emotion(self, emotion_str):
        """Map complex emotions to 7 basic emotions"""
        emotion_str = emotion_str.lower().strip()
        
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
    
    def __len__(self) -> int:
        return len(self.data)
    
    def _load_video_frames(self, video_filename: str) -> torch.Tensor:
        """Load and sample frames from video."""
        video_path = os.path.join(self.video_root, video_filename)
        
        if not os.path.exists(video_path):
            # Return blank frames if video doesn't exist
            blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            frames = [Image.fromarray(blank_frame) for _ in range(self.num_frames)]
            pixel_values = self.owlvit_processor(images=frames, return_tensors="pt")['pixel_values']
            return pixel_values
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            # Return blank frames if video can't be read
            blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            for _ in range(self.num_frames):
                frames.append(Image.fromarray(blank_frame))
        else:
            # Sample frames uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                else:
                    # Use last valid frame or blank
                    if frames:
                        frames.append(frames[-1])
                    else:
                        blank_frame = np.zeros((224, 224, 3), dtype=np.uint8)
                        frames.append(Image.fromarray(blank_frame))
        
        cap.release()
        
        # Process frames with OwlViT processor
        pixel_values = self.owlvit_processor(images=frames, return_tensors="pt")['pixel_values']
        return pixel_values  # (num_frames, 3, H, W)
    
    def _load_audio(self, audio_filename: str) -> torch.Tensor:
        """Load and process audio waveform."""
        audio_path = os.path.join(self.audio_root, audio_filename)
        
        try:
            if not os.path.exists(audio_path):
                # Return silence if audio doesn't exist
                return torch.zeros(self.audio_max_length)
            
            waveform, sr = librosa.load(audio_path, sr=self.audio_sample_rate, mono=True)
            
            # Pad or truncate to fixed length
            if len(waveform) < self.audio_max_length:
                waveform = np.pad(waveform, (0, self.audio_max_length - len(waveform)))
            else:
                waveform = waveform[:self.audio_max_length]
            
            # Process with Wav2Vec2 processor
            inputs = self.wav2vec_processor(
                waveform,
                sampling_rate=self.audio_sample_rate,
                return_tensors="pt",
                padding=True
            )
            return inputs['input_values'].squeeze(0)
        
        except Exception as e:
            print(f"Error loading audio {audio_filename}: {e}")
            # Return silence
            return torch.zeros(self.audio_max_length)
    
    def _encode_metadata(self, sample: Dict) -> torch.Tensor:
        """Encode metadata as fixed-size vector."""
        metadata_vector = []
        
        # Speaker profile features
        speaker_profile = sample.get('speaker_profile', {})
        
        # Age encoding (0=young, 1=middle, 2=old)
        age_map = {'young': 0.0, 'middle': 0.5, 'old': 1.0}
        age = speaker_profile.get('age', 'young')
        metadata_vector.append(age_map.get(age, 0.0))
        
        # Gender encoding (0=female, 1=male)
        gender_map = {'female': 0.0, 'male': 1.0}
        gender = speaker_profile.get('gender', 'female')
        metadata_vector.append(gender_map.get(gender, 0.0))
        
        # Timbre encoding (0=low, 1=mid, 2=high)
        timbre_map = {'low': 0.0, 'mid': 0.5, 'high': 1.0}
        timbre = speaker_profile.get('timbre', 'mid')
        metadata_vector.append(timbre_map.get(timbre, 0.5))
        
        # Topic encoding (hash-based)
        topic = sample.get('topic', 'unknown')
        metadata_vector.append(float(hash(topic) % 100) / 100.0)
        
        # Chain of empathy features (hash-based encoding for text features)
        chain = sample.get('chain_of_empathy', {})
        event_scenario = chain.get('event_scenario', '')
        emotion_cause = chain.get('emotion_cause', '')
        goal_response = chain.get('goal_to_response', '')
        
        metadata_vector.append(float(hash(event_scenario) % 100) / 100.0)
        metadata_vector.append(float(hash(emotion_cause) % 100) / 100.0)
        metadata_vector.append(float(hash(goal_response) % 100) / 100.0)
        
        # Pad to fixed size (10 dimensions)
        while len(metadata_vector) < 10:
            metadata_vector.append(0.0)
        
        return torch.tensor(metadata_vector[:10], dtype=torch.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        
        # Load video frames
        video_filename = sample.get('video_name', '')
        video_frames = self._load_video_frames(video_filename)
        
        # Load audio
        audio_filename = sample.get('audio_name', '')
        audio_waveform = self._load_audio(audio_filename)
        
        # Process text
        text = sample.get('text', '')
        text_inputs = self.owlvit_processor.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_text_length,
            return_tensors='pt'
        )
        
        # Encode metadata
        metadata = self._encode_metadata(sample)
        
        # Get emotion label
        chain_of_empathy = sample.get('chain_of_empathy', {})
        speaker_emotion = chain_of_empathy.get('speaker_emotion', 'neutral')
        label = self._map_to_basic_emotion(speaker_emotion)
        
        # Print data info every 1000 samples to track progress
        if idx % 1000 == 0:
            print(f"\n{'='*80}")
            print(f"SAMPLE #{idx} DATA VERIFICATION")
            print(f"{'='*80}")
            
            # Video info
            print(f"\n📹 VIDEO DATA:")
            print(f"  File: {video_filename}")
            print(f"  Shape: {video_frames.shape}")
            print(f"  Min/Max values: {video_frames.min():.3f} / {video_frames.max():.3f}")
            print(f"  Mean: {video_frames.mean():.3f}, Std: {video_frames.std():.3f}")
            
            # Audio info
            print(f"\n🔊 AUDIO DATA:")
            print(f"  File: {audio_filename}")
            print(f"  Shape: {audio_waveform.shape}")
            print(f"  Length: {len(audio_waveform)} samples ({len(audio_waveform)/self.audio_sample_rate:.2f} seconds)")
            print(f"  Min/Max values: {audio_waveform.min():.3f} / {audio_waveform.max():.3f}")
            print(f"  Mean: {audio_waveform.mean():.3f}, Std: {audio_waveform.std():.3f}")
            
            # Text info
            print(f"\n📝 TEXT DATA:")
            print(f"  Original text: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
            print(f"  Text length: {len(text)} characters")
            print(f"  Token IDs shape: {text_inputs['input_ids'].shape}")
            print(f"  Number of tokens: {text_inputs['attention_mask'].sum().item()}")
            print(f"  Token IDs: {text_inputs['input_ids'].squeeze(0).tolist()[:10]}...")
            
            # Metadata info
            speaker_profile = sample.get('speaker_profile', {})
            print(f"\n🏷️  METADATA:")
            print(f"  Speaker Profile:")
            print(f"    - Age: {speaker_profile.get('age', 'N/A')}")
            print(f"    - Gender: {speaker_profile.get('gender', 'N/A')}")
            print(f"    - Timbre: {speaker_profile.get('timbre', 'N/A')}")
            print(f"    - ID: {speaker_profile.get('ID', 'N/A')}")
            print(f"  Topic: {sample.get('topic', 'N/A')}")
            print(f"  Metadata vector shape: {metadata.shape}")
            print(f"  Metadata values: {metadata.tolist()}")
            
            # Label info
            print(f"\n🎯 LABEL:")
            print(f"  Raw emotion: {speaker_emotion}")
            print(f"  Mapped label: {label} ({list(self.emotion_mapping.keys())[label]})")
            print(f"  Chain of empathy:")
            print(f"    - Event: {chain_of_empathy.get('event_scenario', 'N/A')[:60]}...")
            print(f"    - Cause: {chain_of_empathy.get('emotion_cause', 'N/A')[:60]}...")
            print(f"{'='*80}\n")
        
        return {
            'video_frames': video_frames,
            'text_input_ids': text_inputs['input_ids'].squeeze(0),
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'audio_waveform': audio_waveform,
            'metadata': metadata,
            'label': torch.tensor(label, dtype=torch.long)
        }


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1
) -> Tuple[float, float, float, float, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        video_frames = batch['video_frames'].to(device)
        text_input_ids = batch['text_input_ids'].to(device)
        text_attention_mask = batch['text_attention_mask'].to(device)
        audio_waveform = batch['audio_waveform'].to(device)
        metadata = batch['metadata'].to(device)
        labels = batch['label'].to(device)
        
        # Print detailed batch info every 100 batches
        if batch_idx % 100 == 0:
            print(f"\n{'='*80}")
            print(f"BATCH #{batch_idx} - FORWARD PASS DATA CHECK")
            print(f"{'='*80}")
            print(f"📹 VIDEO: shape={video_frames.shape}, device={video_frames.device}, dtype={video_frames.dtype}")
            print(f"   Range: [{video_frames.min():.3f}, {video_frames.max():.3f}]")
            print(f"🔊 AUDIO: shape={audio_waveform.shape}, device={audio_waveform.device}, dtype={audio_waveform.dtype}")
            print(f"   Range: [{audio_waveform.min():.3f}, {audio_waveform.max():.3f}]")
            print(f"📝 TEXT:  input_ids shape={text_input_ids.shape}, attention_mask shape={text_attention_mask.shape}")
            print(f"   Active tokens: {text_attention_mask.sum().item()}")
            print(f"🏷️  META: shape={metadata.shape}, device={metadata.device}, dtype={metadata.dtype}")
            print(f"   Values: {metadata[0].cpu().tolist()}")
            print(f"🎯 LABEL: shape={labels.shape}, value={labels[0].item()} ({list(model.emotion_mapping.keys())[labels[0].item()] if hasattr(model, 'emotion_mapping') else 'N/A'})")
            print(f"{'='*80}\n")
        
        # Forward pass with mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(
                    video_frames=video_frames,
                    text_input_ids=text_input_ids,
                    text_attention_mask=text_attention_mask,
                    audio_waveform=audio_waveform,
                    metadata=metadata,
                    verbose=(batch_idx == 0)  # Verbose for first batch only
                )
                loss = criterion(outputs['logits'], labels)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Only step optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        else:
            outputs = model(
                video_frames=video_frames,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_waveform=audio_waveform,
                metadata=metadata,
                verbose=(batch_idx == 0)  # Verbose for first batch only
            )
            loss = criterion(outputs['logits'], labels)
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            # Only step optimizer every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler is not None:
                    scheduler.step()
        
        # Track metrics (use unscaled loss for logging)
        total_loss += loss.item() * gradient_accumulation_steps
        predictions = outputs['logits'].argmax(dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
        
        # Clear GPU cache periodically
        if (batch_idx + 1) % 100 == 0:
            torch.cuda.empty_cache()
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move to device
            video_frames = batch['video_frames'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            audio_waveform = batch['audio_waveform'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(
                video_frames=video_frames,
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                audio_waveform=audio_waveform,
                metadata=metadata
            )
            loss = criterion(outputs['logits'], labels)
            
            # Track metrics
            total_loss += loss.item()
            predictions = outputs['logits'].argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    return avg_loss, accuracy, precision, recall, f1, np.array(all_predictions), np.array(all_labels)


def main():
    """Main training script."""
    
    # Configuration
    config = {
        # Model
        'num_classes': 7,  # neutral, joy, sadness, anger, fear, disgust, surprise
        'owlvit_model_name': 'google/owlvit-base-patch32',
        'wav2vec_model_name': 'facebook/wav2vec2-base-960h',
        'metadata_dim': 10,
        'hidden_dim': 512,  # Changed to 512 to match OwlViT dimensions
        'num_transformer_layers': 6,
        'num_heads': 8,  # Changed to 8 for 512-dim
        'ff_dim': 2048,  # Changed to 2048
        'dropout': 0.1,
        'freeze_owlvit': False,
        'freeze_wav2vec': True,
        
        # Dataset
        'train_json': 'json/mapped_train_data_video_aligned.json',
        'val_json': 'json/mapped_val_data_video_aligned.json',
        'test_json': 'json/mapped_test_data_video_aligned.json',
        'video_root': 'data/train_video',
        'audio_root': 'data/train_audio',
        'num_frames': 4,  # Reduced from 8 to save memory
        'audio_sample_rate': 16000,
        'audio_max_length': 48000,  # Reduced to 3 seconds from 5
        'max_text_length': 16,  # OwlViT max sequence length is 16
        
        # Training
        'batch_size': 1,  # Reduced from 4 to 1 for memory
        'num_epochs': 30,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'use_amp': True,
        'gradient_accumulation_steps': 16,  # Increased to maintain effective batch size
        
        # Paths
        'checkpoint_dir': 'owlvit_multimodal_checkpoints',
        'results_dir': 'owlvit_multimodal_results'
    }
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Initialize model
    print("\n" + "="*80)
    print("Initializing OwlViT Multimodal Transformer")
    print("="*80)
    
    model = OwlViTMultimodalEmotionModel(
        num_classes=config['num_classes'],
        owlvit_model_name=config['owlvit_model_name'],
        wav2vec_model_name=config['wav2vec_model_name'],
        metadata_dim=config['metadata_dim'],
        hidden_dim=config['hidden_dim'],
        num_transformer_layers=config['num_transformer_layers'],
        num_heads=config['num_heads'],
        ff_dim=config['ff_dim'],
        dropout=config['dropout'],
        freeze_owlvit=config['freeze_owlvit'],
        freeze_wav2vec=config['freeze_wav2vec']
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create datasets
    print("\n" + "="*80)
    print("Loading Datasets")
    print("="*80)
    
    train_dataset = MultimodalEmotionDataset(
        data_json_path=config['train_json'],
        video_root=config['video_root'],
        audio_root=config['audio_root'],
        owlvit_processor=model.owlvit_processor,
        wav2vec_processor=model.wav2vec_processor,
        num_frames=config['num_frames'],
        audio_sample_rate=config['audio_sample_rate'],
        audio_max_length=config['audio_max_length'],
        max_text_length=config['max_text_length']
    )
    
    val_dataset = MultimodalEmotionDataset(
        data_json_path=config['val_json'],
        video_root=config['video_root'],
        audio_root=config['audio_root'],
        owlvit_processor=model.owlvit_processor,
        wav2vec_processor=model.wav2vec_processor,
        num_frames=config['num_frames'],
        audio_sample_rate=config['audio_sample_rate'],
        audio_max_length=config['audio_max_length'],
        max_text_length=config['max_text_length']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # Reduced from 2 to save memory
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,  # Reduced from 2 to save memory
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Print dataset statistics
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    
    # Count emotions in training set
    emotion_counts = {}
    for i in range(min(len(train_dataset), 1000)):  # Sample first 1000 for quick stats
        sample = train_dataset.data[i]
        emotion = sample.get('chain_of_empathy', {}).get('speaker_emotion', 'neutral')
        basic_emotion = train_dataset._map_to_basic_emotion(emotion)
        emotion_name = list(train_dataset.emotion_mapping.keys())[basic_emotion]
        emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
    
    print(f"\n🎯 Emotion Distribution (first 1000 samples):")
    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sum(emotion_counts.values())) * 100
        print(f"  {emotion:>10}: {count:>4} ({percentage:>5.2f}%)")
    
    # Sample data verification
    print(f"\n📊 Sample Data Verification:")
    sample_idx = 0
    sample = train_dataset[sample_idx]
    print(f"  Video frames shape: {sample['video_frames'].shape}")
    print(f"  Audio waveform shape: {sample['audio_waveform'].shape}")
    print(f"  Text input IDs shape: {sample['text_input_ids'].shape}")
    print(f"  Metadata shape: {sample['metadata'].shape}")
    print(f"  Label: {sample['label'].item()}")
    
    print(f"{'='*80}\n")
    
    # Loss and optimizer
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    num_training_steps = len(train_loader) * config['num_epochs']
    num_warmup_steps = int(num_training_steps * config['warmup_ratio'])
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    print(f"\nTraining steps: {num_training_steps}")
    print(f"Warmup steps: {num_warmup_steps}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            use_amp=config['use_amp'],
            gradient_accumulation_steps=config['gradient_accumulation_steps']
        )
        
        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1, val_preds, val_labels = validate_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )
        
        # Log metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(
                config['checkpoint_dir'],
                f'best_owlvit_multimodal_acc{val_acc:.4f}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(config['results_dir'], 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print("="*80)


if __name__ == '__main__':
    main()
