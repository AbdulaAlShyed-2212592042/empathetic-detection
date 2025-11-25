"""
Test script for OwlViT Multimodal Transformer
Loads the best checkpoint and evaluates on test set
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import OwlViTProcessor, OwlViTModel, Wav2Vec2Processor, Wav2Vec2Model
import json
import cv2
import librosa
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import os


# Define the same model architecture as training
class AudioProjection(nn.Module):
    def __init__(self, wav2vec_dim=768, owlvit_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(wav2vec_dim, owlvit_dim * 2),
            nn.LayerNorm(owlvit_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(owlvit_dim * 2, owlvit_dim),
            nn.LayerNorm(owlvit_dim)
        )
    
    def forward(self, audio_features):
        return self.projection(audio_features)


class MetadataEmbedding(nn.Module):
    def __init__(self, metadata_dim=10, owlvit_dim=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(metadata_dim, owlvit_dim),
            nn.LayerNorm(owlvit_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(owlvit_dim, owlvit_dim),
            nn.LayerNorm(owlvit_dim)
        )
    
    def forward(self, metadata):
        return self.embedding(metadata).unsqueeze(1)


class UnifiedMultimodalTransformer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, num_layers=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.modality_embeddings = nn.Parameter(torch.randn(4, hidden_dim))
    
    def forward(self, video_features, text_features, audio_features, metadata_features):
        batch_size = video_features.size(0)
        
        video_features = video_features + self.modality_embeddings[0].unsqueeze(0).unsqueeze(0)
        text_features = text_features + self.modality_embeddings[1].unsqueeze(0).unsqueeze(0)
        audio_features = audio_features + self.modality_embeddings[2].unsqueeze(0).unsqueeze(0)
        metadata_features = metadata_features + self.modality_embeddings[3].unsqueeze(0).unsqueeze(0)
        
        combined_features = torch.cat([video_features, text_features, audio_features, metadata_features], dim=1)
        transformed_features = self.transformer(combined_features)
        normalized_features = self.norm(transformed_features)
        
        return normalized_features.mean(dim=1)


class OwlViTMultimodalEmotionModel(nn.Module):
    def __init__(self, num_classes=7, hidden_dim=512):
        super().__init__()
        
        self.owlvit = OwlViTModel.from_pretrained("google/owlvit-base-patch32")
        self.owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        
        for param in self.wav2vec.parameters():
            param.requires_grad = False
        
        # Vision projection (768 -> 512)
        self.vision_projection = nn.Linear(768, hidden_dim)
        
        # Text projection (512 -> 512, identity since already 512)
        self.text_projection = nn.Identity()
        
        self.audio_projection = AudioProjection(wav2vec_dim=768, owlvit_dim=hidden_dim, dropout=0.1)
        self.metadata_embedding = MetadataEmbedding(metadata_dim=10, owlvit_dim=hidden_dim, dropout=0.1)
        
        self.multimodal_transformer = UnifiedMultimodalTransformer(
            hidden_dim=hidden_dim,
            num_heads=8,
            num_layers=6,
            ff_dim=2048,
            dropout=0.1
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, videos, text_input_ids, text_attention_mask, audio_waveforms, metadata):
        batch_size, num_frames, C, H, W = videos.shape
        videos_flat = videos.view(batch_size * num_frames, C, H, W)
        
        with torch.no_grad():
            vision_outputs = self.owlvit.vision_model(pixel_values=videos_flat)
        
        vision_features = vision_outputs.last_hidden_state
        vision_features = vision_features.view(batch_size, num_frames * vision_features.size(1), vision_features.size(2))
        vision_features = self.vision_projection(vision_features)
        
        with torch.no_grad():
            text_outputs = self.owlvit.text_model(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask
            )
        
        text_features = text_outputs.last_hidden_state
        text_features = self.text_projection(text_features)
        
        with torch.no_grad():
            audio_outputs = self.wav2vec(audio_waveforms)
        
        audio_features = audio_outputs.last_hidden_state
        audio_features = self.audio_projection(audio_features)
        
        metadata_features = self.metadata_embedding(metadata)
        
        unified_features = self.multimodal_transformer(
            vision_features, text_features, audio_features, metadata_features
        )
        
        logits = self.classifier(unified_features)
        return logits


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, video_root, audio_root):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        emotion_mapping = {
            'neutral': 0, 'joyful': 1, 'sad': 2, 'angry': 3, 'fearful': 4, 'disgusted': 5, 'surprised': 6,
            'surprised_update': 6, 'excited': 1, 'proud': 1, 'grateful': 1, 'hopeful': 1, 'confident': 1, 'content': 1,
            'embarrassed': 2, 'disappointed': 2, 'lonely': 2, 'ashamed': 2, 'guilty': 2, 'devastated': 2,
            'annoyed': 3, 'furious': 3, 'jealous': 3, 'terrified': 4, 'apprehensive': 4, 'anxious': 4,
            'disgusted_update': 5, 'faithful': 1, 'anticipating': 1, 'caring': 1, 'trusting': 1, 'impressed': 1,
            'prepared': 1, 'sentimental': 1, 'nostalgic': 2
        }
        
        for conversation in data:
            turn = conversation.get('turn', {})
            dialogue_items = turn.get('dialogue', [])
            chain_of_empathy = turn.get('chain_of_empathy', {})
            speaker_profile = conversation.get('speaker_profile', {})
            topic = conversation.get('topic', '')
            
            for utterance in dialogue_items:
                if utterance.get('role') == 'speaker':
                    video_name = utterance.get('video_name', '')
                    audio_name = utterance.get('audio_name', '')
                    
                    video_file = os.path.join(video_root, video_name)
                    audio_file = os.path.join(audio_root, audio_name)
                    
                    if os.path.exists(video_file) and os.path.exists(audio_file):
                        raw_emotion = chain_of_empathy.get('speaker_emotion', '').lower()
                        label = emotion_mapping.get(raw_emotion, 0)
                        
                        self.samples.append({
                            'video_path': video_file,
                            'audio_path': audio_file,
                            'text': utterance.get('text', ''),
                            'label': label,
                            'metadata': self._encode_metadata(speaker_profile, topic, chain_of_empathy)
                        })
        
        print(f"Loaded {len(self.samples)} test samples")
    
    def _encode_metadata(self, speaker_profile, topic, chain_of_empathy):
        age_map = {'child': 0, 'young': 1, 'middle-aged': 2, 'senior': 3}
        gender_map = {'female': 0, 'male': 1}
        timbre_map = {'low': 0, 'mid': 1, 'high': 2}
        
        age = age_map.get(speaker_profile.get('age', 'young'), 1) / 3.0
        gender = gender_map.get(speaker_profile.get('gender', 'female'), 0)
        timbre = timbre_map.get(speaker_profile.get('timbre', 'mid'), 1) / 2.0
        
        topic_encoding = self._encode_topic(topic)
        event_encoding = self._encode_field(chain_of_empathy.get('event_scenario', ''))
        cause_encoding = self._encode_field(chain_of_empathy.get('emotion_cause', ''))
        goal_encoding = self._encode_field(chain_of_empathy.get('goal_to_response', ''))
        
        metadata = [age, gender, timbre] + topic_encoding + [event_encoding, cause_encoding, goal_encoding]
        return metadata
    
    def _encode_topic(self, topic):
        topics = ['Personal Struggles', 'Relationships', 'Life Events', 'Work', 'Health', 
                  'Achievements', 'Support', 'Social', 'General']
        encoding = [0.0, 0.0, 0.0, 0.0]
        
        for i, t in enumerate(topics[:4]):
            if t.lower() in topic.lower():
                encoding[i] = 1.0
                break
        
        return encoding
    
    def _encode_field(self, field):
        if not field or len(field) < 10:
            return 0.0
        return min(len(field) / 100.0, 1.0)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load video (4 frames)
        cap = cv2.VideoCapture(sample['video_path'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, 4, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (768, 768))
                frames.append(Image.fromarray(frame))
        cap.release()
        
        # Load audio (3 seconds)
        audio_waveform, sr = librosa.load(sample['audio_path'], sr=16000, duration=3.0)
        if len(audio_waveform) < 48000:
            audio_waveform = np.pad(audio_waveform, (0, 48000 - len(audio_waveform)))
        else:
            audio_waveform = audio_waveform[:48000]
        
        return {
            'frames': frames,
            'text': sample['text'],
            'audio': torch.FloatTensor(audio_waveform),
            'metadata': torch.FloatTensor(sample['metadata']),
            'label': sample['label']
        }


def collate_fn(batch):
    frames_list = [item['frames'] for item in batch]
    texts = [item['text'] for item in batch]
    audios = torch.stack([item['audio'] for item in batch])
    metadata = torch.stack([item['metadata'] for item in batch])
    labels = torch.LongTensor([item['label'] for item in batch])
    
    return {
        'frames': frames_list,
        'texts': texts,
        'audios': audios,
        'metadata': metadata,
        'labels': labels
    }


def test_model(checkpoint_path, test_json, video_root, audio_root, device='cuda'):
    print("=" * 80)
    print("OWLVIT MULTIMODAL TRANSFORMER - TEST EVALUATION")
    print("=" * 80)
    print(f"\n Loading checkpoint: {checkpoint_path}")
    
    # Load model
    model = OwlViTMultimodalEmotionModel(num_classes=7, hidden_dim=512)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f" Model loaded successfully!")
    print(f"   Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    val_acc = checkpoint.get('val_accuracy', None)
    if val_acc is not None:
        print(f"   Validation accuracy: {val_acc:.4f}")
    else:
        print(f"   Validation accuracy: N/A")
    
    # Load test dataset
    print(f"\n Loading test dataset from: {test_json}")
    test_dataset = TestDataset(test_json, video_root, audio_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"   Total test samples: {len(test_dataset)}")
    
    # Initialize processors
    owlvit_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    
    # Test loop
    print("\n Running inference on test set...")
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            # Process video frames
            videos_processed = []
            for frames in batch['frames']:
                video_inputs = owlvit_processor(images=frames, return_tensors="pt")
                videos_processed.append(video_inputs['pixel_values'])
            videos = torch.cat(videos_processed, dim=0).unsqueeze(0).to(device)
            
            # Process text
            text_inputs = owlvit_processor(
                text=batch['texts'],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=16
            )
            text_input_ids = text_inputs['input_ids'].to(device)
            text_attention_mask = text_inputs['attention_mask'].to(device)
            
            # Audio and metadata
            audios = batch['audios'].to(device)
            metadata = batch['metadata'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(videos, text_input_ids, text_attention_mask, audios, metadata)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    print(f"\n Overall Metrics:")
    print(f"   Accuracy:  {accuracy * 100:.2f}%")
    print(f"   Precision: {precision * 100:.2f}%")
    print(f"   Recall:    {recall * 100:.2f}%")
    print(f"   F1 Score:  {f1 * 100:.2f}%")
    
    # Per-class metrics
    emotion_names = ['Neutral', 'Joy', 'Sadness', 'Anger', 'Fear', 'Disgust', 'Surprise']
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    print(f"\n Per-Class Metrics:")
    print(f"{'Emotion':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 56)
    for i, emotion in enumerate(emotion_names):
        print(f"{emotion:<12} {precision_per_class[i]*100:>9.2f}% {recall_per_class[i]*100:>9.2f}% "
              f"{f1_per_class[i]*100:>9.2f}% {support_per_class[i]:>10}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\n Confusion Matrix:")
    print("     " + "  ".join([f"{e[:3]:>5}" for e in emotion_names]))
    for i, row in enumerate(cm):
        print(f"{emotion_names[i][:3]:>3}: " + "  ".join([f"{val:>5}" for val in row]))
    
    # Save results
    results = {
        'checkpoint': checkpoint_path,
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
        'per_class_metrics': {
            emotion_names[i]: {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            for i in range(len(emotion_names))
        },
        'confusion_matrix': cm.tolist()
    }
    
    output_file = 'owlvit_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n Results saved to: {output_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    # Configuration
    CHECKPOINT_PATH = "owlvit_multimodal_checkpoints/best_owlvit_multimodal_acc0.6389.pth"
    TEST_JSON = "json/mapped_test_data_video_aligned.json"
    VIDEO_ROOT = "data/for testing"
    AUDIO_ROOT = "data/train_audio"  # Assuming audio files are in train_audio
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {DEVICE}")
    
    # Run test
    results = test_model(CHECKPOINT_PATH, TEST_JSON, VIDEO_ROOT, AUDIO_ROOT, DEVICE)
