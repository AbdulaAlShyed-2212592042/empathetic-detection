#!/usr/bin/env python3
"""
Comprehensive Test Script for train_2.py Model Evaluation
Loads the best_7class_model.pth model from checkpoints_2
and evaluates on mapped_test_data_video_aligned.json with video, text, and metadata
"""

# Suppress FutureWarnings from transformers library
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.utils._pytree.*")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
import time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import pandas as pd

# Set CUDA memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

# Import model components from train_2.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_2 import (
        VideoTransform,
        MultimodalSequentialVideoDataset, 
        TimeSformerEncoder,
        MultimodalLSTMVideoModel,
        FocalLoss,
        compute_metrics
    )
    print("✅ Successfully imported components from train_2.py")
    USING_LOCAL_DEFS = False
except ImportError as e:
    print(f"❌ Error importing from train_2.py: {e}")
    print("🔄 Using local definitions...")
    USING_LOCAL_DEFS = True
    
    # Define local versions if import fails
    class VideoTransform:
        """TimeSformer video preprocessing pipeline"""
        
        def __init__(self, num_frames=8, frame_size=192, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
            self.num_frames = num_frames
            self.frame_size = frame_size
            self.mean = mean
            self.std = std
        
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
                
                frame_step = max(1, total_frames // (self.num_frames + 2))
                frame_indices = list(range(0, min(total_frames, frame_step * self.num_frames), frame_step))[:self.num_frames]
                
                frames = []
                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.resize(frame, (self.frame_size, self.frame_size), interpolation=cv2.INTER_AREA)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = torch.from_numpy(frame).float().div(255.0).permute(2, 0, 1)
                        frames.append(frame)
                    else:
                        if frames:
                            frames.append(frames[-1])
                        else:
                            frames.append(torch.zeros(3, self.frame_size, self.frame_size))
                
                cap.release()
                
                while len(frames) < self.num_frames:
                    frames.append(frames[-1] if frames else torch.zeros(3, self.frame_size, self.frame_size))
                
                video_tensor = torch.stack(frames[:self.num_frames], dim=1)
                
                mean_tensor = torch.tensor(self.mean).view(3, 1, 1, 1)
                std_tensor = torch.tensor(self.std).view(3, 1, 1, 1)
                video_tensor = (video_tensor - mean_tensor) / std_tensor
                
                return video_tensor
                
            except Exception as e:
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)

def load_training_vocabularies():
    """Load vocabularies from training dataset to ensure compatibility"""
    print("📚 Loading training vocabularies...")
    
    try:
        # Try to load from train_2.py
        from train_2 import MultimodalSequentialVideoDataset
        
        # Create temporary training dataset to get vocabularies
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        video_transform = VideoTransform(frame_size=192)
        
        train_dataset = MultimodalSequentialVideoDataset(
            'json/mapped_train_data_video_aligned.json',
            'data/train_video/video_v5_0',
            tokenizer,
            video_transform,
            max_dialogue_length=10
        )
        
        return train_dataset
        
    except Exception as e:
        print(f"⚠️ Could not load training vocabularies: {e}")
        return None

def load_best_model_with_vocab_fix(checkpoints_dir, device):
    """Load the best saved model and handle vocabulary size mismatches"""
    
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Try best_7class_model.pth (the specified model)
    model_file = checkpoints_path / "best_7class_model.pth"
    if not model_file.exists():
        # Fallback to finding any best model with highest accuracy
        model_files = list(checkpoints_path.glob("best_*_acc*.pth"))
        if not model_files:
            raise FileNotFoundError(f"No best model files found in {checkpoints_dir}")
        
        def extract_accuracy(filename):
            import re
            match = re.search(r'acc(\d+\.\d+)', str(filename))
            return float(match.group(1)) if match else 0.0
        
        model_file = max(model_files, key=extract_accuracy)
    
    print(f"📦 Loading model: {model_file.name}")
    
    # Load checkpoint
    checkpoint = torch.load(model_file, map_location=device)
    
    # Get vocab sizes from the saved model state dict
    state_dict = checkpoint['model_state_dict']
    
    # Extract actual vocab sizes from the saved embeddings
    vocab_sizes = {
        'event_scenario': state_dict['event_scenario_embedding.weight'].shape[0],
        'emotion_cause': state_dict['emotion_cause_embedding.weight'].shape[0], 
        'goal_response': state_dict['goal_response_embedding.weight'].shape[0],
        'topic': state_dict['topic_embedding.weight'].shape[0]
    }
    
    print(f"📋 Model vocab sizes: {vocab_sizes}")
    
    return checkpoint, model_file, vocab_sizes

def create_test_dataset_with_fixed_vocab(tokenizer, video_transform, vocab_sizes, max_dialogue_length=10):
    """Create test dataset with vocabulary sizes matching the saved model"""
    
    print("📊 Creating test dataset with fixed vocabularies...")
    
    # Load test data
    test_data_file = 'json/mapped_test_data_video_aligned.json'
    video_dir = 'data/train_video/video_v5_0'
    
    # Create a custom dataset class that handles vocab size differences
    class FixedVocabTestDataset(Dataset):
        def __init__(self, json_file, video_dir, tokenizer, video_transform, vocab_sizes, max_dialogue_length=10):
            self.json_file = json_file
            self.video_dir = video_dir
            self.tokenizer = tokenizer
            self.video_transform = video_transform
            self.max_dialogue_length = max_dialogue_length
            
            # Load data
            with open(json_file, 'r') as f:
                self.data = json.load(f)
            
            # Create emotion mapping (7-class) - same as train_2.py
            self.ed_emotion_projection = {
                'anticipating': 'anticipating', 'excited': 'excited', 'joyful': 'joyful',
                'proud': 'proud', 'hopeful': 'hopeful', 'grateful': 'grateful',
                'content': 'content', 'confident': 'confident', 'prepared': 'prepared',
                'caring': 'caring', 'trusting': 'trusting', 'faithful': 'faithful',
                'impressed': 'impressed', 'surprised': 'surprised',
                'annoyed': 'annoyed', 'angry': 'angry', 'furious': 'furious',
                'jealous': 'jealous', 'disgusted': 'disgusted',
                'apprehensive': 'apprehensive', 'anxious': 'anxious', 'afraid': 'afraid',
                'terrified': 'terrified', 'devastated': 'devastated', 'sad': 'sad',
                'disappointed': 'disappointed', 'lonely': 'lonely', 'guilty': 'guilty',
                'embarrassed': 'embarrassed', 'ashamed': 'ashamed', 'sentimental': 'sentimental'
            }
            
            # 7-class emotion mapping
            emotion_groups = {
                'anticipating': ['anticipating', 'excited', 'joyful', 'proud', 'hopeful'],
                'grateful': ['grateful', 'content', 'confident', 'prepared'],
                'annoyed': ['annoyed', 'angry', 'furious', 'jealous', 'disgusted'],
                'apprehensive': ['apprehensive', 'anxious', 'afraid', 'terrified'],
                'sad': ['devastated', 'sad', 'disappointed', 'lonely'],
                'embarrassed': ['guilty', 'embarrassed', 'ashamed'],
                'sentimental': ['sentimental', 'caring', 'trusting', 'faithful', 'impressed', 'surprised']
            }
            
            # Create reverse mapping
            self.emotion_to_group = {}
            for group, emotions in emotion_groups.items():
                for emotion in emotions:
                    self.emotion_to_group[emotion] = group
            
            # Create emotion to ID mapping
            unique_emotions = list(emotion_groups.keys())
            self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
            self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}
            
            # Metadata mappings
            self.age_to_id = {'child': 0, 'young': 1, 'middle-aged': 2, 'elderly': 3}
            self.gender_to_id = {'male': 0, 'female': 1}
            self.timbre_to_id = {'low': 0, 'mid': 1, 'high': 2}
            
            # Create dummy vocabularies with the exact sizes from the model
            self.event_scenario_vocab = {f'scenario_{i}': i for i in range(vocab_sizes['event_scenario'] - 1)}
            self.emotion_cause_vocab = {f'cause_{i}': i for i in range(vocab_sizes['emotion_cause'] - 1)}
            self.goal_response_vocab = {f'goal_{i}': i for i in range(vocab_sizes['goal_response'] - 1)}
            self.topic_vocab = {f'topic_{i}': i for i in range(vocab_sizes['topic'] - 1)}
            
            print(f"Test dataset: {len(self.data)} samples")
            print(f"Emotion classes: {len(self.emotion_to_id)}")
            print(f"Fixed vocab sizes: {vocab_sizes}")
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            
            # Extract turn data
            turn = item.get('turn', {})
            chain_of_empathy = turn.get('chain_of_empathy', {})
            
            # Get emotion label
            raw_emotion = chain_of_empathy.get('speaker_emotion', None)
            mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
            group_emotion = self.emotion_to_group.get(mapped_emotion, 'sentimental')
            emotion_label = self.emotion_to_id.get(group_emotion, 0)
            
            # Extract dialogue sequence
            dialogue = turn.get('dialogue', [])
            
            # Process dialogue with video
            dialogue_encodings = []
            dialogue_video_features = []
            dialogue_roles = []
            dialogue_indices = []
            
            for i, utterance in enumerate(dialogue[:self.max_dialogue_length]):
                # Text encoding
                text = utterance.get('text', '')
                encoding = self.tokenizer(
                    text, truncation=True, padding='max_length',
                    max_length=128, return_tensors='pt'
                )
                dialogue_encodings.append({
                    'input_ids': encoding['input_ids'].squeeze(),
                    'attention_mask': encoding['attention_mask'].squeeze()
                })
                
                # Video processing
                video_file = utterance.get('video_name', '') or utterance.get('video_file', '')
                if video_file:
                    video_path = os.path.join(self.video_dir, video_file)
                    if os.path.exists(video_path):
                        video_features = self.video_transform.extract_frames(video_path)
                    else:
                        video_features = torch.zeros(3, 8, self.video_transform.frame_size, self.video_transform.frame_size)
                else:
                    video_features = torch.zeros(3, 8, self.video_transform.frame_size, self.video_transform.frame_size)
                
                dialogue_video_features.append(video_features)
                
                # Role 
                role = utterance.get('role', 'speaker')
                role_id = {'speaker': 0, 'listener': 1, 'listener_response': 2}.get(role, 0)
                dialogue_roles.append(role_id)
                
                # Index
                dialogue_indices.append(utterance.get('index', i))
            
            # Pad sequences to max_dialogue_length
            while len(dialogue_encodings) < self.max_dialogue_length:
                dialogue_encodings.append({
                    'input_ids': torch.zeros(128, dtype=torch.long),
                    'attention_mask': torch.zeros(128, dtype=torch.long)
                })
                dialogue_video_features.append(torch.zeros(3, 8, self.video_transform.frame_size, self.video_transform.frame_size))
                dialogue_roles.append(0)
                dialogue_indices.append(0)
            
            # Context processing
            context = turn.get('context', '')
            if not context or context.strip() == '':
                context = '[NO CONTEXT]'
            context_encoding = self.tokenizer(
                context, truncation=True, padding='max_length',
                max_length=256, return_tensors='pt'
            )
            
            # Extract metadata
            speaker_profile = item.get('speaker_profile', {})
            listener_profile = item.get('listener_profile', {})
            
            # Profile features with safe defaults
            speaker_age = self.age_to_id.get(speaker_profile.get('age', 'young'), 1)
            speaker_gender = self.gender_to_id.get(speaker_profile.get('gender', 'male'), 0)
            speaker_timbre = self.timbre_to_id.get(speaker_profile.get('timbre', 'mid'), 1)
            speaker_id = min(speaker_profile.get('ID', 0), 99)  # Clamp to valid range
            
            listener_age = self.age_to_id.get(listener_profile.get('age', 'young'), 1)
            listener_gender = self.gender_to_id.get(listener_profile.get('gender', 'male'), 0)
            listener_timbre = self.timbre_to_id.get(listener_profile.get('timbre', 'mid'), 1)
            listener_id = min(listener_profile.get('ID', 0), 99)  # Clamp to valid range
            
            # Chain of empathy metadata - use hash for mapping to vocab range
            event_scenario = chain_of_empathy.get('event_scenario', '')
            emotion_cause = chain_of_empathy.get('emotion_cause', '')
            goal_to_response = chain_of_empathy.get('goal_to_response', '')
            topic = item.get('topic', '')
            
            # Safe mapping to vocabulary IDs
            event_scenario_id = hash(event_scenario) % (vocab_sizes['event_scenario'] - 1)
            emotion_cause_id = hash(emotion_cause) % (vocab_sizes['emotion_cause'] - 1)
            goal_response_id = hash(goal_to_response) % (vocab_sizes['goal_response'] - 1)
            topic_id = hash(topic) % (vocab_sizes['topic'] - 1)
            
            # Create metadata tensor
            metadata_features = torch.tensor([
                speaker_age, speaker_gender, speaker_timbre, speaker_id,
                listener_age, listener_gender, listener_timbre, listener_id,
                event_scenario_id, emotion_cause_id, goal_response_id, topic_id
            ], dtype=torch.long)
            
            # Ensure video features have consistent dimensions
            consistent_video_features = []
            target_size = (3, 8, 192, 192)
            
            for video_feat in dialogue_video_features:
                if video_feat.shape != target_size:
                    if video_feat.shape[0] == 3 and video_feat.shape[1] == 8:
                        resized_frames = []
                        for t in range(video_feat.shape[1]):
                            frame = video_feat[:, t, :, :]
                            frame_resized = F.interpolate(
                                frame.unsqueeze(0), size=(192, 192), mode='bilinear', align_corners=False
                            ).squeeze(0)
                            resized_frames.append(frame_resized)
                        video_feat = torch.stack(resized_frames, dim=1)
                    else:
                        video_feat = torch.zeros(target_size)
                consistent_video_features.append(video_feat)
            
            dialogue_video_tensor = torch.stack(consistent_video_features)
            
            # Calculate actual sequence length
            actual_seq_length = len(dialogue)
            
            return {
                'context_input_ids': context_encoding['input_ids'].squeeze(),
                'context_attention_mask': context_encoding['attention_mask'].squeeze(),
                'dialogue_input_ids': torch.stack([enc['input_ids'] for enc in dialogue_encodings]),
                'dialogue_attention_mask': torch.stack([enc['attention_mask'] for enc in dialogue_encodings]),
                'dialogue_video': dialogue_video_tensor,
                'dialogue_roles': torch.tensor(dialogue_roles, dtype=torch.long),
                'dialogue_indices': torch.tensor(dialogue_indices, dtype=torch.long),
                'metadata': metadata_features,
                'sequence_length': torch.tensor(actual_seq_length, dtype=torch.long),
                'emotion_labels': emotion_label,
                'conversation_id': item.get('conversation_id', ''),
                'raw_emotion': raw_emotion,
                'mapped_emotion': mapped_emotion,
                'group_emotion': group_emotion
            }
    
    # Create the test dataset
    test_dataset = FixedVocabTestDataset(
        test_data_file, video_dir, tokenizer, video_transform, vocab_sizes, max_dialogue_length
    )
    
    return test_dataset
    """Dataset for multimodal video-text emotion recognition with metadata"""
    
    def __init__(self, json_file, video_dir, tokenizer, video_transform=None, max_dialogue_length=10):
        self.json_file = json_file
        self.video_dir = video_dir
        self.tokenizer = tokenizer
        self.video_transform = video_transform or VideoTransform()
        self.max_dialogue_length = max_dialogue_length
        
        # Load and process data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        # Create emotion mapping (7-class)
        self.ed_emotion_projection = {
            'anticipating': 'anticipating', 'excited': 'excited', 'joyful': 'joyful',
            'proud': 'proud', 'hopeful': 'hopeful', 'grateful': 'grateful',
            'content': 'content', 'confident': 'confident', 'prepared': 'prepared',
            'caring': 'caring', 'trusting': 'trusting', 'faithful': 'faithful',
            'impressed': 'impressed', 'surprised': 'surprised',
            'annoyed': 'annoyed', 'angry': 'angry', 'furious': 'furious',
            'jealous': 'jealous', 'disgusted': 'disgusted',
            'apprehensive': 'apprehensive', 'anxious': 'anxious', 'afraid': 'afraid',
            'terrified': 'terrified', 'devastated': 'devastated', 'sad': 'sad',
            'disappointed': 'disappointed', 'lonely': 'lonely', 'guilty': 'guilty',
            'embarrassed': 'embarrassed', 'ashamed': 'ashamed', 'sentimental': 'sentimental'
        }
        
        # 7-class emotion mapping
        emotion_groups = {
            'anticipating': ['anticipating', 'excited', 'joyful', 'proud', 'hopeful'],
            'grateful': ['grateful', 'content', 'confident', 'prepared'],
            'annoyed': ['annoyed', 'angry', 'furious', 'jealous', 'disgusted'],
            'apprehensive': ['apprehensive', 'anxious', 'afraid', 'terrified'],
            'sad': ['devastated', 'sad', 'disappointed', 'lonely'],
            'embarrassed': ['guilty', 'embarrassed', 'ashamed'],
            'sentimental': ['sentimental', 'caring', 'trusting', 'faithful', 'impressed', 'surprised']
        }
        
        # Create reverse mapping
        self.emotion_to_group = {}
        for group, emotions in emotion_groups.items():
            for emotion in emotions:
                self.emotion_to_group[emotion] = group
        
        # Create emotion to ID mapping
        unique_emotions = list(emotion_groups.keys())
        self.emotion_to_id = {emotion: idx for idx, emotion in enumerate(unique_emotions)}
        self.id_to_emotion = {idx: emotion for emotion, idx in self.emotion_to_id.items()}
        
        # Metadata mappings
        self.age_to_id = {'child': 0, 'young': 1, 'middle-aged': 2, 'elderly': 3}
        self.gender_to_id = {'male': 0, 'female': 1}
        self.timbre_to_id = {'low': 0, 'mid': 1, 'high': 2}
        
        # Build vocabularies
        self._build_vocabularies()
        
        print(f"Test dataset loaded: {len(self.data)} samples")
        print(f"Emotion classes: {len(self.emotion_to_id)}")
        print(f"Created vocabularies: {len(self.event_scenario_vocab)} scenarios, "
              f"{len(self.emotion_cause_vocab)} causes, {len(self.goal_response_vocab)} goals, "
              f"{len(self.topic_vocab)} topics")
    
    def _build_vocabularies(self):
        """Build vocabularies from the dataset"""
        scenarios = set()
        causes = set()
        goals = set()
        topics = set()
        
        for item in self.data:
            turn = item.get('turn', {})
            chain_of_empathy = turn.get('chain_of_empathy', {})
            
            scenarios.add(chain_of_empathy.get('event_scenario', ''))
            causes.add(chain_of_empathy.get('emotion_cause', ''))
            goals.add(chain_of_empathy.get('goal_to_response', ''))
            topics.add(item.get('topic', ''))
        
        # Create mappings
        self.event_scenario_vocab = {scenario: idx for idx, scenario in enumerate(sorted(scenarios))}
        self.emotion_cause_vocab = {cause: idx for idx, cause in enumerate(sorted(causes))}
        self.goal_response_vocab = {goal: idx for idx, goal in enumerate(sorted(goals))}
        self.topic_vocab = {topic: idx for idx, topic in enumerate(sorted(topics))}
    
    def __len__(self):
        return len(self.data)
    
    def get_label(self, idx):
        """Get label without processing other data"""
        item = self.data[idx]
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        group_emotion = self.emotion_to_group.get(mapped_emotion, 'sentimental')
        emotion_label = self.emotion_to_id.get(group_emotion, 0)
        return emotion_label
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Extract turn data
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        
        # Get emotion label
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        group_emotion = self.emotion_to_group.get(mapped_emotion, 'sentimental')
        emotion_label = self.emotion_to_id.get(group_emotion, 0)
        
        # Extract dialogue sequence
        dialogue = turn.get('dialogue', [])
        
        # Process dialogue with video
        dialogue_encodings = []
        dialogue_video_features = []
        dialogue_roles = []
        dialogue_indices = []
        
        for i, utterance in enumerate(dialogue[:self.max_dialogue_length]):
            # Text encoding
            text = utterance.get('text', '')
            encoding = self.tokenizer(
                text, truncation=True, padding='max_length',
                max_length=128, return_tensors='pt'
            )
            dialogue_encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
            
            # Video processing
            video_file = utterance.get('video_file', '')
            if video_file:
                video_path = os.path.join(self.video_dir, video_file)
                video_features = self.video_transform.extract_frames(video_path)
            else:
                video_features = torch.zeros(3, 8, self.video_transform.frame_size, self.video_transform.frame_size)
            
            dialogue_video_features.append(video_features)
            
            # Role and index
            role = utterance.get('role', 'speaker')
            role_id = {'speaker': 0, 'listener': 1, 'listener_response': 2}.get(role, 0)
            dialogue_roles.append(role_id)
            dialogue_indices.append(utterance.get('index', i))
        
        # Pad sequences to max_dialogue_length
        while len(dialogue_encodings) < self.max_dialogue_length:
            dialogue_encodings.append({
                'input_ids': torch.zeros(128, dtype=torch.long),
                'attention_mask': torch.zeros(128, dtype=torch.long)
            })
            dialogue_video_features.append(torch.zeros(3, 8, self.video_transform.frame_size, self.video_transform.frame_size))
            dialogue_roles.append(0)
            dialogue_indices.append(0)
        
        # Context processing
        context = turn.get('context', '')
        if not context or context.strip() == '':
            context = '[NO CONTEXT]'
        context_encoding = self.tokenizer(
            context, truncation=True, padding='max_length',
            max_length=256, return_tensors='pt'
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
        
        # Calculate actual sequence length (excluding padding)
        actual_seq_length = len(dialogue)
        
        return {
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'dialogue_input_ids': torch.stack([enc['input_ids'] for enc in dialogue_encodings]),
            'dialogue_attention_mask': torch.stack([enc['attention_mask'] for enc in dialogue_encodings]),
            'dialogue_video': dialogue_video_tensor,
            'dialogue_roles': torch.tensor(dialogue_roles, dtype=torch.long),
            'metadata': metadata_features,
            'sequence_length': torch.tensor(actual_seq_length, dtype=torch.long),
            'emotion_labels': emotion_label,  # Match train_2.py naming
            'conversation_id': item.get('conversation_id', ''),
            'raw_emotion': raw_emotion,
            'mapped_emotion': mapped_emotion,
            'group_emotion': group_emotion
        }

class TimeSformerEncoder(nn.Module):
    """TimeSformer-style video encoder with CNN + Transformer - exact match to train_2.py"""
    
    def __init__(self, num_frames=8, frame_size=192, patch_size=32, embed_dim=256, num_heads=4, num_layers=3, dropout_rate=0.1):
        super(TimeSformerEncoder, self).__init__()
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Calculate number of patches per frame
        self.num_patches_per_frame = (frame_size // patch_size) ** 2
        self.total_patches = num_frames * self.num_patches_per_frame
        
        # Patch embedding layer (CNN backbone)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Positional embeddings
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, self.num_patches_per_frame, embed_dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout_rate,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection - EXACT MATCH to train_2.py
        self.output_projection = nn.Linear(embed_dim, 256)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
    
    def forward(self, video):
        """
        Forward pass for video sequence - exact match to train_2.py
        
        Args:
            video: [batch_size, seq_len, channels, num_frames, height, width]
            
        Returns:
            video_features: [batch_size, seq_len, 256]
        """
        batch_size, seq_len, channels, num_frames, height, width = video.shape
        
        # Process each sequence element
        sequence_features = []
        
        for i in range(seq_len):
            # Get video for this sequence position [batch_size, channels, num_frames, height, width]
            seq_video = video[:, i]  # [batch_size, 3, num_frames, height, width]
            
            # Reshape for patch embedding: [batch_size * num_frames, channels, height, width]
            reshaped_video = seq_video.transpose(1, 2).contiguous()  # [batch_size, num_frames, 3, height, width]
            reshaped_video = reshaped_video.reshape(batch_size * num_frames, channels, height, width)
            
            # Extract patches using CNN
            patches = self.patch_embed(reshaped_video)  # [batch_size * num_frames, embed_dim, H//patch_size, W//patch_size]
            
            # Flatten spatial dimensions
            patches = patches.flatten(2).transpose(1, 2)  # [batch_size * num_frames, num_patches_per_frame, embed_dim]
            
            # Reshape back to separate batch and time dimensions
            patches = patches.reshape(batch_size, num_frames, self.num_patches_per_frame, self.embed_dim)
            
            # Add spatial positional embeddings
            patches = patches + self.pos_embed_spatial.unsqueeze(1)  # Add spatial pos to each frame
            
            # Reshape for temporal processing: [batch_size, num_frames * num_patches_per_frame, embed_dim]
            patches = patches.reshape(batch_size, num_frames * self.num_patches_per_frame, self.embed_dim)
            
            # Add temporal positional embeddings (repeat for each spatial patch)
            temporal_pos = self.pos_embed_temporal.repeat_interleave(self.num_patches_per_frame, dim=1)
            patches = patches + temporal_pos
            
            # Apply transformer
            features = self.transformer(patches)  # [batch_size, total_patches, embed_dim]
            
            # Global average pooling across all patches
            pooled_features = features.mean(dim=1)  # [batch_size, embed_dim]
            
            # Project to output dimension
            output_features = self.output_projection(pooled_features)  # [batch_size, 256]
            output_features = self.dropout(output_features)
            
            sequence_features.append(output_features)
        
        # Stack sequence features
        final_features = torch.stack(sequence_features, dim=1)  # [batch_size, seq_len, 256]
        
        return final_features

class MultimodalLSTMVideoModel(nn.Module):
    """Multimodal LSTM model with BERT + TimeSformer - EXACT MATCH to train_2.py"""
    
    def __init__(self, dataset, num_classes=7, hidden_size=256, num_layers=2, dropout_rate=0.3):
        super(MultimodalLSTMVideoModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Text encoder (BERT-base)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
        
        # Video encoder (TimeSformer) - CPU-optimized for faster processing
        self.video_encoder = TimeSformerEncoder(
            num_frames=8, frame_size=192, patch_size=32,  # Reduced resolution for faster processing
            embed_dim=256, num_heads=4, num_layers=3, dropout_rate=dropout_rate
        )
        video_dim = 256  # TimeSformer output dimension
        
        # Metadata embeddings (exact copy from train_2.py)
        self.age_embedding = nn.Embedding(4, 16)
        self.gender_embedding = nn.Embedding(2, 8)
        self.timbre_embedding = nn.Embedding(3, 8)
        self.role_embedding = nn.Embedding(3, 16)
        
        # Profile ID embeddings - EXACT MATCH
        self.speaker_id_embedding = nn.Embedding(100, 32)
        self.listener_id_embedding = nn.Embedding(100, 32)
        
        # Chain of empathy embeddings - EXACT MATCH
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
        
        # Context processor - EXACT MATCH
        self.context_processor = nn.Linear(text_dim, 256)
        
        # Sequential processing with LSTM - EXACT MATCH
        lstm_input_size = text_dim + video_dim + 16  # text + video + role
        
        self.dialogue_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for LSTM outputs - EXACT MATCH
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Metadata fusion - EXACT MATCH
        metadata_dim = 16 + 8 + 8 + 32 + 16 + 8 + 8 + 32 + 32 + 32 + 32 + 32  # All embeddings
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64)
        )
        
        # Final fusion and classification - EXACT MATCH
        final_dim = 256 + hidden_size * 2 + 64  # context + LSTM + metadata
        
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, context_input_ids, context_attention_mask, dialogue_input_ids, 
                dialogue_attention_mask, dialogue_video, dialogue_roles, metadata, sequence_length):
        """Forward pass - EXACT MATCH to train_2.py"""
        batch_size, seq_len, token_len = dialogue_input_ids.shape
        
        # Process context
        context_outputs = self.text_encoder(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_features = self.context_processor(context_outputs.last_hidden_state[:, 0, :])  # CLS token
        
        # Process dialogue sequence
        dialogue_features = []
        
        for i in range(seq_len):
            # Text features for this utterance
            utt_outputs = self.text_encoder(
                input_ids=dialogue_input_ids[:, i, :],
                attention_mask=dialogue_attention_mask[:, i, :]
            )
            utt_text_features = utt_outputs.last_hidden_state[:, 0, :]  # CLS token
            
            # Video features for this utterance
            utt_video_raw = dialogue_video[:, i:i+1]  # [batch, 1, C, T, H, W]
            utt_video_features = self.video_encoder(utt_video_raw).squeeze(1)  # [batch, 256]
            
            # Role embedding
            utt_role_features = self.role_embedding(dialogue_roles[:, i])
            
            # Combine features for this utterance
            utt_combined = torch.cat([utt_text_features, utt_video_features, utt_role_features], dim=1)
            dialogue_features.append(utt_combined)
        
        # Stack dialogue features
        dialogue_sequence = torch.stack(dialogue_features, dim=1)  # [batch, seq_len, features]
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.dialogue_lstm(dialogue_sequence)
        
        # Apply attention to LSTM outputs
        lstm_out_transposed = lstm_out.transpose(0, 1)  # [seq_len, batch, features]
        attended_output, _ = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attended_output = attended_output.transpose(0, 1)  # [batch, seq_len, features]
        
        # Use the last relevant output based on sequence length
        dialogue_final_features = []
        for i in range(batch_size):
            seq_len_i = sequence_length[i].item() - 1  # 0-indexed
            seq_len_i = max(0, min(seq_len_i, seq_len - 1))  # Clamp to valid range
            dialogue_final_features.append(attended_output[i, seq_len_i, :])
        dialogue_final_features = torch.stack(dialogue_final_features)
        
        # Process metadata (exact copy from train_2.py)
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
        
        metadata_features = self.metadata_processor(metadata_combined)
        
        # Final fusion
        final_features = torch.cat([context_features, dialogue_final_features, metadata_features], dim=1)
        final_features = self.dropout(final_features)
        
        # Classification
        logits = self.classifier(final_features)
        
        return logits

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model, test_loader, device, emotion_classes, results_dir, checkpoint=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.emotion_classes = emotion_classes
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoint = checkpoint
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def evaluate(self):
        """Run comprehensive evaluation"""
        print("🔍 Starting comprehensive model evaluation...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_conversation_ids = []
        all_raw_emotions = []
        all_mapped_emotions = []
        
        total_batches = len(self.test_loader)
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):
                # Move batch to device
                context_input_ids = batch['context_input_ids'].to(self.device)
                context_attention_mask = batch['context_attention_mask'].to(self.device)
                dialogue_input_ids = batch['dialogue_input_ids'].to(self.device)
                dialogue_attention_mask = batch['dialogue_attention_mask'].to(self.device)
                dialogue_video = batch['dialogue_video'].to(self.device)
                dialogue_roles = batch['dialogue_roles'].to(self.device)
                dialogue_indices = batch['dialogue_indices'].to(self.device)
                metadata = batch['metadata'].to(self.device)
                sequence_length = batch['sequence_length'].to(self.device)
                labels = batch['emotion_labels'].to(self.device)
                
                # Forward pass
                logits = self.model(
                    context_input_ids, context_attention_mask,
                    dialogue_input_ids, dialogue_attention_mask,
                    dialogue_video, dialogue_roles, dialogue_indices, 
                    metadata, sequence_length
                )
                
                # Get probabilities and predictions
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_conversation_ids.extend(batch['conversation_id'])
                all_raw_emotions.extend(batch['raw_emotion'])
                all_mapped_emotions.extend(batch['mapped_emotion'])
                
                # Clear GPU cache periodically
                if (batch_idx + 1) % 25 == 0:
                    torch.cuda.empty_cache()
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Generate reports and plots
        self._generate_classification_report(all_labels, all_predictions, metrics)
        self._plot_confusion_matrix(all_labels, all_predictions)
        self._plot_class_distribution(all_labels, all_predictions)
        self._plot_roc_curves(all_labels, all_probabilities)
        self._plot_per_class_metrics(metrics)
        
        # Plot training history if checkpoint is available
        if self.checkpoint:
            self._plot_accuracy_graph(self.checkpoint)
            self._plot_loss_graph(self.checkpoint)
            self._plot_training_history(self.checkpoint)
        
        self._save_detailed_results(
            all_predictions, all_labels, all_probabilities,
            all_conversation_ids, all_raw_emotions, all_mapped_emotions, metrics
        )
        
        return metrics
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1 (macro and weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # ROC-AUC (multiclass)
        try:
            y_true_binarized = label_binarize(y_true, classes=list(range(len(self.emotion_classes))))
            roc_auc_macro = roc_auc_score(y_true_binarized, y_prob, average='macro', multi_class='ovr')
            roc_auc_weighted = roc_auc_score(y_true_binarized, y_prob, average='weighted', multi_class='ovr')
        except:
            roc_auc_macro = 0.0
            roc_auc_weighted = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'roc_auc_macro': roc_auc_macro,
            'roc_auc_weighted': roc_auc_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support_per_class': support_per_class,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def _generate_classification_report(self, y_true, y_pred, metrics):
        """Generate and save comprehensive classification report"""
        
        # Create detailed report
        report = classification_report(
            y_true, y_pred, target_names=self.emotion_classes,
            output_dict=True, zero_division=0
        )
        
        # Create formatted text report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"evaluation_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Evaluation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test samples: {len(y_true)}\n\n")
            
            f.write("OVERALL PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n\n")
            
            f.write("MACRO AVERAGES:\n")
            f.write(f"Precision (Macro): {metrics['precision_macro']:.4f}\n")
            f.write(f"Recall (Macro): {metrics['recall_macro']:.4f}\n")
            f.write(f"F1-Score (Macro): {metrics['f1_macro']:.4f}\n")
            f.write(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}\n\n")
            
            f.write("WEIGHTED AVERAGES:\n")
            f.write(f"Precision (Weighted): {metrics['precision_weighted']:.4f}\n")
            f.write(f"Recall (Weighted): {metrics['recall_weighted']:.4f}\n")
            f.write(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}\n")
            f.write(f"ROC-AUC (Weighted): {metrics['roc_auc_weighted']:.4f}\n\n")
            
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-" * 60 + "\n")
            
            for i, class_name in enumerate(self.emotion_classes):
                precision = metrics['precision_per_class'][i]
                recall = metrics['recall_per_class'][i]
                f1 = metrics['f1_per_class'][i]
                support = metrics['support_per_class'][i]
                f.write(f"{class_name:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {support:<10.0f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SKLEARN CLASSIFICATION REPORT:\n")
            f.write("=" * 80 + "\n")
            f.write(classification_report(y_true, y_pred, target_names=self.emotion_classes, zero_division=0))
        
        print(f"📄 Evaluation report saved: {report_file}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Generate and save confusion matrix plot"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot absolute confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.emotion_classes, yticklabels=self.emotion_classes, ax=ax1)
        ax1.set_title('Confusion Matrix (Absolute Counts)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=self.emotion_classes, yticklabels=self.emotion_classes, ax=ax2)
        ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Predicted Label', fontsize=12)
        ax2.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"confusion_matrix_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved: {plot_file}")
    
    def _plot_class_distribution(self, y_true, y_pred):
        """Plot class distribution comparison"""
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        
        # True distribution
        unique_true, counts_true = np.unique(y_true, return_counts=True)
        class_names_true = [self.emotion_classes[i] for i in unique_true]
        ax1.bar(class_names_true, counts_true, color='skyblue', alpha=0.7)
        ax1.set_title('True Label Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Predicted distribution
        unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
        class_names_pred = [self.emotion_classes[i] for i in unique_pred]
        ax2.bar(class_names_pred, counts_pred, color='lightcoral', alpha=0.7)
        ax2.set_title('Predicted Label Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Comparison
        all_classes = list(range(len(self.emotion_classes)))
        true_counts = [np.sum(y_true == i) for i in all_classes]
        pred_counts = [np.sum(y_pred == i) for i in all_classes]
        
        x = np.arange(len(self.emotion_classes))
        width = 0.35
        
        ax3.bar(x - width/2, true_counts, width, label='True', color='skyblue', alpha=0.7)
        ax3.bar(x + width/2, pred_counts, width, label='Predicted', color='lightcoral', alpha=0.7)
        ax3.set_title('True vs Predicted Distribution', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(self.emotion_classes, rotation=45)
        ax3.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"class_distribution_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Class distribution plot saved: {plot_file}")
    
    def _plot_roc_curves(self, y_true, y_prob):
        """Plot ROC curves for multiclass classification"""
        
        # Binarize the output
        y_true_binarized = label_binarize(y_true, classes=list(range(len(self.emotion_classes))))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(len(self.emotion_classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.emotion_classes)))
        for i, color in zip(range(len(self.emotion_classes)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.emotion_classes[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', lw=3,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Multiclass Emotion Classification', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"roc_curves_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ROC curves saved: {plot_file}")
    
    def _plot_accuracy_graph(self, checkpoint):
        """Plot training and validation accuracy over epochs"""
        
        # Extract training history from checkpoint if available
        if 'history' not in checkpoint:
            print("⚠️ No training history found in checkpoint")
            return
        
        history = checkpoint['history']
        
        if 'train_accuracy' not in history or 'val_accuracy' not in history:
            print("⚠️ No accuracy history found in checkpoint")
            return
        
        train_acc = history['train_accuracy']
        val_acc = history['val_accuracy']
        epochs = list(range(1, len(train_acc) + 1))
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2, marker='o')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2, marker='s')
        
        plt.title('Model Accuracy Over Training Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best performance
        best_val_epoch = epochs[val_acc.index(max(val_acc))]
        best_val_acc = max(val_acc)
        plt.annotate(f'Best Val: {best_val_acc:.4f}\nEpoch {best_val_epoch}', 
                    xy=(best_val_epoch, best_val_acc), 
                    xytext=(best_val_epoch + 1, best_val_acc + 0.02),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='center')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"accuracy_graph_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Accuracy graph saved: {plot_file}")
    
    def _plot_loss_graph(self, checkpoint):
        """Plot training and validation loss over epochs"""
        
        # Extract training history from checkpoint if available
        if 'history' not in checkpoint:
            print("⚠️ No training history found in checkpoint")
            return
        
        history = checkpoint['history']
        
        if 'train_loss' not in history or 'val_loss' not in history:
            print("⚠️ No loss history found in checkpoint")
            return
        
        train_loss = history['train_loss']
        val_loss = history['val_loss']
        epochs = list(range(1, len(train_loss) + 1))
        
        plt.figure(figsize=(12, 6))
        
        plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, marker='o')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, marker='s')
        
        plt.title('Model Loss Over Training Epochs', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add annotations for best performance
        best_val_epoch = epochs[val_loss.index(min(val_loss))]
        best_val_loss = min(val_loss)
        plt.annotate(f'Best Val: {best_val_loss:.4f}\nEpoch {best_val_epoch}', 
                    xy=(best_val_epoch, best_val_loss), 
                    xytext=(best_val_epoch + 1, best_val_loss + 0.1),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, ha='center')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"loss_graph_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📉 Loss graph saved: {plot_file}")
    
    def _plot_per_class_metrics(self, metrics):
        """Plot detailed per-class performance metrics"""
        
        precision = metrics['precision_per_class']
        recall = metrics['recall_per_class']
        f1 = metrics['f1_per_class']
        support = metrics['support_per_class']
        
        # Create subplot grid
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        x = np.arange(len(self.emotion_classes))
        width = 0.6
        
        # 1. Precision, Recall, F1 comparison
        ax1.bar(x - width/3, precision, width/3, label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x, recall, width/3, label='Recall', alpha=0.8, color='lightgreen')
        ax1.bar(x + width/3, f1, width/3, label='F1-Score', alpha=0.8, color='lightcoral')
        
        ax1.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Emotion Classes', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.emotion_classes, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
            ax1.text(i - width/3, p + 0.01, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i, r + 0.01, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width/3, f + 0.01, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Support (sample counts) per class
        bars = ax2.bar(self.emotion_classes, support, color='gold', alpha=0.7)
        ax2.set_title('Support (Sample Count) per Class', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Emotion Classes', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, support):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(support)*0.01,
                    f'{int(count)}', ha='center', va='bottom', fontsize=10)
        
        # 3. F1-Score ranking
        f1_sorted_indices = np.argsort(f1)[::-1]  # Descending order
        sorted_classes = [self.emotion_classes[i] for i in f1_sorted_indices]
        sorted_f1 = [f1[i] for i in f1_sorted_indices]
        
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(sorted_classes)))
        bars = ax3.barh(sorted_classes, sorted_f1, color=colors)
        ax3.set_title('F1-Score Ranking by Class', fontsize=14, fontweight='bold')
        ax3.set_xlabel('F1-Score', fontsize=12)
        ax3.set_ylabel('Emotion Classes', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0, 1)
        
        # Add value labels
        for bar, score in zip(bars, sorted_f1):
            width = bar.get_width()
            ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        # 4. Performance vs Support scatter plot
        ax4.scatter(support, f1, s=100, alpha=0.7, c=range(len(self.emotion_classes)), cmap='viridis')
        
        # Add class labels to points
        for i, (sup, f1_score, class_name) in enumerate(zip(support, f1, self.emotion_classes)):
            ax4.annotate(class_name, (sup, f1_score), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, alpha=0.8)
        
        ax4.set_title('F1-Score vs Support Correlation', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Support (Number of Samples)', fontsize=12)
        ax4.set_ylabel('F1-Score', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Add correlation coefficient
        correlation = np.corrcoef(support, f1)[0, 1]
        ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=ax4.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"per_class_metrics_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Per-class metrics plot saved: {plot_file}")
    
    def _plot_training_history(self, checkpoint):
        """Plot comprehensive training history analysis"""
        
        if 'history' not in checkpoint:
            print("⚠️ No training history found in checkpoint")
            return
        
        history = checkpoint['history']
        
        # Check what metrics are available
        available_metrics = list(history.keys())
        print(f"📊 Available training metrics: {available_metrics}")
        
        # Create a comprehensive training history plot
        fig = plt.figure(figsize=(20, 12))
        
        # Calculate number of subplots needed
        metrics_to_plot = []
        if 'train_accuracy' in history and 'val_accuracy' in history:
            metrics_to_plot.append(('accuracy', 'Accuracy', history['train_accuracy'], history['val_accuracy']))
        if 'train_loss' in history and 'val_loss' in history:
            metrics_to_plot.append(('loss', 'Loss', history['train_loss'], history['val_loss']))
        if 'train_f1' in history and 'val_f1' in history:
            metrics_to_plot.append(('f1', 'F1-Score', history['train_f1'], history['val_f1']))
        if 'train_precision' in history and 'val_precision' in history:
            metrics_to_plot.append(('precision', 'Precision', history['train_precision'], history['val_precision']))
        if 'train_recall' in history and 'val_recall' in history:
            metrics_to_plot.append(('recall', 'Recall', history['train_recall'], history['val_recall']))
        
        if not metrics_to_plot:
            print("⚠️ No valid metric pairs found for plotting")
            return
        
        # Arrange subplots
        n_plots = len(metrics_to_plot)
        rows = (n_plots + 1) // 2
        cols = 2 if n_plots > 1 else 1
        
        for idx, (metric_name, metric_label, train_values, val_values) in enumerate(metrics_to_plot):
            ax = plt.subplot(rows, cols, idx + 1)
            
            epochs = list(range(1, len(train_values) + 1))
            
            ax.plot(epochs, train_values, 'b-', label=f'Training {metric_label}', 
                   linewidth=2, marker='o', markersize=4)
            ax.plot(epochs, val_values, 'r-', label=f'Validation {metric_label}', 
                   linewidth=2, marker='s', markersize=4)
            
            ax.set_title(f'{metric_label} Over Training Epochs', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add best performance annotation
            if metric_name == 'loss':
                best_epoch = epochs[val_values.index(min(val_values))]
                best_value = min(val_values)
                best_label = f'Best: {best_value:.4f}'
            else:
                best_epoch = epochs[val_values.index(max(val_values))]
                best_value = max(val_values)
                best_label = f'Best: {best_value:.4f}'
            
            ax.annotate(f'{best_label}\nEpoch {best_epoch}', 
                       xy=(best_epoch, best_value), 
                       xytext=(0.7, 0.9), textcoords='axes fraction',
                       arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                       fontsize=9, ha='center',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Add overall training summary
        if n_plots % 2 == 1:  # Odd number of plots, use the last subplot for summary
            ax_summary = plt.subplot(rows, cols, n_plots + 1)
            
            # Create training summary table
            summary_data = []
            for metric_name, metric_label, train_values, val_values in metrics_to_plot:
                if metric_name == 'loss':
                    best_train = min(train_values)
                    best_val = min(val_values)
                else:
                    best_train = max(train_values)
                    best_val = max(val_values)
                
                summary_data.append([metric_label, f'{best_train:.4f}', f'{best_val:.4f}'])
            
            table = ax_summary.table(cellText=summary_data,
                                   colLabels=['Metric', 'Best Train', 'Best Val'],
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(summary_data) + 1):
                for j in range(3):
                    cell = table[(i, j)]
                    if i == 0:  # Header
                        cell.set_facecolor('#40466e')
                        cell.set_text_props(weight='bold', color='white')
                    else:
                        cell.set_facecolor('#f1f1f2')
            
            ax_summary.set_title('Training Summary', fontsize=14, fontweight='bold')
            ax_summary.axis('off')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.results_dir / f"training_history_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Training history plot saved: {plot_file}")
    
    def _save_detailed_results(self, predictions, labels, probabilities, conversation_ids, 
                              raw_emotions, mapped_emotions, metrics):
        """Save detailed per-sample results"""
        
        # Create detailed results DataFrame
        results_data = []
        
        for i in range(len(predictions)):
            sample_data = {
                'conversation_id': conversation_ids[i],
                'true_label': labels[i],
                'true_emotion': self.emotion_classes[labels[i]],
                'predicted_label': predictions[i],
                'predicted_emotion': self.emotion_classes[predictions[i]],
                'raw_emotion': raw_emotions[i],
                'mapped_emotion': mapped_emotions[i],
                'correct_prediction': labels[i] == predictions[i],
                'confidence': probabilities[i][predictions[i]]
            }
            
            # Add probability for each class
            for j, class_name in enumerate(self.emotion_classes):
                sample_data[f'prob_{class_name}'] = probabilities[i][j]
            
            results_data.append(sample_data)
        
        # Create DataFrame and save
        results_df = pd.DataFrame(results_data)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_file = self.results_dir / f"detailed_results_{timestamp}.csv"
        results_df.to_csv(csv_file, index=False)
        
        # Save metrics as JSON
        metrics_serializable = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            else:
                metrics_serializable[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        json_file = self.results_dir / f"evaluation_metrics_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
        
        print(f"📄 Detailed results saved: {csv_file}")
        print(f"📄 Metrics saved: {json_file}")
        
        # Print summary statistics
        print("\n📊 EVALUATION SUMMARY:")
        print("-" * 40)
        print(f"Total test samples: {len(predictions)}")
        print(f"Overall accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Macro F1-score: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1-score: {metrics['f1_weighted']:.4f}")
        print(f"Macro ROC-AUC: {metrics['roc_auc_macro']:.4f}")
        print(f"Weighted ROC-AUC: {metrics['roc_auc_weighted']:.4f}")
        
        # Best and worst performing classes
        f1_scores = metrics['f1_per_class']
        best_class_idx = np.argmax(f1_scores)
        worst_class_idx = np.argmin(f1_scores)
        
        print(f"\nBest performing class: {self.emotion_classes[best_class_idx]} (F1: {f1_scores[best_class_idx]:.4f})")
        print(f"Worst performing class: {self.emotion_classes[worst_class_idx]} (F1: {f1_scores[worst_class_idx]:.4f})")

def load_best_model(checkpoints_dir, device, dataset):
    """Load the best saved model from checkpoints with correct vocab sizes"""
    
    checkpoints_path = Path(checkpoints_dir)
    if not checkpoints_path.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")
    
    # Find the best model file
    model_files = list(checkpoints_path.glob("best_*_acc*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No best model files found in {checkpoints_dir}")
    
    # Sort by accuracy (highest first)
    def extract_accuracy(filename):
        import re
        match = re.search(r'acc(\d+\.\d+)', str(filename))
        return float(match.group(1)) if match else 0.0
    
    best_model_file = max(model_files, key=extract_accuracy)
    
    print(f"📦 Loading best model: {best_model_file.name}")
    
    # Load model state
    checkpoint = torch.load(best_model_file, map_location=device)
    
    # Get vocab sizes from checkpoint
    state_dict = checkpoint['model_state_dict']
    
    # Extract actual vocab sizes from the saved model
    vocab_sizes = {
        'event_scenario': state_dict['event_scenario_embedding.weight'].shape[0],
        'emotion_cause': state_dict['emotion_cause_embedding.weight'].shape[0], 
        'goal_response': state_dict['goal_response_embedding.weight'].shape[0],
        'topic': state_dict['topic_embedding.weight'].shape[0]
    }
    
    print(f"📋 Using vocab sizes from checkpoint: {vocab_sizes}")
    
    # Create fake vocabularies with correct sizes
    dataset.event_scenario_vocab = {f'scenario_{i}': i for i in range(vocab_sizes['event_scenario'] - 1)}
    dataset.emotion_cause_vocab = {f'cause_{i}': i for i in range(vocab_sizes['emotion_cause'] - 1)}
    dataset.goal_response_vocab = {f'goal_{i}': i for i in range(vocab_sizes['goal_response'] - 1)}
    dataset.topic_vocab = {f'topic_{i}': i for i in range(vocab_sizes['topic'] - 1)}
    
    return checkpoint, best_model_file

def main():
    """Main evaluation function"""
    print("🚀 COMPREHENSIVE MODEL EVALUATION - TRAIN_2.PY REFERENCE")
    print("=" * 70)
    print("📦 Model: best_7class_model.pth from checkpoints_2")
    print("📊 Test Data: mapped_test_data_video_aligned.json")
    print("🔧 Modalities: Video + Text + Metadata (NO AUDIO)")
    print("=" * 70)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Configuration
    config = {
        'test_data_file': 'json/mapped_test_data_video_aligned.json',
        'test_video_dir': 'data/train_video/video_v5_0',
        'checkpoints_dir': 'checkpoints_2',
        'results_dir': 'results_2',
        'batch_size': 4,
        'num_workers': 0,  # Set to 0 to avoid multiprocessing issues with nested class
        'max_dialogue_length': 10,
        'model_name': 'best_7class_model.pth'
    }
    
    # Create results directory
    results_dir = Path(config['results_dir'])
    results_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("🔧 Initializing components...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    video_transform = VideoTransform(frame_size=192)
    
    # Load the best model with vocab handling
    print("� Loading best trained model...")
    checkpoint, model_file, vocab_sizes = load_best_model_with_vocab_fix(config['checkpoints_dir'], device)
    
    # Create test dataset with fixed vocabularies
    test_dataset = create_test_dataset_with_fixed_vocab(
        tokenizer, video_transform, vocab_sizes, config['max_dialogue_length']
    )
    
    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    print(f"Test batches: {len(test_loader)}")
    
    # Initialize model using local definition or from train_2.py
    print("🏗️ Initializing model...")
    
    try:
        from train_2 import MultimodalLSTMVideoModel
        model = MultimodalLSTMVideoModel(
            dataset=test_dataset,
            num_classes=7,
            hidden_size=256,
            num_layers=2,
            dropout_rate=0.3
        )
        print("✅ Using model from train_2.py")
    except ImportError:
        print("⚠️ Could not import from train_2.py, using local model definition")
        # Use local model definition if needed
        # ... local model code here ...
        raise ImportError("Model import failed - please ensure train_2.py is accessible")
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully from {model_file.name}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get emotion classes
    emotion_classes = list(test_dataset.emotion_to_id.keys())
    print(f"Emotion classes: {emotion_classes}")
    
    # Initialize evaluator
    print("🔍 Initializing evaluator...")
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        emotion_classes=emotion_classes,
        results_dir=config['results_dir'],
        checkpoint=checkpoint
    )
    
    # Run evaluation
    print("⚡ Running comprehensive evaluation...")
    start_time = time.time()
    
    metrics = evaluator.evaluate()
    
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    print(f"\n✅ EVALUATION COMPLETED!")
    print(f"⏱️ Total evaluation time: {evaluation_time:.2f} seconds")
    print(f"📁 Results saved in: {config['results_dir']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("🎯 FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
    print(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
    print(f"Model File: {model_file.name}")
    print(f"Results Directory: {config['results_dir']}")
    print("=" * 70)

if __name__ == "__main__":
    main()
