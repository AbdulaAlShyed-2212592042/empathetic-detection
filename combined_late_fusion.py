import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import librosa
import soundfile as sf
import copy
import time
import sys
import warnings
import importlib.util
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    get_linear_schedule_with_warmup
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Hyperparameter optimization
try:
    import optuna
    import optuna.visualization.matplotlib
    OPTUNA_AVAILABLE = True
    print("✅ Optuna available for hyperparameter optimization")
except ImportError:
    print("Warning: Optuna not available. Install with: pip install optuna")
    OPTUNA_AVAILABLE = False

# Additional warning suppression for audio processing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
os.environ['NUMBA_DISABLE_JIT'] = '1'  # Disable Numba JIT as fallback

# Mixed precision training support
from torch.cuda.amp import autocast, GradScaler

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
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return torch.zeros(3, self.num_frames, self.frame_size, self.frame_size)
            
            # Calculate frame indices for uniform sampling
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

class CombinedFusionDataset(Dataset):
    """Combined dataset for enhanced late fusion with both audio models"""
    
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
        
        # Comprehensive emotion mapping (7 classes)
        self.emotion_to_id = {
            "happy": 0,
            "surprised": 1,
            "angry": 2,
            "fear": 3,
            "sad": 4,
            "disgusted": 5,
            "contempt": 6
        }
        
        # Enhanced ED emotion projection combining both models
        self.ed_emotion_projection = {
            # From audio training model
            'conflicted': 'anxious', 'vulnerability': 'afraid', 'helplessness': 'afraid',
            'sadness': 'sad', 'pensive': 'sentimental', 'frustration': 'annoyed',
            'weary': 'tired', 'anxiety': 'anxious', 'reflective': 'sentimental',
            'upset': 'disappointed', 'worried': 'anxious', 'fear': 'afraid',
            'frustrated': 'sad', 'fatigue': 'tired', 'lost': 'jealous',
            'disappointment': 'disappointed', 'nostalgia': 'nostalgic',
            'exhaustion': 'tired', 'uneasy': 'anxious', 'loneliness': 'lonely',
            'fragile': 'afraid', 'confused': 'jealous', 'vulnerable': 'afraid',
            'thoughtful': 'sentimental', 'stressed': 'anxious', 'concerned': 'anxious',
            'tiredness': 'tired', 'burdened': 'anxious', 'melancholy': 'sad',
            'overwhelmed': 'anxious', 'worry': 'anxious', 'heavy-hearted': 'sad',
            'melancholic': 'sad', 'nervous': 'anxious', 'fearful': 'afraid',
            'stress': 'anxious', 'confusion': 'anxious', 'inadequacy': 'ashamed',
            'regret': 'guilty', 'helpless': 'afraid', 'concern': 'anxious',
            'exhausted': 'tired', 'overwhelm': 'anxious', 'tired': 'tired',
            'disappointed': 'sad', 'surprised': 'surprised', 'excited': 'happy',
            'angry': 'angry', 'proud': 'happy', 'annoyed': 'angry',
            'grateful': 'happy', 'lonely': 'sad', 'afraid': 'fear',
            'terrified': 'fear', 'guilty': 'sad', 'impressed': 'surprised',
            'disgusted': 'disgusted', 'hopeful': 'happy', 'confident': 'happy',
            'furious': 'angry', 'anxious': 'sad', 'anticipating': 'happy',
            'joyful': 'happy', 'nostalgic': 'sad', 'prepared': 'happy',
            'jealous': 'contempt', 'content': 'happy', 'devastated': 'surprised',
            'embarrassed': 'sad', 'caring': 'happy', 'sentimental': 'sad',
            'trusting': 'happy', 'ashamed': 'sad', 'apprehensive': 'fear',
            'faithful': 'happy'
        }
        
        # Final mapping to 7 basic emotions
        self.final_emotion_mapping = {
            'happy': 'happy', 'joyful': 'happy', 'excited': 'happy', 'content': 'happy',
            'grateful': 'happy', 'proud': 'happy', 'hopeful': 'happy', 'confident': 'happy',
            'caring': 'happy', 'trusting': 'happy', 'faithful': 'happy', 'prepared': 'happy',
            'anticipating': 'happy',
            
            'surprised': 'surprised', 'impressed': 'surprised', 'devastated': 'surprised',
            
            'angry': 'angry', 'annoyed': 'angry', 'furious': 'angry',
            
            'fear': 'fear', 'afraid': 'fear', 'terrified': 'fear', 'anxious': 'fear',
            'worried': 'fear', 'nervous': 'fear', 'fearful': 'fear', 'apprehensive': 'fear',
            'vulnerable': 'fear', 'fragile': 'fear', 'helpless': 'fear', 'uneasy': 'fear',
            'stressed': 'fear', 'concerned': 'fear', 'burdened': 'fear', 'overwhelmed': 'fear',
            'overwhelm': 'fear', 'confusion': 'fear', 'concern': 'fear',
            
            'sad': 'sad', 'sadness': 'sad', 'disappointed': 'sad', 'lonely': 'sad',
            'guilty': 'sad', 'melancholy': 'sad', 'melancholic': 'sad', 'heavy-hearted': 'sad',
            'embarrassed': 'sad', 'sentimental': 'sad', 'ashamed': 'sad', 'nostalgic': 'sad',
            'anxious': 'sad', 'frustrated': 'sad', 'disappointment': 'sad', 'upset': 'sad',
            'nostalgia': 'sad', 'loneliness': 'sad', 'regret': 'sad', 'inadequacy': 'sad',
            
            'disgusted': 'disgusted',
            
            'contempt': 'contempt', 'jealous': 'contempt', 'lost': 'contempt', 'confused': 'contempt',
            
            # Additional mappings for states/conditions
            'tired': 'sad', 'weary': 'sad', 'fatigue': 'sad', 'exhaustion': 'sad',
            'exhausted': 'sad', 'tiredness': 'sad',
            'pensive': 'sad', 'reflective': 'sad', 'thoughtful': 'sad'
        }
        
        # Create comprehensive vocabularies
        self._build_vocabularies()
        
        # Role mapping
        self.role_to_id = {'speaker': 0, 'listener': 1, 'listener_response': 2}
        
        print(f"Loaded {len(self.data)} samples from {json_file}")
        
    def _build_vocabularies(self):
        """Build vocabularies from data"""
        event_scenarios = set()
        emotion_causes = set()
        goal_responses = set()
        topics = set()
        
        for item in self.data:
            turn = item.get('turn', {})
            chain = turn.get('chain_of_empathy', {})
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

    def load_audio(self, audio_path):
        """Load and preprocess audio file with robust error handling"""
        try:
            if not os.path.exists(audio_path):
                return None
            
            # Try librosa first (for basic loading only)
            try:
                audio, sr = librosa.load(audio_path, sr=self.sample_rate)
                
                if len(audio) == 0:
                    return None
                
                # Normalize audio
                audio = audio / (np.max(np.abs(audio)) + 1e-8)
                
                # Pad or truncate to fixed length (10 seconds for comprehensive processing)
                max_length = self.sample_rate * 10
                if len(audio) > max_length:
                    audio = audio[:max_length]
                else:
                    audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
                
                return audio
                
            except Exception as librosa_error:
                print(f"Librosa failed for {audio_path}: {librosa_error}")
                # Try alternative loading method using soundfile
                try:
                    audio, sr = sf.read(audio_path)
                    
                    # Resample if needed (simple downsampling)
                    if sr != self.sample_rate:
                        # Simple resampling by taking every nth sample
                        step = sr // self.sample_rate
                        if step > 1:
                            audio = audio[::step]
                    
                    if len(audio) == 0:
                        return None
                    
                    # Convert to mono if stereo
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    
                    # Normalize audio
                    audio = audio / (np.max(np.abs(audio)) + 1e-8)
                    
                    # Pad or truncate to fixed length
                    max_length = self.sample_rate * 10
                    if len(audio) > max_length:
                        audio = audio[:max_length]
                    else:
                        audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
                    
                    return audio.astype(np.float32)
                    
                except Exception as sf_error:
                    print(f"Soundfile also failed for {audio_path}: {sf_error}")
                    # Fallback: return silent audio of correct length
                    return np.zeros(self.sample_rate * 10, dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silent audio as fallback
            return np.zeros(self.sample_rate * 10, dtype=np.float32)

    def extract_traditional_audio_features(self, audio):
        """Extract traditional audio features with fallback methods"""
        try:
            if audio is None or len(audio) == 0:
                return np.zeros(39, dtype=np.float32)
            
            # Check if audio is all zeros (silent)
            if np.allclose(audio, 0):
                return np.zeros(39, dtype=np.float32)
            
            # Use basic numpy-based features instead of librosa for compatibility
            features = []
            
            # Basic statistical features
            features.extend([
                np.mean(audio), np.std(audio),
                np.max(audio), np.min(audio),
                np.median(audio)
            ])
            
            # Simple spectral features using FFT
            try:
                # Compute FFT
                fft = np.fft.fft(audio)
                magnitude = np.abs(fft[:len(fft)//2])
                
                if len(magnitude) > 0 and np.sum(magnitude) > 0:
                    # Spectral centroid approximation
                    freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)[:len(magnitude)]
                    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                    
                    # Spectral rolloff approximation
                    cumulative_magnitude = np.cumsum(magnitude)
                    total_magnitude = cumulative_magnitude[-1]
                    rolloff_point = 0.85 * total_magnitude
                    rolloff_idx = np.where(cumulative_magnitude >= rolloff_point)[0]
                    spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0
                    
                    # Spectral bandwidth approximation
                    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude))
                    
                    features.extend([
                        spectral_centroid, spectral_bandwidth, spectral_rolloff
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
                    
            except Exception:
                features.extend([0.0, 0.0, 0.0])
            
            # Zero crossing rate (manual implementation)
            try:
                zero_crossings = np.where(np.diff(np.signbit(audio)))[0]
                zcr = len(zero_crossings) / len(audio)
                features.extend([zcr, 0.0])  # mean and std (std=0 for single value)
            except:
                features.extend([0.0, 0.0])
            
            # Energy features
            try:
                # Frame-based energy
                frame_length = 512
                hop_length = 256
                frames = []
                for i in range(0, len(audio) - frame_length, hop_length):
                    frame = audio[i:i + frame_length]
                    energy = np.sum(frame ** 2)
                    frames.append(energy)
                
                if len(frames) > 0:
                    frame_energies = np.array(frames)
                    features.extend([
                        np.mean(frame_energies), np.std(frame_energies),
                        np.max(frame_energies), np.min(frame_energies)
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])
            except:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # Pad or truncate to exactly 39 features
            features = np.array(features, dtype=np.float32)
            if len(features) < 39:
                features = np.pad(features, (0, 39 - len(features)), 'constant')
            elif len(features) > 39:
                features = features[:39]
            
            # Replace any NaN or infinite values
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            return features
                
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return np.zeros(39, dtype=np.float32)

    def _map_emotion_comprehensive(self, emotion_str):
        """Comprehensive emotion mapping using both models' approaches"""
        if not emotion_str:
            return 0  # Default to happy
            
        emotion_str = emotion_str.lower().strip()
        
        # Direct mapping first
        if emotion_str in self.emotion_to_id:
            return self.emotion_to_id[emotion_str]
        
        # Apply ED emotion projection
        projected_emotion = self.ed_emotion_projection.get(emotion_str, emotion_str)
        
        # Apply final mapping
        final_emotion = self.final_emotion_mapping.get(projected_emotion, projected_emotion)
        
        # Return ID
        return self.emotion_to_id.get(final_emotion, 0)  # Default to happy

    def __len__(self):
        return len(self.data)
    
    def get_label(self, idx):
        """Get label for computing class weights"""
        item = self.data[idx]
        turn = item.get('turn', {})
        chain_of_empathy = turn.get('chain_of_empathy', {})
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        return self._map_emotion_comprehensive(raw_emotion)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        turn = item.get('turn', {})
        
        # Get emotion label
        chain_of_empathy = turn.get('chain_of_empathy', {})
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        emotion_label = self._map_emotion_comprehensive(raw_emotion)
        
        # Extract dialogue sequence
        dialogue = turn.get('dialogue', [])
        
        # Initialize collections for all modalities
        dialogue_videos = []
        dialogue_texts = []
        dialogue_audio_raw = []  # For Wav2Vec2
        dialogue_audio_features = []  # For traditional features
        dialogue_roles_video = []
        dialogue_roles_audio = []
        dialogue_indices = []
        
        # Process each utterance
        for i, utt in enumerate(dialogue[:self.max_dialogue_length]):
            # Common data
            role = utt.get('role', 'speaker')
            role_id = self.role_to_id.get(role, 0)
            text = utt.get('text', '')
            
            dialogue_texts.append(text)
            dialogue_roles_video.append(role_id)
            dialogue_roles_audio.append(role_id)
            dialogue_indices.append(utt.get('index', 0))
            
            # Video processing
            video_name = utt.get('video_name', None)
            if video_name:
                try:
                    video_path = os.path.join(self.video_dir, video_name)
                    if os.path.exists(video_path):
                        video_tensor = self.video_transform(video_path)
                    else:
                        video_tensor = torch.zeros(3, 8, 224, 224)
                except Exception as e:
                    print(f"Error loading video {video_name}: {e}")
                    video_tensor = torch.zeros(3, 8, 224, 224)
            else:
                video_tensor = torch.zeros(3, 8, 224, 224)
            
            dialogue_videos.append(video_tensor)
            
            # Audio processing (both raw and features)
            audio_name = utt.get('audio_name', None)
            if audio_name:
                audio_path = os.path.join(self.audio_dir, audio_name)
                audio = self.load_audio(audio_path)
                
                if audio is not None:
                    # Raw audio for Wav2Vec2
                    dialogue_audio_raw.append(audio)
                    # Traditional features for MIMAMO
                    traditional_features = self.extract_traditional_audio_features(audio)
                    dialogue_audio_features.append(traditional_features)
                else:
                    # Silent audio and zero features
                    dialogue_audio_raw.append(np.zeros(self.sample_rate * 10, dtype=np.float32))
                    dialogue_audio_features.append(np.zeros(39, dtype=np.float32))
            else:
                # Silent audio and zero features
                dialogue_audio_raw.append(np.zeros(self.sample_rate * 10, dtype=np.float32))
                dialogue_audio_features.append(np.zeros(39, dtype=np.float32))
        
        # Pad sequences to max_dialogue_length
        while len(dialogue_texts) < self.max_dialogue_length:
            dialogue_texts.append('[EMPTY]')
            dialogue_videos.append(torch.zeros(3, 8, 224, 224))
            dialogue_audio_raw.append(np.zeros(self.sample_rate * 10, dtype=np.float32))
            dialogue_audio_features.append(np.zeros(39, dtype=np.float32))
            dialogue_roles_video.append(2)  # padding role
            dialogue_roles_audio.append(2)  # padding role
            dialogue_indices.append(0)
        
        # Convert to tensors
        dialogue_video = torch.stack(dialogue_videos)
        dialogue_roles_video = torch.tensor(dialogue_roles_video, dtype=torch.long)
        dialogue_roles_audio = torch.tensor(dialogue_roles_audio, dtype=torch.long)
        dialogue_indices = torch.tensor(dialogue_indices, dtype=torch.long)
        
        # Process audio - ensure consistent length
        max_audio_length = self.sample_rate * 10  # 10 seconds
        processed_audio_raw = []
        for audio in dialogue_audio_raw:
            if len(audio) > max_audio_length:
                audio = audio[:max_audio_length]
            elif len(audio) < max_audio_length:
                audio = np.pad(audio, (0, max_audio_length - len(audio)), 'constant')
            processed_audio_raw.append(audio)
        
        dialogue_audio_raw_tensor = torch.tensor(processed_audio_raw, dtype=torch.float)
        dialogue_audio_features_tensor = torch.tensor(dialogue_audio_features, dtype=torch.float)
        
        # Text processing
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
        
        # Dialogue encoding
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
        
        # Process metadata
        speaker_profile = item.get('speaker_profile', {})
        listener_profile = item.get('listener_profile', {})
        
        # Age, gender, timbre mappings
        age_to_id = {"child": 0, "young": 1, "middle-aged": 2, "elderly": 3}
        gender_to_id = {"male": 0, "female": 1}
        timbre_to_id = {"high": 0, "mid": 1, "low": 2}
        
        speaker_age = age_to_id.get(speaker_profile.get('age', 'young'), 1)
        speaker_gender = gender_to_id.get(speaker_profile.get('gender', 'male'), 0)
        speaker_timbre = timbre_to_id.get(speaker_profile.get('timbre', 'mid'), 1)
        speaker_id = speaker_profile.get('ID', 0)
        
        listener_age = age_to_id.get(listener_profile.get('age', 'young'), 1)
        listener_gender = gender_to_id.get(listener_profile.get('gender', 'male'), 0)
        listener_timbre = timbre_to_id.get(listener_profile.get('timbre', 'mid'), 1)
        listener_id = listener_profile.get('ID', 0)
        
        # Chain of empathy metadata
        event_scenario = chain_of_empathy.get('event_scenario', '')
        emotion_cause = chain_of_empathy.get('emotion_cause', '')
        goal_to_response = chain_of_empathy.get('goal_to_response', '')
        topic = item.get('topic', '')
        
        event_scenario_id = self.event_scenario_vocab.get(event_scenario, 0)
        emotion_cause_id = self.emotion_cause_vocab.get(emotion_cause, 0)
        goal_response_id = self.goal_response_vocab.get(goal_to_response, 0)
        topic_id = self.topic_vocab.get(topic, 0)
        
        metadata_features = torch.tensor([
            speaker_age, speaker_gender, speaker_timbre, speaker_id,
            listener_age, listener_gender, listener_timbre, listener_id,
            event_scenario_id, emotion_cause_id, goal_response_id, topic_id
        ], dtype=torch.long)
        
        sequence_length = torch.tensor(min(len(dialogue), self.max_dialogue_length), dtype=torch.long)
        label = torch.tensor(emotion_label, dtype=torch.long)
        
        return {
            # Video data (for MIMAMO model)
            'dialogue_video': dialogue_video,
            'dialogue_roles_video': dialogue_roles_video,
            
            # Audio-text data (for multimodal LSTM model)
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'dialogue_input_ids': torch.stack([enc['input_ids'] for enc in dialogue_encodings]),
            'dialogue_attention_mask': torch.stack([enc['attention_mask'] for enc in dialogue_encodings]),
            'dialogue_audio_raw': dialogue_audio_raw_tensor,  # For Wav2Vec2
            'dialogue_audio_features': dialogue_audio_features_tensor,  # For traditional features
            'dialogue_roles_audio': dialogue_roles_audio,
            'dialogue_indices': dialogue_indices,
            
            # Common data
            'metadata': metadata_features,
            'sequence_length': sequence_length,
            'label': label,
            'conversation_id': item.get('conversation_id', ''),
            'raw_emotion': raw_emotion
        }

class EnhancedLateFusionModel(nn.Module):
    """Enhanced Late Fusion Model combining MIMAMO Net and Multimodal LSTM"""
    
    def __init__(self, mimamo_model, multimodal_lstm_model, num_classes=7, 
                 fusion_method='weighted', initial_weights=None, dropout_rate=0.3):
        super(EnhancedLateFusionModel, self).__init__()
        
        self.mimamo_model = mimamo_model
        self.multimodal_lstm_model = multimodal_lstm_model
        self.num_classes = num_classes
        self.fusion_method = fusion_method
        
        # Freeze pretrained models (keep them stable)
        for param in self.mimamo_model.parameters():
            param.requires_grad = False
        for param in self.multimodal_lstm_model.parameters():
            param.requires_grad = False
        
        # Simple weighted fusion (most effective and stable)
        if initial_weights is None:
            initial_weights = [0.6, 0.4]  # Give slightly more weight to MIMAMO (video)
        
        self.fusion_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        self.fusion_bias = nn.Parameter(torch.zeros(num_classes))
        
        print(f"Enhanced late fusion initialized with {fusion_method} fusion")
        print(f"Initial weights: MIMAMO={initial_weights[0]:.3f}, Multimodal LSTM={initial_weights[1]:.3f}")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"📊 Total trainable parameters: {trainable_params:,}")
    
    def forward(self, mimamo_data, multimodal_data, return_features=False):
        # Extract logits from pretrained models (frozen)
        with torch.no_grad():
            mimamo_logits = self.mimamo_model(
                mimamo_data['dialogue_video'],
                mimamo_data['dialogue_roles'],
                mimamo_data['metadata'],
                mimamo_data['sequence_length']
            )
            
            multimodal_logits = self.multimodal_lstm_model(
                multimodal_data['context_input_ids'],
                multimodal_data['context_attention_mask'],
                multimodal_data['dialogue_input_ids'],
                multimodal_data['dialogue_attention_mask'],
                multimodal_data['dialogue_audio'],
                multimodal_data['dialogue_roles'],
                multimodal_data['metadata'],
                multimodal_data['sequence_length']
            )
        
        # Simple weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused_logits = weights[0] * mimamo_logits + weights[1] * multimodal_logits
        fused_logits = fused_logits + self.fusion_bias  # Add learnable bias
        
        if return_features:
            fusion_weights = weights.unsqueeze(0).repeat(mimamo_logits.shape[0], 1)
            return {
                'fused_logits': fused_logits,
                'mimamo_logits': mimamo_logits,
                'multimodal_logits': multimodal_logits,
                'fusion_weights': fusion_weights
            }
        
        return fused_logits, mimamo_logits, multimodal_logits

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

def load_pretrained_models(mimamo_checkpoint_path, multimodal_checkpoint_path, dataset):
    """Load pretrained MIMAMO and Multimodal LSTM models"""
    
    print("Loading pretrained models...")
    
    # Load MIMAMO Net (video model) - trained on video_aligned data
    print("Loading MIMAMO Net checkpoint...")
    mimamo_model = None
    try:
        # Import the MIMAMO model from video_MIMAMO_Net.py
        import importlib.util
        video_module_path = os.path.join(os.path.dirname(__file__), 'video_MIMAMO_Net.py')
        spec = importlib.util.spec_from_file_location("video_MIMAMO_Net", video_module_path)
        video_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(video_module)
        
        MIMAMONet = video_module.MIMAMONet
        VideoSequentialDataset = video_module.VideoSequentialDataset
        
        # Create original training dataset that MIMAMO was trained on
        print("Creating original MIMAMO training dataset for correct vocabularies...")
        original_mimamo_dataset = VideoSequentialDataset(
            r'json\mapped_train_data_video_aligned.json',
            r'data\train_video\video_v5_0',
            max_dialogue_length=10
        )
        
        # Load checkpoint first
        mimamo_checkpoint = torch.load(mimamo_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create the model with original dataset vocabularies
        print("Creating MIMAMO model with original vocabulary sizes...")
        mimamo_model = MIMAMONet(original_mimamo_dataset, num_classes=7)
        
        # Load state dict
        print("Loading MIMAMO checkpoint...")
        if 'model_state_dict' in mimamo_checkpoint:
            mimamo_model.load_state_dict(mimamo_checkpoint['model_state_dict'])
            print(f"MIMAMO model loaded successfully - Original Validation Accuracy: {mimamo_checkpoint.get('val_accuracy', 'N/A'):.4f}")
        else:
            mimamo_model.load_state_dict(mimamo_checkpoint)
            print("MIMAMO model loaded successfully from direct state dict")
        
    except Exception as e:
        print(f"Error loading MIMAMO model: {e}")
        print(f"Exception type: {type(e)}")
        mimamo_model = None
    
    # Load Multimodal LSTM (audio-text model) - trained on original data
    print("Loading Multimodal LSTM checkpoint...")
    multimodal_model = None
    try:
        # Import the Multimodal LSTM model
        audio_module_path = os.path.join(os.path.dirname(__file__), 'audio train and test', 'train_audio_text_metadata.py')
        spec = importlib.util.spec_from_file_location("train_audio_text_metadata", audio_module_path)
        audio_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audio_module)
        
        MultimodalLSTMModel = audio_module.MultimodalLSTMModel
        MultimodalSequentialDataset = audio_module.MultimodalSequentialDataset
        
        # Create original training dataset that Multimodal LSTM was trained on
        print("Creating original Multimodal LSTM training dataset for correct vocabularies...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
        
        original_multimodal_dataset = MultimodalSequentialDataset(
            data_path='json/mapped_train_data.json',
            audio_dir='data/train_audio/audio_v5_0',
            tokenizer=tokenizer,
            wav2vec_feature_extractor=wav2vec_feature_extractor,
            max_length=384,
            max_dialogue_length=10
        )
        
        # Load checkpoint
        multimodal_checkpoint = torch.load(multimodal_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Create model with original dataset vocabularies
        print("Creating Multimodal LSTM model with original vocabulary sizes...")
        multimodal_model = MultimodalLSTMModel(original_multimodal_dataset, num_classes=7, use_wav2vec=True)
        
        # Load state dict
        print("Loading Multimodal LSTM checkpoint...")
        if 'model_state_dict' in multimodal_checkpoint:
            multimodal_model.load_state_dict(multimodal_checkpoint['model_state_dict'])
            print(f"Multimodal LSTM loaded successfully - Original Validation Accuracy: {multimodal_checkpoint.get('val_accuracy', 'N/A'):.4f}")
        else:
            multimodal_model.load_state_dict(multimodal_checkpoint)
            print("Multimodal LSTM loaded successfully from direct state dict")
            
    except Exception as e:
        print(f"Error loading Multimodal LSTM model: {e}")
        print(f"Exception type: {type(e)}")
        multimodal_model = None
    
    # Check if both models loaded successfully
    if mimamo_model is None:
        print("❌ Failed to load MIMAMO model")
        return None, None
    if multimodal_model is None:
        print("❌ Failed to load Multimodal LSTM model")
        return None, None
    
    print("✅ Both pretrained models loaded successfully with correct vocabularies!")
    return mimamo_model, multimodal_model

def train_combined_fusion_model(fusion_model, train_loader, val_loader, device, 
                               num_epochs=50, learning_rate=1e-4, use_advanced_training=True):
    """Train the enhanced late fusion model"""
    
    # Mixed precision training
    scaler = GradScaler()
    print("🚀 Mixed precision training enabled")
    
    # Compute class weights
    print("Computing class weights...")
    all_labels = []
    max_samples = min(2000, len(train_loader.dataset))
    
    for i in range(0, max_samples, 10):
        try:
            sample = train_loader.dataset[i]
            all_labels.append(sample['label'].item())
        except:
            continue
        if len(all_labels) >= 500:
            break
    
    try:
        if len(all_labels) > 0:
            unique_labels = np.unique(all_labels)
            class_weights = compute_class_weight('balanced', classes=unique_labels, y=all_labels)
            full_class_weights = np.ones(7)
            for i, label in enumerate(unique_labels):
                if label < 7:
                    full_class_weights[label] = class_weights[i]
            class_weights = torch.tensor(full_class_weights, dtype=torch.float32).to(device)
            print(f"Computed balanced class weights: {class_weights.cpu().numpy()}")
        else:
            raise ValueError("No labels found")
    except Exception as e:
        print(f"Warning: Could not compute class weights ({e}), using equal weights")
        class_weights = torch.ones(7, dtype=torch.float32).to(device)
    
    # Loss function
    criterion = FocalLoss(alpha=1.0, gamma=2.0, num_classes=7)
    
    # Optimizer - only train fusion parameters
    optimizer_params = []
    for name, param in fusion_model.named_parameters():
        if param.requires_grad:
            optimizer_params.append({'params': param, 'lr': learning_rate})
            print(f"Training parameter: {name}")
    
    if not optimizer_params:
        print("Warning: No trainable parameters found!")
        return None, None
    
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=1e-4)
    
    # Scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 3,
        num_training_steps=len(train_loader) * num_epochs
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_path = None
    patience = 7
    patience_counter = 0
    
    print(f"Starting enhanced late fusion training for {num_epochs} epochs...")
    print(f"Trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")
    print("="*60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}...")
        
        # Training phase
        fusion_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        torch.cuda.empty_cache()
        
        for batch_idx, batch in enumerate(train_loader):
            # Prepare MIMAMO data
            mimamo_data = {
                'dialogue_video': batch['dialogue_video'].to(device),
                'dialogue_roles': batch['dialogue_roles_video'].to(device),
                'metadata': batch['metadata'].to(device),
                'sequence_length': batch['sequence_length'].to(device)
            }
            
            # Prepare Multimodal LSTM data
            multimodal_data = {
                'context_input_ids': batch['context_input_ids'].to(device),
                'context_attention_mask': batch['context_attention_mask'].to(device),
                'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                'dialogue_audio': batch['dialogue_audio_raw'].to(device),  # Use raw audio for Wav2Vec2
                'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                'metadata': batch['metadata'].to(device),
                'sequence_length': batch['sequence_length'].to(device)
            }
            
            labels = batch['label'].to(device)
            
            # Forward pass with mixed precision
            with autocast():
                fused_logits, mimamo_logits, multimodal_logits = fusion_model(
                    mimamo_data, multimodal_data
                )
                loss = criterion(fused_logits, labels)
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(fused_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                current_acc = 100. * train_correct / train_total if train_total > 0 else 0
                if batch_idx % 50 == 0:  # Print every 50 batches
                    print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, Acc={current_acc:.2f}%")
        
        # Validation phase
        fusion_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch in val_loader:
                mimamo_data = {
                    'dialogue_video': batch['dialogue_video'].to(device),
                    'dialogue_roles': batch['dialogue_roles_video'].to(device),
                    'metadata': batch['metadata'].to(device),
                    'sequence_length': batch['sequence_length'].to(device)
                }
                
                multimodal_data = {
                    'context_input_ids': batch['context_input_ids'].to(device),
                    'context_attention_mask': batch['context_attention_mask'].to(device),
                    'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                    'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                    'dialogue_audio': batch['dialogue_audio_raw'].to(device),
                    'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                    'metadata': batch['metadata'].to(device),
                    'sequence_length': batch['sequence_length'].to(device)
                }
                
                labels = batch['label'].to(device)
                
                with autocast():
                    fused_logits, _, _ = fusion_model(mimamo_data, multimodal_data)
                    loss = criterion(fused_logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(fused_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Print detailed epoch results
        print(f'\n📊 Epoch {epoch+1}/{num_epochs} Results:')
        print(f'  🔹 Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  🔹 Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Print fusion weights if using weighted fusion
        if fusion_model.fusion_method == 'weighted':
            weights = F.softmax(fusion_model.fusion_weights, dim=0)
            print(f'  🔹 Fusion weights: MIMAMO={weights[0]:.3f}, Multimodal={weights[1]:.3f}')
        
        # Save best model (save on EVERY new best validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save to three locations
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path_main = f'checkpoint_4_mimamo/best_combined_fusion_model_{timestamp}_epoch{epoch+1}_acc{val_acc:.4f}.pth'
            best_model_path_results = f'using_mimo_late_result/checkpoints/best_combined_fusion_model_{timestamp}_epoch{epoch+1}_acc{val_acc:.4f}.pth'
            best_model_path_mimamo = f'mimamo_net_result/checkpoints/best_combined_fusion_model_{timestamp}_epoch{epoch+1}_acc{val_acc:.4f}.pth'
            
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'history': history,
                'fusion_method': fusion_model.fusion_method,
                'best_val_accuracy': best_val_acc
            }
            
            # Save to all three checkpoint directories
            torch.save(checkpoint_data, best_model_path_main)
            torch.save(checkpoint_data, best_model_path_results)
            torch.save(checkpoint_data, best_model_path_mimamo)
            best_model_path = best_model_path_main  # Use main path as primary
            
            print(f'  ✅ 🏆 NEW BEST MODEL SAVED! Accuracy: {val_acc:.4f}%')
            print(f'     📁 Main: {best_model_path_main}')
            print(f'     📁 Results: {best_model_path_results}')
            print(f'     📁 MIMAMO: {best_model_path_mimamo}')
            patience_counter = 0
        else:
            print(f'  📊 Best Val Acc so far: {best_val_acc:.2f}% (no improvement)')
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\n⏹️ Early stopping triggered after {patience} epochs without improvement')
            break
    
    print(f'\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%')
    return history, best_model_path

def objective(trial, train_loader, val_loader, mimamo_model, multimodal_model, device):
    """Optuna objective function for hyperparameter optimization"""
    
    print(f"🔬 Trial {trial.number}: Starting hyperparameter optimization...")
    
    # Suggest basic hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [4, 6, 8])
    mimamo_weight = trial.suggest_float('mimamo_weight', 0.3, 0.8)
    initial_weights = [mimamo_weight, 1.0 - mimamo_weight]
    
    print(f"   🎯 LR: {learning_rate:.6f}, Batch: {batch_size}, MIMAMO weight: {mimamo_weight:.3f}")
    
    # Create simple fusion model
    fusion_model = EnhancedLateFusionModel(
        mimamo_model=mimamo_model,
        multimodal_lstm_model=multimodal_model,
        num_classes=7,
        fusion_method='weighted',
        initial_weights=initial_weights
    ).to(device)
    
    # Simple training setup
    criterion = FocalLoss(alpha=1.0, gamma=2.0, num_classes=7)
    optimizer = torch.optim.AdamW(
        [p for p in fusion_model.parameters() if p.requires_grad], 
        lr=learning_rate, weight_decay=1e-4
    )
    
    # Short training for hyperparameter search
    max_epochs = 3
    best_val_acc = 0.0
    
    for epoch in range(max_epochs):
        # Training
        fusion_model.train()
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if train_total >= 500:  # Quick training for speed
                break
                
            # Prepare data
            mimamo_data = {
                'dialogue_video': batch['dialogue_video'].to(device),
                'dialogue_roles': batch['dialogue_roles_video'].to(device),
                'metadata': batch['metadata'].to(device),
                'sequence_length': batch['sequence_length'].to(device)
            }
            
            multimodal_data = {
                'context_input_ids': batch['context_input_ids'].to(device),
                'context_attention_mask': batch['context_attention_mask'].to(device),
                'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                'dialogue_audio': batch['dialogue_audio_raw'].to(device),
                'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                'metadata': batch['metadata'].to(device),
                'sequence_length': batch['sequence_length'].to(device)
            }
            
            labels = batch['label'].to(device)
            
            # Forward pass
            fused_logits, _, _ = fusion_model(mimamo_data, multimodal_data)
            loss = criterion(fused_logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            _, predicted = torch.max(fused_logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        fusion_model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if val_total >= 200:  # Quick validation
                    break
                    
                mimamo_data = {
                    'dialogue_video': batch['dialogue_video'].to(device),
                    'dialogue_roles': batch['dialogue_roles_video'].to(device),
                    'metadata': batch['metadata'].to(device),
                    'sequence_length': batch['sequence_length'].to(device)
                }
                
                multimodal_data = {
                    'context_input_ids': batch['context_input_ids'].to(device),
                    'context_attention_mask': batch['context_attention_mask'].to(device),
                    'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                    'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                    'dialogue_audio': batch['dialogue_audio_raw'].to(device),
                    'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                    'metadata': batch['metadata'].to(device),
                    'sequence_length': batch['sequence_length'].to(device)
                }
                
                labels = batch['label'].to(device)
                fused_logits, _, _ = fusion_model(mimamo_data, multimodal_data)
                
                _, predicted = torch.max(fused_logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0.0
        best_val_acc = max(best_val_acc, val_acc)
        
        # Report to Optuna
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    print(f"   🏁 Trial {trial.number} completed with best val acc: {best_val_acc:.2f}%")
    return best_val_acc

def save_training_results(history, best_model_path, best_params, fusion_model, results_dir='using_mimo_late_result'):
    """Save comprehensive training results and analysis"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save training history
    history_path = f'{results_dir}/logs/training_history_{timestamp}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save hyperparameters
    hyperparams_path = f'{results_dir}/logs/hyperparameters_{timestamp}.json'
    with open(hyperparams_path, 'w') as f:
        json.dump({
            'best_hyperparameters': best_params,
            'fusion_method': fusion_model.fusion_method,
            'total_parameters': sum(p.numel() for p in fusion_model.parameters()),
            'trainable_parameters': sum(p.numel() for p in fusion_model.parameters() if p.requires_grad),
            'timestamp': timestamp
        }, f, indent=2)
    
    # Create comprehensive training plots
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.plot(history['val_loss'], label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(2, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', color='blue')
    plt.plot(history['val_acc'], label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Learning curve
    plt.subplot(2, 3, 3)
    epochs = range(1, len(history['train_acc']) + 1)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    plt.fill_between(epochs, history['train_acc'], alpha=0.3, color='blue')
    plt.fill_between(epochs, history['val_acc'], alpha=0.3, color='red')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    # Loss convergence
    plt.subplot(2, 3, 4)
    plt.semilogy(history['train_loss'], label='Train Loss', color='blue')
    plt.semilogy(history['val_loss'], label='Val Loss', color='red')
    plt.title('Loss Convergence (Log Scale)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True)
    
    # Overfitting analysis
    plt.subplot(2, 3, 5)
    overfitting_gap = [train_acc - val_acc for train_acc, val_acc in zip(history['train_acc'], history['val_acc'])]
    plt.plot(overfitting_gap, color='orange', linewidth=2)
    plt.title('Overfitting Gap (Train - Val Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy Gap (%)')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True)
    
    # Final metrics summary
    plt.subplot(2, 3, 6)
    plt.axis('off')
    final_metrics = f"""
    Final Training Results
    =====================
    Best Val Accuracy: {max(history['val_acc']):.2f}%
    Final Train Accuracy: {history['train_acc'][-1]:.2f}%
    Final Val Accuracy: {history['val_acc'][-1]:.2f}%
    Final Train Loss: {history['train_loss'][-1]:.4f}
    Final Val Loss: {history['val_loss'][-1]:.4f}
    
    Model Configuration
    ==================
    Fusion Method: {fusion_model.fusion_method}
    Total Parameters: {sum(p.numel() for p in fusion_model.parameters()):,}
    Trainable Parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}
    """
    plt.text(0.1, 0.9, final_metrics, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plot_path = f'{results_dir}/plots/comprehensive_training_analysis_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary report
    summary_path = f'{results_dir}/logs/training_summary_{timestamp}.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MIMO LATE FUSION TRAINING SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Training Duration: {len(history['train_acc'])} epochs\n\n")
        
        f.write("PERFORMANCE METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Best Validation Accuracy: {max(history['val_acc']):.4f}%\n")
        f.write(f"Final Training Accuracy: {history['train_acc'][-1]:.4f}%\n")
        f.write(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}%\n")
        f.write(f"Final Training Loss: {history['train_loss'][-1]:.6f}\n")
        f.write(f"Final Validation Loss: {history['val_loss'][-1]:.6f}\n")
        f.write(f"Overfitting Gap: {history['train_acc'][-1] - history['val_acc'][-1]:.4f}%\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Fusion Method: {fusion_model.fusion_method}\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in fusion_model.parameters()):,}\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}\n")
        
        if fusion_model.fusion_method == 'weighted':
            weights = F.softmax(fusion_model.fusion_weights, dim=0)
            f.write(f"Final Fusion Weights: MIMAMO={weights[0]:.4f}, Multimodal LSTM={weights[1]:.4f}\n")
        
        f.write("\nHYPERPARAMETERS:\n")
        f.write("-"*40 + "\n")
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
        
        f.write(f"\nMODEL CHECKPOINTS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Best Model Path (Main): checkpoint_4_mimamo/\n")
        f.write(f"Best Model Path (Results): {best_model_path}\n")
        
        f.write("\nFILES GENERATED:\n")
        f.write("-"*40 + "\n")
        f.write(f"Training History: {history_path}\n")
        f.write(f"Hyperparameters: {hyperparams_path}\n")
        f.write(f"Training Plots: {plot_path}\n")
        f.write(f"Summary Report: {summary_path}\n")
    
    print(f"\n📁 Comprehensive results saved:")
    print(f"   📊 Training plots: {plot_path}")
    print(f"   📈 Training history: {history_path}")
    print(f"   ⚙️  Hyperparameters: {hyperparams_path}")
    print(f"   📝 Summary report: {summary_path}")
    print(f"   💾 Best model: {best_model_path}")
    
    return {
        'plots': plot_path,
        'history': history_path,
        'hyperparams': hyperparams_path,
        'summary': summary_path,
        'model': best_model_path
    }

def run_hyperparameter_optimization(train_loader, val_loader, mimamo_model, multimodal_model, device, n_trials=50):
    """Run hyperparameter optimization using Optuna"""
    
    if not OPTUNA_AVAILABLE:
        print("❌ Optuna not available. Skipping hyperparameter optimization.")
        return None
    
    print("\n" + "="*60)
    print("🔍 STARTING HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    # Create study
    study_name = f"late_fusion_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study = optuna.create_study(
        direction='maximize',
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    )
    
    # Optimize
    def timeout_objective(trial):
        try:
            return objective(trial, train_loader, val_loader, mimamo_model, multimodal_model, device)
        except Exception as e:
            print(f"❌ Trial {trial.number} failed: {e}")
            return 0.0  # Return minimum score for failed trials
    
    study.optimize(
        timeout_objective,
        n_trials=n_trials,
        show_progress_bar=True,
        timeout=3600  # 1 hour timeout
    )
    
    # Results
    print(f"\n✅ Hyperparameter optimization completed!")
    print(f"📊 Best trial:")
    print(f"  Value: {study.best_value:.4f}")
    print(f"  Params: {study.best_params}")
    
    # Save results
    results_path = f'using_mimo_late_result/optuna_studies/optuna_study_{study_name}.json'
    with open(results_path, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'study_name': study_name
        }, f, indent=2)
    
    print(f"📁 Optimization results saved to: {results_path}")
    
    # Create visualization if possible
    try:
        # Optimization history plot
        fig = optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title('Hyperparameter Optimization History')
        plt.savefig(f'using_mimo_late_result/plots/optuna_history_{study_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Parameter importances
        fig = optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title('Parameter Importances')
        plt.savefig(f'using_mimo_late_result/plots/optuna_importances_{study_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Optimization plots saved")
        
    except Exception as e:
        print(f"Warning: Could not create optimization plots: {e}")
    
    return study.best_params

def main():
    """Main function for combined late fusion training with hyperparameter optimization"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Create directories
    results_dir = 'using_mimo_late_result'
    checkpoint_dir = 'checkpoint_4_mimamo'
    mimamo_results_dir = 'mimamo_net_result'
    
    # Create all necessary directories
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f'{results_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{results_dir}/plots', exist_ok=True)
    os.makedirs(f'{results_dir}/logs', exist_ok=True)
    os.makedirs(f'{results_dir}/optuna_studies', exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)  # Main checkpoint directory
    os.makedirs(mimamo_results_dir, exist_ok=True)  # MIMAMO results directory
    os.makedirs(f'{mimamo_results_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{mimamo_results_dir}/plots', exist_ok=True)
    os.makedirs(f'{mimamo_results_dir}/logs', exist_ok=True)
    
    print(f"✅ Created directories:")
    print(f"   📂 Results: {results_dir}/")
    print(f"   📂 Main Checkpoints: {checkpoint_dir}/")
    print(f"   📂 MIMAMO Results: {mimamo_results_dir}/")
    
    # Data paths - both models now use aligned data
    # Video model uses video-aligned data
    train_json_video = 'json/mapped_train_data_video_aligned.json'
    val_json_video = 'json/mapped_val_data_video_aligned.json'
    test_json_video = 'json/mapped_test_data_video_aligned.json'
    
    # Audio model also uses aligned data
    train_json_audio = 'json/mapped_train_data_video_aligned.json'
    val_json_audio = 'json/mapped_val_data_video_aligned.json'
    test_json_audio = 'json/mapped_test_data_video_aligned.json'
    
    video_dir = 'data/train_video/video_v5_0'
    audio_dir = 'data/train_audio/audio_v5_0'
    
    # Model checkpoint paths
    mimamo_checkpoint_path = 'checkpoint_3/best_mimamo_model_20251030_135754_acc0.5804.pth'
    multimodal_checkpoint_path = 'checkpoints/best_7class_model.pth'
    
    # Initialize components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    video_transform = VideoTransform()
    
    # Create datasets using video-aligned JSON files (contains both video and audio data)
    print("Creating combined datasets using video-aligned data...")
    train_dataset = CombinedFusionDataset(
        json_file=train_json_video,
        video_dir=video_dir,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
        video_transform=video_transform,
        max_dialogue_length=10
    )
    
    val_dataset = CombinedFusionDataset(
        json_file=val_json_video,
        video_dir=video_dir,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
        video_transform=video_transform,
        max_dialogue_length=10
    )
    
    # Load pretrained models
    print("Loading pretrained models...")
    mimamo_model, multimodal_model = load_pretrained_models(
        mimamo_checkpoint_path, multimodal_checkpoint_path, train_dataset
    )
    
    # Check if models loaded successfully
    if mimamo_model is None or multimodal_model is None:
        print("❌ Failed to load required models. Exiting...")
        return
    
    # Create initial data loaders for hyperparameter optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,  # Initial batch size
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Test data loading
    print("Testing data loading...")
    try:
        test_batch = next(iter(train_loader))
        print(f"✅ Data loading successful - batch size: {test_batch['label'].shape[0]}")
        del test_batch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return
    
    # Ask user for hyperparameter optimization
    use_optimization = input("\n🔍 Run hyperparameter optimization? (y/n, default=y): ").strip().lower()
    if use_optimization == '' or use_optimization == 'y':
        # Run hyperparameter optimization
        n_trials = int(input("Number of optimization trials (default=30): ") or "30")
        
        best_params = run_hyperparameter_optimization(
            train_loader, val_loader, mimamo_model, multimodal_model, device, n_trials=n_trials
        )
        
        if best_params is None:
            print("Using default hyperparameters...")
            best_params = {
                'fusion_method': 'weighted',
                'learning_rate': 1e-3,
                'batch_size': 4,
                'mimamo_weight': 0.58,
                'focal_alpha': 1.0,
                'focal_gamma': 2.0,
                'weight_decay': 1e-4
            }
    else:
        print("Skipping hyperparameter optimization, using default parameters...")
        best_params = {
            'fusion_method': 'weighted',
            'learning_rate': 1e-3,
            'batch_size': 4,
            'mimamo_weight': 0.58,
            'focal_alpha': 1.0,
            'focal_gamma': 2.0,
            'weight_decay': 1e-4
        }
    
    # Create optimized model with best hyperparameters
    print(f"\n🚀 Creating optimized model with best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # Extract hyperparameters
    fusion_method = best_params.get('fusion_method', 'weighted')
    if fusion_method == 'weighted':
        mimamo_weight = best_params.get('mimamo_weight', 0.58)
        initial_weights = [mimamo_weight, 1.0 - mimamo_weight]
        dropout_rate = 0.3
        hidden_dim = 256
        fusion_dim = 512
    else:
        initial_weights = None
        dropout_rate = best_params.get('dropout_rate', 0.3)
        hidden_dim = best_params.get('hidden_dim', 256)
        fusion_dim = best_params.get('fusion_dim', 512)
    
    fusion_model = EnhancedLateFusionModel(
        mimamo_model=mimamo_model,
        multimodal_lstm_model=multimodal_model,
        num_classes=7,
        fusion_method=fusion_method,
        initial_weights=initial_weights,
        dropout_rate=dropout_rate,
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim
    ).to(device)
    
    print(f"Enhanced late fusion model created")
    print(f"Total parameters: {sum(p.numel() for p in fusion_model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,}")
    
    # Create optimized data loaders
    optimal_batch_size = best_params.get('batch_size', 4)
    train_loader_final = DataLoader(
        train_dataset,
        batch_size=optimal_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader_final = DataLoader(
        val_dataset,
        batch_size=optimal_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Train the model with optimized hyperparameters
    print("\n" + "="*60)
    print("🚀 STARTING OPTIMIZED LATE FUSION TRAINING")
    print("="*60)
    
    # Create custom training function with optimized parameters
    def train_optimized_model():
        # Mixed precision training
        scaler = GradScaler()
        
        # Optimized loss function
        criterion = FocalLoss(
            alpha=best_params.get('focal_alpha', 1.0),
            gamma=best_params.get('focal_gamma', 2.0),
            num_classes=7
        )
        
        # Optimized optimizer
        optimizer_params = []
        for param in fusion_model.parameters():
            if param.requires_grad:
                optimizer_params.append({
                    'params': param, 
                    'lr': best_params.get('learning_rate', 1e-3)
                })
        
        if not optimizer_params:
            print("Warning: No trainable parameters found!")
            return None, None
        
        optimizer = torch.optim.AdamW(
            optimizer_params, 
            weight_decay=best_params.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader_final) // 3,
            num_training_steps=len(train_loader_final) * 20
        )
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_path = None
        patience = 7
        patience_counter = 0
        num_epochs = 20
        
        print(f"Starting training with optimized hyperparameters...")
        print(f"Learning rate: {best_params.get('learning_rate', 1e-3)}")
        print(f"Batch size: {optimal_batch_size}")
        print(f"Focal loss - Alpha: {best_params.get('focal_alpha', 1.0)}, Gamma: {best_params.get('focal_gamma', 2.0)}")
        
        for epoch in range(num_epochs):
            # Training phase
            fusion_model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            torch.cuda.empty_cache()
            
            pbar = tqdm(train_loader_final, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                # Prepare data
                mimamo_data = {
                    'dialogue_video': batch['dialogue_video'].to(device),
                    'dialogue_roles': batch['dialogue_roles_video'].to(device),
                    'metadata': batch['metadata'].to(device),
                    'sequence_length': batch['sequence_length'].to(device)
                }
                
                multimodal_data = {
                    'context_input_ids': batch['context_input_ids'].to(device),
                    'context_attention_mask': batch['context_attention_mask'].to(device),
                    'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                    'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                    'dialogue_audio': batch['dialogue_audio_raw'].to(device),
                    'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                    'metadata': batch['metadata'].to(device),
                    'sequence_length': batch['sequence_length'].to(device)
                }
                
                labels = batch['label'].to(device)
                
                # Forward pass
                with autocast():
                    fused_logits, _, _ = fusion_model(mimamo_data, multimodal_data)
                    loss = criterion(fused_logits, labels)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(fused_logits.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if batch_idx % 20 == 0:
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{100.*train_correct/train_total:.2f}%',
                        'GPU': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
                    })
            
            # Validation phase
            fusion_model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                for batch in tqdm(val_loader_final, desc='Validation'):
                    mimamo_data = {
                        'dialogue_video': batch['dialogue_video'].to(device),
                        'dialogue_roles': batch['dialogue_roles_video'].to(device),
                        'metadata': batch['metadata'].to(device),
                        'sequence_length': batch['sequence_length'].to(device)
                    }
                    
                    multimodal_data = {
                        'context_input_ids': batch['context_input_ids'].to(device),
                        'context_attention_mask': batch['context_attention_mask'].to(device),
                        'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                        'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                        'dialogue_audio': batch['dialogue_audio_raw'].to(device),
                        'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                        'metadata': batch['metadata'].to(device),
                        'sequence_length': batch['sequence_length'].to(device)
                    }
                    
                    labels = batch['label'].to(device)
                    
                    with autocast():
                        fused_logits, _, _ = fusion_model(mimamo_data, multimodal_data)
                        loss = criterion(fused_logits, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(fused_logits.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Update history
            history['train_loss'].append(train_loss / len(train_loader_final))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss / len(val_loader_final))
            history['val_acc'].append(val_acc)
            
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss/len(train_loader_final):.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss/len(val_loader_final):.4f}, Val Acc: {val_acc:.2f}%')
            
            # Print fusion weights if using weighted fusion
            if fusion_model.fusion_method == 'weighted':
                weights = F.softmax(fusion_model.fusion_weights, dim=0)
                print(f'  Fusion weights: MIMAMO={weights[0]:.3f}, Multimodal={weights[1]:.3f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save to three locations
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path_main = f'checkpoint_4_mimamo/best_optimized_fusion_model_{timestamp}_acc{val_acc:.4f}.pth'
                best_model_path_results = f'using_mimo_late_result/checkpoints/best_optimized_fusion_model_{timestamp}_acc{val_acc:.4f}.pth'
                best_model_path_mimamo = f'mimamo_net_result/checkpoints/best_optimized_fusion_model_{timestamp}_acc{val_acc:.4f}.pth'
                
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_loss': train_loss / len(train_loader_final),
                    'val_loss': val_loss / len(val_loader_final),
                    'history': history,
                    'best_hyperparameters': best_params,
                    'fusion_method': fusion_model.fusion_method
                }
                
                # Save to all three checkpoint directories
                torch.save(checkpoint_data, best_model_path_main)
                torch.save(checkpoint_data, best_model_path_results)
                torch.save(checkpoint_data, best_model_path_mimamo)
                best_model_path = best_model_path_main  # Use main path as primary
                
                print(f'  ✅ New best optimized model saved!')
                print(f'     📁 Main: {best_model_path_main}')
                print(f'     📁 Results: {best_model_path_results}')
                print(f'     📁 MIMAMO: {best_model_path_mimamo}')
                print(f'     Accuracy: {val_acc:.2f}%')
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs without improvement')
                break
        
        return history, best_model_path
    
    history, best_model_path = train_optimized_model()
    
    if history is None:
        print("Training failed!")
        return
    
    # Save comprehensive training results
    print("\n" + "="*60)
    print("💾 SAVING COMPREHENSIVE TRAINING RESULTS")
    print("="*60)
    
    # Save to both result directories
    result_files = save_training_results(history, best_model_path, best_params, fusion_model, 'using_mimo_late_result')
    mimamo_result_files = save_training_results(history, best_model_path, best_params, fusion_model, 'mimamo_net_result')
    
    print(f"\n✅ Optimized MIMO Late Fusion training completed!")
    print(f"📊 Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"🔧 Optimized hyperparameters: {best_params}")
    
    print("\n" + "="*60)
    print("🎉 MIMO LATE FUSION TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Results saved in MULTIPLE directories:")
    print(f"   📂 Main Checkpoints: checkpoint_4_mimamo/")
    print(f"   📂 Primary Results: using_mimo_late_result/")
    print(f"   📂 MIMAMO Results: mimamo_net_result/")
    print(f"   📈 Plots: using_mimo_late_result/plots/ & mimamo_net_result/plots/")
    print(f"   📝 Logs: using_mimo_late_result/logs/ & mimamo_net_result/logs/")
    print(f"   🔍 Optuna Studies: using_mimo_late_result/optuna_studies/")
    print("   � Training results saved to directories above")
    print("="*60)

def test_combined_fusion_model(fusion_model, test_loader, device, results_dir='using_mimo_late_result'):
    """Test the combined fusion model on test dataset with comprehensive evaluation"""
    
    print(f"\n🧪 Starting comprehensive test evaluation...")
    fusion_model.eval()
    
    all_predictions = []
    all_labels = []
    all_mimamo_predictions = []
    all_multimodal_predictions = []
    all_fusion_weights = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Prepare MIMAMO data
            mimamo_data = {
                'dialogue_video': batch['dialogue_video'].to(device),
                'dialogue_roles': batch['dialogue_roles_video'].to(device),
                'metadata': batch['metadata'].to(device),
                'sequence_length': batch['sequence_length'].to(device)
            }
            
            # Prepare Multimodal LSTM data
            multimodal_data = {
                'context_input_ids': batch['context_input_ids'].to(device),
                'context_attention_mask': batch['context_attention_mask'].to(device),
                'dialogue_input_ids': batch['dialogue_input_ids'].to(device),
                'dialogue_attention_mask': batch['dialogue_attention_mask'].to(device),
                'dialogue_audio': batch['dialogue_audio_raw'].to(device),
                'dialogue_roles': batch['dialogue_roles_audio'].to(device),
                'metadata': batch['metadata'].to(device),
                'sequence_length': batch['sequence_length'].to(device)
            }
            
            labels = batch['label'].to(device)
            
            # Forward pass with feature extraction
            outputs = fusion_model(mimamo_data, multimodal_data, return_features=True)
            
            # Get predictions
            main_predictions = torch.max(outputs['fused_logits'], 1)[1]
            mimamo_predictions = torch.max(outputs['mimamo_logits'], 1)[1]
            multimodal_predictions = torch.max(outputs['multimodal_logits'], 1)[1]
            
            # Store results
            all_predictions.extend(main_predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_mimamo_predictions.extend(mimamo_predictions.cpu().numpy())
            all_multimodal_predictions.extend(multimodal_predictions.cpu().numpy())
            all_fusion_weights.extend(outputs['fusion_weights'].cpu().numpy())
    
    # Calculate metrics
    labels = np.array(all_labels)
    main_preds = np.array(all_predictions)
    mimamo_preds = np.array(all_mimamo_predictions)
    multimodal_preds = np.array(all_multimodal_predictions)
    fusion_weights = np.array(all_fusion_weights)
    
    # Main fusion model metrics
    main_accuracy = accuracy_score(labels, main_preds)
    main_precision, main_recall, main_f1, _ = precision_recall_fscore_support(
        labels, main_preds, average='weighted', zero_division=0
    )
    
    # Component model metrics
    mimamo_accuracy = accuracy_score(labels, mimamo_preds)
    multimodal_accuracy = accuracy_score(labels, multimodal_preds)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        labels, main_preds, average=None, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, main_preds)
    
    emotion_classes = ['happy', 'surprised', 'angry', 'fear', 'sad', 'disgusted', 'contempt']
    
    # Print test results
    print(f"\n📊 COMBINED FUSION TEST RESULTS:")
    print("=" * 60)
    print(f"🎯 Combined Fusion Accuracy: {main_accuracy:.4f} ({main_accuracy*100:.2f}%)")
    print(f"📈 Combined Fusion F1-Score: {main_f1:.4f}")
    print(f"📈 MIMAMO Net Accuracy:      {mimamo_accuracy:.4f} ({mimamo_accuracy*100:.2f}%)")
    print(f"📈 Multimodal LSTM Accuracy: {multimodal_accuracy:.4f} ({multimodal_accuracy*100:.2f}%)")
    
    improvement_acc = (main_accuracy - max(mimamo_accuracy, multimodal_accuracy)) * 100
    print(f"\n🎯 Fusion Improvement: +{improvement_acc:.2f}% accuracy")
    
    print(f"\n🏋️ Fusion Weight Analysis:")
    if fusion_weights.shape[1] >= 2:
        print(f"   Mean MIMAMO Weight:      {np.mean(fusion_weights[:, 0]):.3f} ± {np.std(fusion_weights[:, 0]):.3f}")
        print(f"   Mean Multimodal Weight:  {np.mean(fusion_weights[:, 1]):.3f} ± {np.std(fusion_weights[:, 1]):.3f}")
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create confusion matrix visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_classes, yticklabels=emotion_classes)
    plt.title('Combined Fusion Model - Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/combined_fusion_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'mimamo_net_result/plots/combined_fusion_confusion_matrix_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model comparison visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    models = ['Combined Fusion', 'MIMAMO Net', 'Multimodal LSTM']
    accuracies = [main_accuracy, mimamo_accuracy, multimodal_accuracy]
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
    ax2.set_title('Per-Class F1-Score (Combined Fusion)')
    ax2.set_xlabel('Emotion Classes')
    ax2.set_ylabel('F1-Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(emotion_classes, rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/plots/combined_fusion_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'mimamo_net_result/plots/combined_fusion_performance_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed JSON results
    test_results = {
        'test_info': {
            'timestamp': timestamp,
            'test_samples': len(labels),
            'emotion_classes': emotion_classes
        },
        'performance_metrics': {
            'combined_fusion': {
                'accuracy': float(main_accuracy),
                'precision': float(main_precision),
                'recall': float(main_recall),
                'f1_score': float(main_f1)
            },
            'mimamo_net': {'accuracy': float(mimamo_accuracy)},
            'multimodal_lstm': {'accuracy': float(multimodal_accuracy)}
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
            'mean_mimamo_weight': float(np.mean(fusion_weights[:, 0])) if fusion_weights.shape[1] >= 2 else 0,
            'mean_multimodal_weight': float(np.mean(fusion_weights[:, 1])) if fusion_weights.shape[1] >= 2 else 0,
            'mimamo_weight_std': float(np.std(fusion_weights[:, 0])) if fusion_weights.shape[1] >= 2 else 0,
            'multimodal_weight_std': float(np.std(fusion_weights[:, 1])) if fusion_weights.shape[1] >= 2 else 0
        },
        'confusion_matrix': cm.tolist()
    }
    
    # Save to both result directories
    with open(f'{results_dir}/logs/combined_fusion_test_results_{timestamp}.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    with open(f'mimamo_net_result/logs/combined_fusion_test_results_{timestamp}.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\n✅ Test results saved to {results_dir}/ and mimamo_net_result/")
    print(f"   📊 Confusion Matrix: combined_fusion_confusion_matrix_{timestamp}.png")
    print(f"   📈 Performance Charts: combined_fusion_performance_{timestamp}.png")
    print(f"   📄 Detailed Results: combined_fusion_test_results_{timestamp}.json")
    
    return test_results

def main():
    """Main training function with comprehensive testing"""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create comprehensive directory structure
    directories = [
        'using_mimo_late_result', 'using_mimo_late_result/checkpoints', 'using_mimo_late_result/plots',
        'using_mimo_late_result/logs', 'using_mimo_late_result/optuna_studies',
        'checkpoint_4_mimamo', 'mimamo_net_result', 'mimamo_net_result/checkpoints',
        'mimamo_net_result/plots', 'mimamo_net_result/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created directories:")
    print("   📂 Results: using_mimo_late_result/")
    print("   📂 Main Checkpoints: checkpoint_4_mimamo/")
    print("   📂 MIMAMO Results: mimamo_net_result/")
    
    # Dataset paths
    # Data paths - both models now use aligned data
    # Video model uses video-aligned data
    train_json_video = 'json/mapped_train_data_video_aligned.json'
    val_json_video = 'json/mapped_val_data_video_aligned.json'
    test_json_video = 'json/mapped_test_data_video_aligned.json'
    
    # Audio model also uses aligned data
    train_json_audio = 'json/mapped_train_data_video_aligned.json'
    val_json_audio = 'json/mapped_val_data_video_aligned.json'
    test_json_audio = 'json/mapped_test_data_video_aligned.json'
    
    video_dir = 'data/train_video/video_v5_0'
    audio_dir = 'data/train_audio/audio_v5_0'
    
    # Pretrained model paths
    mimamo_checkpoint_path = 'checkpoint_3/best_mimamo_model_20251030_135754_acc0.5804.pth'
    multimodal_checkpoint_path = 'checkpoints/best_7class_model.pth'
    
    # Initialize components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-base')
    video_transform = VideoTransform()
    
    print("Creating combined datasets using video-aligned data...")
    # Create datasets
    train_dataset = CombinedFusionDataset(
        json_file=train_json_video,
        video_dir=video_dir,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
        video_transform=video_transform,
        max_dialogue_length=10
    )
    
    val_dataset = CombinedFusionDataset(
        json_file=val_json_video,
        video_dir=video_dir,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
        video_transform=video_transform,
        max_dialogue_length=10
    )
    
    # Load pretrained models
    mimamo_model, multimodal_model = load_pretrained_models(
        mimamo_checkpoint_path, multimodal_checkpoint_path, train_dataset
    )
    
    if mimamo_model is None or multimodal_model is None:
        print("❌ Failed to load pretrained models. Exiting.")
        return
    
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Test data loading
    print("Testing data loading...")
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    try:
        test_batch = next(iter(train_loader))
        print("✅ Optuna available for hyperparameter optimization")
        print("✅ Optuna available for hyperparameter optimization")
        print(f"✅ Data loading successful - batch size: {test_batch['label'].shape[0]}")
        del test_batch
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return
    
    # Hyperparameter optimization or direct training
    if OPTUNA_AVAILABLE:
        try:
            optimize_hyperparameters = input("🔍 Run hyperparameter optimization? (y/n, default=y): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # Fallback for non-interactive environments
            optimize_hyperparameters = 'y'
            print("🔍 Run hyperparameter optimization? (y/n, default=y): y")
        
        if optimize_hyperparameters in ['', 'y', 'yes']:
            try:
                n_trials_input = input("Number of optimization trials (default=30): ").strip()
                n_trials = int(n_trials_input) if n_trials_input else 30
            except (EOFError, KeyboardInterrupt, ValueError):
                # Fallback for non-interactive environments
                n_trials = 30
                print("Number of optimization trials (default=30): 30")
            
            print("\n" + "="*60)
            print("🔍 STARTING HYPERPARAMETER OPTIMIZATION")
            print("="*60)
            
            # Create study
            study_name = f"late_fusion_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
            )
            
            # Optimize with enhanced progress tracking
            def timeout_objective(trial):
                try:
                    return objective(trial, train_loader, val_loader, mimamo_model, multimodal_model, device)
                except Exception as e:
                    print(f"❌ Trial {trial.number} failed: {e}")
                    return 0.0  # Return minimum score for failed trials
            
            # Custom progress bar for trials
            trial_pbar = tqdm(total=n_trials, desc="🔬 Hyperparameter Optimization", 
                             bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                             dynamic_ncols=True, position=0)
            
            def progress_callback(study, trial):
                trial_pbar.update(1)
                trial_pbar.set_postfix({
                    'Best': f'{study.best_value:.4f}' if study.best_value else 'N/A',
                    'Current': f'{trial.value:.4f}' if trial.value else 'Failed',
                    'Trial': f'{trial.number+1}/{n_trials}'
                })
            
            study.optimize(
                timeout_objective,
                n_trials=n_trials,
                callbacks=[progress_callback],
                show_progress_bar=False,  # Use our custom progress bar
                timeout=3600  # 1 hour timeout
            )
            
            trial_pbar.close()  # Close the progress bar
            
            # Results
            print(f"\n✅ Hyperparameter optimization completed!")
            print(f"📊 Best trial:")
            print(f"  Value: {study.best_value:.4f}")
            print(f"  Params: {study.best_params}")
            
            # Save results
            results_path = f'using_mimo_late_result/optuna_studies/optuna_study_{study_name}.json'
            with open(results_path, 'w') as f:
                json.dump({
                    'best_value': study.best_value,
                    'best_params': study.best_params,
                    'n_trials': len(study.trials),
                    'study_name': study_name
                }, f, indent=2)
            
            print(f"📁 Optuna study saved: {results_path}")
            
            # Use best parameters for final training
            best_params = study.best_params
        else:
            # Use default parameters
            best_params = {
                'fusion_method': 'weighted',
                'learning_rate': 1e-3,
                'batch_size': 4,
                'mimamo_weight': 0.6
            }
    else:
        # Use default parameters if Optuna not available
        best_params = {
            'fusion_method': 'weighted',
            'learning_rate': 1e-3,
            'batch_size': 4,
            'mimamo_weight': 0.6
        }
    
    print("\n" + "="*60)
    print("🚀 STARTING FINAL TRAINING WITH OPTIMAL PARAMETERS")
    print("="*60)
    print(f"📊 Using parameters: {best_params}")
    
    # Create final model with ADVANCED settings for best performance
    fusion_model = EnhancedLateFusionModel(
        mimamo_model=mimamo_model,
        multimodal_lstm_model=multimodal_model,
        num_classes=7,
        fusion_method=best_params.get('fusion_method', 'weighted'),
        initial_weights=[best_params.get('mimamo_weight', 0.6), 1.0 - best_params.get('mimamo_weight', 0.6)] if best_params.get('fusion_method') == 'weighted' else None,
        dropout_rate=best_params.get('dropout_rate', 0.3),
        hidden_dim=best_params.get('hidden_dim', 512),  # 🔥 Increased default
        fusion_dim=best_params.get('fusion_dim', 1024),  # 🔥 Increased default
        use_advanced_fusion=True  # 🔥 ALWAYS use advanced fusion for final training
    ).to(device)
    
    # Create final data loaders
    batch_size = best_params.get('batch_size', 4)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Train final model
    print(f"🏗️ Final model created with ADVANCED fusion and {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):,} trainable parameters")
    
    # 🔥 **ENHANCED FINAL TRAINING** for best performance  
    history, best_model_path = train_combined_fusion_model(
        fusion_model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=100,  # 🚀 Increased epochs for best performance
        learning_rate=best_params.get('learning_rate', 1e-4),  # 🔥 Lower learning rate for stability
        use_advanced_training=True  # 🔥 Enable all advanced features
    )
    
    # Save training results
    save_training_results(history, best_model_path, best_params, fusion_model, 'using_mimo_late_result')
    save_training_results(history, best_model_path, best_params, fusion_model, 'mimamo_net_result')
    
    print("\n" + "="*60)
    print("🎉 MIMO LATE FUSION TRAINING COMPLETE!")
    print("="*60)
    print(f"📁 Results saved in MULTIPLE directories:")
    print(f"   📂 Main Checkpoints: checkpoint_4_mimamo/")
    print(f"   📂 Primary Results: using_mimo_late_result/")
    print(f"   📂 MIMAMO Results: mimamo_net_result/")
    print(f"   📈 Plots: using_mimo_late_result/plots/ & mimamo_net_result/plots/")
    print(f"   📝 Logs: using_mimo_late_result/logs/ & mimamo_net_result/logs/")
    print(f"   🔍 Optuna Studies: using_mimo_late_result/optuna_studies/")
    print("   � Training results saved to directories above")
    print("="*60)
    
    # 🧪 Testing Phase
    print("\n" + "="*60)
    print("🧪 STARTING AUTOMATIC TEST EVALUATION")
    print("="*60)
    
    # Create test dataset and loader
    test_dataset = CombinedFusionDataset(
        json_file=test_json_video,
        video_dir=video_dir,
        audio_dir=audio_dir,
        tokenizer=tokenizer,
        wav2vec_feature_extractor=wav2vec_feature_extractor,
        video_transform=video_transform,
        max_dialogue_length=10
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,  # Larger batch for testing
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"📋 Test samples: {len(test_dataset)}")
    
    # Load best model for testing
    print(f"📂 Loading best model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run comprehensive test evaluation
    test_results = test_combined_fusion_model(
        fusion_model=fusion_model,
        test_loader=test_loader,
        device=device,
        results_dir='using_mimo_late_result'
    )
    
    # Print final summary
    print("\n" + "="*60)
    print("🎉 COMBINED FUSION PIPELINE COMPLETE!")
    print("="*60)
    print(f"📊 Final Test Accuracy: {test_results['performance_metrics']['combined_fusion']['accuracy']:.4f} ({test_results['performance_metrics']['combined_fusion']['accuracy']*100:.2f}%)")
    print(f"📁 All results saved in: using_mimo_late_result/ and mimamo_net_result/")
    print(f"💾 Best model saved at: {best_model_path}")
    print("="*60)

if __name__ == "__main__":
    main()
    main()