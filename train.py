import copy
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import librosa
import time
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel,
    Wav2Vec2FeatureExtractor, Wav2Vec2Model,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

class MultimodalSequentialDataset(Dataset):
    def __init__(self, data_path, audio_dir, tokenizer, wav2vec_feature_extractor=None, max_length=512, max_dialogue_length=10, sample_rate=16000):
        """
        Dataset for multimodal empathetic dialogue with sequential processing
        One sample per conversation (not per utterance)
        """
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.wav2vec_feature_extractor = wav2vec_feature_extractor
        self.max_length = max_length
        self.max_dialogue_length = max_dialogue_length
        self.sample_rate = sample_rate
        
        # Emotion mapping from data_mapping.py
        # Emotion mapping to EmpatheticDialogues 32 emotions
        self.ed_emotion_projection = {
            'conflicted': 'anxious', 'vulnerability': 'afraid', 'helplessness': 'afraid',
            'sadness': 'sad', 'pensive': 'sentimental', 'frustration': 'annoyed',
            'weary': 'tired', 'anxiety': 'anxious', 'reflective': 'sentimental',
            'upset': 'disappointed', 'worried': 'anxious', 'fear': 'afraid',
            'frustrated': 'annoyed', 'fatigue': 'tired', 'lost': 'lonely',
            'disappointment': 'disappointed', 'nostalgia': 'nostalgic', 'exhaustion': 'tired',
            'uneasy': 'anxious', 'loneliness': 'lonely', 'fragile': 'afraid',
            'confused': 'surprised', 'vulnerable': 'afraid', 'thoughtful': 'sentimental',
            'stressed': 'anxious', 'concerned': 'anxious', 'tiredness': 'tired',
            'burdened': 'anxious', 'melancholy': 'sad', 'overwhelmed': 'anxious',
            'worry': 'anxious', 'heavy-hearted': 'sad', 'melancholic': 'sad',
            'nervous': 'anxious', 'fearful': 'afraid', 'stress': 'anxious',
            'confusion': 'surprised', 'inadequacy': 'ashamed', 'regret': 'guilty',
            'helpless': 'afraid', 'concern': 'anxious', 'exhausted': 'tired',
            'overwhelm': 'anxious', 'tired': 'tired', 'disappointed': 'disappointed',
            'surprised': 'surprised', 'excited': 'excited', 'angry': 'angry',
            'proud': 'proud', 'annoyed': 'annoyed', 'grateful': 'grateful',
            'lonely': 'lonely', 'afraid': 'afraid', 'terrified': 'terrified',
            'guilty': 'guilty', 'impressed': 'impressed', 'disgusted': 'disgusted',
            'hopeful': 'hopeful', 'confident': 'confident', 'furious': 'furious',
            'anxious': 'anxious', 'anticipating': 'anticipating', 'joyful': 'joyful',
            'nostalgic': 'nostalgic', 'prepared': 'prepared', 'jealous': 'jealous',
            'content': 'content', 'devastated': 'devastated', 'embarrassed': 'embarrassed',
            'caring': 'caring', 'sentimental': 'sentimental', 'trusting': 'trusting',
            'ashamed': 'ashamed', 'apprehensive': 'apprehensive', 'faithful': 'faithful'
        }
        
        # EmpatheticDialogues 32 emotion classes
        self.emotion_to_id = {
            'afraid': 0, 'angry': 1, 'annoyed': 2, 'anticipating': 3, 
            'anxious': 4, 'apprehensive': 5, 'ashamed': 6, 'caring': 7, 
            'confident': 8, 'content': 9, 'devastated': 10, 'disappointed': 11, 
            'disgusted': 12, 'embarrassed': 13, 'excited': 14, 'faithful': 15, 
            'furious': 16, 'grateful': 17, 'guilty': 18, 'hopeful': 19, 
            'impressed': 20, 'jealous': 21, 'joyful': 22, 'lonely': 23, 
            'nostalgic': 24, 'prepared': 25, 'proud': 26, 'sad': 27, 
            'sentimental': 28, 'surprised': 29, 'terrified': 30, 'trusting': 31
        }
        
        # Profile mappings
        self.age_to_id = {"child": 0, "young": 1, "middle-aged": 2, "elderly": 3}
        self.gender_to_id = {"male": 0, "female": 1}
        self.timbre_to_id = {"high": 0, "mid": 1, "low": 2}
        self.role_to_id = {"speaker": 0, "listener": 1, "listener_response": 2}
        
        # Create vocabulary for text features
        self.create_text_vocabulary()
        
        print(f"Loaded {len(self.data)} samples from {data_path}")

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

    def extract_audio_features(self, audio):
        """Extract audio features using Wav2Vec2 or fallback to librosa"""
        try:
            if audio is None or len(audio) == 0:
                # Return zeros based on whether we're using Wav2Vec2 or librosa
                if self.wav2vec_feature_extractor is not None:
                    return np.zeros(768, dtype=np.float32)  # Wav2Vec2-base hidden size
                else:
                    return np.zeros(39, dtype=np.float32)  # Original librosa features
            
            # Use Wav2Vec2 if feature extractor is available
            if self.wav2vec_feature_extractor is not None:
                try:
                    # Process audio with Wav2Vec2 feature extractor
                    inputs = self.wav2vec_feature_extractor(
                        audio, 
                        sampling_rate=self.sample_rate, 
                        return_tensors="pt",
                        padding=True
                    )
                    
                    # Return the processed input values for later use by the model
                    return inputs['input_values'].squeeze().numpy()
                except Exception as e:
                    print(f"Wav2Vec2 processing failed: {e}, falling back to librosa")
            
            # Fallback to original librosa features
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=self.sample_rate, n_mels=128, hop_length=512
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Statistical features
            features = []
            features.extend([
                np.mean(mel_spec_db), np.std(mel_spec_db),
                np.max(mel_spec_db), np.min(mel_spec_db),
                np.median(mel_spec_db)
            ])
            
            # MFCCs with error handling
            try:
                mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
                for i in range(13):
                    features.extend([np.mean(mfccs[i]), np.std(mfccs[i])])
            except:
                features.extend([0.0] * 26)  # 13 * 2 features
            
            # Zero crossing rate with error handling
            try:
                zcr = librosa.feature.zero_crossing_rate(audio)
                features.extend([np.mean(zcr), np.std(zcr)])
            except:
                features.extend([0.0, 0.0])
            
            # Spectral features with error handling
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
                spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
                
                features.extend([
                    np.mean(spectral_centroids), np.std(spectral_centroids),
                    np.mean(spectral_rolloff), np.std(spectral_rolloff),
                    np.mean(spectral_bandwidth), np.std(spectral_bandwidth)
                ])
            except:
                features.extend([0.0] * 6)  # 6 spectral features
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            if self.wav2vec_feature_extractor is not None:
                return np.zeros(768, dtype=np.float32)
            else:
                return np.zeros(39, dtype=np.float32)

    def load_audio(self, audio_path):
        """Load and preprocess audio file"""
        try:
            if not os.path.exists(audio_path):
                return None
            
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if len(audio) == 0:
                return None
            
            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            
            # Pad or truncate to fixed length (5 seconds for faster processing)
            max_length = self.sample_rate * 5
            if len(audio) > max_length:
                audio = audio[:max_length]
            else:
                audio = np.pad(audio, (0, max_length - len(audio)), mode='constant')
            
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None

    def __len__(self):
        return len(self.data)
    
    def get_label(self, idx):
        """Get label without processing other data (for efficient class weight computation)"""
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
        
        # Get emotion label (conversation-level)
        raw_emotion = chain_of_empathy.get('speaker_emotion', None)
        mapped_emotion = self.ed_emotion_projection.get(raw_emotion, raw_emotion)
        emotion_label = self.emotion_to_id.get(mapped_emotion, 0)  # Default to 'afraid'
        
        # Extract dialogue sequence (all utterances in conversation)
        dialogue = turn.get('dialogue', [])
        
        # Process sequential dialogue data
        dialogue_texts = []
        dialogue_audio_features = []
        dialogue_roles = []
        dialogue_indices = []
        
        for utt in dialogue[:self.max_dialogue_length]:  # Limit sequence length
            # Text
            text = utt.get('text', '')
            dialogue_texts.append(text)
            
            # Audio
            audio_name = utt.get('audio_name', None)
            if audio_name:
                audio_path = os.path.join(self.audio_dir, audio_name)
                audio = self.load_audio(audio_path)
                audio_features = self.extract_audio_features(audio)
            else:
                # Zero features with appropriate dimension
                if self.wav2vec_feature_extractor is not None:
                    # For Wav2Vec2, store raw audio (will be processed by model)
                    # Use a small silent audio segment
                    audio_features = np.zeros(16000, dtype=np.float32)  # 1 second of silence
                else:
                    audio_features = np.zeros(39, dtype=np.float32)
            dialogue_audio_features.append(audio_features)
            
            # Role and index
            role = utt.get('role', 'speaker')
            dialogue_roles.append(self.role_to_id.get(role, 0))
            dialogue_indices.append(utt.get('index', 0))
        
        # Determine padding dimensions
        if self.wav2vec_feature_extractor is not None:
            # For Wav2Vec2, we need to handle variable-length audio
            # Pad with silent audio segments
            audio_pad_shape = 16000  # 1 second of silence
        else:
            audio_pad_shape = 39  # Librosa features
        
        # Pad sequences to max_dialogue_length
        while len(dialogue_texts) < self.max_dialogue_length:
            dialogue_texts.append('[EMPTY]')  # Use placeholder instead of empty string
            dialogue_audio_features.append(np.zeros(audio_pad_shape, dtype=np.float32))
            dialogue_roles.append(0)
            dialogue_indices.append(0)
        
        # Tokenize all dialogue texts
        dialogue_encodings = []
        for text in dialogue_texts:
            # Ensure text is not empty for tokenizer
            if not text or text.strip() == '':
                text = '[EMPTY]'
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,  # Shorter for individual utterances
                return_tensors='pt'
            )
            dialogue_encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })
        
        # Context encoding (use conversation context)
        context = turn.get('context', '')
        # Ensure context is not empty for tokenizer
        if not context or context.strip() == '':
            context = '[NO CONTEXT]'  # Fallback for empty context
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
        
        # Create comprehensive metadata tensor (12 features total)
        metadata_features = torch.tensor([
            speaker_age, speaker_gender, speaker_timbre, speaker_id,
            listener_age, listener_gender, listener_timbre, listener_id,
            event_scenario_id, emotion_cause_id, goal_response_id, topic_id
        ], dtype=torch.long)
        
        # Process audio features for tensor creation
        if self.wav2vec_feature_extractor is not None:
            # For Wav2Vec2, we have variable-length raw audio
            # Pad/truncate to consistent length for batching
            max_audio_length = 16000  # 1 second at 16kHz
            processed_audio = []
            for audio_feat in dialogue_audio_features:
                if len(audio_feat) > max_audio_length:
                    audio_feat = audio_feat[:max_audio_length]
                elif len(audio_feat) < max_audio_length:
                    audio_feat = np.pad(audio_feat, (0, max_audio_length - len(audio_feat)), 'constant')
                processed_audio.append(audio_feat)
            dialogue_audio_tensor = torch.tensor(processed_audio, dtype=torch.float)
        else:
            # Traditional librosa features - all same length (39)
            dialogue_audio_tensor = torch.tensor(dialogue_audio_features, dtype=torch.float)
        
        return {
            'context_input_ids': context_encoding['input_ids'].squeeze(),
            'context_attention_mask': context_encoding['attention_mask'].squeeze(),
            'dialogue_input_ids': torch.stack([enc['input_ids'] for enc in dialogue_encodings]),
            'dialogue_attention_mask': torch.stack([enc['attention_mask'] for enc in dialogue_encodings]),
            'dialogue_audio': dialogue_audio_tensor,
            'dialogue_roles': torch.tensor(dialogue_roles, dtype=torch.long),
            'dialogue_indices': torch.tensor(dialogue_indices, dtype=torch.long),
            'metadata': metadata_features,
            'label': torch.tensor(emotion_label, dtype=torch.long),
            'conversation_id': item.get('conversation_id', ''),
            'raw_emotion': raw_emotion,
            'sequence_length': torch.tensor(min(len(dialogue), self.max_dialogue_length), dtype=torch.long)
        }

class MultimodalLSTMModel(nn.Module):
    def __init__(self, dataset, num_classes=32, hidden_size=256, num_layers=2, dropout_rate=0.3, use_wav2vec=False):
        super(MultimodalLSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_wav2vec = use_wav2vec
        
        # Text encoder (BERT-base)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        text_dim = self.text_encoder.config.hidden_size  # 768 for BERT-base
        
        # Audio feature processor
        if use_wav2vec:
            # Wav2Vec2-base encoder
            self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            audio_feature_dim = self.wav2vec_model.config.hidden_size  # 768 for wav2vec2-base
            
            # Audio feature processor for Wav2Vec2 features
            self.audio_processor = nn.Sequential(
                nn.Linear(audio_feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        else:
            # Traditional audio features (librosa)
            audio_feature_dim = 39
            self.audio_processor = nn.Sequential(
                nn.Linear(audio_feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
        
        # Metadata embeddings
        self.age_embedding = nn.Embedding(4, 16)
        self.gender_embedding = nn.Embedding(2, 8)
        self.timbre_embedding = nn.Embedding(3, 8)
        self.role_embedding = nn.Embedding(3, 16)
        
        # Profile ID embeddings (assuming max 100 IDs)
        self.speaker_id_embedding = nn.Embedding(100, 32)
        self.listener_id_embedding = nn.Embedding(100, 32)
        
        # Chain of empathy embeddings
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
        
        # Context processor
        self.context_processor = nn.Linear(text_dim, 256)
        
        # Sequential processing with LSTM
        # Input: text + audio + role embeddings
        lstm_input_size = text_dim + 256 + 16  # text + audio + role
        
        self.dialogue_lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for LSTM outputs
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout_rate
        )
        
        # Metadata fusion
        metadata_dim = 16 + 8 + 8 + 32 + 16 + 8 + 8 + 32 + 32 + 32 + 32 + 32  # All embeddings + topic
        self.metadata_processor = nn.Sequential(
            nn.Linear(metadata_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64)
        )
        
        # Final fusion and classification
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
                dialogue_attention_mask, dialogue_audio, dialogue_roles, metadata, sequence_length):
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
            
            # Audio features for this utterance
            if self.use_wav2vec:
                # For Wav2Vec2, dialogue_audio contains raw waveforms
                # We need to process them through Wav2Vec2 model
                if dialogue_audio.dim() == 3:  # [batch, seq, audio_length]
                    utt_audio_raw = dialogue_audio[:, i, :]  # [batch, audio_length]
                    
                    # Process through Wav2Vec2 (extract features only, no classification head)
                    with torch.no_grad():
                        wav2vec_outputs = self.wav2vec_model(utt_audio_raw)
                        # Use the last hidden state and take mean pooling
                        utt_audio_features_raw = wav2vec_outputs.last_hidden_state.mean(dim=1)  # [batch, 1024]
                    
                    # Process through our audio processor
                    utt_audio_features = self.audio_processor(utt_audio_features_raw)
                else:
                    # Fallback if audio format is unexpected
                    utt_audio_features = self.audio_processor(dialogue_audio[:, i, :])
            else:
                # Traditional librosa features
                utt_audio_features = self.audio_processor(dialogue_audio[:, i, :])
            
            # Role embedding
            utt_role_features = self.role_embedding(dialogue_roles[:, i])
            
            # Combine features for this utterance
            utt_combined = torch.cat([utt_text_features, utt_audio_features, utt_role_features], dim=1)
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
    def __init__(self, alpha=1.0, gamma=1.0, num_classes=32):
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
            dialogue_audio = batch['dialogue_audio'].to(device)
            dialogue_roles = batch['dialogue_roles'].to(device)
            metadata = batch['metadata'].to(device)
            sequence_length = batch['sequence_length'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                         dialogue_attention_mask, dialogue_audio, dialogue_roles, metadata, sequence_length)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            batch_count += 1
    
    avg_loss = total_loss / batch_count
    expected_loss = np.log(32)  # -log(1/32) for 32 emotion classes
    
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

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history(history, save_dir='results'):
    """Plot and save training history graphs"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Loss Graph
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 2. Accuracy Graph
    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], [acc*100 for acc in history['train_accuracy']], 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(history['epoch'], [acc*100 for acc in history['val_accuracy']], 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 3. F1 Score Graph
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], [f1*100 for f1 in history['train_f1']], 'b-', label='Training F1', linewidth=2)
    plt.plot(history['epoch'], [f1*100 for f1 in history['val_f1']], 'r-', label='Validation F1', linewidth=2)
    plt.title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score (%)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 4. Learning Rate (if available)
    plt.subplot(2, 2, 4)
    if 'learning_rate' in history and history['learning_rate']:
        plt.plot(history['epoch'], history['learning_rate'], 'g-', linewidth=2)
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Learning Rate', fontsize=12)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
    else:
        # Show epoch time instead
        plt.plot(history['epoch'], history['epoch_time'], 'g-', linewidth=2)
        plt.title('Training Time per Epoch', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Time (seconds)', fontsize=12)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_path = os.path.join(save_dir, f'training_history_{timestamp}.jpg')
    plt.savefig(combined_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"📊 Training history saved: {combined_path}")
    
    # Save individual plots
    
    # Individual Loss Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], history['train_loss'], 'b-', label='Training Loss', linewidth=3, marker='o', markersize=6)
    plt.plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss', linewidth=3, marker='s', markersize=6)
    plt.title('Training and Validation Loss Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    loss_path = os.path.join(save_dir, f'loss_graph_{timestamp}.jpg')
    plt.savefig(loss_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"📈 Loss graph saved: {loss_path}")
    
    # Individual Accuracy Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['epoch'], [acc*100 for acc in history['train_accuracy']], 'b-', 
             label='Training Accuracy', linewidth=3, marker='o', markersize=6)
    plt.plot(history['epoch'], [acc*100 for acc in history['val_accuracy']], 'r-', 
             label='Validation Accuracy', linewidth=3, marker='s', markersize=6)
    plt.title('Training and Validation Accuracy Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.tight_layout()
    
    accuracy_path = os.path.join(save_dir, f'accuracy_graph_{timestamp}.jpg')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print(f"🎯 Accuracy graph saved: {accuracy_path}")
    
    return combined_path, loss_path, accuracy_path

def save_training_summary(history, config, model_info, save_dir='results'):
    """Save a comprehensive training summary"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    summary_path = os.path.join(save_dir, f'training_summary_{timestamp}.txt')
    
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("MULTIMODAL EMPATHETIC EMOTION DETECTION - TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        for key, value in config.items():
            f.write(f"{key:25}: {value}\n")
        f.write(f"{'Model Parameters':25}: {model_info.get('parameters', 'N/A'):,}\n\n")
        
        f.write("TRAINING RESULTS:\n")
        f.write("-" * 20 + "\n")
        if history['val_accuracy']:
            best_val_acc = max(history['val_accuracy'])
            best_epoch = history['val_accuracy'].index(best_val_acc) + 1
            f.write(f"{'Best Validation Accuracy':25}: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)\n")
            f.write(f"{'Best Epoch':25}: {best_epoch}\n")
            f.write(f"{'Final Training Accuracy':25}: {history['train_accuracy'][-1]:.4f} ({history['train_accuracy'][-1]*100:.2f}%)\n")
            f.write(f"{'Final Validation Accuracy':25}: {history['val_accuracy'][-1]:.4f} ({history['val_accuracy'][-1]*100:.2f}%)\n")
            f.write(f"{'Final Training Loss':25}: {history['train_loss'][-1]:.4f}\n")
            f.write(f"{'Final Validation Loss':25}: {history['val_loss'][-1]:.4f}\n")
            f.write(f"{'Total Epochs Trained':25}: {len(history['epoch'])}\n")
            f.write(f"{'Total Training Time':25}: {sum(history['epoch_time']):.2f} seconds\n")
            f.write(f"{'Average Time per Epoch':25}: {sum(history['epoch_time'])/len(history['epoch_time']):.2f} seconds\n")
        
        f.write(f"\nEPOCH-BY-EPOCH DETAILS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<10} {'Train Acc':<11} {'Val Acc':<9} {'Time(s)':<8}\n")
        f.write("-" * 60 + "\n")
        
        for i in range(len(history['epoch'])):
            f.write(f"{history['epoch'][i]:<6} ")
            f.write(f"{history['train_loss'][i]:<12.4f} ")
            f.write(f"{history['val_loss'][i]:<10.4f} ")
            f.write(f"{history['train_accuracy'][i]:<11.4f} ")
            f.write(f"{history['val_accuracy'][i]:<9.4f} ")
            f.write(f"{history['epoch_time'][i]:<8.2f}\n")
    
    print(f"📋 Training summary saved: {summary_path}")
    return summary_path

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, scaler=None, gradient_accumulation_steps=1, max_grad_norm=1.0, epoch_num=1):
    """Train for one epoch with gradient accumulation and clipping"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num}", 
                       bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}',
                       dynamic_ncols=True)
    optimizer.zero_grad()  # Initialize gradients
    
    for batch_idx, batch in enumerate(progress_bar):
        
        # Move batch to device
        context_input_ids = batch['context_input_ids'].to(device)
        context_attention_mask = batch['context_attention_mask'].to(device)
        dialogue_input_ids = batch['dialogue_input_ids'].to(device)
        dialogue_attention_mask = batch['dialogue_attention_mask'].to(device)
        dialogue_audio = batch['dialogue_audio'].to(device)
        dialogue_roles = batch['dialogue_roles'].to(device)
        metadata = batch['metadata'].to(device)
        sequence_length = batch['sequence_length'].to(device)
        labels = batch['label'].to(device)
        
        if scaler:
            with torch.cuda.amp.autocast():
                logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                             dialogue_attention_mask, dialogue_audio, dialogue_roles, metadata, sequence_length)
                loss = criterion(logits, labels)  # Don't scale loss here
            
            # Scale loss only for backward pass
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            
            # Update parameters every gradient_accumulation_steps
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
                         dialogue_attention_mask, dialogue_audio, dialogue_roles, metadata, sequence_length)
            loss = criterion(logits, labels)  # Don't scale loss here
            
            # Scale loss only for backward pass
            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            
            # Update parameters every gradient_accumulation_steps
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
        
        total_loss += loss.item()  # Use original loss for logging
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
            dialogue_audio = batch['dialogue_audio'].to(device)
            dialogue_roles = batch['dialogue_roles'].to(device)
            metadata = batch['metadata'].to(device)
            sequence_length = batch['sequence_length'].to(device)
            labels = batch['label'].to(device)
            
            # Use mixed precision for inference speedup
            with torch.cuda.amp.autocast():
                logits = model(context_input_ids, context_attention_mask, dialogue_input_ids,
                             dialogue_attention_mask, dialogue_audio, dialogue_roles, metadata, sequence_length)
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy, precision, recall, f1 = compute_metrics(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1, all_predictions, all_labels

def main():
    # Optimized configuration for base models with mixed precision
    config = {
        'batch_size': 6,  # Increased for base models - more memory efficient
        'gradient_accumulation_steps': 2,  # Effective batch size of 12
        'learning_rate': 1e-5,  # Slightly higher for base models
        'num_epochs': 25,  # Efficient training with base models
        'patience': 5,  # Base models converge faster
        'dropout_rate': 0.2,  # Reduced dropout for base models
        'weight_decay': 0.01,  # Standard weight decay
        'warmup_steps': 500,  # Shorter warmup for base models
        'max_length': 384,  # Longer sequences possible with base models
        'max_dialogue_length': 10,  # More dialogue context
        'hidden_size': 256,  # Increased for base model capacity
        'num_layers': 2,  # More layers for base models
        'max_grad_norm': 1.0,  # Standard gradient clipping
        'use_focal_loss': True,  # Handle class imbalance
        'focal_alpha': 1.0,  # Standard focal loss
        'focal_gamma': 2.0,   # Standard focal loss gamma
        'use_wav2vec': True,  # Enable Wav2Vec2-base for better audio understanding
        'use_mixed_precision': True  # Enable mixed precision training
    }
    
    # Device configuration - Force GPU usage
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize tokenizer (BERT-base)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Initialize Wav2Vec2 feature extractor if needed
    wav2vec_processor = None
    if config.get('use_wav2vec', False):
        print("Loading Wav2Vec2 feature extractor...")
        wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalSequentialDataset(
        'json/mapped_train_data.json', 
        'data/train_audio/audio_v5_0',
        tokenizer,
        wav2vec_feature_extractor=wav2vec_processor,
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length']
    )
    
    val_dataset = MultimodalSequentialDataset(
        'json/mapped_val_data.json',
        'data/train_audio/audio_v5_0', 
        tokenizer,
        wav2vec_feature_extractor=wav2vec_processor,
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length']
    )
    
    test_dataset = MultimodalSequentialDataset(
        'json/mapped_test_data.json',
        'data/train_audio/audio_v5_0',
        tokenizer,
        wav2vec_feature_extractor=wav2vec_processor,
        max_length=config['max_length'],
        max_dialogue_length=config['max_dialogue_length']
    )
    
    # Create data loaders (reduced workers for Windows compatibility)
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=0,  # Reduced for Windows stability
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,  # Reduced for Windows stability
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=0,  # Reduced for Windows stability
        pin_memory=True
    )
    print("Data loaders created successfully!")
    
    # Compute class weights for balanced training (using full dataset)
    print("Computing class weights from full training dataset...")
    all_labels = []
    for i in tqdm(range(len(train_dataset)), desc="Getting labels for class weights"):
        all_labels.append(train_dataset.get_label(i))
    
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights computed from {len(train_dataset)} samples: {class_weights}")
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalLSTMModel(
        train_dataset,
        num_classes=32, 
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout_rate=config['dropout_rate'],
        use_wav2vec=config.get('use_wav2vec', False)  # Default to False for backward compatibility
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Model initialized successfully!")
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    print(f"Mixed precision training: {'Enabled' if scaler else 'Disabled (CPU mode)'}")
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    total_steps = len(train_loader) * config['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Loss function - use Focal Loss if specified, otherwise weighted CrossEntropy
    if config.get('use_focal_loss', False):
        criterion = FocalLoss(
            alpha=config.get('focal_alpha', 1.0),
            gamma=config.get('focal_gamma', 1.0),
            num_classes=32
        )
        print(f"Using Focal Loss (alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropy Loss")
    
    # Validate initial loss before training
    print("\nValidating initial model state...")
    initial_loss = validate_initial_loss(model, train_loader, criterion, device)
    
    # Mixed precision scaler for GPU
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'])
    
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
    
    class_names = ["happy", "surprised", "angry", "fear", "sad", "disgusted", "contempt"]
    
    print("Starting training...")
    best_val_accuracy = 0
    
    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)
        
        # Training
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, scaler,
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
            max_grad_norm=config.get('max_grad_norm', 1.0),
            epoch_num=epoch+1
        )
        
        # Validation
        val_loss, val_acc, val_prec, val_rec, val_f1, val_predictions, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        epoch_time = time.time() - epoch_start_time
        
        # Log metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save history
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
        
        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_acc,
                'epoch': epoch + 1,
                'config': config
            }, 'checkpoints/best_model.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.4f}")
        
        # Save epoch history
        with open('results/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Save graphs every 5 epochs and at the last epoch
        if (epoch + 1) % 5 == 0 or epoch == config['num_epochs'] - 1:
            try:
                combined_path, loss_path, accuracy_path = plot_training_history(history)
                print(f"📊 Training graphs saved at epoch {epoch + 1}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save training graphs - {e}")
        
        # Early stopping
        if early_stopping(val_loss, model):
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
    
    print("\nTraining completed!")
    
    # Save final training graphs and summary
    try:
        # Prepare model config for summary
        model_config = {
            'model_type': 'Multimodal LSTM',
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers'],
            'dropout_rate': config['dropout_rate'],
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'num_epochs': config['num_epochs'],
            'use_focal_loss': config.get('use_focal_loss', False),
            'focal_alpha': config.get('focal_alpha', 1.0),
            'focal_gamma': config.get('focal_gamma', 1.0),
            'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
            'device': str(device)
        }
        
        model_info = {
            'parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        combined_path, loss_path, accuracy_path = plot_training_history(history)
        summary_path = save_training_summary(history, model_config, model_info)
        print(f"📈 Final training visualizations and summary saved!")
        print(f"📊 Best validation accuracy achieved: {best_val_accuracy:.4f}")
    except Exception as e:
        print(f"⚠️ Warning: Could not save final graphs/summary - {e}")
    
    # Load best model for testing
    print("Loading best model for testing...")
    checkpoint = torch.load('checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    print("Evaluating on test set...")
    test_loss, test_acc, test_prec, test_rec, test_f1, test_predictions, test_labels = validate_epoch(
        model, test_loader, criterion, device
    )
    
    # Save test results
    test_results = {
        'test_accuracy': test_acc,
        'test_precision': test_prec,
        'test_recall': test_rec,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'total_test_samples': len(test_dataset),
        'evaluation_date': datetime.now().isoformat(),
        'best_epoch': checkpoint['epoch'],
        'best_val_accuracy': checkpoint['val_accuracy']
    }
    
    with open('results/test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    
    # Generate and save confusion matrix
    plot_confusion_matrix(
        test_labels, 
        test_predictions, 
        class_names, 
        'results/confusion_matrix.png'
    )
    
    print("Confusion matrix saved to results/confusion_matrix.png")
    print("Training history saved to results/training_history.json")
    print("Test results saved to results/test_results.json")
    print("Best model saved to checkpoints/best_model.pth")

if __name__ == "__main__":
    main()
 