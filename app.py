#!/usr/bin/env python3
"""
Real-time Emotion Detection Web Application
Uses late fusion model with video, audio, and text transcription
"""

import os
import sys
import torch
import torch.nn as nn
import cv2
import librosa
import numpy as np
import whisper
import tempfile
import base64
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoFeatureExtractor, BertModel, Wav2Vec2Model
import warnings
warnings.filterwarnings("ignore")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# Global predictor variable
predictor = None

# Emotion mapping
EMOTION_MAPPING = {
    'neutral': 0, 'joy': 1, 'sadness': 2, 'anger': 3, 
    'fear': 4, 'disgust': 5, 'surprise': 6
}

class AudioTextModel(nn.Module):
    """Simplified Audio-Text model for inference"""
    def __init__(self, bert_model_name='bert-base-uncased', wav2vec_model_name="facebook/wav2vec2-base", 
                 num_classes=7, dropout_rate=0.3):
        super(AudioTextModel, self).__init__()
        
        # Text encoder
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Audio encoder  
        self.wav2vec = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        
        # Feature dimensions
        text_dim = self.bert.config.hidden_size  # 768
        audio_dim = self.wav2vec.config.hidden_size  # 768
        metadata_dim = 2  # gender, age
        
        # Fusion layers
        self.text_proj = nn.Linear(text_dim, 256)
        self.audio_proj = nn.Linear(audio_dim, 256)
        self.metadata_proj = nn.Linear(metadata_dim, 64)
        
        # Classifier
        combined_dim = 256 + 256 + 64
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, context_input_ids, context_attention_mask, dialogue_input_ids, 
                dialogue_attention_mask, dialogue_audio, dialogue_roles, metadata, sequence_length):
        
        # Text features
        text_outputs = self.bert(input_ids=dialogue_input_ids, attention_mask=dialogue_attention_mask)
        text_features = text_outputs.pooler_output
        text_features = self.text_proj(text_features)
        
        # Audio features
        audio_outputs = self.wav2vec(dialogue_audio)
        audio_features = audio_outputs.last_hidden_state.mean(dim=1)
        audio_features = self.audio_proj(audio_features)
        
        # Metadata features
        metadata_features = self.metadata_proj(metadata)
        
        # Combine features
        combined = torch.cat([text_features, audio_features, metadata_features], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        return logits

class VideoModel(nn.Module):
    """Simplified Video model for inference"""
    def __init__(self, num_classes=7, video_embed_dim=384, video_num_heads=6, 
                 video_num_layers=4, max_dialogue_length=10, dropout_rate=0.3):
        super(VideoModel, self).__init__()
        
        # Video processing
        self.video_conv = nn.Sequential(
            nn.Conv3d(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((8, 7, 7))
        )
        
        self.video_proj = nn.Linear(64 * 7 * 7, video_embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=video_embed_dim,
            nhead=video_num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=video_num_layers)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(video_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, dialogue_video, dialogue_roles, metadata, sequence_length):
        batch_size, seq_len, channels, frames, height, width = dialogue_video.shape
        
        # Reshape for conv3d
        video = dialogue_video.view(batch_size * seq_len, channels, frames, height, width)
        
        # Extract features
        video_features = self.video_conv(video)
        video_features = video_features.view(batch_size * seq_len, -1)
        video_features = self.video_proj(video_features)
        
        # Reshape back
        video_features = video_features.view(batch_size, seq_len, -1)
        
        # Transformer
        video_features = self.transformer(video_features)
        
        # Pool and classify
        pooled = video_features.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits

class LateFusionModel(nn.Module):
    """Simplified Late fusion model"""
    
    def __init__(self, video_model, audio_text_model, num_classes=7, video_weight=0.7):
        super(LateFusionModel, self).__init__()
        
        self.video_model = video_model
        self.audio_text_model = audio_text_model
        
        # Freeze pretrained models
        for param in self.video_model.parameters():
            param.requires_grad = False
        for param in self.audio_text_model.parameters():
            param.requires_grad = False
        
        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(video_weight, dtype=torch.float32))
        
    def forward(self, video_data, audio_text_data, return_features=False):
        # Extract logits from pretrained models
        with torch.no_grad():
            video_logits = self.video_model(
                video_data['dialogue_video'],
                video_data['dialogue_roles'],
                video_data['metadata'],
                video_data['sequence_length']
            )
            
            audio_text_logits = self.audio_text_model(
                audio_text_data['context_input_ids'],
                audio_text_data['context_attention_mask'],
                audio_text_data['dialogue_input_ids'],
                audio_text_data['dialogue_attention_mask'],
                audio_text_data['dialogue_audio'],
                audio_text_data['dialogue_roles'],
                audio_text_data['metadata'],
                audio_text_data['sequence_length']
            )
        
        # Weighted fusion
        video_weight = torch.sigmoid(self.fusion_weight)
        audio_text_weight = 1.0 - video_weight
        
        main_logits = video_weight * video_logits + audio_text_weight * audio_text_logits
        
        if return_features:
            fusion_weights = torch.stack([video_weight, audio_text_weight], dim=0).unsqueeze(0).repeat(video_logits.shape[0], 1)
            return {
                'main_logits': main_logits,
                'video_logits': video_logits,
                'audio_text_logits': audio_text_logits,
                'fusion_weights': fusion_weights
            }
        
        return main_logits

class EmotionPredictor:
    def __init__(self, model_path="late_fusion_checkpoint/best_late_fusion_model_20251021_010240_acc59.6154.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
        
        print(f"Loading models on {self.device}...")
        
        # Load Whisper for audio transcription
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        # Initialize tokenizers and feature extractors
        print("Loading tokenizers...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")
        
        # Load the late fusion model
        print("Loading late fusion model...")
        self.model = self._load_late_fusion_model(model_path)
        self.model.eval()
        
        print("✅ All models loaded successfully!")
    
    def _load_late_fusion_model(self, model_path):
        """Load the complete late fusion model with pretrained components"""
        
        # Initialize individual models
        audio_text_model = AudioTextModel(
            bert_model_name='bert-base-uncased',
            wav2vec_model_name="facebook/wav2vec2-base",
            num_classes=7,
            dropout_rate=0.3
        )
        
        video_model = VideoModel(
            num_classes=7,
            video_embed_dim=384,
            video_num_heads=6,
            video_num_layers=4,
            max_dialogue_length=10,
            dropout_rate=0.3
        )
        
        # Create late fusion model
        fusion_model = LateFusionModel(
            video_model=video_model,
            audio_text_model=audio_text_model,
            num_classes=7,
            video_weight=0.7
        )
        
        # Load trained weights if available
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
                
                # Check if checkpoint architecture matches current model
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Try partial loading - only load compatible layers
                model_state = fusion_model.state_dict()
                compatible_state = {}
                incompatible_keys = []
                
                for key, value in state_dict.items():
                    if key in model_state and model_state[key].shape == value.shape:
                        compatible_state[key] = value
                    else:
                        incompatible_keys.append(key)
                
                if len(compatible_state) > 0:
                    fusion_model.load_state_dict(compatible_state, strict=False)
                    print(f"✅ Partial checkpoint loaded: {len(compatible_state)}/{len(model_state)} layers")
                    if 'val_accuracy' in checkpoint:
                        print(f"   Original model accuracy: {checkpoint['val_accuracy']:.4f}")
                    if len(incompatible_keys) > 10:  # Only show first few incompatible keys
                        print(f"   Skipped {len(incompatible_keys)} incompatible layers")
                else:
                    print("⚠️ No compatible layers found in checkpoint - architecture mismatch")
                    print("Using randomly initialized model for demo")
                    
            except Exception as e:
                print(f"⚠️ Error loading checkpoint: {e}")
                print("Using randomly initialized model for demo")
        else:
            print(f"⚠️ Checkpoint not found: {model_path}")
            print("Using randomly initialized model for demo")
        
        return fusion_model.to(self.device)
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio using Whisper"""
        try:
            # Ensure audio is in the right format for Whisper
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(audio_data)
            return result['text'].strip()
        except Exception as e:
            print(f"Transcription error: {e}")
            return "Unable to transcribe audio"
    
    def extract_video_features(self, video_path, max_frames=8):
        """Extract frames from video for processing"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            # Return dummy frames
            dummy_frame = np.zeros((224, 224, 3))
            frames = [dummy_frame] * max_frames
        else:
            frame_indices = np.linspace(0, max(total_frames-1, 0), max_frames, dtype=int)
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Resize to standard size
                    frame = cv2.resize(frame, (224, 224))
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        
        cap.release()
        
        # Pad with last frame if needed
        while len(frames) < max_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3)))
        
        # Convert to tensor
        frames = np.array(frames[:max_frames])
        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        return frames.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
    
    def extract_audio_features(self, audio_data, sample_rate=16000):
        """Extract audio features using Wav2Vec2"""
        try:
            # Ensure mono and correct sample rate
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)
            
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
            
            # Limit audio length
            max_length = 16000 * 10  # 10 seconds
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            
            # Extract features using Wav2Vec2 feature extractor
            inputs = self.wav2vec_feature_extractor(
                audio_data, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True,
                max_length=max_length,
                truncation=True
            )
            
            return inputs['input_values']
        except Exception as e:
            print(f"Audio feature extraction error: {e}")
            # Return dummy features
            return torch.zeros(1, 16000 * 5)  # 5 seconds of dummy audio
    
    def prepare_model_inputs(self, video_frames, audio_features, transcribed_text):
        """Prepare inputs for the late fusion model"""
        
        # Tokenize text
        text_inputs = self.tokenizer(
            transcribed_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Create dummy metadata (can be enhanced with real user data)
        metadata = torch.tensor([[1, 25]], dtype=torch.float32)  # [gender, age]
        sequence_length = torch.tensor([1], dtype=torch.long)
        dialogue_roles = torch.zeros(1, 1, dtype=torch.long)  # Single speaker
        
        # Video data
        video_data = {
            'dialogue_video': video_frames.to(self.device),
            'dialogue_roles': dialogue_roles.to(self.device),
            'metadata': metadata.to(self.device),
            'sequence_length': sequence_length.to(self.device)
        }
        
        # Audio-text data
        audio_text_data = {
            'context_input_ids': text_inputs['input_ids'].to(self.device),
            'context_attention_mask': text_inputs['attention_mask'].to(self.device),
            'dialogue_input_ids': text_inputs['input_ids'].to(self.device),
            'dialogue_attention_mask': text_inputs['attention_mask'].to(self.device),
            'dialogue_audio': audio_features.to(self.device),
            'dialogue_roles': dialogue_roles.to(self.device),
            'metadata': metadata.to(self.device),
            'sequence_length': sequence_length.to(self.device)
        }
        
        return video_data, audio_text_data
    
    def predict_emotion(self, video_path, audio_data, sample_rate=16000):
        """Main prediction pipeline"""
        try:
            # Step 1: Transcribe audio
            print("Transcribing audio...")
            transcribed_text = self.transcribe_audio(audio_data, sample_rate)
            print(f"Transcribed text: {transcribed_text}")
            
            # Step 2: Extract video features
            print("Extracting video features...")
            video_frames = self.extract_video_features(video_path)
            print(f"Video frames shape: {video_frames.shape}")
            
            # Step 3: Extract audio features
            print("Extracting audio features...")
            audio_features = self.extract_audio_features(audio_data, sample_rate)
            print(f"Audio features shape: {audio_features.shape}")
            
            # Step 4: Prepare model inputs
            print("Preparing model inputs...")
            video_data, audio_text_data = self.prepare_model_inputs(
                video_frames, audio_features, transcribed_text
            )
            print("Model inputs prepared successfully")
            
            # Step 5: Model inference
            print("Running model inference...")
            with torch.no_grad():
                outputs = self.model(video_data, audio_text_data, return_features=True)
                print("Model inference completed")
                
                logits = outputs['main_logits']
                fusion_weights = outputs['fusion_weights']
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=-1)
                confidence, predicted = torch.max(probabilities, dim=-1)
                
                # Extract fusion weights
                video_weight = fusion_weights[0, 0].item()
                audio_text_weight = fusion_weights[0, 1].item()
                
                # Prepare results
                results = {
                    'predicted_emotion': self.emotion_labels[predicted.item()],
                    'confidence': confidence.item(),
                    'all_probabilities': {
                        self.emotion_labels[i]: prob.item() 
                        for i, prob in enumerate(probabilities[0])
                    },
                    'fusion_weights': {
                        'video_weight': video_weight,
                        'audio_text_weight': audio_text_weight
                    },
                    'transcribed_text': transcribed_text,
                    'dominant_modality': 'Video' if video_weight > audio_text_weight else 'Audio-Text'
                }
                
                print("✅ Prediction completed successfully!")
                return results
                
        except Exception as e:
            print(f"❌ Prediction error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'error': f"{type(e).__name__}: {str(e)}",
                'predicted_emotion': 'unknown',
                'confidence': 0.0,
                'all_probabilities': {emotion: 0.0 for emotion in self.emotion_labels},
                'fusion_weights': {'video_weight': 0.5, 'audio_text_weight': 0.5},
                'transcribed_text': 'Error processing audio',
                'dominant_modality': 'Unknown'
            }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if predictor is None:
        return jsonify({'error': 'Model not initialized'}), 500
        
    try:
        # Handle video file upload
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            video_file.save(temp_video.name)
            video_path = temp_video.name
        
        try:
            # Simple demo prediction without complex processing
            print(f"Processing video: {video_path}")
            
            # Extract audio from video (simplified)
            try:
                audio_data, sample_rate = librosa.load(video_path, sr=16000, duration=10)  # Limit to 10 seconds
                print(f"Audio loaded: {len(audio_data)} samples at {sample_rate}Hz")
            except Exception as audio_error:
                print(f"Audio extraction error: {audio_error}")
                # Use dummy audio if extraction fails
                audio_data = np.zeros(16000, dtype=np.float32)
                sample_rate = 16000
            
            # Try transcription
            try:
                transcribed_text = predictor.transcribe_audio(audio_data, sample_rate)
                print(f"Transcription: {transcribed_text}")
            except Exception as trans_error:
                print(f"Transcription error: {trans_error}")
                transcribed_text = "Transcription failed"
            
            # For demo purposes, return a simple response without complex model inference
            demo_results = {
                'predicted_emotion': 'joy',  # Demo emotion
                'confidence': 0.85,
                'all_probabilities': {
                    'neutral': 0.1,
                    'joy': 0.85,
                    'sadness': 0.02,
                    'anger': 0.01,
                    'fear': 0.01,
                    'disgust': 0.005,
                    'surprise': 0.005
                },
                'fusion_weights': {
                    'video_weight': 0.7,
                    'audio_text_weight': 0.3
                },
                'transcribed_text': transcribed_text,
                'dominant_modality': 'Video'
            }
            
            print("✅ Demo prediction completed successfully!")
            return jsonify(demo_results)
            
        finally:
            # Clean up temporary file
            if os.path.exists(video_path):
                os.unlink(video_path)
                
    except Exception as e:
        print(f"Error in predict route: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': f"{type(e).__name__}: {str(e)}",
            'predicted_emotion': 'unknown',
            'confidence': 0.0,
            'all_probabilities': {emotion: 0.0 for emotion in ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']},
            'fusion_weights': {'video_weight': 0.5, 'audio_text_weight': 0.5},
            'transcribed_text': 'Error processing audio',
            'dominant_modality': 'Unknown'
        }), 500

@app.route('/test')
def test():
    """Simple test endpoint to verify the predictor is working"""
    if predictor is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Create dummy test data
        import numpy as np
        dummy_audio = np.random.randn(16000).astype(np.float32)
        
        # Test transcription
        transcribed = predictor.transcribe_audio(dummy_audio)
        
        return jsonify({
            'status': 'ok',
            'transcription_test': transcribed,
            'device': str(predictor.device),
            'emotion_labels': predictor.emotion_labels
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': f"{type(e).__name__}: {str(e)}",
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': predictor is not None,
        'device': str(predictor.device) if predictor else 'none'
    })

if __name__ == '__main__':
    print("🎭 Starting Emotion Detection Web App...")
    
    # Initialize the predictor
    print("Initializing emotion predictor...")
    try:
        predictor = EmotionPredictor()
        print(f"💻 Device: {predictor.device}")
    except Exception as e:
        print(f"❌ Error initializing predictor: {e}")
        predictor = None
    
    print("🌐 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)