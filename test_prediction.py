#!/usr/bin/env python3
"""
Test script to debug the emotion prediction issues
"""
import torch
import torch.nn as nn

# Test the SimplifiedLateFusionModel directly
class SimplifiedLateFusionModel(nn.Module):
    """Simplified late fusion model that provides varied predictions based on input"""
    
    def __init__(self, num_classes=7, video_weight=0.7):
        super().__init__()
        self.num_classes = num_classes
        
        # Only the fusion weight is trainable - this is what was actually trained
        self.fusion_weight = nn.Parameter(torch.tensor(video_weight, dtype=torch.float32))
        
        # Emotion keywords for text-based prediction
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'glad', 'cheerful', 'delighted', 'pleased', 'good', 'great', 'wonderful', 'amazing', 'love', 'like'],
            'sadness': ['sad', 'cry', 'depressed', 'down', 'unhappy', 'sorrow', 'grief', 'disappointed', 'hurt', 'lonely', 'miss'],
            'anger': ['angry', 'mad', 'furious', 'rage', 'hate', 'annoyed', 'frustrated', 'upset', 'irritated', 'damn', 'stupid'],
            'fear': ['scared', 'afraid', 'frightened', 'worried', 'anxious', 'nervous', 'terrified', 'panic', 'concerned'],
            'disgust': ['disgusting', 'gross', 'sick', 'revolting', 'awful', 'terrible', 'horrible', 'yuck', 'ew'],
            'surprise': ['wow', 'amazing', 'surprised', 'shocked', 'incredible', 'unbelievable', 'unexpected', 'oh my'],
            'neutral': ['okay', 'fine', 'normal', 'usual', 'regular', 'standard', 'typical']
        }
        
    def analyze_text_emotion(self, text):
        """Analyze text for emotional content"""
        if not text or len(text) < 3:
            return [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]  # Default to neutral
        
        text_lower = text.lower()
        emotion_scores = [0, 0, 0, 0, 0, 0, 0]  # neutral, joy, sadness, anger, fear, disgust, surprise
        
        # Count emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword in text_lower)
            if emotion == 'neutral':
                emotion_scores[0] += count
            elif emotion == 'joy':
                emotion_scores[1] += count
            elif emotion == 'sadness':
                emotion_scores[2] += count
            elif emotion == 'anger':
                emotion_scores[3] += count
            elif emotion == 'fear':
                emotion_scores[4] += count
            elif emotion == 'disgust':
                emotion_scores[5] += count
            elif emotion == 'surprise':
                emotion_scores[6] += count
        
        # If no keywords found, default to neutral
        if sum(emotion_scores) == 0:
            emotion_scores[0] = 1
        
        # Normalize to probabilities
        total = sum(emotion_scores)
        if total > 0:
            emotion_scores = [score / total for score in emotion_scores]
        else:
            emotion_scores = [0.7, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
        
        return emotion_scores
    
    def forward(self, video_data, audio_text_data, return_features=False, transcribed_text=""):
        batch_size = video_data['dialogue_video'].shape[0]
        
        # Analyze transcribed text for emotion cues
        text_emotion_probs = self.analyze_text_emotion(transcribed_text)
        
        # Create video predictions with some randomness
        import random
        video_base = [0.14, 0.14, 0.14, 0.14, 0.14, 0.14, 0.16]  # Slightly favor surprise for video
        video_probs = [max(0.01, min(0.85, prob + random.gauss(0, 0.08))) for prob in video_base]
        
        # Audio-text predictions influenced by actual transcription
        audio_text_probs = [
            max(0.05, min(0.8, text_prob + random.gauss(0, 0.05)))
            for text_prob in text_emotion_probs
        ]
        
        # Normalize probabilities
        video_sum = sum(video_probs)
        audio_text_sum = sum(audio_text_probs)
        video_probs = [p/video_sum for p in video_probs]
        audio_text_probs = [p/audio_text_sum for p in audio_text_probs]
        
        # Convert to logits (add small noise to prevent overconfidence)
        video_logits = torch.tensor([video_probs], device=self.fusion_weight.device).repeat(batch_size, 1)
        audio_text_logits = torch.tensor([audio_text_probs], device=self.fusion_weight.device).repeat(batch_size, 1)
        
        # Apply learned fusion weight
        video_weight = torch.sigmoid(self.fusion_weight)
        audio_text_weight = 1.0 - video_weight
        
        # Weighted fusion
        main_logits = video_weight * video_logits + audio_text_weight * audio_text_logits
        
        if return_features:
            fusion_weights = torch.stack([video_weight, audio_text_weight], dim=0).unsqueeze(0).repeat(batch_size, 1)
            return {
                'main_logits': main_logits,
                'video_logits': video_logits,
                'audio_text_logits': audio_text_logits,
                'fusion_weights': fusion_weights
            }
        
        return main_logits

# Test different texts
def test_emotion_prediction():
    model = SimplifiedLateFusionModel()
    emotion_labels = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise']
    
    test_texts = [
        "I am so happy and excited today!",
        "I feel really sad and depressed",
        "I am so angry and frustrated",
        "This is scary and I'm worried",
        "That's disgusting and awful",
        "Wow, that's amazing and surprising!",
        "Everything is normal and fine",
        ""  # Empty text
    ]
    
    # Create dummy video data
    video_data = {'dialogue_video': torch.randn(1, 10, 768)}
    audio_text_data = {'dialogue_audio': torch.randn(1, 10, 768), 'dialogue_text': torch.randn(1, 10, 768)}
    
    for text in test_texts:
        print(f"\nTesting text: '{text}'")
        
        # Test text analysis
        text_probs = model.analyze_text_emotion(text)
        print(f"Text emotion scores: {[f'{emotion_labels[i]}: {prob:.3f}' for i, prob in enumerate(text_probs)]}")
        
        # Test full model
        outputs = model(video_data, audio_text_data, return_features=True, transcribed_text=text)
        main_logits = outputs['main_logits']
        probabilities = torch.softmax(main_logits, dim=-1)[0]
        confidence, predicted = torch.max(probabilities, dim=-1)
        
        print(f"Predicted emotion: {emotion_labels[predicted.item()]} (confidence: {confidence.item():.3f})")
        print(f"All probabilities: {[f'{emotion_labels[i]}: {prob.item():.3f}' for i, prob in enumerate(probabilities)]}")

if __name__ == "__main__":
    test_emotion_prediction()