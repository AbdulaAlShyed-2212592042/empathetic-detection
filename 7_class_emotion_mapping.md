# 7-Class Emotion Mapping Configuration

## Changes Made

The emotion detection model has been successfully reduced from 32 EmpatheticDialogues classes to 7 basic emotion classes.

### 1. Updated Emotion Mapping (train.py)

**New emotion projection mapping:**
- Emotions are now mapped to 7 basic categories: `happy`, `surprised`, `angry`, `fear`, `sad`, `disgusted`, `contempt`
- Complex emotions are grouped into these fundamental categories

**Key mappings:**
- Positive emotions → `happy`: excited, proud, grateful, confident, hopeful, joyful, content, caring, trusting, faithful
- Negative emotions → `sad`: sadness, lonely, guilty, nostalgic, embarrassed, sentimental, ashamed, anxious
- Fear-related → `fear`: afraid, terrified, apprehensive
- Anger-related → `angry`: angry, furious, annoyed
- Social emotions → `contempt`: jealous
- Cognitive → `surprised`: surprised, impressed, devastated
- Disgust → `disgusted`: disgusted

### 2. Model Architecture Updates

**Files modified:**
- `train.py`: Updated model class and configuration
- `test.py`: Updated evaluation script

**Key changes:**
- Model output layer: 32 → 7 classes
- FocalLoss: Updated for 7 classes
- Expected loss calculation: log(32) → log(7) ≈ 1.95
- Default emotion fallback: 'afraid' → 'happy' (index 0)

### 3. Visualization Updates

**Confusion matrix plotting:**
- Reduced figure size from (16,14) to (10,8)
- Updated emotion labels list to 7 classes

## Benefits of 7-Class System

1. **Simplified model**: Fewer parameters in output layer
2. **Better generalization**: More robust categories
3. **Faster training**: Reduced complexity
4. **Better interpretability**: Basic emotions are more intuitive
5. **Improved class balance**: Grouping reduces extreme class imbalances

## Usage

The model can now be trained and tested with the 7-class system:

```bash
# Train with 7 emotion classes
python train.py

# Test with 7 emotion classes  
python test.py
```

## Expected Performance Impact

- **Training speed**: Faster convergence due to simplified task
- **Memory usage**: Slightly reduced due to smaller output layer
- **Accuracy**: May improve due to better class balance and clearer boundaries
- **Generalization**: Better performance on unseen data due to fundamental emotion categories

## Emotion Class Distribution

The 7 classes follow Ekman's basic emotions framework:
1. **Happy** (0): Joy, excitement, gratitude, confidence
2. **Surprised** (1): Surprise, being impressed
3. **Angry** (2): Anger, fury, annoyance
4. **Fear** (3): Fear, terror, apprehension
5. **Sad** (4): Sadness, loneliness, anxiety, shame
6. **Disgusted** (5): Disgust
7. **Contempt** (6): Jealousy, social rejection

This mapping provides a good balance between emotion recognition capability and model simplicity.
