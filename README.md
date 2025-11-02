# 🎭 Multimodal Emotion Detection with Deep Fusion

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-RTX3060-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Advanced deep fusion architecture for multimodal emotion recognition combining audio, video, and text modalities for 7-class emotion detection**

---

## 🎯 Overview

This project implements an advanced multimodal emotion recognition system with deep fusion architecture that processes audio, video, and text modalities for emotion detection across 7 emotion classes: **neutral, joy, sadness, anger, fear, disgust, surprise**.

The system features three distinct models with a novel late fusion approach that combines pretrained MIMAMO Net (video) and Multimodal LSTM (audio-text) models through learnable weighted fusion with only 9 trainable parameters.

### 📊 Performance Results

| Model | Test Accuracy | Test Samples | Architecture |
|:-----:|:-------------:|:------------:|:-------------|
| **Audio-Text** | **83.15%** | 4,939 | BERT + Wav2Vec2 + Metadata |
| **Video** | **72.24%** | 4,939 | Video Transformer + Text |
| **Deep Fusion** | **58.04%** | 4,939 | Enhanced Late Fusion (9 parameters) |
| **Late Fusion** | **60.42%** | 4,939 | Learnable Weighted Fusion |

---

## 🚀 Installation

```bash
# Clone repository
git clone https://github.com/AbdulaAlShyed-2212592042/empathetic-detection.git
cd empathetic-detection

# Install dependencies
pip install -r requirements.txt
```

### System Requirements
- **GPU**: NVIDIA RTX 3060 12GB (or equivalent)
- **Python**: 3.9+
- **CUDA**: Compatible version for PyTorch 2.4.0

---

## 📁 Project Structure

```
empathetic-detection/
├── audio train and test/
│   ├── train_audio_text_metadata.py    # Audio model training
│   └── test_audio_text_metadata.py     # Audio model testing
├── checkpoints/
│   └── best_7class_model.pth           # Audio model checkpoint (83.15%)
├── checkpoint_3/
│   └── best_mimamo_model_*.pth         # MIMAMO Net checkpoint (58.04%)
├── checkpoint_4_mimamo/
│   └── best_combined_fusion_model_*.pth # Deep fusion checkpoint
├── combined_late_fusion.py             # Deep fusion architecture
├── late_fusion_checkpoint/
│   └── best_late_fusion_model_*.pth    # Late fusion checkpoint (60.42%)
├── data/
│   ├── train_audio/audio_v5_0/         # Audio files (.wav)
│   └── train_video/video_v5_0/         # Video files (.mp4)
├── json/
│   ├── mapped_train_data_video_aligned.json
│   ├── mapped_val_data_video_aligned.json
│   └── mapped_test_data_video_aligned.json
├── late_fusion.py                      # Complete fusion system
├── late_fusion_test.py                 # Fusion testing script
├── app.py                              # Web application for real-time detection
├── templates/index.html                # Web interface
├── run_webapp.bat/.sh                  # Web app startup scripts
├── video_training.py                   # Video model training
└── requirements.txt                    # Core dependencies
```

---

## 🎯 Usage

### Audio-Text Model
```bash
# Training
python "audio train and test/train_audio_text_metadata.py"

# Testing  
python "audio train and test/test_audio_text_metadata.py"
```

### Video Model
```bash
# Training/Testing
python video_training.py
```

### Deep Fusion Model
```bash
# Training with hyperparameter optimization
python combined_late_fusion.py

# Testing
python late_fusion_test.py
```

### Late Fusion Model  
```bash
# Training
python late_fusion.py

# Testing
python late_fusion_test.py
```

### Web Application 🌐
```bash
# Start the emotion detection web app
python app.py

# Or use startup scripts:
# Windows: run_webapp.bat
# Linux/Mac: ./run_webapp.sh

# Access at: http://localhost:5000
```

---

## 🔬 Technical Implementation

### Deep Fusion Model (Enhanced Late Fusion)
- **Architecture**: Frozen MIMAMO Net (video) + Multimodal LSTM (audio-text)
- **Fusion Method**: Learnable weighted fusion with only 9 trainable parameters
- **Innovation**: 2 fusion weights + 7 bias terms for optimal model combination
- **Memory Efficiency**: Mixed precision training optimized for RTX 3060 12GB
- **Optimization**: Optuna hyperparameter search with early stopping

### Audio-Text Model (83.15% Accuracy)
- **Text Processing**: BERT-base-uncased tokenization and embedding
- **Audio Features**: Wav2Vec2-base feature extraction 
- **Metadata Integration**: Gender, age, and speaker context features
- **Architecture**: Multi-head attention fusion with dropout
- **Optimization**: AdamW optimizer with warmup scheduling

### Video Model (72.24% Accuracy)  
- **Video Processing**: Frame extraction and temporal encoding
- **Text Integration**: Dialogue context through BERT embedding
- **Architecture**: Enhanced TimeSformer for spatial-temporal processing
- **Features**: Visual expressions and temporal dynamics (8 frames/sequence)
- **Training**: Mixed precision with gradient accumulation

### Late Fusion Model (60.42% Accuracy)
- **Fusion Strategy**: Learnable weighted logit-level combination  
- **Architecture**: Frozen pretrained models + 1 trainable fusion weight
- **Memory Efficiency**: RTX 3060 12GB optimized with mixed precision
- **Training Results**: Video weight 89.5%, Audio-text weight 10.5%
- **Analysis**: Individual models outperformed fusion combination

---

## 📊 Detailed Performance Metrics

## 📊 Detailed Performance Metrics

### Audio-Text Model Results
```
Test Accuracy: 83.15%
Test Precision: 82.76%
Test F1-Score: 82.37%
Test Samples: 4,939
Architecture: BERT + Wav2Vec2 + Metadata
Checkpoint: checkpoints/best_7class_model.pth
```

### Video Model Results  
```
Test Accuracy: 72.24%
Test Precision: 72.31%
Test F1-Score: 69.84%
Test Samples: 4,939
Architecture: Video Transformer + Text encoding
Checkpoint: checkpoints_2/best_video_model_*.pth
```

### Deep Fusion Results
```
Test Accuracy: 58.04%
Test Precision: 57.89%
Test F1-Score: 56.12%
Test Samples: 4,939
Fusion Weights: Optimized through hyperparameter search
Training Parameters: Only 9 trainable (2 weights + 7 bias)
Checkpoint: checkpoint_3/best_mimamo_model_*_acc0.5804.pth
```

### Late Fusion Results
```
Test Accuracy: 60.42%
Test Precision: 60.39%
Test F1-Score: 55.98%
Test Samples: 4,939
Fusion Weights: Video 89.5%, Audio-Text 10.5%
Training Epochs: 14
Checkpoint: late_fusion_checkpoint/best_late_fusion_model_*.pth
```

### Emotion Classes
**7 Categories**: neutral, joy, sadness, anger, fear, disgust, surprise

---

## 💾 Core Dependencies

From `requirements.txt`:
```python
torch==2.4.0                    # Deep learning framework
transformers==4.44.2            # BERT, Wav2Vec2 models
librosa==0.10.2                 # Audio processing
scikit-learn==1.5.2             # Metrics and evaluation
numpy==1.26.4                   # Numerical computing
pandas==2.2.2                   # Data manipulation
```

---

## 🔧 RTX 3060 12GB Optimizations

✅ **Mixed Precision Training**: FP16 reduces memory usage by ~50%  
✅ **Batch Processing**: Optimized batch sizes for 12GB VRAM  
✅ **Gradient Accumulation**: Effective larger batch sizes without memory overflow  
✅ **Model Freezing**: Reduced trainable parameters for fusion training  
✅ **Memory Efficiency**: Optimized data loading and preprocessing

---

## 📁 Key Files

| File | Purpose | Modalities |
|:----:|:-------:|:----------:|
| `late_fusion.py` | Complete late fusion system | 🎧🎥📝📊 |
| `late_fusion_test.py` | Memory-optimized testing | 🎧🎥📝📊 |
| `video_training.py` | Video-only model implementation | 🎥 |
| `audio train and test/train_audio_text_metadata.py` | Audio model training | 🎧📝📊 |
| `audio train and test/test_audio_text_metadata.py` | Audio model testing | 🎧📝📊 |

---

## 🔧 Technical Specifications

### System Requirements
- **GPU**: NVIDIA RTX 3060 12GB (optimized) or better
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for dataset and models
- **CUDA**: 11.8+ for optimal performance

### Dependencies
```
torch>=2.0.0
transformers>=4.21.0
librosa>=0.9.2
opencv-python>=4.6.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.64.0
numpy>=1.21.0
```

---

## 📈 Dataset

**[AvaMERG Dataset](https://huggingface.co/datasets/ZhangHanXD/AvaMERG)** - A comprehensive multimodal emotion recognition dataset

| Modality | Format | Details |
|:--------:|:------:|---------|
| 🎧 **Audio** | `.wav` | 16kHz sampling, emotional prosody |
| 🎥 **Video** | `.mp4` | 224×224 resolution, 8 frames/sequence |
| 📝 **Text** | `JSON` | BERT tokenization, dialogue context |
| 📊 **Metadata** | `JSON` | Speaker profiles, empathy chains |

---

## 🏆 Key Technical Achievements

1. **Enhanced Late Fusion Architecture** - Learnable weighted fusion combining frozen MIMAMO Net (video) + Multimodal LSTM (audio-text) with only 9 trainable parameters
2. **Mixed Precision Training Optimization** - FP16 memory efficiency for RTX 3060 12GB with gradient scaling and automatic loss scaling
3. **Optuna Hyperparameter Search** - Automated optimization for fusion weights, learning rates, and batch sizes with pruning strategies
4. **Multimodal Feature Extraction** - BERT + Wav2Vec2 + TimeSformer integration for text, audio, and video processing
5. **Focal Loss Implementation** - Class imbalance handling with alpha-gamma weighting for 7-class emotion detection
6. **Real-time Web Interface** - Flask application with live video/audio processing and Whisper transcription
7. **Memory-Efficient Architecture** - Frozen pretrained models strategy reducing trainable parameters from 232M+ to 9
8. **Comprehensive Evaluation Framework** - Confusion matrices, per-class metrics, and cross-modal performance analysis

---

## 🏆 Key Innovations

1. **🔄 Enhanced Deep Fusion**: Memory-efficient learnable weighted fusion combining MIMAMO Net and Multimodal LSTM with only 9 trainable parameters
2. **🚀 Mixed Precision Training**: RTX 3060 optimization with FP16 for deep fusion model
3. **🔬 Hyperparameter Optimization**: Optuna-based automated search for optimal fusion weights
4. **🎧 Enhanced Audio Processing**: Wav2Vec2 + BERT + metadata integration  
5. **🎥 Pure Video Processing**: Enhanced TimeSformer for video-only emotion recognition
6. **⚡ Ultra-Efficient Training**: Single parameter fusion approach vs 232M+ parameter models

---

## 🌐 Web Application

A real-time emotion detection web interface that uses your trained late fusion model:

### Features
- 📹 **Video Upload**: Analyze emotion from uploaded video files
- 📷 **Live Camera**: Real-time webcam emotion detection  
- 🎤 **Audio Transcription**: Automatic speech-to-text using Whisper
- 📊 **Detailed Results**: Emotion predictions with confidence and fusion weights
- 🎯 **Multimodal Analysis**: Combines video, audio, and transcribed text

### Quick Start
```bash
# Start web application
python app.py

# Or use startup scripts:
run_webapp.bat    # Windows
./run_webapp.sh   # Linux/Mac

# Access at: http://localhost:5000
```

### Model Integration
- Uses your trained checkpoint: `checkpoint_4_mimamo/best_combined_fusion_model_*.pth`
- Real-time Whisper transcription for audio processing
- Video frame extraction and processing
- Live fusion weight visualization with deep fusion architecture

---

## 📈 Results Analysis

### Model Comparison
| Model | Accuracy | Strengths | Limitations |
|-------|----------|-----------|-------------|
| Audio-Text | 83.15% | Rich prosodic + linguistic features | Audio quality dependent |
| Video | 72.24% | Visual emotion cues | Lighting/angle sensitive |
| Deep Fusion | 58.04% | Combined modalities with minimal parameters | Requires optimal hyperparameter tuning |
| Late Fusion | 60.42% | Combined modalities | Underperformed individual models |

### Key Findings
- **Audio-text model** achieved highest accuracy due to rich prosodic and linguistic features
- **Video model** showed moderate performance with visual emotion recognition  
- **Deep fusion** demonstrates efficient parameter usage with competitive performance using only 9 trainable parameters
- **Late fusion** resulted in lower accuracy than individual models, suggesting overlapping rather than complementary feature learning
- **Fusion weights** optimization through Optuna improved model combination strategies

---

## 📂 Results Storage

- **Audio Results**: `result_1/` - Contains audio model test results and metrics
- **Video Results**: `test_results/` - Contains video model evaluation outputs  
- **Deep Fusion Results**: `checkpoint_4_mimamo/`, `using_mimo_late_result/`, `mimamo_net_result/` - Contains enhanced fusion training and testing results
- **Late Fusion Results**: `late_fusion_results/` - Contains original fusion training and testing results

Each folder includes:
- Performance metrics (JSON)
- Confusion matrices (PNG)  
- Training histories (JSON)
- Classification reports (TXT)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## � License

This project is licensed under the MIT License - see the LICENSE file for details.

---

*Built with PyTorch, Transformers, and advanced deep fusion techniques for multimodal emotion recognition.*

⭐ Star this repo if it helped you! | 🐛 Report issues | 💡 Suggest improvements

</div>