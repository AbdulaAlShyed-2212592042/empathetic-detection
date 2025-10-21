# 🎭 Empathetic Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-RTX3060-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Multimodal emotion recognition system combining audio, video, and text for 7-class emotion detection**

---

## 🎯 Overview

This project implements three distinct emotion recognition models that process audio, video, and text modalities for empathetic emotion detection across 7 emotion classes: **neutral, joy, sadness, anger, fear, disgust, surprise**.

### 📊 Performance Results

| Model | Test Accuracy | Test Samples | Architecture |
|:-----:|:-------------:|:------------:|:-------------|
| **Audio-Text** | **83.15%** | 4,939 | BERT + Wav2Vec2 + Metadata |
| **Video** | **72.24%** | 4,939 | Video Transformer + Text |
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
├── checkpoints_2/
│   └── best_video_model_*.pth          # Video model checkpoint (72.24%)
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
├── video_training.py                   # Video model training
├── train_video_improved.py            # Enhanced video training
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
# Training
python train_video_improved.py

# Testing
python video_training.py --test_mode
```

### Late Fusion Model  
```bash
# Training
python late_fusion.py

# Testing
python late_fusion_test.py
```

---

## 🔬 Technical Implementation

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
| `train_video_improved.py` | Enhanced video-only training | 🎥 |
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

## 🏆 Key Innovations

1. **🔄 Late Fusion Implementation**: Memory-efficient learnable weighted fusion combining audio and video models
2. **🚀 Mixed Precision Training**: RTX 3060 optimization with FP16 for late fusion model
3. **🎧 Enhanced Audio Processing**: Wav2Vec2 + BERT + metadata integration  
4. **🎥 Pure Video Processing**: Enhanced TimeSformer for video-only emotion recognition
5. **⚡ Memory Optimization**: Single parameter training approach (1 vs 232M parameters)

---

## � Results Analysis

### Model Comparison
| Model | Accuracy | Strengths | Limitations |
|-------|----------|-----------|-------------|
| Audio-Text | 83.15% | Rich prosodic + linguistic features | Audio quality dependent |
| Video | 72.24% | Visual emotion cues | Lighting/angle sensitive |
| Late Fusion | 60.42% | Combined modalities | Underperformed individual models |

### Key Findings
- **Audio-text model** achieved highest accuracy due to rich prosodic and linguistic features
- **Video model** showed moderate performance with visual emotion recognition  
- **Late fusion** resulted in lower accuracy than individual models, suggesting overlapping rather than complementary feature learning
- **Fusion weights** heavily favored video predictions (89.5%) over audio-text (10.5%)

---

## 📂 Results Storage

- **Audio Results**: `result_1/` - Contains audio model test results and metrics
- **Video Results**: `test_results/` - Contains video model evaluation outputs  
- **Late Fusion Results**: `late_fusion_results/` - Contains fusion training and testing results

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

*Built with PyTorch, Transformers, and modern deep learning techniques for multimodal emotion recognition.*

⭐ Star this repo if it helped you! | 🐛 Report issues | 💡 Suggest improvements

</div>