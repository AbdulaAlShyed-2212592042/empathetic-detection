# 🤖 Empathetic Detection - Multimodal Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Advanced multimodal emotion classification system using deep learning for empathy detection**

---

## 🎯 Project Overview

This project implements a comprehensive multimodal neural network system that combines **🎧 audio, 🎥 video, 📝 text, and 📊 metadata** to recognize empathetic emotions from the **AvaMERG dataset**. We employ a sophisticated **late fusion strategy** with specialized models for optimal performance.

### 🏆 Key Achievements

| Model | Modalities | Accuracy | Status |
|:-----:|:----------:|:--------:|:------:|
| **🎧 Audio Model** | 🎧📝📊 | **85.18%** | ✅ Complete |
| **🎥 Video Model** | 🎥📝📊 | **72.81%** | ✅ Complete |
| **🔄 Late Fusion** | 🎧🎥📝📊 | **Implemented** | ✅ RTX 3060 Optimized |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/AbdulaAlShyed-2212592042/empathetic-detection.git
cd empathetic-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Structure

```
empathetic-detection/
├── data/
│   ├── train_audio/audio_v5_0/     # Audio files (.wav)
│   └── train_video/video_v5_0/     # Video files (.mp4)
├── json/
│   ├── mapped_train_data_video_aligned.json
│   ├── mapped_val_data_video_aligned.json
│   └── mapped_test_data_video_aligned.json
├── checkpoints/                     # Audio model checkpoints
├── checkpoints_2/                   # Video model checkpoints
└── late_fusion_checkpoint/          # Late fusion checkpoints
```

### 3. Testing & Training

```bash
# Test audio model (85.18% accuracy)
cd "audio train and test"
python test_audio_text_metadata.py

# Test video model (72.81% accuracy)
python test_2.py

# Train late fusion model (RTX 3060 optimized)
python late_fusion.py

# Memory-optimized testing
python late_fusion_test.py
```

---

## 🧠 Model Architecture

### Audio Model (🎧📝📊)
- **Wav2Vec2-base**: Speech representation learning
- **BERT-base**: Text understanding
- **Metadata**: Speaker profiles & empathy chains
- **Performance**: 85.18% accuracy

### Video Model (🎥📝📊)
- **Enhanced TimeSformer**: Spatial-temporal video processing
- **BERT-base**: Text understanding
- **Metadata**: Speaker profiles & empathy chains
- **Performance**: 72.81% accuracy

### Late Fusion (🔄)
- **Architecture**: Learnable weighted logit fusion
- **Optimization**: RTX 3060 12GB optimized with mixed precision
- **Memory**: Only 0.93GB VRAM usage
- **Training**: Single parameter (fusion weight) optimization

---

## ⚡ RTX 3060 12GB Optimizations

✅ **Mixed Precision Training**: FP16 reduces memory by ~50%  
✅ **Memory Usage**: Only **0.93GB VRAM** for late fusion training  
✅ **Batch Optimization**: 2×2 gradient accumulation (effective batch 4)  
✅ **Model Freezing**: Only fusion weight trainable (1 parameter vs 114M)  
✅ **Speed**: 1.5-2x faster training with mixed precision  

---

## 📊 Performance Results

### Audio Model Performance
```
Overall Test Accuracy: 85.18%
Overall Precision:     85.12%
Overall F1-Score:      85.15%
```

### Video Model Performance  
```
Overall Test Accuracy: 72.81%
Overall Precision:     72.45%
Overall F1-Score:      72.63%
```

### Emotion Classes
**Happy**, **Surprised**, **Angry**, **Fear**, **Sad**, **Disgusted**, **Contempt**

---

## 📁 Key Files

| File | Purpose | Modalities |
|:----:|:-------:|:----------:|
| `late_fusion.py` | Complete late fusion system | 🎧🎥📝📊 |
| `late_fusion_test.py` | Memory-optimized testing | 🎧🎥📝📊 |
| `video_training.py` | Video model implementation | 🎥📝📊 |
| `train_video_improved.py` | Enhanced video training | 🎥📝📊 |
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

1. **🔄 Late Fusion Implementation**: Memory-efficient learnable weighted fusion
2. **🚀 Mixed Precision Training**: RTX 3060 optimization with FP16
3. **🎧 Enhanced Audio Processing**: Wav2Vec2 + BERT integration  
4. **🎥 Improved Video Processing**: Enhanced TimeSformer architecture
5. **⚡ Memory Optimization**: Single parameter training approach

---

## 📚 References

- **AvaMERG Dataset**: [Hugging Face](https://huggingface.co/datasets/ZhangHanXD/AvaMERG)
- **TimeSformer**: "Is Space-Time Attention All You Need for Video Understanding?"
- **Wav2Vec2**: "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"
- **BERT**: "Bidirectional Encoder Representations from Transformers"

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**🤖 Built for advancing empathetic AI and emotion understanding**

⭐ Star this repo if it helped you! | 🐛 Report issues | 💡 Suggest improvements

</div>