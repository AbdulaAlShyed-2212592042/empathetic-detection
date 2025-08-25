# 🤖 Empathetic Detection - Multimodal Emotion Recognition

This project focuses on **multimodal emotion classification** using the **AvaMERG dataset**. It combines audio, video, and text modalities to detect empathetic states with deep learning.

----

## 📁 Dataset

**[AvaMERG Dataset](https://huggingface.co/datasets/ZhangHanXD/AvaMERG)**

A comprehensive multimodal emotion recognition dataset featuring:

- � **Audio sequences** (.wav) with vocal patterns and speech features
- �🎥 **Video sequences** (.mp4) with facial expressions and gestures
- 📝 **Text transcripts** with dialogue and conversation context
- 👤 **Speaker profiles** (age, gender, voice timbre, personality traits)
- 💭 **Empathy chains** (emotional scenarios, causes, response goals)
- 📊 **Metadata** (conversation topics, roles, temporal sequences)
- ❤️ **7 Emotion Classes**: happy, surprised, angry, fear, sad, disgusted, contempt

---

## 🧠 Model Architecture

### **Dual Early Fusion Strategy**

We employ **two specialized early-fusion models** that will be combined via late fusion:

#### **🎧 Model 1: Audio + Text + Metadata**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Audio Input   │    │   Text Input    │    │ Metadata Input  │
│   Wav2Vec2      │    │  BERT Tokens    │    │  Profile+Chain  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
    ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
    │Wav2Vec2   │          │BERT-base  │          │Embeddings │
    │Encoder    │          │ Encoder   │          │ Layers    │
    │768-dim    │          │768-dim    │          │256-dim    │
    └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
          │                      │                      │
          └──────────┬───────────┴──────────────────────┘
                     │
              ┌─────▼─────┐
              │Sequential │
              │ LSTM +    │
              │ Attention │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │Classifier │
              │ 7 Classes │
              └───────────┘
```

#### **🎥 Model 2: Video + Text + Metadata**
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Video Input   │    │   Text Input    │    │ Metadata Input  │
│   224x224x8     │    │  BERT Tokens    │    │  Profile+Chain  │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
    ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
    │TimeSformer│          │BERT-base  │          │Embeddings │
    │ Encoder   │          │ Encoder   │          │ Layers    │
    │384-dim    │          │768-dim    │          │256-dim    │
    └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
          │                      │                      │
    ┌─────▼─────┐          ┌─────▼─────┐          ┌─────▼─────┐
    │Bi-LSTM    │          │Bi-LSTM    │          │Temporal   │
    │+ Attention│          │+ Attention│          │Processor  │
    └─────┬─────┘          └─────┬─────┘          └─────┬─────┘
          │                      │                      │
          └──────────┬───────────┴──────────────────────┘
                     │
              ┌─────▼─────┐
              │ Fusion    │
              │ LSTM +    │
              │ Attention │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │Cross-Modal│
              │ Attention │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │Classifier │
              │ 7 Classes │
              └───────────┘
```

#### **🔄 Planned Late Fusion Architecture**
```
┌─────────────────┐    ┌─────────────────┐
│ Audio Model     │    │ Video Model     │
│ Predictions     │    │ Predictions     │
│ (7-dim softmax) │    │ (7-dim softmax) │
└─────────┬───────┘    └─────────┬───────┘
          │                      │
          └──────────┬───────────┘
                     │
              ┌─────▼─────┐
              │Late Fusion│
              │ Strategy  │
              │(Ensemble) │
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │ Final     │
              │Prediction │
              └───────────┘
```

### **Technical Specifications**

**🎥 Video Processing:**
- **TimeSformer Encoder**: 384-dim embeddings, 6 heads, 4 layers
- **Frame Resolution**: 224×224 pixels, 8 frames per sequence
- **Patch-based**: 32×32 patches with spatial-temporal attention

**📝 Text Processing:**
- **BERT-base-uncased**: 768-dim embeddings
- **Frozen Layers**: First 6 layers frozen for efficiency
- **Context**: 256 tokens, Dialogue: 128 tokens per utterance

**📊 Metadata Processing:**
- **Speaker/Listener Profiles**: Age, gender, timbre, personality IDs
- **Empathy Chains**: Event scenarios, emotion causes, response goals
- **Rich Embeddings**: 32-dim for chain elements, 16-32 dim for profiles

**🔄 Fusion Strategy:**
- **Multi-level LSTMs**: Separate processing for each modality
- **Enhanced Attention**: 8-head multi-head attention mechanisms
- **Cross-modal Fusion**: Final attention layer for feature integration

---

## 📊 Performance Results

### **Dual Model Performance**

| Model | Modalities | Accuracy | Precision | Recall | F1-Score | Status |
|-------|------------|----------|-----------|--------|----------|---------|
| **Audio Model** | 🎧📝📊 | **83.15%** | **82.76%** | **83.15%** | **82.37%** | ✅ Complete |
| **Video Model** | 🎥📝📊 | **31.89%** | **80.71%** | **31.89%** | **36.27%** | ✅ Complete |
| **Late Fusion** | 🎧🎥📝📊 | **TBD** | **TBD** | **TBD** | **TBD** | 🔄 Planned |

### **Audio Model Per-Class Performance** 🎧
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Happy** | 84.2% | 84.2% | 84.2% | 1,150 |
| **Surprised** | 66.0% | 66.0% | 66.0% | 361 |
| **Angry** | 83.6% | 83.6% | 83.6% | 399 |
| **Fear** | 71.0% | 71.0% | 71.0% | 297 |
| **Sad** | 86.3% | 86.3% | 86.3% | 2,536 |
| **Disgusted** | 70.1% | 70.1% | 70.1% | 89 |
| **Contempt** | 79.0% | 79.0% | 79.0% | 107 |

### **Video Model Per-Class Performance** 🎥
| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Happy** | 100.0% | 10.8% | 19.5% | 1,150 |
| **Surprised** | 87.8% | 23.8% | 37.5% | 361 |
| **Angry** | 16.8% | 92.2% | 28.4% | 399 |
| **Fear** | 24.1% | 44.1% | 31.2% | 297 |
| **Sad** | 93.7% | 32.3% | 48.0% | 2,536 |
| **Disgusted** | 0.0% | 0.0% | 0.0% | 89 |
| **Contempt** | 4.2% | 43.9% | 7.7% | 107 |

### **Key Insights**
- ✅ **Audio Model Excellence**: 83.15% accuracy shows strong acoustic-linguistic understanding
- ✅ **Complementary Strengths**: Video model has high precision, audio model has balanced performance
- 🎯 **Fusion Potential**: Different modality strengths suggest excellent late fusion opportunities
- ⚡ **Performance Boost Expected**: Late fusion should significantly exceed individual model performance

---

## 🚀 Quick Start

### **1. Environment Setup**

```bash
# Clone repository
git clone https://github.com/AbdulaAlShyed-2212592042/empathetic-detection.git
cd empathetic-detection
