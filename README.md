# 🤖 Empathetic Detection - Multimodal Emotion Recognition

This project focuses on **multimodal emotion classification** using the **AvaMERG dataset**. It combines audio, video, and text modalities to detect empathetic states with deep learning.

---

## 📁 Dataset

**[AvaMERG on Hugging Face](https://huggingface.co/datasets/ZhangHanXD/AvaMERG)**

A multimodal emotion recognition dataset including:

- 🎧 Audio waveforms (.wav)  
- 🎥 Video data  
- 📝 Text transcripts  
- ❤️ Empathy-related emotion labels  

---

## 🎯 Objective

To build a **multimodal neural network** that detects empathy using:

- Audio features extracted via `Wav2Vec2Model`  
- Video features extracted via pretrained vision model (e.g., CNN or transformer-based)  
- Text features extracted via `BERT`  
- Fusion of all modalities for classification  

---

## 🧠 Model Architecture

- **Audio Encoder**: `facebook/wav2vec2-base` pretrained model  
- **Video Encoder**: (Add your chosen pretrained video model here, e.g., ResNet or Video Transformer)  
- **Text Encoder**: `bert-base-uncased`  
- **Fusion**: Concatenation of embeddings → Fully connected layers → Output classifier  
- **Output**: Empathy-related emotion classification  

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/AbdulaAlShyed-2212592042/empathetic-detection.git
cd empathetic-detection
