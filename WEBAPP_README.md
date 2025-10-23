# 🎭 Emotion Detection Web Application

A real-time web application that uses your trained late fusion model to detect emotions from video input with audio transcription.

## 🚀 Features

- **📹 Video Upload**: Upload video files (MP4, WebM, AVI) for emotion analysis
- **📷 Live Camera**: Real-time emotion detection using your webcam
- **🎤 Audio Transcription**: Automatic speech-to-text using OpenAI Whisper
- **🧠 Multimodal AI**: Combines video, audio, and text analysis
- **📊 Detailed Results**: Shows emotion predictions, confidence scores, and fusion weights
- **🎯 Real Model**: Uses your actual trained checkpoint `best_late_fusion_model_20251021_010240_acc59.6154.pth`

## 🛠️ Installation & Setup

### Option 1: Quick Start (Windows)
```bash
# Simply run the startup script
run_webapp.bat
```

### Option 2: Quick Start (Linux/Mac)
```bash
# Make script executable and run
chmod +x run_webapp.sh
./run_webapp.sh
```

### Option 3: Manual Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## 🌐 Usage

1. **Start the application** using one of the methods above
2. **Open your browser** and go to `http://localhost:5000`
3. **Choose input method**:
   - **Upload Video**: Select a video file from your computer
   - **Live Camera**: Use your webcam for real-time analysis
4. **View results**:
   - Predicted emotion with confidence score
   - Audio transcription from the video
   - Fusion weights showing video vs audio-text contribution
   - Probability breakdown for all emotion classes

## 📋 Requirements

- **Python**: 3.9+
- **GPU**: NVIDIA GPU with CUDA (recommended) or CPU
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: 5GB+ for models and dependencies
- **Webcam**: For live camera functionality (optional)

## 🎯 Model Architecture

The web app uses your trained **Late Fusion Model** which combines:

1. **Video Analysis**: Extracts facial expressions and visual cues
2. **Audio Processing**: Analyzes voice tone and prosodic features  
3. **Text Analysis**: Uses Whisper transcription + BERT understanding
4. **Fusion**: Learnable weighted combination (Video: 89.5%, Audio-Text: 10.5%)

## 📊 Emotion Classes

The model predicts 7 emotion categories:
- 😐 **Neutral**
- 😊 **Joy** 
- 😢 **Sadness**
- 😠 **Anger**
- 😨 **Fear**
- 🤢 **Disgust**
- 😲 **Surprise**

## 🔧 Technical Details

### Model Loading
- Automatically loads your checkpoint: `late_fusion_checkpoint/best_late_fusion_model_20251021_010240_acc59.6154.pth`
- Falls back to demo model if checkpoint not found
- Uses GPU acceleration when available

### Audio Processing
- **Whisper Base Model**: For speech transcription
- **Wav2Vec2**: For audio feature extraction
- **Sample Rate**: 16kHz standardized processing

### Video Processing
- **Frame Extraction**: 8 frames per video sequence
- **Resolution**: 224x224 standardized frames
- **Format Support**: MP4, WebM, AVI, and more

## 🚨 Troubleshooting

### Common Issues

**"Model checkpoint not found"**
- Ensure your trained model is at: `late_fusion_checkpoint/best_late_fusion_model_20251021_010240_acc59.6154.pth`
- The app will run with a demo model for testing if checkpoint is missing

**"CUDA out of memory"**
- Reduce video resolution or duration
- Use CPU mode by setting `CUDA_VISIBLE_DEVICES=""`
- Close other GPU-intensive applications

**"Webcam not accessible"**
- Check browser permissions for camera access
- Ensure webcam is not being used by other applications
- Try refreshing the page

**"Audio transcription failed"**
- Ensure video has audio track
- Check audio quality and volume
- Whisper works best with clear speech

### Performance Tips

- **GPU Acceleration**: Use NVIDIA GPU for faster processing
- **Video Quality**: Higher quality videos give better results
- **Audio Clarity**: Clear speech improves transcription accuracy
- **Lighting**: Good lighting improves video emotion detection

## 📁 File Structure

```
empathetic-detection/
├── app.py                              # Main Flask application
├── templates/
│   └── index.html                      # Web interface
├── late_fusion_checkpoint/
│   └── best_late_fusion_model_*.pth    # Your trained model
├── run_webapp.bat                      # Windows startup script
├── run_webapp.sh                       # Linux/Mac startup script
└── requirements.txt                    # Python dependencies
```

## 🔒 Security Notes

- The app runs locally on `localhost:5000`
- No data is sent to external servers
- Temporary files are automatically cleaned up
- Camera access requires explicit browser permission

## 🤝 Development

To modify the web application:

1. **Frontend**: Edit `templates/index.html` for UI changes
2. **Backend**: Modify `app.py` for processing logic
3. **Models**: Update model classes for architecture changes
4. **Styling**: CSS is embedded in the HTML template

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Verify all requirements are installed
- Ensure your model checkpoint is available
- Check console output for detailed error messages

---

*🎭 Real-time emotion detection powered by your trained multimodal AI system!*