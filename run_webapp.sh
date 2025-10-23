#!/bin/bash
# Startup script for Emotion Detection Web App

echo "🎭 Emotion Detection Web Application"
echo "======================================"

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.9+"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements.txt

# Check if checkpoint exists
if [ ! -f "late_fusion_checkpoint/best_late_fusion_model_20251021_010240_acc59.6154.pth" ]; then
    echo "⚠️  Warning: Model checkpoint not found!"
    echo "   The app will run with a demo model for testing."
    echo "   Place your trained model at: late_fusion_checkpoint/best_late_fusion_model_20251021_010240_acc59.6154.pth"
fi

# Start the application
echo "🚀 Starting web application..."
echo "🌐 Open your browser and go to: http://localhost:5000"
echo ""
python app.py