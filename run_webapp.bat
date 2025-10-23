@echo off
REM Startup script for Emotion Detection Web App (Windows)

echo 🎭 Emotion Detection Web Application
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install requirements
echo 📋 Installing requirements...
pip install -r requirements.txt

REM Check if checkpoint exists
if not exist "late_fusion_checkpoint\best_late_fusion_model_20251021_010240_acc59.6154.pth" (
    echo ⚠️  Warning: Model checkpoint not found!
    echo    The app will run with a demo model for testing.
    echo    Place your trained model at: late_fusion_checkpoint\best_late_fusion_model_20251021_010240_acc59.6154.pth
)

REM Start the application
echo 🚀 Starting web application...
echo 🌐 Open your browser and go to: http://localhost:5000
echo.
python app.py

pause