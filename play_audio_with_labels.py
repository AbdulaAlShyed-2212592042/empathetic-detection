import os
import shutil
import librosa
import sounddevice as sd
import time
import json
from datetime import datetime

# Create 'results' folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Audio metadata list
audio_metadata = [
    {
        "audio_path": "data/train_audio/audio_v5_0\\dia02727utt0_51.wav",
        "label": "sad"
    },
    {
        "audio_path": "data/train_audio/audio_v5_0\\dia14041utt0_19.wav",
        "label": "angry"
    },
    {
        "audio_path": "data/train_audio/audio_v5_0\\dia09610utt0_7.wav",
        "label": "happy"
    }
]

# For logging playback info
play_log = []

for entry in audio_metadata:
    original_path = entry["audio_path"].replace("\\", "/")
    label = entry["label"]
    filename = os.path.basename(original_path)
    results_path = f"results/{filename}"

    if not os.path.exists(original_path):
        print(f"[ERROR] File not found: {original_path}")
        continue

    print(f"\n▶️ Now Playing: {filename}")
    print(f"Emotion Label: {label}")

    try:
        # Load and play audio
        y, sr = librosa.load(original_path, sr=None)
        sd.play(y, sr)
        sd.wait()

        # Copy file to results/
        shutil.copy2(original_path, results_path)

        # Add to log
        play_log.append({
            "filename": filename,
            "label": label,
            "played_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        time.sleep(1)

    except Exception as e:
        print(f"[ERROR] Failed to play {filename} — {e}")

# Save play log
with open("results/audio_play_log.json", "w") as f:
    json.dump(play_log, f, indent=4)

print("\n✅ Done! Played audio and saved to 'results/'")
