import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from transformers import Wav2Vec2Processor, BertTokenizer
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Import your definitions from TEST_1.py
from TEST_1 import (
    AudioTextDataset, collate_fn, MultimodalModel,
    emotion_labels, emotion_projection, num_classes
)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ===== Paths =====
json_path = "data/train_audio/audio_v5_0/train.json"
audio_dir = "data/train_audio/audio_v5_0"
model_path = "results/best_multimodal_model.pth"
conf_matrix_output = "results/confusion_matrix.png"

# ===== Device Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Load Tokenizers =====
processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

# ===== Load Dataset and Perform Stratified Split =====
full_dataset = AudioTextDataset(json_path, audio_dir)
full_labels = [item['label'] for item in full_dataset.data]

splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
_, temp_idx = next(splitter1.split(np.arange(len(full_dataset)), full_labels))
temp_labels = [full_labels[i] for i in temp_idx]

splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
_, test_split = next(splitter2.split(temp_idx, temp_labels))
test_idx = [temp_idx[i] for i in test_split]

test_dataset = AudioTextDataset(json_path, audio_dir, indices=test_idx)

# ===== Prepare DataLoader =====
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text)
)

# ===== Load Model =====
model = MultimodalModel(num_classes=num_classes, device=device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== Inference =====
all_preds, all_labels = [], []

with torch.no_grad():
    for audio_inputs, text_inputs, labels in tqdm(test_loader, desc="Evaluating"):
        audio_inputs = audio_inputs.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        outputs = model(audio_inputs, text_inputs)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ===== Build Detailed Info =====
detailed_test_samples = []
for i, sample in enumerate(test_dataset.data):
    true_label_idx = all_labels[i]
    pred_label_idx = all_preds[i]

    detailed_test_samples.append({
        "audio_path": sample["audio_path"],
        "true_label": emotion_labels[true_label_idx],
        "predicted_label": emotion_labels[pred_label_idx],
        "correct": true_label_idx == pred_label_idx
    })

# ===== Split and Save =====
os.makedirs("results", exist_ok=True)

correct_preds = [s for s in detailed_test_samples if s["correct"]]
incorrect_preds = [s for s in detailed_test_samples if not s["correct"]]

with open("results/correct_predictions.json", "w", encoding="utf-8") as f:
    json.dump(correct_preds, f, indent=4, ensure_ascii=False)

with open("results/incorrect_predictions.json", "w", encoding="utf-8") as f:
    json.dump(incorrect_preds, f, indent=4, ensure_ascii=False)

print("Saved correct predictions to results/correct_predictions.json")
print("Saved incorrect predictions to results/incorrect_predictions.json")

# ===== Confusion Matrix =====
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[emotion_labels[i] for i in range(num_classes)]
)

plt.figure(figsize=(8, 6))
disp.plot(cmap="Blues", xticks_rotation=45, values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(conf_matrix_output, dpi=300)
plt.show()

print(f"Confusion matrix saved to {conf_matrix_output}")
