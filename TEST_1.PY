# ========================== IMPORTS ==========================
import os
import json
import re
import random
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from copy import deepcopy
import subprocess  # <-- Added for git push

# ========================== RANDOM SEED ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ========================== DATASET ==========================
class AudioTextDataset(Dataset):
    def __init__(self, json_path, audio_dir, max_audio_length=16000 * 10, indices=None):
        self.audio_dir = audio_dir
        self.max_audio_length = max_audio_length
        self.data = []

        with open(json_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        pattern = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.wav")

        self.turn_lookup = {}
        for conv in conversations:
            conv_id = str(conv["conversation_id"])
            speaker_id = str(conv["speaker_profile"]["ID"])
            for turn in conv["turns"]:
                turn_id = str(turn["turn_id"])
                self.turn_lookup[(conv_id, turn_id, speaker_id)] = (conv, turn)

        for filename in os.listdir(audio_dir):
            if not filename.endswith('.wav'):
                continue
            m = pattern.match(filename)
            if not m:
                continue

            conv_id, turn_id, audio_id = m.group(1), m.group(2), m.group(3)
            key = (conv_id, turn_id, audio_id)

            if key in self.turn_lookup:
                conv, turn = self.turn_lookup[key]
                dialogue_history = turn.get('dialogue_history', [])

                speaker_text = next((utt['utterance'] for utt in reversed(dialogue_history) if utt['role'] == 'speaker'), "")
                emotion = turn.get('chain_of_empathy', {}).get('speaker_emotion', 'neutral').lower()

                self.data.append({
                    "audio_path": os.path.join(audio_dir, filename),
                    "text": speaker_text,
                    "label": emotion
                })

        if not self.data:
            raise RuntimeError("No valid dataset entries found in provided paths.")

        if indices is not None:
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ========================== EMOTION LABELS ==========================
emotion_labels = {
    "happy": 0,
    "sad": 1,
    "angry": 2,
    "neutral": 3,
    "excited": 4,
    "frustrated": 5,
    "surprised": 6,
}
num_classes = len(emotion_labels)

# ========================== COLLATE FUNCTION ==========================
def collate_fn(batch, processor_audio, tokenizer_text, max_audio_len=16000 * 10, max_text_len=128):
    audio_paths = [item['audio_path'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([emotion_labels.get(item['label'], emotion_labels['neutral']) for item in batch])

    audio_inputs = []
    for path in audio_paths:
        speech_array, sr = librosa.load(path, sr=16000)
        speech_array = speech_array[:max_audio_len] if len(speech_array) > max_audio_len else np.pad(speech_array, (0, max_audio_len - len(speech_array)), mode='constant')
        audio_inputs.append(speech_array)

    audio_inputs = processor_audio(audio_inputs, sampling_rate=16000, return_tensors="pt", padding=True).input_values
    text_inputs = tokenizer_text(texts, padding='max_length', truncation=True, max_length=max_text_len, return_tensors='pt')

    return audio_inputs, text_inputs, labels

# ========================== MODEL ==========================
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
        self.wav2vec.gradient_checkpointing_enable()

        self.bert = BertModel.from_pretrained("bert-base-uncased", use_safetensors=True)
        self.bert.gradient_checkpointing_enable()

        audio_feat_dim = self.wav2vec.config.hidden_size
        text_feat_dim = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(audio_feat_dim + text_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.to(device)

    def forward(self, audio_inputs, text_inputs):
        audio_inputs = audio_inputs.float()
        audio_feat = self.wav2vec(audio_inputs).last_hidden_state.mean(dim=1)
        text_feat = self.bert(**text_inputs).pooler_output
        combined = torch.cat((audio_feat, text_feat), dim=1)
        return self.classifier(combined)

# ========================== TRAINING / EVAL ==========================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for audio_inputs, text_inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        audio_inputs = audio_inputs.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(audio_inputs, text_inputs)
            loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for audio_inputs, text_inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            audio_inputs = audio_inputs.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, text_inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# ========================== GIT PUSH FUNCTION ==========================
def git_push(files, commit_message):
    try:
        subprocess.run(["git", "add"] + files, check=True)
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        subprocess.run(["git", "push"], check=True)
        print("✅ Git push successful.")
    except subprocess.CalledProcessError as e:
        print("❌ Git push failed:", e)

# ========================== MAIN ==========================
def main():
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"
    batch_size = 8
    learning_rate = 3e-5
    max_audio_len = 16000 * 10
    max_text_len = 128
    best_model_path = "best_multimodal_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len)
    print(f"Full dataset size: {len(full_dataset)}")
    print("First audio file name:", os.path.basename(full_dataset.data[0]['audio_path']))
    print("First text sample:", full_dataset.data[0]['text'])

    full_labels = [emotion_labels.get(item['label'], emotion_labels['neutral']) for item in full_dataset.data]

    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter1.split(full_dataset.data, full_labels))

    temp_labels = [full_labels[i] for i in temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(splitter2.split(temp_idx, temp_labels))

    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [temp_idx[i] for i in test_idx_rel]

    train_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=train_idx)
    val_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=val_idx)
    test_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=test_idx)

    processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text, max_audio_len, max_text_len))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text, max_audio_len, max_text_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text, max_audio_len, max_text_len))

    model = MultimodalModel(num_classes=num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    best_val_acc = 0
    best_model = None
    patience = 5
    no_improve = 0

    for epoch in range(1, 31):
        print(f"\nEpoch {epoch}/30")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, desc="Validation")

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model.state_dict())
            no_improve = 0
            print(f"Saved best model (val acc: {best_val_acc:.4f})")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epochs.")
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_model:
        torch.save(best_model, best_model_path)

    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for audio_inputs, text_inputs, labels in tqdm(test_loader, desc="Testing"):
            audio_inputs = audio_inputs.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, text_inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    report_str = classification_report(all_labels, all_preds, target_names=emotion_labels.keys(), digits=4)
    print(report_str)

    # Save classification report and best val accuracy to a file
    results_path = "training_results.txt"
    with open(results_path, "w") as f:
        f.write(f"Best Val Accuracy: {best_val_acc:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report_str)

    # Automatically commit and push model and results file to GitHub
    git_push([best_model_path, results_path], "Auto commit: Training results and best model")

if __name__ == "__main__":
    main()
