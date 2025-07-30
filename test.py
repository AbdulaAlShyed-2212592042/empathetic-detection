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

# ==========================
# 1. Set Random Seed
# ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==========================
# 2. Dataset Class
# ==========================
class AudioTextDataset(Dataset):
    """
    Custom dataset to handle multimodal audio-text emotion recognition.
    Expects:
    - JSON file with conversation and turn structure
    - Audio directory with filenames like diaXuttY_Z.wav
    """
    def __init__(self, json_path, audio_dir, max_audio_length=16000 * 10, indices=None):
        self.audio_dir = audio_dir
        self.max_audio_length = max_audio_length
        self.data = []

        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        # Regex to match audio files
        pattern = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.wav")

        # Build lookup for turns
        self.turn_lookup = {}
        for conv in conversations:
            conv_id = str(conv["conversation_id"])
            speaker_id = str(conv["speaker_profile"]["ID"])
            for turn in conv["turns"]:
                turn_id = str(turn["turn_id"])
                self.turn_lookup[(conv_id, turn_id, speaker_id)] = (conv, turn)

        # Map audio files to dataset entries
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

                # Get latest speaker utterance
                speaker_text = None
                for utt in reversed(dialogue_history):
                    if utt['role'] == 'speaker':
                        speaker_text = utt['utterance']
                        break
                if speaker_text is None:
                    speaker_text = ""

                emotion = turn.get('chain_of_empathy', {}).get('speaker_emotion', 'neutral').lower()

                self.data.append({
                    "audio_path": os.path.join(audio_dir, filename),
                    "text": speaker_text,
                    "label": emotion
                })

        if not self.data:
            raise RuntimeError("No valid dataset entries found in provided paths.")

        # Apply stratified indices if provided
        if indices is not None:
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==========================
# 3. Emotion Label Mapping
# ==========================
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

# ==========================
# 4. Collate Function
# ==========================
def collate_fn(batch, processor_audio, tokenizer_text, max_audio_len=16000 * 10, max_text_len=128):
    """
    Prepares batch for model input:
    - Loads audio, resamples, pads/truncates
    - Tokenizes text
    - Converts labels to tensor
    """
    audio_paths = [item['audio_path'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([emotion_labels.get(item['label'], emotion_labels['neutral']) for item in batch])

    # Process audio
    audio_inputs = []
    for path in audio_paths:
        speech_array, sr = librosa.load(path, sr=16000)
        if len(speech_array) > max_audio_len:
            speech_array = speech_array[:max_audio_len]
        else:
            speech_array = np.pad(speech_array, (0, max_audio_len - len(speech_array)), mode='constant')
        audio_inputs.append(speech_array)

    audio_inputs = processor_audio(audio_inputs, sampling_rate=16000, return_tensors="pt", padding=True).input_values

    # Process text
    text_inputs = tokenizer_text(texts, padding='max_length', truncation=True,
                                max_length=max_text_len, return_tensors='pt')

    return audio_inputs, text_inputs, labels

# ==========================
# 5. Model Definition
# ==========================
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        # Load pretrained Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
        self.wav2vec.gradient_checkpointing_enable()

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained("bert-base-uncased", use_safetensors=True)
        self.bert.gradient_checkpointing_enable()

        audio_feat_dim = self.wav2vec.config.hidden_size
        text_feat_dim = self.bert.config.hidden_size

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(audio_feat_dim + text_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        self.to(device)

    def forward(self, audio_inputs, text_inputs):
        audio_inputs = audio_inputs.float()

        # Audio features
        audio_outputs = self.wav2vec(audio_inputs).last_hidden_state
        audio_feat = audio_outputs.mean(dim=1)

        # Text features
        text_outputs = self.bert(**text_inputs)
        text_feat = text_outputs.pooler_output

        # Combine features
        combined = torch.cat((audio_feat, text_feat), dim=1)
        logits = self.classifier(combined)
        return logits

# ==========================
# 6. Training & Evaluation
# ==========================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for audio_inputs, text_inputs, labels in pbar:
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

        pbar.set_postfix(loss=total_loss / total, accuracy=correct / total)

    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(dataloader, desc=desc, leave=False)

    with torch.no_grad():
        for audio_inputs, text_inputs, labels in pbar:
            audio_inputs = audio_inputs.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, text_inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=total_loss / total, accuracy=correct / total)

    return total_loss / total, correct / total

# ==========================
# 7. Main Training Routine
# ==========================
def main():
    # ---- Config ----
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"
    batch_size = 4
    epochs = 10
    learning_rate = 3e-5
    max_audio_len = 16000 * 10
    max_text_len = 128
    best_model_path = "best_multimodal_model.pth"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load dataset ----
    full_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len)
    print(f"Full dataset size: {len(full_dataset)}")

    full_labels = [emotion_labels.get(item['label'], emotion_labels['neutral']) for item in full_dataset.data]

    # Stratified split: 70% train, 15% val, 15% test
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter1.split(full_dataset.data, full_labels))

    temp_labels = [full_labels[i] for i in temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_rel, test_idx_rel = next(splitter2.split(temp_idx, temp_labels))

    val_idx = [temp_idx[i] for i in val_idx_rel]
    test_idx = [temp_idx[i] for i in test_idx_rel]

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    # Datasets
    train_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=train_idx)
    val_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=val_idx)
    test_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=test_idx)

    # Processors
    processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda batch: collate_fn(batch, processor_audio, tokenizer_text, max_audio_len, max_text_len))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda batch: collate_fn(batch, processor_audio, tokenizer_text, max_audio_len, max_text_len))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda batch: collate_fn(batch, processor_audio, tokenizer_text, max_audio_len, max_text_len))

    # Model, optimizer, loss
    model = MultimodalModel(num_classes=num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # ---- Training loop ----
    best_val_acc = 0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, desc="Validation")

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model (val acc: {best_val_acc:.4f})")

    # ---- Test evaluation ----
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, desc="Testing")
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")

if __name__ == "__main__":
    main()
# ==========================nvidia-smi
