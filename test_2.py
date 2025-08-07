# IMPORTS ==========================
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from copy import deepcopy
import time

# RANDOM SEED ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# EMOTION PROJECTION DICTS ==========================
ed_emotion_projection = {
    'conflicted': 'anxious', 'vulnerability': 'afraid', 'helplessness': 'afraid',
    'sadness': 'sad', 'pensive': 'sentimental', 'frustration': 'annoyed',
    'weary': 'tired', 'anxiety': 'anxious', 'reflective': 'sentimental',
    'upset': 'disappointed', 'worried': 'anxious', 'fear': 'afraid',
    'frustrated': 'sad', 'fatigue': 'tired', 'lost': 'jealous',
    'disappointment': 'disappointed', 'nostalgia': 'nostalgic', 'exhaustion': 'tired',
    'uneasy': 'anxious', 'loneliness': 'lonely', 'fragile': 'afraid',
    'confused': 'jealous', 'vulnerable': 'afraid', 'thoughtful': 'sentimental',
    'stressed': 'anxious', 'concerned': 'anxious', 'tiredness': 'tired',
    'burdened': 'anxious', 'melancholy': 'sad', 'overwhelmed': 'anxious',
    'worry': 'anxious', 'heavy-hearted': 'sad', 'melancholic': 'sad',
    'nervous': 'anxious', 'fearful': 'afraid', 'stress': 'anxious',
    'confusion': 'anxious', 'inadequacy': 'ashamed', 'regret': 'guilty',
    'helpless': 'afraid', 'concern': 'anxious', 'exhausted': 'tired',
    'overwhelm': 'anxious', 'tired': 'tired', 'disappointed': 'sad',
    'surprised': 'surprised', 'excited': 'happy', 'angry': 'angry',
    'proud': 'happy', 'annoyed': 'angry', 'grateful': 'happy',
    'lonely': 'sad', 'afraid': 'fear', 'terrified': 'fear',
    'guilty': 'sad', 'impressed': 'surprised', 'disgusted': 'disgusted',
    'hopeful': 'happy', 'confident': 'happy', 'furious': 'angry',
    'anxious': 'sad', 'anticipating': 'happy', 'joyful': 'happy',
    'nostalgic': 'sad', 'prepared': 'happy', 'jealous': 'contempt',
    'content': 'happy', 'devastated': 'surprised', 'embarrassed': 'sad',
    'caring': 'happy', 'sentimental': 'sad', 'trusting': 'happy',
    'ashamed': 'sad', 'apprehensive': 'fear', 'faithful': 'happy'
}

emotion_projection = {
    "happy": 0, "surprised": 1, "angry": 2, "fear": 3,
    "sad": 4, "disgusted": 5, "contempt": 6
}

num_classes = len(emotion_projection)
emotion_labels = {v: k for k, v in emotion_projection.items()}

def map_emotion_to_class(label):
    label = label.lower()
    coarse = ed_emotion_projection.get(label, None)
    if coarse is None:
        coarse = label if label in emotion_projection else 'happy'
    return emotion_projection.get(coarse, emotion_projection['happy'])

# SAVE METRICS LOG ==========================
def save_detailed_metrics(
    train_accuracies, train_losses,
    val_accuracies, val_losses,
    test_accuracy, test_precision,
    test_recall, test_f1,
    best_val_accuracy,
    log_file_path="results_1/metrics_log.txt"
):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "w") as f:
        f.write("====== TRAINING AND VALIDATION METRICS ======\n")
        for epoch in range(len(train_accuracies)):
            f.write(
                f"Epoch {epoch+1}: "
                f"Train Loss={train_losses[epoch]:.4f}, Train Acc={train_accuracies[epoch]:.4f}, "
                f"Val Loss={val_losses[epoch]:.4f}, Val Acc={val_accuracies[epoch]:.4f}\n"
            )
        f.write("\n====== BEST VALIDATION ACCURACY ======\n")
        f.write(f"{best_val_accuracy:.4f}\n")
        f.write("\n====== FINAL TEST METRICS ======\n")
        f.write(f"Test Accuracy : {test_accuracy:.4f}\n")
        f.write(f"Precision     : {test_precision:.4f}\n")
        f.write(f"Recall        : {test_recall:.4f}\n")
        f.write(f"F1 Score      : {test_f1:.4f}\n")

# AUDIO AUGMENTATION ==========================
def augment_audio(y, sr):
    if random.random() < 0.3:  # Time-stretch
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y=y, rate=rate)  # <-- FIXED

    if random.random() < 0.3:  # Pitch shift
        steps = random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)

    if random.random() < 0.3:  # Add noise
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise

    if random.random() < 0.3:  # Volume change
        gain = random.uniform(0.8, 1.2)
        y = y * gain

    y = y / (np.max(np.abs(y)) + 1e-9)  # Normalize
    return y

# DATASET ==========================
class AudioTextDataset(Dataset):
    def __init__(self, json_data, audio_dir, conversation_ids=None):
        self.audio_dir = audio_dir
        self.data = []
        pattern = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.wav")

        # Filter conversations by IDs (train/val/test)
        conversations = [conv for conv in json_data if str(conv["conversation_id"]) in conversation_ids]

        # Build lookup per split
        turn_lookup = {}
        for conv in conversations:
            conv_id = str(conv["conversation_id"])
            speaker_id = str(conv["speaker_profile"]["ID"])
            for turn in conv["turns"]:
                turn_id = str(turn["turn_id"])
                turn_lookup[(conv_id, turn_id, speaker_id)] = (conv, turn)

        for filename in tqdm(os.listdir(audio_dir), desc="Loading audio file metadata"):
            if not filename.endswith('.wav'):
                continue
            m = pattern.match(filename)
            if not m:
                continue

            conv_id, turn_id, audio_id = m.group(1), m.group(2), m.group(3)
            key = (conv_id, turn_id, audio_id)

            if key in turn_lookup:
                conv, turn = turn_lookup[key]

                # Only past dialogue history (no leakage)
                dialogue_history = turn.get('dialogue_history', [])
                speaker_text = next(
                    (utt['utterance'] for utt in reversed(dialogue_history) if utt['role'] == 'speaker'),
                    ""
                )
                emotion_raw = turn.get('chain_of_empathy', {}).get('speaker_emotion', 'neutral').lower()
                label = map_emotion_to_class(emotion_raw)

                self.data.append({
                    "audio_path": os.path.join(audio_dir, filename),
                    "text": speaker_text,
                    "label": label
                })

        if not self.data:
            raise RuntimeError("No valid dataset entries found in provided paths.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# COLLATE FUNCTION ==========================
def collate_fn(batch, processor_audio, tokenizer_text, augment=False):
    audio_paths = [item['audio_path'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    audio_inputs_list = []
    for path in audio_paths:
        y, sr = librosa.load(path, sr=16000)
        if augment:
            y = augment_audio(y, sr)
        audio_inputs_list.append(y)

    audio_inputs = processor_audio(audio_inputs_list, sampling_rate=16000, return_tensors="pt", padding=True)
    attention_mask = audio_inputs.attention_mask if hasattr(audio_inputs, "attention_mask") else torch.ones_like(audio_inputs.input_values)

    text_inputs = tokenizer_text(texts, padding=True, truncation=False, return_tensors="pt")

    return audio_inputs.input_values, attention_mask, text_inputs, labels

# MODEL ==========================
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

    def forward(self, audio_inputs, audio_attention_mask, text_inputs):
        audio_inputs = audio_inputs.float()
        audio_feat = self.wav2vec(audio_inputs, attention_mask=audio_attention_mask).last_hidden_state.mean(dim=1)
        text_feat = self.bert(**text_inputs).pooler_output
        combined = torch.cat((audio_feat, text_feat), dim=1)
        return self.classifier(combined)

# TRAINING / EVAL ==========================
def train_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for audio_inputs, audio_attention_mask, text_inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        audio_inputs = audio_inputs.to(device)
        audio_attention_mask = audio_attention_mask.to(device)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            outputs = model(audio_inputs, audio_attention_mask, text_inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total

def eval_epoch(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for audio_inputs, audio_attention_mask, text_inputs, labels in tqdm(dataloader, desc=desc, leave=False):
            audio_inputs = audio_inputs.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, audio_attention_mask, text_inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total

# MAIN ==========================
def main():
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"
    batch_size = 8
    learning_rate = 3e-5
    best_model_path = "results/best_multimodal_model_1.pth"
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading JSON...")
    with open(json_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    # Extract conversation IDs and split (no leakage)
    conv_ids = [str(conv["conversation_id"]) for conv in conversations]
    train_ids, temp_ids = train_test_split(conv_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print("Preparing datasets (no leakage)...")
    train_dataset = AudioTextDataset(conversations, audio_dir, train_ids)
    val_dataset = AudioTextDataset(conversations, audio_dir, val_ids)
    test_dataset = AudioTextDataset(conversations, audio_dir, test_ids)

    # Show sample
    sample = train_dataset[0]
    print(f"Sample: {sample['audio_path']} | {sample['text']} | {sample['label']} ({emotion_labels[sample['label']]})")

    print("Loading tokenizers and processors...")
    processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text, augment=True)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text, augment=False)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text, augment=False)
    )

    print("Initializing model...")
    model = MultimodalModel(num_classes=num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scaler = torch.amp.GradScaler('cuda')

    best_val_acc = 0
    best_model = None
    patience = 3
    no_improve = 0

    train_accuracies, train_losses = [], []
    val_accuracies, val_losses = [], []

    for epoch in range(1, 31):
        start_time = time.time()
        print(f"\nEpoch {epoch}/30")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, desc="Validation")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Epoch Time: {(time.time()-start_time):.2f}s")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, best_model_path)
            print(f"Best model saved at epoch {epoch} with val acc {val_acc:.4f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_model:
        model.load_state_dict(best_model)

    print("\nEvaluating on test set...")
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for audio_inputs, audio_attention_mask, text_inputs, labels in tqdm(test_loader, desc="Testing"):
            audio_inputs = audio_inputs.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, audio_attention_mask, text_inputs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"Test Accuracy: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    save_detailed_metrics(
        train_accuracies, train_losses,
        val_accuracies, val_losses,
        test_acc, precision, recall, f1,
        best_val_acc
    )

if __name__ == "__main__":
    main()