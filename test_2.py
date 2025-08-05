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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from copy import deepcopy

# RANDOM SEED ==========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)

# EMOTION PROJECTION DICTS ==========================
ed_emotion_projection = {
    'conflicted': 'anxious',
    'vulnerability': 'afraid',
    'helplessness': 'afraid',
    'sadness': 'sad',
    'pensive': 'sentimental',
    'frustration': 'annoyed',
    'weary': 'tired',
    'anxiety': 'anxious',
    'reflective': 'sentimental',
    'upset': 'disappointed',
    'worried': 'anxious',
    'fear': 'afraid',
    'frustrated': 'sad',
    'fatigue': 'tired',
    'lost': 'jealous',
    'disappointment': 'disappointed',
    'nostalgia': 'nostalgic',
    'exhaustion': 'tired',
    'uneasy': 'anxious',
    'loneliness': 'lonely',
    'fragile': 'afraid',
    'confused': 'jealous',
    'vulnerable': 'afraid',
    'thoughtful': 'sentimental',
    'stressed': 'anxious',
    'concerned': 'anxious',
    'tiredness': 'tired',
    'burdened': 'anxious',
    'melancholy': 'sad',
    'overwhelmed': 'anxious',
    'worry': 'anxious',
    'heavy-hearted': 'sad',
    'melancholic': 'sad',
    'nervous': 'anxious',
    'fearful': 'afraid',
    'stress': 'anxious',
    'confusion': 'anxious',
    'inadequacy': 'ashamed',
    'regret': 'guilty',
    'helpless': 'afraid',
    'concern': 'anxious',
    'exhausted': 'tired',
    'overwhelm': 'anxious',
    'tired': 'tired',
    'disappointed': 'sad',
    'surprised': 'surprised',
    'excited': 'happy',
    'angry': 'angry',
    'proud': 'happy',
    'annoyed': 'angry',
    'grateful': 'happy',
    'lonely': 'sad',
    'afraid': 'fear',
    'terrified': 'fear',
    'guilty': 'sad',
    'impressed': 'surprised',
    'disgusted': 'disgusted',
    'hopeful': 'happy',
    'confident': 'happy',
    'furious': 'angry',
    'anxious': 'sad',
    'anticipating': 'happy',
    'joyful': 'happy',
    'nostalgic': 'sad',
    'prepared': 'happy',
    'jealous': 'contempt',
    'content': 'happy',
    'devastated': 'surprised',
    'embarrassed': 'sad',
    'caring': 'happy',
    'sentimental': 'sad',
    'trusting': 'happy',
    'ashamed': 'sad',
    'apprehensive': 'fear',
    'faithful': 'happy'       
}

emotion_projection = {
    "happy": 0,
    "surprised": 1,
    "angry": 2,
    "fear": 3,
    "sad": 4,
    "disgusted": 5,
    "contempt": 6
}

num_classes = len(emotion_projection)

# Reverse map for printing labels
emotion_labels = {v: k for k, v in emotion_projection.items()}

def map_emotion_to_class(label):
    label = label.lower()
    coarse = ed_emotion_projection.get(label, None)
    if coarse is None:
        if label in emotion_projection:
            coarse = label
        else:
            coarse = 'happy'  # fallback default
    return emotion_projection.get(coarse, emotion_projection['happy'])

# SAVE METRICS LOG ==========================
def save_detailed_metrics(
    train_accuracies,
    train_losses,
    val_accuracies,
    val_losses,
    test_accuracy,
    test_precision,
    test_recall,
    test_f1,
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
    # Random time-stretch
    if random.random() < 0.3:
        rate = random.uniform(0.9, 1.1)
        try:
            y = librosa.effects.time_stretch(y=y, rate=rate)
        except TypeError:
            y = librosa.effects.time_stretch(y, rate)

    # Random pitch shift
    if random.random() < 0.3:
        steps = random.randint(-2, 2)
        try:
            y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)
        except TypeError:
            y = librosa.effects.pitch_shift(y, sr, steps)

    # Random noise injection
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.005, y.shape)
        y = y + noise

    # Random volume scaling
    if random.random() < 0.3:
        gain = random.uniform(0.8, 1.2)
        y = y * gain

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

# DATASET ==========================
class AudioTextDataset(Dataset):
    def __init__(self, json_path, audio_dir, indices=None):
        self.audio_dir = audio_dir
        self.data = []

        with open(json_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)

        pattern = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.wav")

        # Build lookup for conversation-turn matching
        self.turn_lookup = {}
        for conv in tqdm(conversations, desc="Building conversation lookup"):
            conv_id = str(conv["conversation_id"])
            speaker_id = str(conv["speaker_profile"]["ID"])
            for turn in conv["turns"]:
                turn_id = str(turn["turn_id"])
                self.turn_lookup[(conv_id, turn_id, speaker_id)] = (conv, turn)

        for filename in tqdm(os.listdir(audio_dir), desc="Loading audio file metadata"):
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
                emotion_raw = turn.get('chain_of_empathy', {}).get('speaker_emotion', 'neutral').lower()
                label = map_emotion_to_class(emotion_raw)

                self.data.append({
                    "audio_path": os.path.join(audio_dir, filename),
                    "text": speaker_text,
                    "label": label
                })

        if not self.data:
            raise RuntimeError("No valid dataset entries found in provided paths.")

        if indices is not None:
            self.data = [self.data[i] for i in indices]

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
    
    if not hasattr(audio_inputs, "attention_mask"):
        attention_mask = torch.ones(audio_inputs.input_values.shape, dtype=torch.long)
    else:
        attention_mask = audio_inputs.attention_mask

    text_inputs = tokenizer_text(texts, padding=True, truncation=False, return_tensors="pt")

    return audio_inputs.input_values, attention_mask, text_inputs, labels


# MODEL ==========================
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
        self.wav2vec.gradient_checkpointing_enable()

        self.bert = BertModel.from_pretrained("bert-base-uncased")
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
    # ===== PATHS AND HYPERPARAMETERS =====
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"
    batch_size = 8
    learning_rate = 3e-5
    best_model_path = "results/best_multimodal_model_1.pth"
    os.makedirs("results", exist_ok=True)

    # ===== DEVICE CONFIG =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== LOAD FULL DATASET =====
    print("Loading full dataset...")
    full_dataset = AudioTextDataset(json_path, audio_dir)

    # Extract labels (integers) for stratified splitting
    full_labels = [item['label'] for item in full_dataset.data]

    # ===== STRATIFIED SPLITS (70/15/15) =====
    print("Performing stratified split...")
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter1.split(np.arange(len(full_dataset)), full_labels))

    temp_labels = [full_labels[i] for i in temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_split, test_split = next(splitter2.split(temp_idx, temp_labels))
    val_idx = [temp_idx[i] for i in val_split]
    test_idx = [temp_idx[i] for i in test_split]

    # ===== CREATE DATASET SUBSETS =====
    train_dataset = AudioTextDataset(json_path, audio_dir, indices=train_idx)
    val_dataset = AudioTextDataset(json_path, audio_dir, indices=val_idx)
    test_dataset = AudioTextDataset(json_path, audio_dir, indices=test_idx)

    # ===== Print First Training Sample =====
    print("\n====== Sample from Train Dataset ======")
    sample = train_dataset[0]
    print(f"Audio Path : {sample['audio_path']}")
    print(f"Text       : {sample['text']}")
    print(f"Label      : {sample['label']} ({emotion_labels.get(sample['label'], 'unknown')})")

    # ===== TOKENIZERS / PROCESSORS =====
    print("Loading tokenizers and processors...")
    processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

    # ===== DATALOADERS =====
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

    # ===== MODEL, OPTIMIZER, SCHEDULER =====
    print("Initializing model...")
    model = MultimodalModel(num_classes=num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    # ===== TRAINING VARIABLES =====
    best_val_acc = 0
    best_model = None
    patience = 5
    no_improve = 0

    train_accuracies, train_losses = [], []
    val_accuracies, val_losses = [], []

    # ===== TRAINING LOOP =====
    for epoch in range(1, 31):
        print(f"\nEpoch {epoch}/30")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, desc="Validation")

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, best_model_path)
            no_improve = 0
            print("Best model saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping due to no improvement.")
                break

    # ===== LOAD BEST MODEL AND TEST =====
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, desc="Test")

    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Get detailed test metrics
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for audio_inputs, audio_attention_mask, text_inputs, labels in tqdm(test_loader, desc="Test metrics"):
            audio_inputs = audio_inputs.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, audio_attention_mask, text_inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")

    save_detailed_metrics(train_accuracies, train_losses, val_accuracies, val_losses, accuracy, precision, recall, f1, best_val_acc)

if __name__ == "__main__":
    main()
