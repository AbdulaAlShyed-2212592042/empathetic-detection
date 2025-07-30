import os
import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
import librosa
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
import random
from torch.utils.tensorboard import SummaryWriter

# Fix seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ---------------- Dataset ----------------
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
            raise RuntimeError("No valid dataset entries found.")

        if indices is not None:
            self.data = [self.data[i] for i in indices]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ---------------- Label Encoding ----------------
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

# ---------------- Collate ----------------
def collate_fn(batch, processor_audio, tokenizer_text, max_audio_len=16000 * 10, max_text_len=128):
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

# ---------------- Model ----------------
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.wav2vec.gradient_checkpointing_enable()

        self.bert = BertModel.from_pretrained("bert-large-uncased")
        self.bert.gradient_checkpointing_enable()

        audio_feat_dim = self.wav2vec.config.hidden_size
        text_feat_dim = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(audio_feat_dim + text_feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

        self.to(device)

    def forward(self, audio_inputs, text_inputs):
        audio_inputs = audio_inputs.float()

        audio_outputs = self.wav2vec(audio_inputs).last_hidden_state
        audio_feat = audio_outputs.mean(dim=1)

        text_outputs = self.bert(**text_inputs)
        text_feat = text_outputs.pooler_output

        combined = torch.cat((audio_feat, text_feat), dim=1)
        logits = self.classifier(combined)
        return logits

# ---------------- Train & Eval ----------------
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, writer, epoch):
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
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=total_loss / total, accuracy=correct / total)

    avg_loss = total_loss / total
    avg_acc = correct / total

    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_acc, epoch)

    return avg_loss, avg_acc

def eval_epoch(model, dataloader, criterion, device, writer=None, epoch=None, desc="Evaluating"):
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

    avg_loss = total_loss / total
    avg_acc = correct / total

    if writer and epoch is not None:
        tag = 'Val' if desc.lower().startswith('val') else 'Test'
        writer.add_scalar(f'Loss/{tag}', avg_loss, epoch)
        writer.add_scalar(f'Accuracy/{tag}', avg_acc, epoch)

    return avg_loss, avg_acc

# ---------------- Main ----------------
def main():
    json_path = "data/train_audio/audio_v5_0/train.json"
    audio_dir = "data/train_audio/audio_v5_0"

    batch_size = 4
    epochs = 10
    learning_rate = 3e-5
    max_audio_len = 16000 * 10
    max_text_len = 128

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len)
    print(f"Full dataset size: {len(full_dataset)}")

    full_labels = [emotion_labels.get(item['label'], emotion_labels['neutral']) for item in full_dataset.data]

    # Stratified splits: train (70%), val (15%), test (15%)
    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(splitter1.split(full_dataset.data, full_labels))

    temp_labels = [full_labels[i] for i in temp_idx]

    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx_relative, test_idx_relative = next(splitter2.split(temp_idx, temp_labels))

    val_idx = [temp_idx[i] for i in val_idx_relative]
    test_idx = [temp_idx[i] for i in test_idx_relative]

    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}")

    train_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=train_idx)
    val_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=val_idx)
    test_dataset = AudioTextDataset(json_path, audio_dir, max_audio_length=max_audio_len, indices=test_idx)

    processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    tokenizer_text = BertTokenizer.from_pretrained("bert-large-uncased")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor_audio, tokenizer_text, max_audio_len, max_text_len)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor_audio, tokenizer_text, max_audio_len, max_text_len)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, processor_audio, tokenizer_text, max_audio_len, max_text_len)
    )

    model = MultimodalModel(num_classes=num_classes, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Compute class weights to handle imbalance
    labels_array = np.array(full_labels)
    class_weights_np = compute_class_weight(class_weight='balanced', classes=np.unique(labels_array), y=labels_array)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(log_dir="runs/multimodal_experiment")

    best_val_acc = 0
    best_model_path = "best_multimodal_model.pth"

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scaler, writer, epoch)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device, writer, epoch, desc="Validation")

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with val acc: {best_val_acc:.4f}")

    # Load best model for testing
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device, desc="Test")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

    writer.close()

if __name__ == "__main__":
    main()
