import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 1. Dataset Class
class MultimodalAudioDataset(Dataset):
    def __init__(self, json_path, label_map, max_audio_len=16000*5):
        self.label_map = label_map
        self.data = []

        with open(json_path, 'r', encoding='utf-8') as f:
            entries = json.load(f)

        for entry in entries:
            audio_path = entry.get("audio_path")
            text = entry.get("text")
            label_str = entry.get("label")

            if audio_path is None or text is None or label_str is None:
                continue

            if label_str not in self.label_map:
                continue

            if not os.path.isfile(audio_path):
                print(f"Warning: audio file not found: {audio_path}")
                continue

            self.data.append({
                "audio_path": audio_path,
                "text": text,
                "label": self.label_map[label_str]
            })

        if not self.data:
            raise RuntimeError(f"No valid data found in {json_path}")

        self.max_audio_len = max_audio_len
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(sample["audio_path"])
        waveform = waveform.mean(dim=0)  # mono if stereo

        # Resample if not 16kHz
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Pad or trim to max_audio_len
        if waveform.size(0) > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]
        else:
            pad_len = self.max_audio_len - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Text tokenize
        encoded = self.tokenizer(sample["text"], padding='max_length', truncation=True, max_length=64, return_tensors="pt")
        input_ids = encoded['input_ids'].squeeze(0)  # remove batch dim
        attention_mask = encoded['attention_mask'].squeeze(0)

        label = sample["label"]

        return {
            "waveform": waveform,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }


# 2. Multimodal Model Definition
class MultimodalClassifier(nn.Module):
    def __init__(self, num_classes, audio_emb_dim=128, text_emb_dim=768):
        super().__init__()
        # Audio feature extractor: simple CNN on raw waveform
        self.audio_cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # (B,16,L/2)
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2), # (B,32,L/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # (B,32,1)
        )
        self.audio_fc = nn.Linear(32, audio_emb_dim)

        # Text feature extractor (DistilBERT)
        self.text_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        for param in self.text_model.parameters():
            param.requires_grad = False  # Freeze BERT

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(audio_emb_dim + text_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, waveform, input_ids, attention_mask):
        # Audio pathway
        x_audio = waveform.unsqueeze(1)  # (B,1,L)
        x_audio = self.audio_cnn(x_audio)  # (B,32,1)
        x_audio = x_audio.squeeze(-1)  # (B,32)
        x_audio = self.audio_fc(x_audio)  # (B, audio_emb_dim)

        # Text pathway
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        x_text = text_outputs.last_hidden_state[:,0,:]  # CLS token embedding (B, text_emb_dim)

        # Concatenate
        x = torch.cat((x_audio, x_text), dim=1)
        out = self.classifier(x)
        return out


# 3. Training loop
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        waveform = batch["waveform"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(waveform, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# 4. Validation loop
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Validation", leave=False)
    with torch.no_grad():
        for batch in pbar:
            waveform = batch["waveform"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(waveform, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(loss=running_loss/total, acc=100.*correct/total)

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# 5. Main function
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, help="Path to JSON data file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoint.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    label_map = {
        "sentimental": 0,
        "anxious": 1,
        "angry": 2,
        "sad": 3,
        "happy": 4,
        "neutral": 5,
    }
    num_classes = len(label_map)

    dataset = MultimodalAudioDataset(args.json_path, label_map)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    model = MultimodalClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train(model, dataloader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f}, accuracy: {train_acc:.2f}%")

        # Save checkpoint if best
        if train_acc > best_acc:
            best_acc = train_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

    print("Training completed.")

if __name__ == "__main__":
    main()
