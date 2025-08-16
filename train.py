import os
import json
import datetime
import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, recall_score, precision_score

# ---- CUDA/Backend safety tweaks ----
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# Map emotion labels to integer classes
EMOTION_MAP = {
    'happy': 0, 'surprised': 1, 'angry': 2, 'fear': 3, 'sad': 4, 'disgusted': 5, 'contempt': 6
}

ED_EMOTION_PROJECTION = {
    'conflicted': 'anxious', 'vulnerability': 'afraid', 'helplessness': 'afraid', 'sadness': 'sad',
    'pensive': 'sentimental', 'frustration': 'annoyed', 'weary': 'tired', 'anxiety': 'anxious',
    'reflective': 'sentimental', 'upset': 'disappointed', 'worried': 'anxious', 'fear': 'afraid',
    'frustrated': 'sad', 'fatigue': 'tired', 'lost': 'jealous', 'disappointment': 'disappointed',
    'nostalgia': 'nostalgic', 'exhaustion': 'tired', 'uneasy': 'anxious', 'loneliness': 'lonely',
    'fragile': 'afraid', 'confused': 'jealous', 'vulnerable': 'afraid', 'thoughtful': 'sentimental',
    'stressed': 'anxious', 'concerned': 'anxious', 'tiredness': 'tired', 'burdened': 'anxious',
    'melancholy': 'sad', 'overwhelmed': 'anxious', 'worry': 'anxious', 'heavy-hearted': 'sad',
    'melancholic': 'sad', 'nervous': 'anxious', 'fearful': 'afraid', 'stress': 'anxious',
    'confusion': 'anxious', 'inadequacy': 'ashamed', 'regret': 'guilty', 'helpless': 'afraid',
    'concern': 'anxious', 'exhausted': 'tired', 'overwhelm': 'anxious', 'tired': 'tired',
    'disappointed': 'sad', 'surprised': 'surprised', 'excited': 'happy', 'angry': 'angry',
    'proud': 'happy', 'annoyed': 'angry', 'grateful': 'happy', 'lonely': 'sad', 'afraid': 'fear',
    'terrified': 'fear', 'guilty': 'sad', 'impressed': 'surprised', 'disgusted': 'disgusted',
    'hopeful': 'happy', 'confident': 'happy', 'furious': 'angry', 'anxious': 'sad', 'anticipating': 'happy',
    'joyful': 'happy', 'nostalgic': 'sad', 'prepared': 'happy', 'jealous': 'contempt', 'content': 'happy',
    'devastated': 'surprised', 'embarrassed': 'sad', 'caring': 'happy', 'sentimental': 'sad', 'trusting': 'happy',
    'ashamed': 'sad', 'apprehensive': 'fear', 'faithful': 'happy'
}

META_MAP = {
    'age': {'child': 0, 'young': 1, 'middle-aged': 2, 'elderly': 3},
    'gender': {'male': 0, 'female': 1},
    'timbre': {'high': 0, 'mid': 1, 'low': 2}
}

class MultimodalEmpathyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, audio_dir, tokenizer, processor, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

        valid_data = []
        import librosa
        skipped = 0
        for item in all_data:
            audio_name = item['turn']['dialogue'][-1]['audio_name']
            audio_path = os.path.join(audio_dir, audio_name)
            if os.path.exists(audio_path):
                try:
                    audio_array, sr = librosa.load(audio_path, sr=16000)
                    # Pad or trim to 16000 samples (1 sec)
                    if len(audio_array) < 16000:
                        audio_array = np.pad(audio_array, (0, 16000 - len(audio_array)), mode='constant')
                    else:
                        audio_array = audio_array[:16000]
                    valid_data.append(item)
                except Exception:
                    skipped += 1
        self.data = valid_data
        print(f"Filtered training set: {len(self.data)} samples with valid audio. Skipped {skipped} files that failed to load.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Text: concatenate all utterances in the turn
        text = " ".join([utt['text'] for utt in item['turn']['dialogue'] if utt['text']])
        text_inputs = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt'
        )

        # Audio (last utterance)
        import librosa
        audio_name = item['turn']['dialogue'][-1]['audio_name']
        audio_path = os.path.join(self.audio_dir, audio_name)
        audio_missing = False
        try:
            audio_array, sr = librosa.load(audio_path, sr=16000)
            if len(audio_array) < 16000:
                audio_array = np.pad(audio_array, (0, 16000 - len(audio_array)), mode='constant')
            else:
                audio_array = audio_array[:16000]
        except Exception:
            audio_array = np.zeros((16000,), dtype=np.float32)
            audio_missing = True

        # IMPORTANT: use processor to get normalized float32 in [-1,1]
        audio_inputs = self.processor(audio_array, sampling_rate=16000, return_tensors='pt')

        # Metadata
        age = META_MAP['age'][item['listener_profile']['age']]
        gender = META_MAP['gender'][item['listener_profile']['gender']]
        timbre = META_MAP['timbre'][item['listener_profile']['timbre']]
        meta = torch.tensor([age, gender, timbre], dtype=torch.long)

        # Label mapping
        raw_emotion = item['turn']['chain_of_empathy']['speaker_emotion']
        mapped_emotion = ED_EMOTION_PROJECTION.get(raw_emotion, raw_emotion)
        label = EMOTION_MAP.get(mapped_emotion, 0)

        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'audio': audio_inputs['input_values'].squeeze(0),  # [T]
            'meta': meta,
            'label': torch.tensor(label, dtype=torch.long),
            'audio_missing': audio_missing
        }

def save_terminal_details_json(metadata_list, filename_prefix="terminal_details"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)
    print(f"Saved terminal details to {filename}")

class MultimodalEmpathyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_model = BertModel.from_pretrained('bert-large-uncased')
        self.audio_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-large-960h')
        # Freeze feature extractor (conv stack) — common NaN source with AMP
        for p in self.audio_model.feature_extractor.parameters():
            p.requires_grad = False
        # (Optional) enable gradient checkpointing for memory/stability
        try:
            self.audio_model.gradient_checkpointing_enable()
        except Exception:
            pass

        self.meta_fc = nn.Linear(3, 32)
        self.fc1 = nn.Linear(
            self.text_model.config.hidden_size + self.audio_model.config.hidden_size + 32,
            128
        )
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, audio, meta):
        # Text (BERT)
        text_feat = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)

        # Audio (Wav2Vec2) - force FP32 for stability
        with torch.amp.autocast('cuda', enabled=False):
            audio = audio.to(torch.float32)
            audio_feat = self.audio_model(audio).last_hidden_state.mean(dim=1)

        # Metadata
        meta_feat = self.meta_fc(meta.float())

        # Fusion
        combined = torch.cat((text_feat, audio_feat, meta_feat), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def plot_confusion_matrix(y_true, y_pred, class_names, epoch):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.show()

def train_loop(model, dataloader, optimizer, scheduler, device, class_names=None, epoch=0, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []
    missing_audio_batches = 0
    progress = tqdm(dataloader)

    criterion = nn.CrossEntropyLoss()

    for batch in progress:
        # --- Robust audio tensor checks ---
        audio = batch['audio']
        if audio.dtype != torch.float32:
            audio = audio.float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # ensure [B, T]
        elif audio.dim() > 2:
            print(f"[Error] Audio tensor has invalid shape: {audio.shape}. Skipping batch.")
            missing_audio_batches += 1
            continue
        max_abs = audio.abs().max().item()
        if max_abs > 1.01:
            audio = audio / (max_abs + 1e-8)
        if torch.isnan(audio).any():
            print(f"[Error] NaN detected in audio batch. Skipping batch.")
            missing_audio_batches += 1
            continue
        if torch.isinf(audio).any():
            print(f"[Error] Inf detected in audio batch. Skipping batch.")
            missing_audio_batches += 1
            continue
        # --- End robust audio checks ---

        optimizer.zero_grad(set_to_none=True)

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        audio = audio.to(device, non_blocking=True)
        meta = batch['meta'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        if scaler is not None and device.type == "cuda":
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, audio, meta)
                loss = criterion(logits, labels)
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping backward.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, audio, meta)
            loss = criterion(logits, labels)
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping backward.")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.4f}"})

    print(f"Total batches skipped due to missing/invalid audio: {missing_audio_batches}")
    denom = max(1, len(dataloader) - missing_audio_batches)
    avg_loss = total_loss / denom

    if class_names is not None and len(all_labels) > 0:
        plot_confusion_matrix(all_labels, all_preds, class_names, epoch)

    return avg_loss

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask, audio, meta)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(1, total)
    accuracy = correct / max(1, total)
    f1 = f1_score(all_labels, all_preds, average='weighted') if total > 0 else 0
    recall = recall_score(all_labels, all_preds, average='weighted') if total > 0 else 0
    precision = precision_score(all_labels, all_preds, average='weighted') if total > 0 else 0
    return avg_loss, accuracy, f1, recall, precision

def print_detailed_conversation(sample_raw):
    print(f"Conversation ID: {sample_raw.get('conversation_id')}")
    print(f"Topic: {sample_raw.get('topic')}")
    turn = sample_raw.get('turn', {})
    print(f"Turn ID: {turn.get('turn_id')}")
    print(f"Context: {turn.get('context')}")
    print("Dialogue:")
    for utt in turn.get('dialogue', []):
        print(f"  Index: {utt.get('index')}, Role: {utt.get('role')}")
        print(f"    Text: {utt.get('text')}")
        print(f"    Audio Name: {utt.get('audio_name')}")
        print(f"    Audio Path: {utt.get('audio_path')}")
    print("Chain of Empathy:")
    for k, v in turn.get('chain_of_empathy', {}).items():
        print(f"  {k}: {v}")
    print("Speaker Profile:")
    for k, v in sample_raw.get('speaker_profile', {}).items():
        print(f"  {k}: {v}")
    print("Listener Profile:")
    for k, v in sample_raw.get('listener_profile', {}).items():
        print(f"  {k}: {v}")
    print("---\n")

def print_training_summary(train_raw, batch_size, learning_rate, epochs, model):
    print("\n--- Training Summary ---")
    print(f"Total conversations: {len(train_raw)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Features used: text, audio, metadata (age, gender, timbre)")
    print("Labels: emotion class (7 classes)")
    print("------------------------\n")

def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

def print_all_train_conversations(train_raw):
    print("All conversations used for training:")
    for i, conv in enumerate(train_raw):
        turn = conv.get('turn', {})
        print(f"[{i+1}] ID: {conv.get('conversation_id')}, Topic: {conv.get('topic')}, Emotion: {turn.get('chain_of_empathy', {}).get('speaker_emotion')}")
        print(f"    Context: {turn.get('context')}")
        print(f"    Dialogue turns: {len(turn.get('dialogue', []))}")
    print("---\n")

def print_all_train_conversations_json(train_raw, batch_size, learning_rate, epochs, model, class_names):
    print("All training conversations (JSON format):")
    print(json.dumps(train_raw, ensure_ascii=False, indent=2))
    print(f"Total conversations for training: {len(train_raw)}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {epochs}")
    print(f"Model name: {model.__class__.__name__}")
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Features used: text, audio, metadata (age, gender, timbre)")
    print(f"Label classes: {class_names}")
    print("---\n")

def main():
    # Hyperparameters
    # Advanced hyperparameters
    learning_rates = [5e-6, 1e-5, 2e-5]  # Try multiple LRs
    batch_size = 8  # Increased batch size (adjust if OOM)
    epochs = 50     # More epochs for deeper training
    patience = 8    # Longer patience for early stopping
    grad_clip = 1.0

    # TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=f'runs/exp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h')

    # Stratified split for balanced classes
    import pandas as pd
    from sklearn.model_selection import StratifiedShuffleSplit
    df = pd.read_json('json/mapped_train_data.json')
    labels = df['turn'].apply(lambda x: x['chain_of_empathy']['speaker_emotion'])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, val_idx in sss.split(df, labels):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
    train_df.to_json('json/strat_train.json', orient='records', force_ascii=False)
    val_df.to_json('json/strat_val.json', orient='records', force_ascii=False)
    train_dataset = MultimodalEmpathyDataset(
        'json/strat_train.json',
        'data/train_audio/audio_v5_0',
        tokenizer,
        processor
    )
    val_dataset = MultimodalEmpathyDataset(
        'json/strat_val.json',
        'data/train_audio/audio_v5_0',
        tokenizer,
        processor
    )
    test_dataset = MultimodalEmpathyDataset(
        'json/mapped_test_data.json',
        'data/train_audio/audio_v5_0',
        tokenizer,
        processor
    )

    # For Windows + large I/O, keep workers low; adjust if Linux
    train_loader = DataLoader(
         train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda")
    )

    num_classes = 7
    class_names = ['happy', 'surprised', 'angry', 'fear', 'sad', 'disgusted', 'contempt']

    # Try multiple learning rates
    for lr in learning_rates:
        print(f"\n--- Training with learning rate: {lr} ---")
        model = MultimodalEmpathyModel(num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
        best_loss = float('inf')
        patience_counter = 0
        best_val_acc = 0.0
        torch.autograd.set_detect_anomaly(True)
        with open('json/strat_train.json', 'r', encoding='utf-8') as f:
            train_raw = json.load(f)
        print_all_train_conversations_json(train_raw, batch_size, lr, epochs, model, class_names)
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}")
            start_time = time.time()
            avg_loss = train_loop(model, train_loader, optimizer, scheduler, device, class_names, epoch, scaler, grad_clip)
            val_loss, val_acc, val_f1, val_recall, val_prec = evaluate(model, val_loader, device)
            test_loss, test_acc, test_f1, test_recall, test_prec = evaluate(model, test_loader, device)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} | Time: {epoch_time:.2f}s | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f} | Val Prec: {val_prec:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f} | Test Recall: {test_recall:.4f} | Test Prec: {test_prec:.4f}")
            print(f"Epoch {epoch+1}")
            # TensorBoard logging
            writer.add_scalar(f'Loss/train_lr_{lr}', avg_loss, epoch)
            writer.add_scalar(f'Loss/val_lr_{lr}', val_loss, epoch)
            writer.add_scalar(f'Acc/val_lr_{lr}', val_acc, epoch)
            writer.add_scalar(f'Acc/test_lr_{lr}', test_acc, epoch)
            writer.add_scalar(f'F1/val_lr_{lr}', val_f1, epoch)
            writer.add_scalar(f'F1/test_lr_{lr}', test_f1, epoch)
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f"best_model_lr{lr}.pt")
                print(f"Best model saved at epoch {epoch+1} with Val Acc={val_acc:.4f}")
    # After training loop
        print("\nEvaluating best model on test set...")
        model.load_state_dict(torch.load(f"best_model_lr{lr}.pt", map_location=device))
        test_loss, test_acc, test_f1, test_recall, test_prec = evaluate(model, test_loader, device)
        print(f"Best Model Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f} | Recall: {test_recall:.4f} | Prec: {test_prec:.4f}")
        writer.add_hparams({'lr': lr, 'batch_size': batch_size}, {'test_acc': test_acc, 'test_f1': test_f1})
    writer.close()

if __name__ == '__main__':
    main()
