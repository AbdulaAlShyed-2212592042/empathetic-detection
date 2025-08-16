import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim.swa_utils import AveragedModel, SWALR

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
        # Only keep samples with valid audio .npy files
        valid_data = []
        import librosa
        skipped = 0
        print("Checking audio paths (first 10):")
        for idx, item in enumerate(all_data):
            audio_name = item['turn']['dialogue'][-1]['audio_name']
            audio_path = os.path.join(audio_dir, audio_name)
            if idx < 10:
                print(audio_path)
            if os.path.exists(audio_path):
                try:
                    audio_array, sr = librosa.load(audio_path, sr=16000)
                    # Pad or trim to 16000 samples
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
        text_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        # Audio: load and process audio file for the last utterance in the turn
        import librosa
        audio_name = item['turn']['dialogue'][-1]['audio_name']
        audio_path = os.path.join(self.audio_dir, audio_name)
        audio_missing = False
        try:
            audio_array, sr = librosa.load(audio_path, sr=16000)
            # Pad or trim to 16000 samples (1 second)
            if len(audio_array) < 16000:
                audio_array = np.pad(audio_array, (0, 16000 - len(audio_array)), mode='constant')
            else:
                audio_array = audio_array[:16000]
        except Exception:
            audio_array = np.zeros((16000,), dtype=np.float32)
            audio_missing = True
        audio_inputs = self.processor(audio_array, sampling_rate=16000, return_tensors='pt')
        # Metadata: age, gender, timbre (as integers)
        age = META_MAP['age'][item['listener_profile']['age']]
        gender = META_MAP['gender'][item['listener_profile']['gender']]
        timbre = META_MAP['timbre'][item['listener_profile']['timbre']]
        meta = torch.tensor([age, gender, timbre], dtype=torch.long)
        # Label: map emotion to integer
        raw_emotion = item['turn']['chain_of_empathy']['speaker_emotion']
        mapped_emotion = ED_EMOTION_PROJECTION.get(raw_emotion, raw_emotion)
        label = EMOTION_MAP.get(mapped_emotion, 0)
        # Only return tensors/scalars for batching
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'audio': audio_inputs['input_values'].squeeze(0),
            'meta': meta,
            'label': torch.tensor(label, dtype=torch.long),
            'audio_missing': audio_missing
        }

# Function to save metadata for each batch to JSON after training
import datetime

def save_terminal_details_json(metadata_list, filename_prefix="terminal_details"):
    import json
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
        self.meta_fc = nn.Linear(3, 32)
        self.fc = nn.Linear(self.text_model.config.hidden_size + self.audio_model.config.hidden_size + 32, num_classes)

    def forward(self, input_ids, attention_mask, audio, meta):
        text_feat = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        audio_feat = self.audio_model(audio).last_hidden_state.mean(dim=1)
        meta_feat = self.meta_fc(meta.float())
        combined = torch.cat([text_feat, audio_feat, meta_feat], dim=1)
        logits = self.fc(combined)
        return logits

def plot_confusion_matrix(y_true, y_pred, class_names, epoch):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Epoch {epoch+1}")
    plt.show()

def train_loop(model, dataloader, optimizer, scheduler, device, class_names=None, epoch=0, scaler=None, grad_clip=1.0):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    missing_audio_batches = 0
    progress = tqdm(dataloader)
    for batch in progress:
        if batch['audio'] is None or (hasattr(batch['audio'], 'shape') and batch['audio'].shape[0] == 0):
            missing_audio_batches += 1
            print("Warning: Skipping batch with missing audio.")
            continue
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        audio = batch['audio'].to(device)
        meta = batch['meta'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                logits = model(input_ids, attention_mask, audio, meta)
                loss = nn.CrossEntropyLoss()(logits, labels)
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping backward.")
                continue
            scaler.scale(loss).backward()
            # Gradient clipping for vanishing/exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(input_ids, attention_mask, audio, meta)
            loss = nn.CrossEntropyLoss()(logits, labels)
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
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        progress.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total):.4f}"})
    print(f"Total batches skipped due to missing audio: {missing_audio_batches}")
    avg_loss = total_loss / max(1, len(dataloader) - missing_audio_batches)
    if class_names is not None:
        plot_confusion_matrix(all_labels, all_preds, class_names, epoch)
    return avg_loss

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
    import json
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
    # Hyperparameters (easy to adjust)
    learning_rate = 2e-5  # Lowered LR to prevent NaN loss
    batch_size = 4
    epochs = 30  # Increased to 30 epochs
    patience = 4  # Early stopping patience
    grad_clip = 1.0  # Gradient clipping value
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h')
    train_dataset = MultimodalEmpathyDataset('json/mapped_train_data.json', 'data/train_audio/audio_v5_0', tokenizer, processor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_classes = 7
    class_names = ['happy', 'surprised', 'angry', 'fear', 'sad', 'disgusted', 'contempt']
    model = MultimodalEmpathyModel(num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    best_loss = float('inf')
    patience_counter = 0
    torch.autograd.set_detect_anomaly(True)
    with open('json/mapped_train_data.json', 'r', encoding='utf-8') as f:
        train_raw = json.load(f)
    print_all_train_conversations_json(train_raw, batch_size, learning_rate, epochs, model, class_names)
    for epoch in range(epochs):
        avg_loss = train_loop(model, train_loader, optimizer, scheduler, device, class_names, epoch, scaler, grad_clip)
        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

if __name__ == '__main__':
    main()
