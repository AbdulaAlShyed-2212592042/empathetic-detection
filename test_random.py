import os
import re
import json
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, BertTokenizer, Wav2Vec2Model, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ================== Emotion Mapping ==================
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

# To track unknown/unmapped labels for debug
unknown_labels = set()

def map_emotion_to_class(label):
    label = label.lower()
    coarse = ed_emotion_projection.get(label, None)
    if coarse is None:
        if label not in emotion_projection:
            unknown_labels.add(label)
        coarse = label if label in emotion_projection else 'happy'
    return emotion_projection.get(coarse, emotion_projection['happy'])

# ================== Dataset ==================
class AudioTextDataset(Dataset):
    def __init__(self, json_data, audio_dir, conversation_ids=None):
        self.audio_dir = audio_dir
        self.data = []
        pattern = re.compile(r"dia(\d+)utt(\d+)_(\d+)\.wav")

        conversations = [conv for conv in json_data if str(conv["conversation_id"]) in conversation_ids]

        turn_lookup = {}
        for conv in conversations:
            conv_id = str(conv["conversation_id"])
            for turn in conv["turns"]:
                turn_id = str(turn["turn_id"])
                turn_lookup[(conv_id, turn_id)] = (conv, turn)

        for filename in tqdm(os.listdir(audio_dir), desc="Loading audio files"):
            if not filename.endswith('.wav'):
                continue
            m = pattern.match(filename)
            if not m:
                continue
            conv_id, turn_id, _ = m.group(1), m.group(2), m.group(3)
            key = (conv_id, turn_id)
            if key in turn_lookup:
                conv, turn = turn_lookup[key]
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
            raise RuntimeError("No valid data found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ================== Collate function ==================
def collate_fn(batch, processor_audio, tokenizer_text):
    audio_paths = [item['audio_path'] for item in batch]
    texts = [item['text'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])

    audio_inputs_list = []
    for path in audio_paths:
        y, sr = librosa.load(path, sr=16000)
        audio_inputs_list.append(y)

    audio_inputs = processor_audio(audio_inputs_list, sampling_rate=16000, return_tensors="pt", padding=True)
    attention_mask = audio_inputs.attention_mask if hasattr(audio_inputs, "attention_mask") else torch.ones_like(audio_inputs.input_values)

    text_inputs = tokenizer_text(texts, padding=True, truncation=True, return_tensors="pt")

    return audio_inputs.input_values, attention_mask, text_inputs, labels

# ================== Model ==================
class MultimodalModel(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)
        self.bert = BertModel.from_pretrained("bert-base-uncased", use_safetensors=True)

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

# ================== Test function ==================
def test_model(json_path, audio_dir, model_path, batch_size=8, save_dir="results_1"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading JSON test data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    conv_ids = [str(conv["conversation_id"]) for conv in conversations]

    test_dataset = AudioTextDataset(conversations, audio_dir, conv_ids)

    # Print unknown labels found during mapping
    if unknown_labels:
        print(f"Warning: Found unknown emotions during mapping: {unknown_labels}")

    if len(test_dataset) > 0:
        first_sample = test_dataset[0]
        print("First sample details:")
        print(f"Audio file: {first_sample['audio_path']}")
        print(f"Text: {first_sample['text']}")
        label_idx = first_sample['label']
        print(f"Label (index): {label_idx}")
        print(f"Label (emotion): {emotion_labels[label_idx]}")
    else:
        print("Test dataset is empty!")
        return

    processor_audio = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    tokenizer_text = BertTokenizer.from_pretrained("bert-base-uncased")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor_audio, tokenizer_text)
    )

    print("Loading model...")
    model = MultimodalModel(num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true, y_pred = [], []
    audio_names = []

    with torch.no_grad():
        for batch_idx, (audio_inputs, audio_attention_mask, text_inputs, labels) in enumerate(tqdm(test_loader, desc="Testing")):
            audio_inputs = audio_inputs.to(device)
            audio_attention_mask = audio_attention_mask.to(device)
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            labels = labels.to(device)

            outputs = model(audio_inputs, audio_attention_mask, text_inputs)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            start_idx = batch_idx * batch_size
            end_idx = start_idx + labels.size(0)
            batch_audio_paths = [test_dataset[i]['audio_path'] for i in range(start_idx, min(end_idx, len(test_dataset)))]
            audio_names.extend(batch_audio_paths)

    # Print distribution of true and predicted labels for debugging
    print("True label distribution:", Counter(y_true))
    print("Predicted label distribution:", Counter(y_pred))

    # Calculate metrics with zero_division=0 to avoid warnings
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    os.makedirs(save_dir, exist_ok=True)

    # Save metrics to a text file
    metrics_path = os.path.join(save_dir, "metrics_1.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    # Save detailed predictions: audio filename, true label, predicted label
    pred_path = os.path.join(save_dir, "predictions_1.txt")
    with open(pred_path, "w", encoding="utf-8") as f:
        f.write("AudioFile\tTrueLabel\tPredLabel\n")
        for audio_path, true_label, pred_label in zip(audio_names, y_true, y_pred):
            audio_file = os.path.basename(audio_path)
            f.write(f"{audio_file}\t{emotion_labels[true_label]}\t{emotion_labels[pred_label]}\n")

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(num_classes),
           yticks=np.arange(num_classes),
           xticklabels=[emotion_labels[i] for i in range(num_classes)],
           yticklabels=[emotion_labels[i] for i in range(num_classes)],
           ylabel='True label',
           xlabel='Predicted label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix_1.jpg")
    plt.savefig(cm_path)
    plt.close(fig)

if __name__ == "__main__":
    json_path = "data/audio_test/test_audio/test.json"
    audio_dir = "data/audio_test/test_audio"
    model_path = "results/best_multimodal_model.pth"

    test_model(json_path, audio_dir, model_path)
