import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, Wav2Vec2Processor
import json
import numpy as np
from train import MultimodalEmpathyDataset, MultimodalEmpathyModel, EMOTION_MAP, ED_EMOTION_PROJECTION, META_MAP

class MultimodalEmpathyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, audio_dir, tokenizer, processor, max_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.audio_dir = audio_dir
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Text: concatenate all utterances in the turn
        text = " ".join([utt['text'] for utt in item['turn']['dialogue'] if utt['text']])
        text_inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        # Audio: load and process audio file for the last utterance in the turn
        audio_name = item['turn']['dialogue'][-1]['audio_name']
        audio_path = os.path.join(self.audio_dir, audio_name.replace('.wav', '.npy'))
        # Handle missing audio files gracefully
        audio_array = None
        try:
            audio_array = np.load(audio_path)
        except Exception:
            # If missing, return zeros of expected shape for Wav2Vec2 (16000,)
            audio_array = np.zeros((16000,), dtype=np.float32)
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
        return {
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'audio': audio_inputs['input_values'].squeeze(0),
            'meta': meta,
            'label': torch.tensor(label, dtype=torch.long)
        }

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            meta = batch['meta'].to(device)
            labels = batch['label'].to(device)
            logits = model(input_ids, attention_mask, audio, meta)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total if total > 0 else 0
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-large-960h')
    test_dataset = MultimodalEmpathyDataset('json/mapped_test_data.json', 'data/train_audio/audio_v5_0', tokenizer, processor)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    num_classes = 7
    model = MultimodalEmpathyModel(num_classes).to(device)
    # Load your trained model weights here if you saved them
    # model.load_state_dict(torch.load('path_to_checkpoint.pt'))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    main()
