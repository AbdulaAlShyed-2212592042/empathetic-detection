import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from typing import List

class AudioFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        self.model.eval()

    def extract(self, audio_tensor: torch.Tensor, sampling_rate: int) -> torch.Tensor:
        """
        Extract audio features from waveform tensor.
        Args:
            audio_tensor: 1D tensor (num_samples,)
            sampling_rate: Sampling rate of the audio
        Returns:
            Tensor of shape (1, hidden_size)
        """
        inputs = self.processor(audio_tensor, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch_size=1, seq_len, hidden_size)
            pooled = hidden_states.mean(dim=1)  # mean pooling over seq_len
        return pooled  # (1, hidden_size)

class TextFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.model.eval()

    def extract(self, text_list: List[str]) -> torch.Tensor:
        """
        Extract BERT features for a list of texts.
        Args:
            text_list: List of strings
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        inputs = self.tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            pooled = hidden_states[:, 0, :]  # CLS token pooling
        return pooled  # (batch_size, hidden_size)

class SentenceTransformerExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def extract(self, text_list: List[str]) -> torch.Tensor:
        """
        Extract sentence embeddings using SentenceTransformer.
        Args:
            text_list: List of strings
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        embeddings = self.model.encode(text_list, convert_to_tensor=True, device=self.device)
        return embeddings  # (batch_size, hidden_size)
