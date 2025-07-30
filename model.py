import torch
import torch.nn as nn

class MultimodalEmpathyModel(nn.Module):
    def __init__(
        self,
        audio_feature_dim: int = 768,
        text_feature_dim: int = 768,
        sentence_emb_dim: int = 384,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()

        # Fully connected layers for each modality
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.bert_text_fc = nn.Sequential(
            nn.Linear(text_feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.sent_trans_fc = nn.Sequential(
            nn.Linear(sentence_emb_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Fusion and classifier
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 + 256 + 128, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)  # logits output, no activation here
        )

    def forward(
        self,
        audio_feat: torch.Tensor,         # shape: (batch_size, audio_feature_dim)
        bert_text_feat: torch.Tensor,     # shape: (batch_size, text_feature_dim)
        sent_trans_feat: torch.Tensor     # shape: (batch_size, sentence_emb_dim)
    ) -> torch.Tensor:
        audio_out = self.audio_fc(audio_feat)
        bert_out = self.bert_text_fc(bert_text_feat)
        sent_out = self.sent_trans_fc(sent_trans_feat)

        fusion = torch.cat((audio_out, bert_out, sent_out), dim=1)
        logits = self.fusion_fc(fusion)
        return logits
