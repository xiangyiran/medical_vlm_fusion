"""Text-only baseline model."""

import torch.nn as nn

from ..utils import pool_tokens
from .encoders import TextEncoder, classifier_block, freeze_module

class TextOnlyModel(nn.Module):
    """Text-only baseline using only the report encoder output."""

    def __init__(self, text_model_name, num_labels, dropout=0.1, freeze_text_encoder=True):
        """Create the text encoder and classifier head."""
        super().__init__()
        self.text_encoder = TextEncoder(text_model_name)
        if freeze_text_encoder:
            freeze_module(self.text_encoder)
        self.classifier = classifier_block(self.text_encoder.out_dim, num_labels, dropout)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None):
        """Predict labels from text features only."""
        t_tokens = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(pool_tokens(t_tokens, attention_mask.bool()))
        return {"logits": logits}
