"""Image-only baseline model."""

import torch
import torch.nn as nn

from ..utils import pool_tokens
from .encoders import RadDinoEncoder, classifier_block, freeze_module

class ImageOnlyModel(nn.Module):
    """Image-only baseline using only the visual encoder output."""

    def __init__(self, vision_model_name, num_labels, dropout=0.1, freeze_vision_encoder=True):
        """Create the image encoder and classifier head."""
        super().__init__()
        self.image_encoder = RadDinoEncoder(vision_model_name)
        if freeze_vision_encoder:
            freeze_module(self.image_encoder)
        self.classifier = classifier_block(self.image_encoder.out_dim, num_labels, dropout)

    def forward(self, pixel_values, input_ids=None, attention_mask=None):
        """Predict labels from image features only."""
        v_tokens = self.image_encoder(pixel_values)
        v_mask = torch.ones(v_tokens.shape[:2], dtype=torch.bool, device=v_tokens.device)
        logits = self.classifier(pool_tokens(v_tokens, v_mask))
        return {"logits": logits}
