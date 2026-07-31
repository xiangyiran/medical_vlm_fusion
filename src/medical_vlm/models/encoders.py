"""Shared RAD-DINO/CXR-BERT encoders and two-tower utilities."""

import torch
import torch.nn as nn
from transformers import AutoModel

from ..utils import pool_tokens

class RadDinoEncoder(nn.Module):
    """Wrap the RAD-DINO vision encoder and return image token embeddings."""

    def __init__(self, vision_model_name):
        """Load a pretrained vision encoder."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(vision_model_name)
        self.out_dim = self.backbone.config.hidden_size

    def forward(self, pixel_values):
        """Encode image tensors into visual tokens."""
        return self.backbone(pixel_values=pixel_values).last_hidden_state

class TextEncoder(nn.Module):
    """Wrap the CXR-BERT text encoder and return text token embeddings."""

    def __init__(self, text_model_name):
        """Load a pretrained text encoder."""
        super().__init__()
        self.backbone = AutoModel.from_pretrained(text_model_name, trust_remote_code=True)
        self.out_dim = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        """Encode token ids into text tokens."""
        return self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

def freeze_module(module):
    """Freeze a module so that its weights are not updated during training."""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()

def classifier_block(input_dim, num_labels, dropout=0.1):
    """Build a small MLP classifier for multi-label prediction."""
    return nn.Sequential(
        nn.Linear(input_dim, input_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(input_dim, num_labels),
    )

class TwoTowerBase(nn.Module):
    """Shared image-text encoder backbone for post-encoder fusion models."""

    def __init__(self, text_model_name, vision_model_name, hidden_dim, freeze_image_encoder=True, freeze_text_encoder=True):
        """Create both encoders and projection layers into a shared hidden dimension."""
        super().__init__()
        self.image_encoder = RadDinoEncoder(vision_model_name)
        self.text_encoder = TextEncoder(text_model_name)

        if freeze_image_encoder:
            freeze_module(self.image_encoder)
        if freeze_text_encoder:
            freeze_module(self.text_encoder)

        self.image_proj = nn.Linear(self.image_encoder.out_dim, hidden_dim)
        self.text_proj = nn.Linear(self.text_encoder.out_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def encode_projected_tokens(self, pixel_values, input_ids, attention_mask):
        """Encode image/text inputs and project both modalities into the same feature space."""
        v_tokens = self.image_proj(self.image_encoder(pixel_values))
        t_tokens = self.text_proj(self.text_encoder(input_ids, attention_mask))
        v_mask = torch.ones(v_tokens.shape[:2], dtype=torch.bool, device=v_tokens.device)
        t_mask = attention_mask.bool()
        return v_tokens, t_tokens, v_mask, t_mask

    def pooled_features(self, pixel_values, input_ids, attention_mask):
        """Return pooled image and text features after projection."""
        v_tokens, t_tokens, v_mask, t_mask = self.encode_projected_tokens(pixel_values, input_ids, attention_mask)
        v_feat = pool_tokens(v_tokens, v_mask)
        t_feat = pool_tokens(t_tokens, t_mask)
        return v_feat, t_feat, v_tokens, t_tokens, v_mask, t_mask
