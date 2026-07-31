"""Learnable gated fusion model."""

import torch
import torch.nn as nn

from .encoders import TwoTowerBase, classifier_block

class GatedFusionModel(TwoTowerBase):
    """Learnable fusion model that dynamically weights image and text features."""

    def __init__(self, text_model_name, vision_model_name, num_labels, hidden_dim=512, dropout=0.1, freeze_image_encoder=True, freeze_text_encoder=True):
        """Create a gate network and classifier for dynamic feature fusion."""
        super().__init__(text_model_name, vision_model_name, hidden_dim, freeze_image_encoder, freeze_text_encoder)
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid())
        self.classifier = classifier_block(hidden_dim, num_labels, dropout)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Fuse pooled features with a learned vector gate."""
        v_feat, t_feat, *_ = self.pooled_features(pixel_values, input_ids, attention_mask)
        gate = self.gate(torch.cat([v_feat, t_feat], dim=-1))
        fused = gate * v_feat + (1.0 - gate) * t_feat
        logits = self.classifier(fused)
        return {"logits": logits, "gate_mean": gate.mean().detach()}
