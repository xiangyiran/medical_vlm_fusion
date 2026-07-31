"""Bidirectional cross-attention model with alignment loss outputs."""

import torch
import torch.nn as nn

from ..utils import pool_tokens
from .encoders import TwoTowerBase, classifier_block

class BidirectionalCrossAttentionWithAlignmentLossModel(TwoTowerBase):
    """Full bidirectional cross-attention with an extra paired image-text alignment loss."""

    def __init__(
        self,
        text_model_name,
        vision_model_name,
        num_labels,
        hidden_dim=512,
        num_heads=8,
        dropout=0.1,
        freeze_image_encoder=True,
        freeze_text_encoder=True,
        alignment_loss_weight=0.1,
    ):
        """Create the same full bidirectional cross-attention model and expose pooled features for alignment loss."""
        super().__init__(text_model_name, vision_model_name, hidden_dim, freeze_image_encoder, freeze_text_encoder)
        self.alignment_loss_weight = alignment_loss_weight
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.v_norm = nn.LayerNorm(hidden_dim)
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.classifier = classifier_block(hidden_dim * 2, num_labels, dropout)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Return logits plus pooled modality features used by the alignment loss."""
        v_tokens, t_tokens, v_mask, t_mask = self.encode_projected_tokens(pixel_values, input_ids, attention_mask)
        v_attn, _ = self.image_to_text(v_tokens, t_tokens, t_tokens, key_padding_mask=~t_mask, need_weights=False)
        t_attn, _ = self.text_to_image(t_tokens, v_tokens, v_tokens, key_padding_mask=~v_mask, need_weights=False)
        v_tokens = self.v_norm(v_tokens + v_attn)
        t_tokens = self.t_norm(t_tokens + t_attn)
        v_feat = pool_tokens(v_tokens, v_mask)
        t_feat = pool_tokens(t_tokens, t_mask)
        logits = self.classifier(torch.cat([v_feat, t_feat], dim=-1))
        return {
            "logits": logits,
            "image_feat": v_feat,
            "text_feat": t_feat,
        }
