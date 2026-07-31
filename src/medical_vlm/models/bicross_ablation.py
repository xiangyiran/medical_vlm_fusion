"""Configurable BiCross direction, residual, and normalization ablation model."""

import torch
import torch.nn as nn

from ..utils import pool_tokens
from .encoders import TwoTowerBase, classifier_block

class BidirectionalCrossAttentionAblationModel(TwoTowerBase):
    """Configurable version of bidirectional cross-attention for ablation."""

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
        cross_attention_mode="bidirectional",
        use_residual_update=True,
        use_layer_norm=True,
    ):
        """Create a cross-attention fusion model with configurable direction and update style."""
        super().__init__(text_model_name, vision_model_name, hidden_dim, freeze_image_encoder, freeze_text_encoder)
        valid_modes = {"bidirectional", "image_to_text", "text_to_image", "none"}
        if cross_attention_mode not in valid_modes:
            raise ValueError(f"cross_attention_mode must be one of {valid_modes}, got {cross_attention_mode!r}")

        self.cross_attention_mode = cross_attention_mode
        self.use_residual_update = use_residual_update
        self.use_layer_norm = use_layer_norm

        use_i2t = cross_attention_mode in {"bidirectional", "image_to_text"}
        use_t2i = cross_attention_mode in {"bidirectional", "text_to_image"}
        self.image_to_text = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True) if use_i2t else None
        self.text_to_image = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True) if use_t2i else None

        self.v_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.t_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.classifier = classifier_block(hidden_dim * 2, num_labels, dropout)

    def _update_tokens(self, original_tokens, attended_tokens, norm_layer):
        """Apply either residual update or direct attention replacement."""
        if attended_tokens is None:
            return norm_layer(original_tokens)
        if self.use_residual_update:
            return norm_layer(original_tokens + attended_tokens)
        return norm_layer(attended_tokens)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Run the selected cross-attention ablation variant."""
        v_tokens, t_tokens, v_mask, t_mask = self.encode_projected_tokens(pixel_values, input_ids, attention_mask)

        v_attn = None
        t_attn = None
        if self.image_to_text is not None:
            v_attn, _ = self.image_to_text(v_tokens, t_tokens, t_tokens, key_padding_mask=~t_mask, need_weights=False)
        if self.text_to_image is not None:
            t_attn, _ = self.text_to_image(t_tokens, v_tokens, v_tokens, key_padding_mask=~v_mask, need_weights=False)

        v_tokens = self._update_tokens(v_tokens, v_attn, self.v_norm)
        t_tokens = self._update_tokens(t_tokens, t_attn, self.t_norm)

        v_feat = pool_tokens(v_tokens, v_mask)
        t_feat = pool_tokens(t_tokens, t_mask)
        logits = self.classifier(torch.cat([v_feat, t_feat], dim=-1))
        return {"logits": logits}
