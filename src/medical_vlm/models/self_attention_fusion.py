"""Joint self-attention fusion model."""

import torch
import torch.nn as nn

from .encoders import TwoTowerBase, classifier_block

class SelfAttentionFusionModel(TwoTowerBase):
    """Fuse image and text tokens with a joint self-attention encoder."""

    def __init__(self, text_model_name, vision_model_name, num_labels, hidden_dim=512, num_heads=8, num_layers=1, dropout=0.1, freeze_image_encoder=True, freeze_text_encoder=True):
        """Create a Transformer encoder over concatenated image-text tokens."""
        super().__init__(text_model_name, vision_model_name, hidden_dim, freeze_image_encoder, freeze_text_encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True, activation="gelu")
        self.self_attention = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fusion_cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.classifier = classifier_block(hidden_dim, num_labels, dropout)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Run self-attention over all image and text tokens and classify the fusion CLS token."""
        v_tokens, t_tokens, v_mask, t_mask = self.encode_projected_tokens(pixel_values, input_ids, attention_mask)
        batch_size = v_tokens.size(0)
        cls_token = self.fusion_cls.expand(batch_size, -1, -1)
        joint_tokens = torch.cat([cls_token, v_tokens, t_tokens], dim=1)
        cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=joint_tokens.device)
        joint_mask = torch.cat([cls_mask, v_mask, t_mask], dim=1)
        fused_tokens = self.self_attention(joint_tokens, src_key_padding_mask=~joint_mask)
        logits = self.classifier(fused_tokens[:, 0, :])
        return {"logits": logits}
