"""Average fusion model."""

from .encoders import TwoTowerBase, classifier_block

class AverageFusionModel(TwoTowerBase):
    """Baseline that averages projected image and text features before classification."""

    def __init__(self, text_model_name, vision_model_name, num_labels, hidden_dim=512, dropout=0.1, freeze_image_encoder=True, freeze_text_encoder=True):
        """Create a two-tower encoder and a classifier over averaged features."""
        super().__init__(text_model_name, vision_model_name, hidden_dim, freeze_image_encoder, freeze_text_encoder)
        self.classifier = classifier_block(hidden_dim, num_labels, dropout)

    def forward(self, pixel_values, input_ids, attention_mask):
        """Fuse modalities by simple averaging."""
        v_feat, t_feat, *_ = self.pooled_features(pixel_values, input_ids, attention_mask)
        logits = self.classifier(0.5 * (v_feat + t_feat))
        return {"logits": logits}
