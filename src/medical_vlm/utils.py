"""Reproducibility, pooling, and frozen-encoder helpers."""

import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .config import USE_AMP

def seed_everything(seed=42):
    """Fix random seeds so that train/validation/test splits and training are more reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def autocast_context():
    """Return the correct mixed-precision context for the current hardware."""
    if USE_AMP:
        return torch.cuda.amp.autocast()
    return nullcontext()

def masked_mean(x, mask, dim=1, eps=1e-6):
    """Compute a mean over valid tokens only."""
    mask = mask.float().unsqueeze(-1)
    return (x * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=eps)

def pool_tokens(tokens, mask):
    """Pool tokens by combining the CLS token and the masked mean of non-CLS tokens."""
    cls_feature = tokens[:, 0, :]
    if tokens.size(1) == 1:
        return cls_feature
    mean_feature = masked_mean(tokens[:, 1:, :], mask[:, 1:])
    return 0.5 * (cls_feature + mean_feature)

def recursive_find_images(root):
    """Find all image files under the dataset root."""
    exts = {".png", ".jpg", ".jpeg"}
    return [p for p in Path(root).rglob("*") if p.suffix.lower() in exts]

def set_frozen_encoders_eval(model):
    """Keep frozen encoders in eval mode after model.train() is called."""
    for attr in ["image_encoder", "text_encoder"]:
        module = getattr(model, attr, None)
        if module is not None and not any(p.requires_grad for p in module.parameters()):
            module.eval()
