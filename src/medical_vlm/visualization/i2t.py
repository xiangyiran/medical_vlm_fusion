"""Original I2T fusion visualisation functions."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from IPython.display import display

from ..config import *
from ..data import image_processor, test_df, test_ds, tokenizer, train_df, train_ds, val_df, val_ds
from ..models import BidirectionalCrossAttentionAblationModel
from ..training.checkpointing import load_checkpoint_into_model
from ..utils import pool_tokens

MODEL_KEY = "bicross_i2t_only"
VIS_DIR = os.path.join(RESULTS_DIR, "i2t_attention_visualisation")
os.makedirs(VIS_DIR, exist_ok=True)
model = None

def build_i2t_model():
    """Build the image-to-text only ablation model exactly as in the ablation notebook."""
    return BidirectionalCrossAttentionAblationModel(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        num_labels=len(LABEL_COLS),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        freeze_image_encoder=FREEZE_IMAGE_ENCODER,
        freeze_text_encoder=FREEZE_TEXT_ENCODER,
        cross_attention_mode="image_to_text",
        use_residual_update=True,
        use_layer_norm=True,
    )

def load_i2t_visualisation_model():
    """Load the configured seed checkpoint for notebook visualisation."""
    global model
    model = build_i2t_model().to(DEVICE)
    ckpt, loaded_path = load_checkpoint_into_model(model, MODEL_KEY)
    if ckpt is None:
        raise FileNotFoundError(f"No checkpoint found for {MODEL_KEY} in {LOCAL_CKPT_DIR}")
    model.eval()
    print("Loaded checkpoint:", loaded_path)
    print("Checkpoint epoch:", ckpt.get("epoch", "unknown"))
    return model, ckpt, loaded_path

def image_tensor_to_numpy(pixel_values):
    """Convert processor-normalised image tensor back to an approximate displayable RGB image."""
    x = pixel_values.detach().cpu().float().clone()
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    x = x * std + mean
    x = x.clamp(0, 1)
    return x.permute(1, 2, 0).numpy()

def normalise_map(x, eps=1e-8):
    """Min-max normalise one heatmap."""
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + eps)

def tokens_to_patch_grid(scores, remove_cls=True):
    """Map image token scores back to a square patch grid."""
    scores = scores.detach().float().cpu()
    if remove_cls:
        scores = scores[1:]
    n = scores.numel()
    grid_size = int(np.sqrt(n))
    if grid_size * grid_size != n:
        raise ValueError(f"Cannot reshape {n} patch tokens into a square grid.")
    return scores.reshape(grid_size, grid_size).numpy()

def resize_heatmap_to_image(heatmap, image_hw):
    """Resize patch-level heatmap to image size using bilinear interpolation."""
    h, w = image_hw
    heat = torch.tensor(heatmap)[None, None, :, :].float()
    heat = F.interpolate(heat, size=(h, w), mode="bilinear", align_corners=False)
    return heat[0, 0].numpy()

def overlay_heatmap(ax, image_np, heatmap, title, alpha=0.45):
    """Overlay heatmap on the X-ray image."""
    ax.imshow(image_np, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")

def get_i2t_visualisation_outputs(model, batch):
    """Return before-fusion, after-fusion, delta maps, logits, and text-token attention scores."""
    pixel_values = batch["pixel_values"].to(DEVICE)
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)

    with torch.inference_mode():
        v_before, t_tokens, v_mask, t_mask = model.encode_projected_tokens(pixel_values, input_ids, attention_mask)

        # In image-to-text mode: image tokens are query; text tokens are key/value.
        # attn_weights shape: [batch, heads, image_query_tokens, text_key_tokens]
        v_attn, attn_weights = model.image_to_text(
            v_before,
            t_tokens,
            t_tokens,
            key_padding_mask=~t_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        v_after = model._update_tokens(v_before, v_attn, model.v_norm)

        v_feat = pool_tokens(v_after, v_mask)
        t_feat = pool_tokens(t_tokens, t_mask)
        logits = model.classifier(torch.cat([v_feat, t_feat], dim=-1))
        probs = torch.sigmoid(logits)[0].detach().cpu().numpy()

    # Use token vector magnitude as a simple visual representation strength.
    before_score = torch.norm(v_before[0], dim=-1)
    after_score = torch.norm(v_after[0], dim=-1)

    # Delta map is the most direct answer to "how fusion changed image tokens".
    delta_score = torch.norm(v_after[0] - v_before[0], dim=-1)

    before_grid = normalise_map(tokens_to_patch_grid(before_score))
    after_grid = normalise_map(tokens_to_patch_grid(after_score))
    delta_grid = normalise_map(tokens_to_patch_grid(delta_score))

    # Text-token importance: average cross-attention over heads and image patches, excluding image CLS query.
    text_attn = attn_weights[0, :, 1:, :].mean(dim=(0, 1)).detach().cpu()
    valid_len = int(attention_mask[0].sum().item())
    text_attn = text_attn[:valid_len]
    token_ids = input_ids[0, :valid_len].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return {
        "before_grid": before_grid,
        "after_grid": after_grid,
        "delta_grid": delta_grid,
        "probs": probs,
        "text_tokens": tokens,
        "text_attn": text_attn.numpy(),
    }

def show_top_predictions(probs, k=5):
    """Display top predicted labels."""
    top_idx = np.argsort(probs)[::-1][:k]
    rows = [{"label": LABEL_COLS[i], "probability": float(probs[i])} for i in top_idx]
    return pd.DataFrame(rows)

def show_top_text_tokens(tokens, scores, k=12):
    """Display the report tokens most attended by image patches."""
    rows = []
    special = set(tokenizer.all_special_tokens)
    for token, score in zip(tokens, scores):
        if token in special:
            continue
        rows.append({"token": token, "attention_score": float(score)})
    out = pd.DataFrame(rows).sort_values("attention_score", ascending=False).head(k)
    return out.reset_index(drop=True)

def visualise_i2t_sample(sample_index=0, split="test", save=True):
    """Visualise one sample from train/val/test split."""
    if model is None:
        raise RuntimeError("Call load_i2t_visualisation_model() before visualise_i2t_sample().")
    split_df = {"train": train_df, "val": val_df, "validate": val_df, "test": test_df}[split]
    row = split_df.reset_index(drop=True).iloc[sample_index]
    ds = {"train": train_ds, "val": val_ds, "validate": val_ds, "test": test_ds}[split]
    sample = ds[sample_index]
    batch = {k: v.unsqueeze(0) if torch.is_tensor(v) else v for k, v in sample.items()}

    out = get_i2t_visualisation_outputs(model, batch)
    image_np = image_tensor_to_numpy(batch["pixel_values"][0])
    h, w = image_np.shape[:2]

    before = resize_heatmap_to_image(out["before_grid"], (h, w))
    after = resize_heatmap_to_image(out["after_grid"], (h, w))
    delta = resize_heatmap_to_image(out["delta_grid"], (h, w))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")
    overlay_heatmap(axes[1], image_np, before, "Before fusion\nimage token magnitude")
    overlay_heatmap(axes[2], image_np, after, "After image-to-text fusion\nupdated image token magnitude")
    overlay_heatmap(axes[3], image_np, delta, "Fusion-induced change\n||after - before||")
    plt.suptitle(f"{split} sample {sample_index} | uid={row['uid']} | file={row['filename']}", y=1.02)
    plt.tight_layout()

    if save:
        fig_path = os.path.join(VIS_DIR, f"i2t_heatmap_{split}_{sample_index}_uid_{row['uid']}.png")
        plt.savefig(fig_path, dpi=220, bbox_inches="tight")
        print("Saved heatmap:", fig_path)
    plt.show()

    print("Report text:")
    print(row["text"])

    print("\nGround-truth positive labels:")
    positives = [c for c in LABEL_COLS if float(row[c]) > 0]
    print(positives if positives else ["No positive labels"])

    print("\nTop predicted labels:")
    display(show_top_predictions(out["probs"], k=8))

    print("\nText tokens most attended by image patches:")
    display(show_top_text_tokens(out["text_tokens"], out["text_attn"], k=15))

    return out

def find_positive_samples(label_name="Lung Opacity", split="test", max_n=10):
    """Return sample indices where a selected label is positive."""
    split_df = {"train": train_df, "val": val_df, "validate": val_df, "test": test_df}[split].reset_index(drop=True)
    idx = split_df.index[split_df[label_name] > 0].tolist()[:max_n]
    print(f"Found {len(idx)} {split} samples positive for {label_name}:", idx)
    return idx
