"""Original training, validation, and test loops."""

import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from ..config import *
from ..data import criterion, test_loader, train_loader, val_loader
from ..utils import autocast_context, set_frozen_encoders_eval
from .checkpointing import load_checkpoint_into_model, save_checkpoint
from .metrics import compute_metrics

def move_batch_to_device(batch, device):
    """Move one dataloader batch to GPU or CPU."""
    return {
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }

def forward_batch(model, batch):
    """Run one model forward pass with the standard multimodal batch format."""
    return model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

def compute_model_loss(model, outputs, labels):
    """Compute BCE classification loss, optionally with paired image-text alignment loss."""
    main_loss = criterion(outputs["logits"], labels)
    total_loss = main_loss
    loss_parts = {"main_loss": main_loss}

    alignment_weight = float(getattr(model, "alignment_loss_weight", 0.0) or 0.0)
    has_alignment_features = "image_feat" in outputs and "text_feat" in outputs

    if alignment_weight > 0 and has_alignment_features:
        # Cosine alignment encourages each image feature to stay close to its paired report feature.
        image_feat = F.normalize(outputs["image_feat"].float(), dim=-1)
        text_feat = F.normalize(outputs["text_feat"].float(), dim=-1)
        alignment_loss = 1.0 - F.cosine_similarity(image_feat, text_feat, dim=-1).mean()
        weighted_alignment_loss = alignment_weight * alignment_loss

        total_loss = main_loss + weighted_alignment_loss
        loss_parts["alignment_loss"] = alignment_loss
        loss_parts["weighted_alignment_loss"] = weighted_alignment_loss

    loss_parts["loss"] = total_loss
    return total_loss, loss_parts

def evaluate_model(model, loader, desc="eval"):
    """Evaluate one model and return loss plus classification metrics."""
    model.eval()
    total_n = 0
    loss_sums = {}
    y_true_list, y_prob_list = [], []

    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, leave=False):
            batch = move_batch_to_device(batch, DEVICE)
            with autocast_context():
                outputs = forward_batch(model, batch)
                loss, loss_parts = compute_model_loss(model, outputs, batch["labels"])

            batch_size = batch["labels"].size(0)
            total_n += batch_size
            for name, value in loss_parts.items():
                loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size

            y_true_list.append(batch["labels"].detach().cpu().numpy())
            y_prob_list.append(torch.sigmoid(outputs["logits"]).detach().float().cpu().numpy())

    y_true = np.concatenate(y_true_list, axis=0)
    y_prob = np.concatenate(y_prob_list, axis=0)
    metrics = compute_metrics(y_true, y_prob)
    for name, value in loss_sums.items():
        metrics[name] = value / max(total_n, 1)
    return metrics

def train_one_epoch(model, loader, optimizer, scaler):
    """Train one model for one epoch."""
    model.train()
    set_frozen_encoders_eval(model)
    total_n = 0
    loss_sums = {}

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch_to_device(batch, DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            outputs = forward_batch(model, batch)
            loss, loss_parts = compute_model_loss(model, outputs, batch["labels"])

        scaler.scale(loss).backward()
        if GRAD_CLIP is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        batch_size = batch["labels"].size(0)
        total_n += batch_size
        for name, value in loss_parts.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.detach().cpu()) * batch_size

    return {f"train_{name}": value / max(total_n, 1) for name, value in loss_sums.items()}

def train_model(model, model_key, epochs=EPOCHS, lr=LR):
    """Train one model, select the best checkpoint by validation Macro AUROC, and test it."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = []
    best_score = -float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler)
        val_metrics = evaluate_model(model, val_loader, desc=f"{model_key} val")
        score = val_metrics["macro_auroc"]
        if pd.isna(score):
            score = -val_metrics["loss"]

        row = {"epoch": epoch}
        row.update(train_metrics)
        row.update({f"val_{k}": v for k, v in val_metrics.items() if not isinstance(v, dict)})
        history.append(row)
        print(json.dumps(row, indent=2))

        if score > best_score:
            best_score = score
            save_checkpoint(model, model_key, epoch, best_score, history)

    ckpt, ckpt_path = load_checkpoint_into_model(model, model_key)
    print("Loaded best checkpoint:", ckpt_path)
    test_metrics = evaluate_model(model, test_loader, desc=f"{model_key} test")
    test_metrics["best_epoch"] = ckpt.get("epoch", np.nan) if ckpt is not None else np.nan
    test_metrics["best_val_score"] = float(best_score)
    return model, pd.DataFrame(history), test_metrics
