"""Seed-specific checkpoint and result persistence on Google Drive."""

import json
import os

import numpy as np
import pandas as pd
import torch

from ..config import *

def checkpoint_paths(model_key, include_legacy=False):
    """Return checkpoint paths for one model"""
    keys = [model_key]
    return [os.path.join(LOCAL_CKPT_DIR, f"{key}_best.pt") for key in keys]

def save_checkpoint(model, model_key, epoch, best_score, history):
    """Save the best model checkpoint to the dissertation checkpoint folder using the current model key."""
    payload = {
        "epoch": epoch,
        "best_score": best_score,
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": {
            "vision_model_name": VISION_MODEL_NAME,
            "text_model_name": TEXT_MODEL_NAME,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "fusion_layers": FUSION_LAYERS,
            "labels": LABEL_COLS,
        },
    }
    for path in checkpoint_paths(model_key, include_legacy=False):
        torch.save(payload, path)

def load_checkpoint_into_model(model, model_key):
    """Load the best checkpoint for a model"""
    for path in checkpoint_paths(model_key, include_legacy=True):
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=DEVICE)
            model.load_state_dict(ckpt["model_state_dict"])
            return ckpt, path
    return None, None

def json_safe(obj):
    """Convert NumPy values into JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def result_paths(model_key, include_legacy=True):
    """Return result JSON paths inside the current experiment result folder."""
    keys = [model_key]
    return [os.path.join(LOCAL_RESULTS_DIR, f"{key}_result.json") for key in keys]

def load_saved_result(model_key):
    """Load saved metrics and history for a model if they already exist."""
    for result_path in result_paths(model_key, include_legacy=True):
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                payload = json.load(f)
            history_path = result_path.replace("_result.json", "_history.csv")
            history = pd.read_csv(history_path) if os.path.exists(history_path) else pd.DataFrame()
            return history, payload.get("test_metrics", None), result_path
    return pd.DataFrame(), None, None

def save_model_result(model_key, display_name, history, test_metrics):
    """Save metrics and validation history to the current experiment result folder."""
    payload = {
        "model_key": model_key,
        "display_name": display_name,
        "test_metrics": json_safe(test_metrics),
        "config": {
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "fusion_layers": FUSION_LAYERS,
            "freeze_image_encoder": FREEZE_IMAGE_ENCODER,
            "freeze_text_encoder": FREEZE_TEXT_ENCODER,
            "max_text_len": MAX_TEXT_LEN,
            "experiment_id": EXPERIMENT_ID,
            "result_dir": RESULTS_DIR,
        },
    }
    output_dirs = [LOCAL_RESULTS_DIR]

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{model_key}_result.json"), "w") as f:
            json.dump(payload, f, indent=2)
        history.to_csv(os.path.join(out_dir, f"{model_key}_history.csv"), index=False)
