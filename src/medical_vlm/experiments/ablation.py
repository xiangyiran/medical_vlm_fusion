"""Single-model ablation-study orchestration used by the Colab notebook."""

import numpy as np
import pandas as pd
import torch

from ..config import *
from ..data import test_loader
from ..models import *
from ..training.checkpointing import load_checkpoint_into_model, load_saved_result, save_model_result
from ..training.engine import evaluate_model, train_model
from ..training.profiling import profile_model_cost

def build_bicross_ablation_model(
    cross_attention_mode="bidirectional",
    use_residual_update=True,
    use_layer_norm=True,
):
    """Build a configurable bidirectional cross-attention ablation model."""
    return BidirectionalCrossAttentionAblationModel(
        text_model_name=TEXT_MODEL_NAME,
        vision_model_name=VISION_MODEL_NAME,
        num_labels=len(LABEL_COLS),
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        freeze_image_encoder=FREEZE_IMAGE_ENCODER,
        freeze_text_encoder=FREEZE_TEXT_ENCODER,
        cross_attention_mode=cross_attention_mode,
        use_residual_update=use_residual_update,
        use_layer_norm=use_layer_norm,
    )

def build_ablation_experiments():
    """Create the full bidirectional cross-attention reference and ablation variants."""
    return [
        (
            "bidirectional_cross_attention",
            "Full bidirectional cross-attention",
            "BiCross ablation",
            lambda: BidirectionalCrossAttentionModel(TEXT_MODEL_NAME, VISION_MODEL_NAME, len(LABEL_COLS), HIDDEN_DIM, NUM_HEADS, DROPOUT, FREEZE_IMAGE_ENCODER, FREEZE_TEXT_ENCODER),
        ),
        (
            "bicross_i2t_only",
            "Image-to-text only",
            "BiCross ablation",
            lambda: build_bicross_ablation_model(cross_attention_mode="image_to_text"),
        ),
        (
            "bicross_t2i_only",
            "Text-to-image only",
            "BiCross ablation",
            lambda: build_bicross_ablation_model(cross_attention_mode="text_to_image"),
        ),
        (
            "bicross_no_cross_attention",
            "No cross-attention",
            "BiCross ablation",
            lambda: build_bicross_ablation_model(cross_attention_mode="none"),
        ),
        (
            "bicross_no_residual_update",
            "No residual update",
            "BiCross ablation",
            lambda: build_bicross_ablation_model(cross_attention_mode="bidirectional", use_residual_update=False),
        ),
        (
            "bicross_no_layer_norm",
            "No layer normalization",
            "BiCross ablation",
            lambda: build_bicross_ablation_model(cross_attention_mode="bidirectional", use_layer_norm=False),
        ),
        (
            "bicross_with_alignment_loss",
            "Full bidirectional cross-attention + alignment loss",
            "BiCross ablation",
            lambda: BidirectionalCrossAttentionWithAlignmentLossModel(
                TEXT_MODEL_NAME,
                VISION_MODEL_NAME,
                len(LABEL_COLS),
                HIDDEN_DIM,
                NUM_HEADS,
                DROPOUT,
                FREEZE_IMAGE_ENCODER,
                FREEZE_TEXT_ENCODER,
                alignment_loss_weight=ALIGNMENT_LOSS_WEIGHT,
            ),
        ),
    ]


def run_ablation_experiment(selected_model_key, action):
    """Train or test one selected ablation-study model for the configured seed."""
    experiments = {item[0]: item for item in build_ablation_experiments()}
    model_key, display_name, experiment_group, build_fn = experiments[selected_model_key]
    print("\n" + "=" * 80)
    print("Running:", display_name)
    print("Group:", experiment_group)
    print("Action:", action)
    print("=" * 80)

    model = None
    history = pd.DataFrame()
    test_metrics = None
    if action == "train":
        model = build_fn()
        model, history, test_metrics = train_model(model, model_key, epochs=EPOCHS, lr=LR)
        test_metrics["trained_this_run"] = True
        test_metrics["result_mode"] = "trained"
    else:
        history, test_metrics, result_path = load_saved_result(model_key)
        if test_metrics is not None:
            print(f"Loaded saved result from: {result_path}")
            test_metrics["trained_this_run"] = False
            test_metrics["result_mode"] = "loaded_result"
        else:
            model = build_fn()
            ckpt, ckpt_path = load_checkpoint_into_model(model, model_key)
            if ckpt is None:
                raise FileNotFoundError(f"No saved result or checkpoint found for {model_key}. Run --action train first.")
            history = pd.DataFrame(ckpt.get("history", []))
            model = model.to(DEVICE)
            test_metrics = evaluate_model(model, test_loader, desc=f"{model_key} test")
            test_metrics["best_epoch"] = ckpt.get("epoch", np.nan)
            test_metrics["trained_this_run"] = False
            test_metrics["result_mode"] = "loaded_checkpoint_evaluated"

    test_metrics["model_key"] = model_key
    test_metrics["experiment_group"] = experiment_group
    need_profile = PROFILE_GFLOPS and (
        FORCE_REPROFILE_GFLOPS
        or test_metrics.get("GFLOPs_per_sample", np.nan) is None
        or pd.isna(test_metrics.get("GFLOPs_per_sample", np.nan))
    )
    if need_profile:
        if model is None:
            model = build_fn()
        test_metrics.update(profile_model_cost(model, test_loader, DEVICE))
    save_model_result(model_key, display_name, history, test_metrics)
    if model is not None:
        del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
