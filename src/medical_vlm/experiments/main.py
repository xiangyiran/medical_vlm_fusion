"""Single-model main-study orchestration used by the Colab notebook."""

import numpy as np
import pandas as pd
import torch

from ..config import *
from ..data import test_loader
from ..models import *
from ..training.checkpointing import load_checkpoint_into_model, load_saved_result, save_model_result
from ..training.engine import evaluate_model, train_model
from ..training.profiling import profile_model_cost

def build_main_experiments():
    """Create the main comparison list."""
    return [
        (
            "image_only",
            "Image-only",
            "Main comparison",
            lambda: ImageOnlyModel(VISION_MODEL_NAME, len(LABEL_COLS), DROPOUT, FREEZE_IMAGE_ENCODER),
        ),
        (
            "text_only",
            "Text-only",
            "Main comparison",
            lambda: TextOnlyModel(TEXT_MODEL_NAME, len(LABEL_COLS), DROPOUT, FREEZE_TEXT_ENCODER),
        ),
        (
            "concat_fusion",
            "Concatenation fusion",
            "Main comparison",
            lambda: ConcatenationFusionModel(TEXT_MODEL_NAME, VISION_MODEL_NAME, len(LABEL_COLS), HIDDEN_DIM, DROPOUT, FREEZE_IMAGE_ENCODER, FREEZE_TEXT_ENCODER),
        ),
        (
            "average_fusion",
            "Average fusion",
            "Main comparison",
            lambda: AverageFusionModel(TEXT_MODEL_NAME, VISION_MODEL_NAME, len(LABEL_COLS), HIDDEN_DIM, DROPOUT, FREEZE_IMAGE_ENCODER, FREEZE_TEXT_ENCODER),
        ),
        (
            "gated_fusion",
            "Gated fusion",
            "Main comparison",
            lambda: GatedFusionModel(TEXT_MODEL_NAME, VISION_MODEL_NAME, len(LABEL_COLS), HIDDEN_DIM, DROPOUT, FREEZE_IMAGE_ENCODER, FREEZE_TEXT_ENCODER),
        ),
        (
            "self_attention_fusion",
            "Self-attention fusion",
            "Main comparison",
            lambda: SelfAttentionFusionModel(TEXT_MODEL_NAME, VISION_MODEL_NAME, len(LABEL_COLS), HIDDEN_DIM, NUM_HEADS, FUSION_LAYERS, DROPOUT, FREEZE_IMAGE_ENCODER, FREEZE_TEXT_ENCODER),
        ),
        (
            "bidirectional_cross_attention",
            "Bidirectional cross-attention",
            "Main comparison",
            lambda: BidirectionalCrossAttentionModel(TEXT_MODEL_NAME, VISION_MODEL_NAME, len(LABEL_COLS), HIDDEN_DIM, NUM_HEADS, DROPOUT, FREEZE_IMAGE_ENCODER, FREEZE_TEXT_ENCODER),
        ),
    ]


def run_main_experiment(selected_model_key, action):
    """Train or test one selected main-study model for the configured seed."""
    experiments = {item[0]: item for item in build_main_experiments()}
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
            print(f"Loaded checkpoint from: {ckpt_path}")
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
