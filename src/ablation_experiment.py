"""Command-line entry for one ablation-study model and seed."""

import argparse
import os


ABLATION_MODEL_KEYS = (
    "bidirectional_cross_attention",
    "bicross_i2t_only",
    "bicross_t2i_only",
    "bicross_no_cross_attention",
    "bicross_no_residual_update",
    "bicross_no_layer_norm",
    "bicross_with_alignment_loss",
)


def parse_args():
    """Read the seed, model, and action supplied by the ablation Colab notebook."""
    parser = argparse.ArgumentParser(description="Run one original ablation-study model.")
    parser.add_argument("--seed", type=int, choices=(42, 95, 1024), required=True)
    parser.add_argument("--model", choices=ABLATION_MODEL_KEYS, required=True)
    parser.add_argument("--action", choices=("train", "test"), default="train")
    return parser.parse_args()


def main():
    """Configure the seed before importing and running the modular experiment."""
    args = parse_args()
    os.environ["MEDICAL_VLM_SEED"] = str(args.seed)
    from medical_vlm.experiments.ablation import run_ablation_experiment

    run_ablation_experiment(args.model, args.action)


if __name__ == "__main__":
    main()

