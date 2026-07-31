"""Command-line entry for one main-study model and seed."""

import argparse
import os


MAIN_MODEL_KEYS = (
    "image_only",
    "text_only",
    "concat_fusion",
    "average_fusion",
    "gated_fusion",
    "self_attention_fusion",
    "bidirectional_cross_attention",
)


def parse_args():
    """Read the seed, model, and action supplied by the main Colab notebook."""
    parser = argparse.ArgumentParser(description="Run one original main-study model.")
    parser.add_argument("--seed", type=int, choices=(42, 95, 1024), required=True)
    parser.add_argument("--model", choices=MAIN_MODEL_KEYS, required=True)
    parser.add_argument("--action", choices=("train", "test"), default="train")
    return parser.parse_args()


def main():
    """Configure the seed before importing and running the modular experiment."""
    args = parse_args()
    os.environ["MEDICAL_VLM_SEED"] = str(args.seed)
    from medical_vlm.experiments.main import run_main_experiment

    run_main_experiment(args.model, args.action)


if __name__ == "__main__":
    main()

