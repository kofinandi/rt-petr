# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR on COCO 2017 with a pretrained DINOv2 backbone.

Everything except the backbone is initialised from scratch.
The DINOv2 weights are pulled automatically from HuggingFace on first run.

Usage (single GPU, e.g. GPU 1):
    CUDA_VISIBLE_DEVICES=1 python train_coco.py

Usage (multi-GPU, e.g. all 8):
    python train_coco.py --devices 8

For a quick smoke-test that stops after a few batches:
    CUDA_VISIBLE_DEVICES=1 python train_coco.py --max_epochs 1 --limit_train_batches 5

Environment:
    conda activate rfdetr
"""

import argparse

from rfdetr import RFDETRSmall

COCO_DIR = "/data/nkoefarago/data/coco"
OUTPUT_DIR = "output/coco_scratch"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training configuration.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train RF-DETR on COCO 2017 from scratch (DINOv2 backbone only).")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use (default: 1).")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device micro-batch size (default: 4).")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps (default: 4).")
    parser.add_argument("--epochs", type=int, default=100, help="Total training epochs (default: 100).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4).")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help=f"Checkpoint output dir (default: {OUTPUT_DIR}).")
    parser.add_argument("--dataset_dir", type=str, default=COCO_DIR, help=f"COCO root dir (default: {COCO_DIR}).")
    return parser.parse_args()


def main() -> None:
    """Entry point: build model and launch training."""
    args = parse_args()

    # pretrain_weights=None  →  skip the RF-DETR checkpoint;
    #                           DINOv2 backbone still loads from HuggingFace
    #                           (load_dinov2_weights=True is triggered automatically).
    model = RFDETRSmall(pretrain_weights=None)

    model.train(
        dataset_dir=args.dataset_dir,
        dataset_file="coco",          # native COCO 2017 layout (train2017 / val2017 / annotations/)
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        lr=args.lr,
        devices=args.devices,
    )


if __name__ == "__main__":
    main()
