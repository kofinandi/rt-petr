# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Train RF-DETR Pose on COCO person keypoints 2017.

The DINOv2 backbone is initialised from HuggingFace pretrained weights.
All other components (transformer decoder, pose head) are trained from scratch.

Usage (single GPU, e.g. GPU 0):
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train_coco_pose.py

Usage (multi-GPU, e.g. GPUs 4-7):
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python train_coco_pose.py --devices 4

For a quick sanity check (a few batches only):
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train_coco_pose.py --limit_train_batches 20 --max_epochs 1

OMP_NUM_THREADS=1 is required for multi-GPU runs: without it, each DDP process
spawns one OpenMP/OpenCV thread per CPU core, quickly exhausting the system thread
limit (EAGAIN / "Can't spawn new thread: res = 11").

Environment:
    conda activate rfdetr
"""

# Limit thread-pool sizes before any library that reads these at import time.
# Must be set via os.environ rather than shell so that DDP child processes also
# inherit the limits.  Each DDP worker spawns its own threadpools; without this
# cap 4 workers × N cores quickly exhausts the system thread limit.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# OpenCV has its own thread pool on top of OMP.  Cap it here before cv2 is
# imported by Albumentations or any other dependency.
try:
    import cv2  # noqa: PLC0415

    cv2.setNumThreads(1)
except ImportError:
    pass

import argparse  # noqa: E402

from rfdetr import RFDETRPose  # noqa: E402

COCO_DIR = "/data/nkoefarago/data/coco"
OUTPUT_DIR = "output/coco_pose_scratch"


def parse_args() -> argparse.Namespace:
    """Parse CLI overrides for the training run.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train RF-DETR Pose on COCO keypoints from scratch.")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs (default: 1).")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device micro-batch size (default: 4).")
    parser.add_argument("--grad_accum_steps", type=int, default=4, help="Gradient accumulation steps (default: 4).")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200).")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (default: 4).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate (default: 1e-4).")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help=f"Output dir (default: {OUTPUT_DIR}).")
    parser.add_argument("--dataset_dir", type=str, default=COCO_DIR, help=f"COCO root dir (default: {COCO_DIR}).")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    """Launch RF-DETR Pose training on COCO keypoints."""
    args = parse_args()

    # pretrain_weights=None → skip the published RF-DETR checkpoint.
    # DINOv2 backbone weights are loaded from HuggingFace automatically because
    # RFDETRPose uses patch_size=14 and positional_encoding_size=37, matching the
    # native DINOv2 configuration exactly (37*14=518px).
    # RFDETRPoseSmall uses patch_size=16 (non-standard), which causes DINOv2 weights
    # to be skipped — use RFDETRPose here to get the pretrained backbone.
    model = RFDETRPose(pretrain_weights=None)

    model.train(
        dataset_dir=args.dataset_dir,
        dataset_file="coco_pose",
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.num_workers,
        lr=args.lr,
        devices=args.devices,
        resume=args.resume,
        # Pose-specific loss coefficients (can be overridden via CLI)
        oks_loss_coef=5.0,
        kpt_l1_loss_coef=0.5,
        kpt_vis_loss_coef=2.0,
        set_cost_oks=2.0,
        # Keep box losses light; they only supervise reference points
        bbox_loss_coef=1.0,
        giou_loss_coef=0.5,
        cls_loss_coef=1.0,
        ia_bce_loss=True,
        use_ema=False,  # EMA disabled initially for simplicity; enable once losses converge
    )


if __name__ == "__main__":
    main()
