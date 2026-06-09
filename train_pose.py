#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Standalone training script for RF-DETR Pose Estimation.

Usage:
    CUDA_VISIBLE_DEVICES=7 python train_pose.py \\
        --coco-path /data/nkoefarago/data/coco \\
        --output-dir output/rfdetr_pose_small \\
        --epochs 150 \\
        --batch-size 8

Or via the Lightning CLI:
    CUDA_VISIBLE_DEVICES=7 rfdetr fit --config configs/rfdetr_pose_small.yaml

This script provides a more direct training entry point that mirrors
what the CLI config does but is easier to iterate on locally.
"""

import argparse
import os
import warnings

# Suppress some noisy deprecation warnings from dependencies
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    parser = argparse.ArgumentParser(description="Train RF-DETR for human pose estimation on COCO Keypoints")
    parser.add_argument("--coco-path", default="/data/nkoefarago/data/coco", help="Path to COCO root directory")
    parser.add_argument("--output-dir", default="output/rfdetr_pose_small", help="Output directory for checkpoints")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--lr-drop", type=int, default=120, help="Epoch at which LR drops 10x")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--model", choices=["small", "medium"], default="small", help="Model size")
    parser.add_argument("--pretrain", default=None, help="Path to pretrained checkpoint (None=train from scratch)")
    parser.add_argument("--resume", default=None, help="Path to Lightning .ckpt to resume from")
    parser.add_argument("--devices", default="7", help="GPU device IDs (e.g. '7' or '0,1')")
    parser.add_argument("--no-multi-scale", action="store_true", help="Disable multi-scale training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Set GPU visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    from rfdetr.config import PoseTrainConfig, RFDETRPoseMediumConfig, RFDETRPoseSmallConfig

    if args.model == "small":
        model_config = RFDETRPoseSmallConfig(pretrain_weights=args.pretrain)
    else:
        model_config = RFDETRPoseMediumConfig(pretrain_weights=args.pretrain)

    train_config = PoseTrainConfig(
        dataset_dir=args.coco_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr * 1.5,
        lr_drop=args.lr_drop,
        num_workers=args.num_workers,
        multi_scale=not args.no_multi_scale,
        square_resize_div_64=True,
        use_ema=True,
        tensorboard=True,
        keypoint_loss_coef=5.0,
        oks_loss_coef=2.0,
        vis_loss_coef=1.0,
        set_cost_oks=2.0,
        cls_loss_coef=2.0,
        ia_bce_loss=False,
        devices=1,
        accelerator="gpu",
        seed=args.seed,
        resume=args.resume,
        eval_interval=5,
    )

    from rfdetr.training.module_data import RFDETRDataModule
    from rfdetr.training.module_model import RFDETRModelModule
    from rfdetr.training.trainer import build_trainer

    print(f"[Train] Model: RF-DETR Pose {args.model.capitalize()}")
    print(f"[Train] COCO data: {args.coco_path}")
    print(f"[Train] Output: {args.output_dir}")
    print(f"[Train] Epochs: {args.epochs}, Batch: {args.batch_size}x{args.grad_accum}={args.batch_size * args.grad_accum} eff.")
    print(f"[Train] Device(s): CUDA_VISIBLE_DEVICES={args.devices}")
    print()

    model_module = RFDETRModelModule(model_config, train_config)
    data_module = RFDETRDataModule(model_config, train_config)
    trainer = build_trainer(train_config, model_config)

    trainer.fit(
        model_module,
        datamodule=data_module,
        ckpt_path=args.resume,
    )
    print("Training complete.")


if __name__ == "__main__":
    main()
