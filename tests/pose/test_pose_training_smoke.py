# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke test for the RF-DETR pose estimation full training loop.

Uses a tiny synthetic COCO keypoints dataset and ``fast_dev_run=2`` to verify
the entire PyTorch Lightning training pipeline runs end-to-end without errors,
without needing a real GPU or real COCO data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image


def _make_tiny_coco_pose_dataset(root: Path) -> Path:
    """Create a minimal COCO keypoints dataset for smoke-testing.

    Args:
        root: Temporary directory to write the dataset into.

    Returns:
        Path to the root of the created dataset (same as *root*).
    """
    train_img_dir = root / "train2017"
    val_img_dir = root / "val2017"
    ann_dir = root / "annotations"
    train_img_dir.mkdir(parents=True)
    val_img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    kpt_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    categories = [
        {
            "id": 1,
            "name": "person",
            "supercategory": "person",
            "keypoints": kpt_names,
            "skeleton": [],
        }
    ]

    def _make_split(img_dir: Path, n_images: int = 4) -> dict:
        images, annotations = [], []
        ann_id = 1
        for img_id in range(1, n_images + 1):
            w, h = 320, 240
            fname = f"{img_id:012d}.jpg"
            Image.new("RGB", (w, h), color=(img_id * 20, 80, 120)).save(img_dir / fname)
            images.append({"id": img_id, "file_name": fname, "width": w, "height": h})
            for _ in range(2):  # 2 persons per image
                kpts_flat: list = []
                for j in range(17):
                    x = float(w // 4 + j * 5)
                    y = float(h // 4 + j * 3)
                    kpts_flat.extend([x, y, 2])
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [50.0, 40.0, 100.0, 150.0],
                        "area": 15000.0,
                        "iscrowd": 0,
                        "num_keypoints": 17,
                        "keypoints": kpts_flat,
                        "segmentation": [],
                    }
                )
                ann_id += 1
        return {"images": images, "annotations": annotations, "categories": categories,
                "info": {"year": 2017}, "licenses": []}

    train_data = _make_split(train_img_dir, n_images=4)
    val_data = _make_split(val_img_dir, n_images=2)

    with open(ann_dir / "person_keypoints_train2017.json", "w") as f:
        json.dump(train_data, f)
    with open(ann_dir / "person_keypoints_val2017.json", "w") as f:
        json.dump(val_data, f)

    return root


@pytest.fixture()
def tiny_coco_pose_dir(tmp_path) -> Path:
    """Fixture providing a minimal COCO pose dataset."""
    return _make_tiny_coco_pose_dataset(tmp_path / "coco_pose")


def test_pose_fast_dev_run(tmp_path: Path, tiny_coco_pose_dir: Path) -> None:
    """Smoke-test: full PTL training loop runs without error on a toy pose dataset.

    Uses ``fast_dev_run=2`` (2 batches of train + 2 of val) so the test
    completes in seconds even on CPU.
    """
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    from rfdetr.config import PoseTrainConfig, RFDETRPoseSmallConfig
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    mc = RFDETRPoseSmallConfig(
        pretrain_weights=None,
        amp=False,
        # Small resolution for speed
        resolution=128,
        positional_encoding_size=8,
        num_windows=1,
    )
    tc = PoseTrainConfig(
        dataset_dir=str(tiny_coco_pose_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        grad_accum_steps=1,
        num_workers=0,
        use_ema=False,
        tensorboard=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        drop_path=0.0,
        set_cost_oks=2.0,
        keypoint_loss_coef=5.0,
        oks_loss_coef=2.0,
        vis_loss_coef=1.0,
        eval_interval=1,
        devices=1,
    )

    module = RFDETRModelModule(mc, tc)
    datamodule = RFDETRDataModule(mc, tc)
    # fast_dev_run=2: run 2 train batches + 2 val batches
    trainer = build_trainer(tc, mc, accelerator="auto", fast_dev_run=2)
    trainer.fit(module, datamodule=datamodule)
    # If we get here without exception the smoke test passed
