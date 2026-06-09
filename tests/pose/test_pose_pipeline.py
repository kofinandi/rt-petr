# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Integration tests for the RF-DETR pose estimation pipeline.

Tests the full pipeline including model construction, forward pass,
loss computation, OKS matching, and dataset components using synthetic
data (no GPU or real COCO data required).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pose_model_config():
    """Minimal pose model config with no pretrained weights."""
    import warnings

    from rfdetr.config import RFDETRPoseSmallConfig

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return RFDETRPoseSmallConfig(pretrain_weights=None)


@pytest.fixture()
def pose_train_config(tmp_path):
    """Minimal PoseTrainConfig for testing."""
    from rfdetr.config import PoseTrainConfig

    return PoseTrainConfig(
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        devices=1,
        epochs=1,
    )


@pytest.fixture()
def pose_model(pose_model_config, pose_train_config):
    """Build the pose model."""
    import warnings

    from rfdetr.models.lwdetr import build_model_from_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_model_from_config(pose_model_config, pose_train_config)


@pytest.fixture()
def pose_criterion_and_postprocess(pose_model_config, pose_train_config):
    """Build the criterion and postprocessor."""
    import warnings

    from rfdetr.models.lwdetr import build_criterion_from_config

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_criterion_from_config(pose_model_config, pose_train_config)


@pytest.fixture()
def fake_batch():
    """A small synthetic batch compatible with the pose model."""
    from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

    B, C, H, W = 2, 3, 512, 512
    images = [torch.rand(C, H, W) for _ in range(B)]
    samples = nested_tensor_from_tensor_list(images)

    targets = []
    for i in range(B):
        N = 2
        kpts = torch.rand(N, 17, 3)
        kpts[:, :, 2] = (kpts[:, :, 2] * 2).long().float()  # 0, 1, 2
        boxes = torch.rand(N, 4) * 0.4 + 0.05
        # ensure cxcywh with positive wh
        boxes[:, 2:] = boxes[:, :2].abs() * 0.2 + 0.05
        boxes = boxes.clamp(0, 1)
        targets.append(
            {
                "boxes": boxes,
                "labels": torch.zeros(N, dtype=torch.int64),
                "keypoints": kpts,
                "image_id": torch.tensor([i + 1]),
                "area": torch.rand(N) * 5000 + 100,
                "iscrowd": torch.zeros(N, dtype=torch.int64),
                "orig_size": torch.tensor([H, W]),
                "size": torch.tensor([H, W]),
            }
        )
    return samples, targets


# ---------------------------------------------------------------------------
# Tests: Config
# ---------------------------------------------------------------------------


class TestPoseConfig:
    def test_pose_small_config_has_pose_head(self, pose_model_config):
        assert pose_model_config.pose_head is True

    def test_pose_small_config_has_17_keypoints(self, pose_model_config):
        assert pose_model_config.num_keypoints == 17

    def test_pose_small_config_single_class(self, pose_model_config):
        assert pose_model_config.num_classes == 1

    def test_pose_train_config_has_oks_cost(self, pose_train_config):
        assert pose_train_config.set_cost_oks == 2.0

    def test_pose_train_config_dataset_file(self, pose_train_config):
        assert pose_train_config.dataset_file == "coco_pose"


# ---------------------------------------------------------------------------
# Tests: Model
# ---------------------------------------------------------------------------


class TestPoseModel:
    def test_pose_head_is_not_none(self, pose_model):
        from rfdetr.models.heads.pose import PoseHead

        assert isinstance(pose_model.pose_head, PoseHead)

    def test_forward_emits_pred_keypoints(self, pose_model, fake_batch):
        samples, targets = fake_batch
        pose_model.eval()
        with torch.no_grad():
            out = pose_model(samples)
        assert "pred_keypoints" in out
        B, Q = out["pred_logits"].shape[:2]
        assert out["pred_keypoints"].shape == (B, Q, 17, 3)

    def test_pred_keypoints_xy_in_unit_interval(self, pose_model, fake_batch):
        samples, _ = fake_batch
        pose_model.eval()
        with torch.no_grad():
            out = pose_model(samples)
        kpts_xy = out["pred_keypoints"][..., :2]
        assert kpts_xy.min() >= 0.0, "Keypoint x/y should be >= 0"
        assert kpts_xy.max() <= 1.0, "Keypoint x/y should be <= 1"

    def test_aux_outputs_include_pred_keypoints(self, pose_model, fake_batch):
        samples, targets = fake_batch
        pose_model.train()
        out = pose_model(samples, targets)
        for aux in out.get("aux_outputs", []):
            assert "pred_keypoints" in aux

    def test_enc_outputs_do_not_include_pred_keypoints(self, pose_model, fake_batch):
        samples, targets = fake_batch
        pose_model.train()
        out = pose_model(samples, targets)
        if "enc_outputs" in out:
            assert "pred_keypoints" not in out["enc_outputs"]


# ---------------------------------------------------------------------------
# Tests: Criterion
# ---------------------------------------------------------------------------


class TestPoseCriterion:
    def test_losses_include_keypoints(self, pose_criterion_and_postprocess):
        criterion, _ = pose_criterion_and_postprocess
        assert "keypoints" in criterion.losses

    def test_weight_dict_has_kpt_keys(self, pose_criterion_and_postprocess):
        criterion, _ = pose_criterion_and_postprocess
        for key in ("loss_kpt_l1", "loss_kpt_oks", "loss_kpt_vis"):
            assert key in criterion.weight_dict, f"Missing key: {key}"

    def test_loss_computation(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        criterion, _ = pose_criterion_and_postprocess
        pose_model.train()
        out = pose_model(samples, targets)
        losses = criterion(out, targets)
        for key in ("loss_kpt_l1", "loss_kpt_oks", "loss_kpt_vis"):
            assert key in losses, f"Missing loss key: {key}"
            assert torch.isfinite(losses[key]), f"Non-finite loss for {key}"

    def test_kpt_l1_is_non_negative(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        criterion, _ = pose_criterion_and_postprocess
        pose_model.train()
        out = pose_model(samples, targets)
        losses = criterion(out, targets)
        assert losses["loss_kpt_l1"].item() >= 0

    def test_kpt_oks_bounded(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        criterion, _ = pose_criterion_and_postprocess
        pose_model.train()
        out = pose_model(samples, targets)
        losses = criterion(out, targets)
        # OKS loss = 1 - OKS, so should be in [0, 1]
        oks_val = losses["loss_kpt_oks"].item()
        assert 0.0 <= oks_val <= 1.1, f"OKS loss out of expected range: {oks_val}"

    def test_total_loss_is_finite(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        criterion, _ = pose_criterion_and_postprocess
        pose_model.train()
        out = pose_model(samples, targets)
        losses = criterion(out, targets)
        total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        assert torch.isfinite(total)


# ---------------------------------------------------------------------------
# Tests: OKS matching cost
# ---------------------------------------------------------------------------


class TestOKSMatcher:
    def test_batch_oks_cost_shape(self):
        from rfdetr.models.matcher import batch_oks_cost

        P, T, K = 5, 3, 17
        pred = torch.rand(P, K, 3)
        tgt = torch.rand(T, K, 3)
        tgt[:, :, 2] = (tgt[:, :, 2] * 2).long().float()
        boxes = torch.rand(T, 4) * 0.5 + 0.1
        cost = batch_oks_cost(pred, tgt, boxes)
        assert cost.shape == (P, T)

    def test_batch_oks_cost_range(self):
        from rfdetr.models.matcher import batch_oks_cost

        P, T, K = 4, 4, 17
        pred = torch.rand(P, K, 3)
        tgt = torch.rand(T, K, 3)
        tgt[:, :, 2] = 2.0  # all visible
        boxes = torch.rand(T, 4) * 0.5 + 0.1
        cost = batch_oks_cost(pred, tgt, boxes)
        # 1 - OKS; OKS in [0, 1] → cost in [0, 1]
        assert cost.min() >= 0.0
        assert cost.max() <= 1.0 + 1e-5

    def test_perfect_match_has_zero_cost(self):
        from rfdetr.models.matcher import batch_oks_cost

        K = 17
        T = 3
        kpts = torch.rand(T, K, 3)
        kpts[:, :, 2] = 2.0  # all visible
        # pred == tgt means d=0 → OKS=1 → cost=0
        cost = batch_oks_cost(kpts.clone(), kpts.clone(), torch.rand(T, 4) * 0.5 + 0.1)
        assert (cost.diag() < 1e-5).all(), "Perfect match should yield cost≈0"

    def test_oks_matcher_included_in_cost(self):
        from rfdetr.models.matcher import HungarianMatcher

        matcher = HungarianMatcher(
            cost_class=2.0, cost_bbox=5.0, cost_giou=2.0, cost_oks=2.0
        )
        assert matcher.cost_oks == 2.0


# ---------------------------------------------------------------------------
# Tests: Dataset
# ---------------------------------------------------------------------------


def _make_fake_coco_keypoints_annotation(tmp_path: Path, n_images: int = 3) -> tuple[Path, Path]:
    """Write a tiny fake COCO keypoints annotation file and dummy images."""
    # Create image directory and dummy images
    img_dir = tmp_path / "train2017"
    img_dir.mkdir(parents=True)
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir(parents=True)

    images = []
    annotations = []
    ann_id = 1
    for img_id in range(1, n_images + 1):
        w, h = 640, 480
        img_path = img_dir / f"{img_id:012d}.jpg"
        Image.new("RGB", (w, h), color=(img_id * 30, 100, 100)).save(img_path)
        images.append({"id": img_id, "file_name": img_path.name, "width": w, "height": h})
        # Add 1-2 person annotations per image
        for _ in range(2):
            kpts_flat = []
            for _ in range(17):
                kpts_flat.extend([float(w // 2), float(h // 2), 2])  # visible
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [100.0, 80.0, 200.0, 250.0],
                    "area": 50000.0,
                    "iscrowd": 0,
                    "num_keypoints": 17,
                    "keypoints": kpts_flat,
                    "segmentation": [],
                }
            )
            ann_id += 1

    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
                    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle",
                ],
                "skeleton": [],
            }
        ],
        "info": {"year": 2017},
        "licenses": [],
    }
    ann_file = ann_dir / "person_keypoints_train2017.json"
    with open(ann_file, "w") as f:
        json.dump(coco_dict, f)
    return img_dir, ann_file


class TestCocoPoseDataset:
    def test_dataset_loads_keypoints(self, tmp_path):
        from rfdetr.datasets.coco_pose import CocoPoseDetection

        img_dir, ann_file = _make_fake_coco_keypoints_annotation(tmp_path)
        ds = CocoPoseDetection(img_dir, ann_file)
        assert len(ds) > 0
        img, target = ds[0]
        assert "keypoints" in target
        assert target["keypoints"].shape[1] == 17
        assert target["keypoints"].shape[2] == 3

    def test_dataset_keypoints_shape(self, tmp_path):
        from rfdetr.datasets.coco_pose import CocoPoseDetection

        img_dir, ann_file = _make_fake_coco_keypoints_annotation(tmp_path)
        ds = CocoPoseDetection(img_dir, ann_file)
        for i in range(min(3, len(ds))):
            _, tgt = ds[i]
            N = tgt["boxes"].shape[0]
            assert tgt["keypoints"].shape == (N, 17, 3)

    def test_dataset_labels_are_zero(self, tmp_path):
        from rfdetr.datasets.coco_pose import CocoPoseDetection

        img_dir, ann_file = _make_fake_coco_keypoints_annotation(tmp_path)
        ds = CocoPoseDetection(img_dir, ann_file)
        for i in range(min(3, len(ds))):
            _, tgt = ds[i]
            assert (tgt["labels"] == 0).all(), "Person labels should be 0"

    def test_normalize_pose_normalizes_keypoints(self, tmp_path):
        import torch

        from rfdetr.datasets.coco_pose import NormalizePose

        norm = NormalizePose()
        # Fake float image tensor (C, H, W)
        H, W = 256, 320
        img = torch.rand(3, H, W)
        kpts = torch.tensor([[[100.0, 150.0, 2.0], [200.0, 100.0, 1.0]]])
        target = {"keypoints": kpts, "boxes": torch.zeros(1, 4)}
        img_out, tgt_out = norm(img, target)
        # x should be / W, y should be / H
        assert abs(float(tgt_out["keypoints"][0, 0, 0]) - 100.0 / W) < 1e-5
        assert abs(float(tgt_out["keypoints"][0, 0, 1]) - 150.0 / H) < 1e-5
        # visibility unchanged
        assert float(tgt_out["keypoints"][0, 0, 2]) == 2.0


# ---------------------------------------------------------------------------
# Tests: PostProcess
# ---------------------------------------------------------------------------


class TestPosePostProcess:
    def test_postprocess_includes_keypoints(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        _, postprocess = pose_criterion_and_postprocess
        pose_model.eval()
        with torch.no_grad():
            out = pose_model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = postprocess(out, orig_sizes)
        for r in results:
            assert "keypoints" in r

    def test_postprocess_keypoints_denormalized(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        _, postprocess = pose_criterion_and_postprocess
        pose_model.eval()
        with torch.no_grad():
            out = pose_model(samples)
        orig_sizes = torch.stack([t["orig_size"] for t in targets])
        results = postprocess(out, orig_sizes)
        H, W = 512.0, 512.0
        for r in results:
            kpts = r["keypoints"]  # (num_select, 17, 3)
            # x should be in [0, W], y in [0, H]
            assert kpts[:, :, 0].max().item() <= W + 1
            assert kpts[:, :, 1].max().item() <= H + 1
            assert kpts[:, :, 0].min().item() >= 0
            # visibility is probability after sigmoid
            vis = kpts[:, :, 2]
            assert vis.min() >= 0.0
            assert vis.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# Tests: Backward pass (gradient flow)
# ---------------------------------------------------------------------------


class TestPoseGradients:
    def test_gradients_flow_to_pose_head(self, pose_model, pose_criterion_and_postprocess, fake_batch):
        samples, targets = fake_batch
        criterion, _ = pose_criterion_and_postprocess
        pose_model.train()
        out = pose_model(samples, targets)
        losses = criterion(out, targets)
        total = sum(v * criterion.weight_dict[k] for k, v in losses.items() if k in criterion.weight_dict)
        total.backward()
        pose_head_grad = pose_model.pose_head.kpt_embed.layers[-1].weight.grad
        assert pose_head_grad is not None, "Pose head should receive gradients"
        assert torch.isfinite(pose_head_grad).all(), "Pose head gradients should be finite"
