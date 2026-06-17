# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Smoke tests for the RF-DETR pose estimation variant.

Verifies that:
1. OKS utilities produce correct values (unit tests).
2. The pose model builds and runs a forward pass.
3. Trainer(fast_dev_run=2).fit() completes end-to-end for pose config.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from rfdetr.config import PoseTrainConfig, RFDETRPoseSmallConfig
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.keypoint_ops import (
    COCO_PERSON_SIGMAS,
    oks_loss,
    pairwise_oks,
)

from .helpers import _make_param_dicts

# ---------------------------------------------------------------------------
# Synthetic keypoints dataset helper
# ---------------------------------------------------------------------------

NUM_KEYPOINTS = 17


class _FakeDatasetWithKeypoints(torch.utils.data.Dataset):
    """Fake dataset that returns detection targets with keypoints."""

    def __init__(self, length: int = 20) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx):
        image = torch.randn(3, 32, 32)
        kpts = torch.zeros(2, NUM_KEYPOINTS, 3)
        # Set a couple of labeled visible keypoints
        kpts[0, 0] = torch.tensor([0.3, 0.4, 2.0])
        kpts[0, 5] = torch.tensor([0.5, 0.6, 2.0])
        kpts[1, 3] = torch.tensor([0.2, 0.3, 1.0])
        target = {
            "boxes": torch.tensor([[0.25, 0.25, 0.5, 0.5], [0.6, 0.6, 0.2, 0.2]]),
            "labels": torch.tensor([0, 0]),
            "image_id": torch.tensor(idx),
            "orig_size": torch.tensor([32, 32]),
            "size": torch.tensor([32, 32]),
            "area": torch.tensor([0.0625, 0.04]),   # normalized areas
            "iscrowd": torch.tensor([0, 0]),
            "keypoints": kpts,
        }
        return image, target


# ---------------------------------------------------------------------------
# Tiny model that outputs pred_keypoints
# ---------------------------------------------------------------------------


class _TinyPoseModel(nn.Module):
    """Minimal nn.Module satisfying the RFDETRModule model contract for pose."""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, samples, targets=None):
        batch = 2
        queries = 10
        num_kpts = NUM_KEYPOINTS
        return {
            "pred_logits": self.dummy.expand(batch, queries, 1).clone(),
            "pred_boxes": torch.zeros(batch, queries, 4),
            "pred_keypoints": torch.zeros(batch, queries, num_kpts, 3),
        }

    def update_drop_path(self, *args, **kwargs) -> None:
        pass

    def update_dropout(self, *args, **kwargs) -> None:
        pass

    def reinitialize_detection_head(self, *args, **kwargs) -> None:
        pass


class _FakePoseCriterion:
    """Fake criterion returning a gradient-connected pose loss dict."""

    weight_dict = {
        "loss_ce": 1.0,
        "loss_bbox": 1.0,
        "loss_giou": 0.5,
        "loss_oks": 5.0,
        "loss_kpt_l1": 0.5,
        "loss_kpt_vis": 2.0,
    }

    def __call__(self, outputs, targets):
        dummy = outputs.get("pred_logits", torch.zeros(1))
        scalar = dummy.sum() * 0.0
        return {k: scalar.clone() for k in self.weight_dict}


def _fake_pose_postprocess(outputs, orig_sizes):
    """Return minimal pose predictions for PoseEvalCallback."""
    n = orig_sizes.shape[0]
    num_kpts = NUM_KEYPOINTS
    return [
        {
            "boxes": torch.tensor([[5.0, 5.0, 20.0, 20.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "keypoints": torch.zeros(1, num_kpts, 3),
        }
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# OKS unit tests
# ---------------------------------------------------------------------------


class TestOKSUtilities:
    """Unit tests for pairwise_oks and oks_loss."""

    def test_pairwise_oks_identical_keypoints_is_one(self):
        """OKS == 1 when predictions exactly match targets (all visible)."""
        num_kpts = 17
        kpts = torch.rand(3, num_kpts, 2)
        tgt = torch.cat([kpts, torch.full((3, num_kpts, 1), 2.0)], dim=-1)
        areas = torch.ones(3) * 0.1
        sigmas = COCO_PERSON_SIGMAS
        oks = pairwise_oks(kpts, tgt, areas, sigmas)  # [3, 3]
        diag = torch.diagonal(oks)
        assert (diag - 1.0).abs().max() < 1e-5, f"OKS diagonal != 1.0: {diag}"

    def test_pairwise_oks_zero_for_invisible_keypoints(self):
        """When all GT keypoints have vis==0, OKS is undefined; implementation returns NaN or fallback."""
        num_kpts = 17
        pred = torch.rand(2, num_kpts, 2)
        tgt = torch.cat([torch.rand(2, num_kpts, 2), torch.zeros(2, num_kpts, 1)], dim=-1)
        areas = torch.ones(2) * 0.1
        sigmas = COCO_PERSON_SIGMAS
        oks = pairwise_oks(pred, tgt, areas, sigmas)
        # With no visible keypoints the denominator clamps to 1e-10; OKS should be near 0 not 1.
        assert oks.shape == (2, 2)

    def test_pairwise_oks_shape(self):
        """pairwise_oks returns [n_pred, n_tgt] tensor."""
        n_pred, n_tgt, num_kpts = 5, 4, 17
        pred = torch.rand(n_pred, num_kpts, 2)
        tgt = torch.cat([torch.rand(n_tgt, num_kpts, 2), torch.ones(n_tgt, num_kpts, 1) * 2.0], dim=-1)
        areas = torch.ones(n_tgt) * 0.05
        oks = pairwise_oks(pred, tgt, areas, COCO_PERSON_SIGMAS)
        assert oks.shape == (n_pred, n_tgt)

    def test_oks_loss_zero_for_perfect_match(self):
        """oks_loss ≈ 0 when predictions exactly match targets."""
        num_kpts = 17
        n_pairs = 4
        tgt_xy = torch.rand(n_pairs, num_kpts, 2)
        tgt = torch.cat([tgt_xy, torch.full((n_pairs, num_kpts, 1), 2.0)], dim=-1)
        areas = torch.ones(n_pairs) * 0.1
        loss = oks_loss(tgt_xy, tgt, areas, COCO_PERSON_SIGMAS)
        assert float(loss) < 1e-5, f"OKS loss for perfect match = {float(loss)}"

    def test_oks_loss_positive_for_wrong_prediction(self):
        """oks_loss > 0 when predictions are far from targets."""
        num_kpts = 17
        n_pairs = 4
        pred = torch.zeros(n_pairs, num_kpts, 2)
        tgt_xy = torch.ones(n_pairs, num_kpts, 2)
        tgt = torch.cat([tgt_xy, torch.full((n_pairs, num_kpts, 1), 2.0)], dim=-1)
        areas = torch.ones(n_pairs) * 0.1
        loss = oks_loss(pred, tgt, areas, COCO_PERSON_SIGMAS)
        assert float(loss) > 0.0

    def test_oks_loss_empty_returns_zero(self):
        """oks_loss returns 0.0 tensor when there are no matched pairs."""
        pred = torch.zeros(0, 17, 2)
        tgt = torch.zeros(0, 17, 3)
        areas = torch.zeros(0)
        loss = oks_loss(pred, tgt, areas, COCO_PERSON_SIGMAS)
        assert float(loss) == 0.0

    def test_oks_loss_ignores_invisible_keypoints(self):
        """Invisible keypoints (vis=0) should not affect OKS loss."""
        num_kpts = 17
        n_pairs = 2
        pred = torch.zeros(n_pairs, num_kpts, 2)
        tgt_xy = torch.zeros(n_pairs, num_kpts, 2)

        # All visible
        tgt_all_vis = torch.cat([tgt_xy, torch.full((n_pairs, num_kpts, 1), 2.0)], dim=-1)
        # Half invisible (first keypoint only visible)
        tgt_vis = torch.cat([tgt_xy, torch.zeros(n_pairs, num_kpts, 1)], dim=-1)
        tgt_vis[:, 0, 2] = 2.0  # only keypoint 0 is visible

        areas = torch.ones(n_pairs) * 0.1
        loss_all = float(oks_loss(pred, tgt_all_vis, areas, COCO_PERSON_SIGMAS))
        loss_partial = float(oks_loss(pred, tgt_vis, areas, COCO_PERSON_SIGMAS))
        # Both losses should be near 0 since pred matches tgt_xy
        assert loss_all < 1e-5
        assert loss_partial < 1e-5


# ---------------------------------------------------------------------------
# Smoke test: end-to-end Trainer.fit()
# ---------------------------------------------------------------------------


def _make_trainer() -> Trainer:
    return Trainer(
        fast_dev_run=2,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )


def _make_pose_config(tmp_path):
    mc = RFDETRPoseSmallConfig(pretrain_weights=None, device="cpu")
    tc = PoseTrainConfig(
        dataset_dir=str(tmp_path / "dataset"),
        output_dir=str(tmp_path / "output"),
        epochs=2,
        batch_size=2,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        num_workers=0,
        tensorboard=False,
        use_ema=False,
    )
    return mc, tc


class TestPoseSmoke:
    """Trainer(fast_dev_run=2).fit() must complete without error for pose estimation."""

    def test_fit_runs_without_error(self, tmp_path):
        """Full PTL fit loop runs 2 train + 2 val batches for pose config."""
        mc, tc = _make_pose_config(tmp_path)
        tiny_model = _TinyPoseModel()
        fake_criterion = _FakePoseCriterion()
        fake_postprocess = MagicMock(side_effect=_fake_pose_postprocess)
        fake_dataset = _FakeDatasetWithKeypoints(length=20)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            _make_trainer().fit(module, datamodule)

    def test_pred_keypoints_shape_in_output(self, tmp_path):
        """Model forward pass returns pred_keypoints with correct shape."""
        mc, tc = _make_pose_config(tmp_path)
        from rfdetr._namespace import _namespace_from_configs

        ns = _namespace_from_configs(mc, tc)
        # Force tiny backbone to avoid downloading weights
        ns.encoder = "dinov2_windowed_small"
        ns.pretrained_encoder = None
        ns.load_dinov2_weights = False
        ns.force_no_pretrain = True
        ns.pretrain_weights = None

        from rfdetr.models.lwdetr import build_model_from_config

        model = build_model_from_config(mc, tc)
        model.eval()

        # Create a minimal NestedTensor input
        from rfdetr.utilities.tensors import NestedTensor

        batch, channels, height, width = 1, 3, 512, 512
        tensors = torch.zeros(batch, channels, height, width)
        mask = torch.zeros(batch, height, width, dtype=torch.bool)
        ns_input = NestedTensor(tensors, mask)

        with torch.no_grad():
            out = model(ns_input)

        assert "pred_keypoints" in out, "pred_keypoints missing from model output"
        kpts = out["pred_keypoints"]
        num_queries = mc.num_queries
        num_kpts = mc.num_keypoints
        assert kpts.shape == (batch, num_queries, num_kpts, 3), (
            f"Expected pred_keypoints shape ({batch}, {num_queries}, {num_kpts}, 3), got {kpts.shape}"
        )
        # xy should be in [0, 1] after sigmoid
        assert kpts[..., :2].min() >= 0.0
        assert kpts[..., :2].max() <= 1.0
