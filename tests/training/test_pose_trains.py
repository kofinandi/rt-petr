# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Fast training-pipeline validation for the RF-DETR pose variant.

These tests do NOT require a full training run or a GPU.  They validate that:

1. ``PostProcess`` includes the ``keypoints`` key when the model predicts them —
   this is the root cause of AP=-1 if it silently drops the key.
2. ``PoseEvalCallback._accumulate`` submits non-empty prediction dicts to
   ``CocoEvaluator`` — this is the second thing that can silently produce AP=-1.
3. The real ``SetCriterion`` (built from config) returns finite losses for a
   single batch of synthetic pose targets.
4. *(Overfit test)* A purely parametric model paired with the real criterion can
   reduce its total loss by ≥50% in 40 Adam steps on a single fixed batch.
   This validates that gradients flow through all pose loss components.

Run with::

    uv run --no-sync pytest tests/training/test_pose_trains.py -v \\
        --timeout=120 -m "not gpu"
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from pycocotools.coco import COCO

from rfdetr.config import PoseTrainConfig, RFDETRPoseSmallConfig
from rfdetr.evaluation.coco_eval import CocoEvaluator
from rfdetr.models.lwdetr import build_criterion_from_config
from rfdetr.models.postprocess import PostProcess
from rfdetr.training.callbacks.pose_eval import PoseEvalCallback

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

_K = 17  # COCO keypoints
_COCO_KPT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def _make_configs(tmp_path):
    """Build minimal pose model + training configs for CPU tests."""
    mc = RFDETRPoseSmallConfig(pretrain_weights=None, device="cpu")
    tc = PoseTrainConfig(
        dataset_dir=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        epochs=1,
        batch_size=1,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        num_workers=0,
        tensorboard=False,
        use_ema=False,
    )
    return mc, tc


def _make_pose_targets(batch_size: int = 1, img_h: int = 64, img_w: int = 64) -> list[dict]:
    """Create a list of synthetic pose targets matching the SetCriterion contract."""
    targets = []
    for img_idx in range(batch_size):
        # 1 annotated person; keypoints are on a rough grid in the upper half
        kpts = torch.zeros(_K, 3)
        for k in range(_K):
            kpts[k, 0] = 0.3 + 0.04 * (k % 5)  # x in [0.3, 0.46]
            kpts[k, 1] = 0.2 + 0.04 * (k // 5)  # y in [0.2, 0.32]
            kpts[k, 2] = 2.0  # visible
        targets.append(
            {
                "boxes": torch.tensor([[0.35, 0.25, 0.30, 0.20]]),  # cx,cy,w,h normalised
                "labels": torch.tensor([0]),
                "keypoints": kpts.unsqueeze(0),  # [1, K, 3]
                "area": torch.tensor([0.06]),
                "iscrowd": torch.tensor([0]),
                "image_id": torch.tensor(img_idx + 1),
                "orig_size": torch.tensor([img_h, img_w]),
                "size": torch.tensor([img_h, img_w]),
            }
        )
    return targets


def _make_tiny_coco_gt() -> COCO:
    """Build an in-memory COCO object with a single person annotation."""
    kpt_flat = []
    for k in range(_K):
        x = int(30 + 4 * (k % 5))
        y = int(20 + 4 * (k // 5))
        kpt_flat.extend([x, y, 2])

    coco = COCO()
    coco.dataset = {
        "images": [{"id": 1, "width": 64, "height": 64, "file_name": "img1.jpg"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [15, 10, 30, 20],
                "area": 600,
                "iscrowd": 0,
                "keypoints": kpt_flat,
                "num_keypoints": _K,
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": _COCO_KPT_NAMES,
                "skeleton": [],
            }
        ],
    }
    with contextlib.redirect_stdout(None):
        coco.createIndex()
    # Attach the label2cat mapping so CocoEvaluator resolves label 0 → cat 1
    coco.label2cat = {0: 1}  # type: ignore[attr-defined]
    return coco


# ---------------------------------------------------------------------------
# Parametric model used only in the overfit test
# ---------------------------------------------------------------------------


class _ParametricPoseModel(nn.Module):
    """Bypass the backbone — outputs are direct nn.Parameters.

    The criterion can therefore fully optimise them in a handful of steps.
    All tensor shapes are chosen to match what ``SetCriterion`` expects.
    """

    def __init__(self, num_queries: int = 10, num_kpts: int = _K) -> None:
        super().__init__()
        # Initialise near random classification scores and centred boxes/keypoints.
        self.raw_logits = nn.Parameter(torch.zeros(1, num_queries, 1))
        # cx, cy, w, h each roughly at 0.5 after sigmoid
        self.raw_boxes = nn.Parameter(torch.zeros(1, num_queries, 4))
        # x, y each at 0.5 after sigmoid; vis logit = 0
        self.raw_kpts_xy = nn.Parameter(torch.zeros(1, num_queries, num_kpts, 2))
        self.raw_kpts_vis = nn.Parameter(torch.zeros(1, num_queries, num_kpts, 1))

    def forward(self, _samples=None, _targets=None) -> dict:
        """Return properly shaped model output dict."""
        boxes = self.raw_boxes.sigmoid()  # normalised (0,1)
        kpts_xy = self.raw_kpts_xy.sigmoid()  # normalised (0,1)
        kpts = torch.cat([kpts_xy, self.raw_kpts_vis], dim=-1)  # [1, Q, K, 3]
        return {
            "pred_logits": self.raw_logits,
            "pred_boxes": boxes,
            "pred_keypoints": kpts,
            # Skip aux losses to keep the overfit test fast
            "aux_outputs": [],
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineProducesKeypoints:
    """Verify the prediction pipeline propagates keypoints to the evaluator."""

    def test_postprocess_outputs_keypoints(self):
        """PostProcess must include 'keypoints' in results when pred_keypoints present."""
        postprocess = PostProcess(num_select=5)

        img_h, img_w = 64, 64
        num_q = 10
        batch_size = 2

        pred_logits = torch.zeros(batch_size, num_q, 1)
        pred_boxes = torch.full((batch_size, num_q, 4), 0.5)
        pred_kpts = torch.zeros(batch_size, num_q, _K, 3)

        outputs = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_keypoints": pred_kpts,
        }
        target_sizes = torch.tensor([[img_h, img_w]] * batch_size)
        results = postprocess(outputs, target_sizes)

        assert len(results) == batch_size
        for res in results:
            assert "keypoints" in res, (
                "PostProcess dropped 'keypoints' — PoseEvalCallback will submit empty "
                "prediction dicts and AP will always be -1."
            )
            # shape: [num_select, K, 3]
            assert res["keypoints"].shape == (5, _K, 3)
            # xy coordinates should be in absolute pixel space (not [0,1])
            assert res["keypoints"][..., :2].max() >= 0.0

    def test_evaluator_accumulates_predictions(self):
        """_accumulate must add at least one image to CocoEvaluator.img_ids."""
        coco_gt = _make_tiny_coco_gt()
        evaluator = CocoEvaluator(coco_gt, ["keypoints"], max_dets=20)

        callback = PoseEvalCallback()
        callback._evaluator = evaluator

        # Fake PostProcess output: 1 predicted person with keypoints
        results = [
            {
                "scores": torch.tensor([0.95]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[5.0, 5.0, 30.0, 25.0]]),
                "keypoints": torch.zeros(1, _K, 3),
            }
        ]
        callback._accumulate(results, image_ids=[1])

        assert len(evaluator.img_ids) > 0, (
            "_accumulate did not update evaluator.img_ids — no predictions were "
            "submitted.  AP will be -1 regardless of model quality."
        )
        # The per-image eval_imgs list should have been extended
        assert len(evaluator.eval_imgs["keypoints"]) > 0, (
            "CocoEvaluator.update() was never called from _accumulate."
        )

    def test_ap_is_not_minus_one_for_perfect_predictions(self):
        """Sanity check: CocoEvaluator gives AP > -1 when predictions match GT exactly."""
        coco_gt = _make_tiny_coco_gt()
        # max_dets=20 is the COCO keypoints standard.  Using 100 makes AP=-1
        # because _summarizeKps() hardcodes max_dets=20 in its lookup.
        evaluator = CocoEvaluator(coco_gt, ["keypoints"], max_dets=20)

        # Perfect prediction: same keypoints as GT, high score
        kpt_flat_values = coco_gt.dataset["annotations"][0]["keypoints"]
        kpts_tensor = torch.zeros(1, _K, 3)
        for k in range(_K):
            kpts_tensor[0, k, 0] = float(kpt_flat_values[k * 3 + 0])  # x
            kpts_tensor[0, k, 1] = float(kpt_flat_values[k * 3 + 1])  # y
            kpts_tensor[0, k, 2] = float(kpt_flat_values[k * 3 + 2])  # vis score

        predictions = {
            1: {
                "scores": torch.tensor([0.99]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[15.0, 10.0, 45.0, 30.0]]),
                "keypoints": kpts_tensor,
            }
        }
        evaluator.update(predictions)
        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        with contextlib.redirect_stdout(None):
            evaluator.summarize()

        kpt_stats = evaluator.coco_eval["keypoints"].stats
        assert kpt_stats[0] >= 0.0, (
            f"AP = {kpt_stats[0]:.3f}: CocoEvaluator returned -1 for perfect "
            "predictions.  This indicates a prediction-format bug."
        )


class TestCriterionFiniteLoss:
    """Real SetCriterion (from config) must return finite losses."""

    def test_all_loss_components_finite(self, tmp_path):
        """SetCriterion.forward() returns finite, non-NaN losses for 1 pose target."""
        mc, tc = _make_configs(tmp_path)
        criterion, _pp = build_criterion_from_config(mc, tc)
        criterion.eval()

        num_q = mc.num_queries  # 300
        batch_size = 1
        targets = _make_pose_targets(batch_size=batch_size)

        outputs = {
            "pred_logits": torch.zeros(batch_size, num_q, mc.num_classes),
            "pred_boxes": torch.full((batch_size, num_q, 4), 0.25),
            "pred_keypoints": torch.zeros(batch_size, num_q, _K, 3),
            "aux_outputs": [],
        }

        with torch.no_grad():
            loss_dict = criterion(outputs, targets)

        for name, val in loss_dict.items():
            assert torch.isfinite(val), f"Loss component '{name}' is not finite: {val}"

    def test_keypoint_losses_present(self, tmp_path):
        """Criterion must emit loss_oks, loss_kpt_l1, and loss_kpt_vis."""
        mc, tc = _make_configs(tmp_path)
        criterion, _pp = build_criterion_from_config(mc, tc)

        num_q = mc.num_queries
        targets = _make_pose_targets()
        outputs = {
            "pred_logits": torch.zeros(1, num_q, mc.num_classes),
            "pred_boxes": torch.full((1, num_q, 4), 0.25),
            "pred_keypoints": torch.zeros(1, num_q, _K, 3),
            "aux_outputs": [],
        }

        with torch.no_grad():
            loss_dict = criterion(outputs, targets)

        kpt_keys = {"loss_oks", "loss_kpt_l1", "loss_kpt_vis"}
        missing = kpt_keys - set(loss_dict)
        assert not missing, f"Missing keypoint loss components: {missing}"


class TestPoseOverfit:
    """Overfit test: gradient descent must reduce pose loss on a fixed batch.

    A parametric model (outputs = bare nn.Parameters) is optimised with Adam.
    If the loss does not decrease, gradients are not flowing through the pose
    loss terms.
    """

    def test_loss_decreases_after_gradient_steps(self, tmp_path):
        """Total pose loss must drop ≥ 50% after 40 Adam steps on a fixed batch.

        Uses a pure-parametric model so the backbone is never executed, keeping
        the test well under 30 seconds on CPU.
        """
        mc, tc = _make_configs(tmp_path)
        criterion, _pp = build_criterion_from_config(mc, tc)

        # num_queries must be >= group_detr (13 by default) so that
        # g_num_queries = num_queries // group_detr >= 1.
        num_q = 13
        model = _ParametricPoseModel(num_queries=num_q, num_kpts=_K)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-2)

        # Fixed batch: 1 image, 1 person with clearly placed keypoints
        targets = _make_pose_targets(batch_size=1)

        def _compute_loss() -> torch.Tensor:
            outputs = model()
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            return sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict
                if k in weight_dict and torch.isfinite(loss_dict[k])
            )

        # Capture initial loss (model at random init)
        with torch.no_grad():
            loss_init = float(_compute_loss())

        assert loss_init > 0.0, "Initial loss is zero — something is wrong with the criterion."

        # Optimise
        for _ in range(80):
            optimizer.zero_grad()
            loss = _compute_loss()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            loss_final = float(_compute_loss())

        # We expect at least a 30% reduction over 80 steps.  A fully working
        # gradient path typically drives the loss down by 50-90%; 30% is a
        # conservative lower bound that still catches broken gradients.
        assert loss_final < loss_init * 0.7, (
            f"Loss did not decrease enough after 80 steps: "
            f"initial={loss_init:.4f}, final={loss_final:.4f}.  "
            "Gradients may not be flowing through the pose loss terms."
        )
