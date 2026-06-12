# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""OKS (Object Keypoint Similarity) utilities for pose estimation."""

from __future__ import annotations

import torch

# COCO 17-keypoint sigmas from pycocotools (kpt_oks_sigmas * 2 for variance)
# Order: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder,
#        left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip,
#        left_knee, right_knee, left_ankle, right_ankle
COCO_SIGMAS: list[float] = [
    0.026,
    0.025,
    0.025,
    0.035,
    0.035,
    0.079,
    0.079,
    0.072,
    0.072,
    0.062,
    0.062,
    0.107,
    0.107,
    0.087,
    0.087,
    0.089,
    0.089,
]

# COCO flip pairs (0-indexed) used to swap left/right keypoints on horizontal flip.
COCO_FLIP_PAIRS: list[tuple[int, int]] = [
    (0, 1),   # left_eye  <-> right_eye
    (2, 3),   # left_ear  <-> right_ear
    (4, 5),   # left_shoulder <-> right_shoulder
    (6, 7),   # left_elbow <-> right_elbow
    (8, 9),   # left_wrist <-> right_wrist
    (10, 11), # left_hip   <-> right_hip
    (12, 13), # left_knee  <-> right_knee
    (14, 15), # left_ankle <-> right_ankle
]


def get_coco_sigmas(device: torch.device | None = None) -> torch.Tensor:
    """Return a [17] tensor of COCO per-keypoint OKS sigmas.

    Args:
        device: Target device.  If ``None`` the tensor is on CPU.

    Returns:
        Float32 tensor of shape ``[17]``.
    """
    t = torch.tensor(COCO_SIGMAS, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t


def compute_oks(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    gt_vis: torch.Tensor,
    gt_areas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Compute OKS for *N* matched prediction–target pairs.

    All keypoint coordinates are expected to be **normalised** to ``[0, 1]``
    (i.e. divided by image width/height).  Areas must also be normalised
    (divided by ``img_w * img_h``).

    Args:
        pred_kpts: Predicted keypoints of shape ``[N, K, 2]`` (x, y).
        gt_kpts: Ground-truth keypoints of shape ``[N, K, 2]`` (x, y).
        gt_vis: Ground-truth visibility of shape ``[N, K]``.  A keypoint is
            included in the OKS sum when ``gt_vis > 0``.
        gt_areas: Ground-truth person areas, normalised, shape ``[N]``.
        sigmas: Per-keypoint OKS sigma values, shape ``[K]``.

    Returns:
        OKS values of shape ``[N]``, one per matched pair.
    """
    # d^2 = squared Euclidean distance in normalised coords
    d_sq = ((pred_kpts - gt_kpts) ** 2).sum(dim=-1)  # [N, K]
    # s^2 = object scale; clamp to avoid division by zero
    s_sq = gt_areas.float().clamp(min=1e-8).unsqueeze(-1)  # [N, 1]
    var = (2.0 * sigmas.to(pred_kpts.device) ** 2).unsqueeze(0)  # [1, K]
    # Per-keypoint OKS contribution
    e = torch.exp(-d_sq / (s_sq * var))  # [N, K]
    visible = (gt_vis > 0).float()  # [N, K]
    num_visible = visible.sum(dim=-1).clamp(min=1.0)  # [N]
    return (e * visible).sum(dim=-1) / num_visible  # [N]


def pairwise_oks(
    pred_kpts: torch.Tensor,
    gt_kpts: torch.Tensor,
    gt_vis: torch.Tensor,
    gt_areas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Compute a pairwise OKS matrix between *Q* predictions and *T* targets.

    Args:
        pred_kpts: Shape ``[Q, K, 2]`` (normalised).
        gt_kpts: Shape ``[T, K, 2]`` (normalised).
        gt_vis: Shape ``[T, K]`` visibility flags.
        gt_areas: Shape ``[T]`` normalised areas.
        sigmas: Shape ``[K]`` per-keypoint sigmas.

    Returns:
        OKS matrix of shape ``[Q, T]``.
    """
    # Expand for broadcasting: [Q, 1, K, 2] vs [1, T, K, 2]
    d_sq = ((pred_kpts.unsqueeze(1) - gt_kpts.unsqueeze(0)) ** 2).sum(dim=-1)  # [Q, T, K]
    s_sq = gt_areas.float().clamp(min=1e-8).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
    var = (2.0 * sigmas.to(pred_kpts.device) ** 2).unsqueeze(0).unsqueeze(0)  # [1, 1, K]
    e = torch.exp(-d_sq / (s_sq * var))  # [Q, T, K]
    visible = (gt_vis > 0).float().unsqueeze(0)  # [1, T, K]
    num_visible = visible.sum(dim=-1).clamp(min=1.0)  # [1, T]
    return (e * visible).sum(dim=-1) / num_visible  # [Q, T]
