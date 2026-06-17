# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""OKS (Object Keypoint Similarity) utilities for pose estimation.

Provides pairwise OKS computation analogous to generalized_box_iou for use in
Hungarian matching and training losses.  Follows the COCO API definition:

    OKS = sum_k [exp(-d_k^2 / (2 s^2 sigma_k^2)) * delta(v_k > 0)]
          -----------------------------------------------------------
                       sum_k [delta(v_k > 0)]

where d_k is the Euclidean distance between predicted and GT keypoint k,
s = sqrt(area) is the object scale, and sigma_k is a per-keypoint constant.
"""

from __future__ import annotations

import torch

# COCO 17-keypoint per-keypoint standard deviations (sigmas), from pycocotools.
# Order: nose, left_eye, right_eye, left_ear, right_ear,
#        left_shoulder, right_shoulder, left_elbow, right_elbow,
#        left_wrist, right_wrist, left_hip, right_hip,
#        left_knee, right_knee, left_ankle, right_ankle
COCO_PERSON_SIGMAS: torch.Tensor = torch.tensor(
    [
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
    ],
    dtype=torch.float32,
)

# Left-right symmetric keypoint index pairs (0-based).
# Swapping these pairs after a HorizontalFlip keeps labels anatomically correct.
COCO_FLIP_PAIRS: list[tuple[int, int]] = [
    (1, 2),    # left_eye  ↔ right_eye
    (3, 4),    # left_ear  ↔ right_ear
    (5, 6),    # left_shoulder ↔ right_shoulder
    (7, 8),    # left_elbow   ↔ right_elbow
    (9, 10),   # left_wrist   ↔ right_wrist
    (11, 12),  # left_hip     ↔ right_hip
    (13, 14),  # left_knee    ↔ right_knee
    (15, 16),  # left_ankle   ↔ right_ankle
]


def pairwise_oks(
    pred_kpts: torch.Tensor,
    tgt_kpts: torch.Tensor,
    tgt_areas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Compute a pairwise OKS matrix between N predicted and M target keypoints.

    Both predicted and target keypoints must use the same coordinate space (either
    fully normalized [0, 1] or absolute pixels).  For normalized coordinates the
    areas must also be normalized (area / (img_w * img_h)).

    Args:
        pred_kpts: Predicted keypoints of shape ``[N, K, 2]`` (x, y).
        tgt_kpts: Target keypoints of shape ``[M, K, 3]`` (x, y, visibility).
            Visibility values: 0 = not labeled, 1 = labeled but occluded,
            2 = labeled and visible.  Keypoints with vis == 0 are excluded from
            the OKS computation.
        tgt_areas: Bounding-box areas for each target instance, shape ``[M]``.
        sigmas: Per-keypoint sigma constants, shape ``[K]``.

    Returns:
        ``[N, M]`` OKS matrix.  Entry ``[i, j]`` is the OKS between prediction
        ``i`` and target ``j``.  Values are in ``[0, 1]``.
    """
    device = pred_kpts.device
    sigmas = sigmas.to(device=device, dtype=pred_kpts.dtype)
    tgt_kpts = tgt_kpts.to(device=device, dtype=pred_kpts.dtype)
    tgt_areas = tgt_areas.to(device=device, dtype=pred_kpts.dtype)

    # Squared Euclidean distances: [N, 1, K, 2] vs [1, M, K, 2] → [N, M, K]
    d_sq = ((pred_kpts[:, None, :, :] - tgt_kpts[None, :, :, :2]) ** 2).sum(-1)

    # Denominator: 2 * s_j^2 * sigma_k^2  →  [1, M, K]
    s_sq = tgt_areas.clamp(min=1e-10)[None, :, None]      # [1, M, 1]
    sigma_sq = (2.0 * sigmas**2)[None, None, :]            # [1, 1, K]
    per_kpt_oks = torch.exp(-d_sq / (s_sq * sigma_sq))     # [N, M, K]

    # Visibility mask: only count keypoints that are labeled
    vis_mask = (tgt_kpts[..., 2] > 0).to(pred_kpts.dtype)  # [M, K]
    numerator = (per_kpt_oks * vis_mask[None, :, :]).sum(-1)    # [N, M]
    denominator = vis_mask.sum(-1).clamp(min=1e-10)[None, :]    # [1, M]
    return numerator / denominator


def oks_loss(
    pred_kpts: torch.Tensor,
    tgt_kpts: torch.Tensor,
    tgt_areas: torch.Tensor,
    sigmas: torch.Tensor,
) -> torch.Tensor:
    """Compute the OKS loss (1 − OKS) for a set of matched prediction–target pairs.

    Args:
        pred_kpts: Predicted keypoints of shape ``[N, K, 2]`` (x, y normalized).
        tgt_kpts: Target keypoints of shape ``[N, K, 3]`` (x, y, visibility).
        tgt_areas: Target bounding-box areas (normalized), shape ``[N]``.
        sigmas: Per-keypoint sigma constants, shape ``[K]``.

    Returns:
        Scalar mean OKS loss (1 − OKS) averaged over all matched pairs.  Returns
        ``0`` when there are no matched pairs.
    """
    if pred_kpts.numel() == 0:
        return pred_kpts.sum()

    device = pred_kpts.device
    sigmas = sigmas.to(device=device, dtype=pred_kpts.dtype)
    tgt_kpts = tgt_kpts.to(device=device, dtype=pred_kpts.dtype)
    tgt_areas = tgt_areas.to(device=device, dtype=pred_kpts.dtype)

    # Squared distances per keypoint: [N, K]
    d_sq = ((pred_kpts - tgt_kpts[..., :2]) ** 2).sum(-1)

    # Denominator per pair and keypoint: [N, K]
    s_sq = tgt_areas.clamp(min=1e-10)[:, None]    # [N, 1]
    sigma_sq = (2.0 * sigmas**2)[None, :]          # [1, K]
    per_kpt_oks = torch.exp(-d_sq / (s_sq * sigma_sq))  # [N, K]

    vis_mask = (tgt_kpts[..., 2] > 0).to(pred_kpts.dtype)  # [N, K]
    numerator = (per_kpt_oks * vis_mask).sum(-1)             # [N]
    denominator = vis_mask.sum(-1).clamp(min=1e-10)          # [N]
    oks = numerator / denominator                             # [N]
    return (1.0 - oks).mean()
