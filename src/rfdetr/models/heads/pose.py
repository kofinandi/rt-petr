# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pose estimation head for RF-DETR.

Predicts human keypoints (x, y, visibility) for each detection query,
following the YOLO-Pose design: each query produces both a bounding box
*and* a dense keypoint prediction.

Reference: https://arxiv.org/pdf/2204.06806
"""

from torch import nn

from rfdetr.models.math import MLP


class PoseHead(nn.Module):
    """Predicts keypoint coordinates and visibility for each detection query.

    For each decoder query the head outputs ``num_keypoints * 3`` values:
    ``(x, y, visibility)`` per keypoint. ``x`` and ``y`` are normalised to
    ``[0, 1]`` by sigmoid, while ``visibility`` is a raw logit (sigmoid is
    applied in the loss and post-process).

    Args:
        hidden_dim: Decoder hidden dimension.
        num_keypoints: Number of keypoints per instance (17 for COCO).
        num_layers: Number of MLP layers (default: 3).
    """

    def __init__(self, hidden_dim: int, num_keypoints: int = 17, num_layers: int = 3) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        # Predict (x, y, vis) per keypoint
        self.kpt_embed = MLP(hidden_dim, hidden_dim, num_keypoints * 3, num_layers)

        # Initialise last layer near zero so predictions start close to 0.5
        nn.init.constant_(self.kpt_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.kpt_embed.layers[-1].bias.data, 0)

    def forward(self, hs):
        """Predict keypoints from decoder hidden states.

        Args:
            hs: Decoder hidden states of shape ``[num_layers, B, Q, hidden_dim]``
                or ``[B, Q, hidden_dim]`` for a single layer.

        Returns:
            Tensor of shape ``[num_layers, B, Q, K, 3]`` (or ``[B, Q, K, 3]``
            for single-layer input) with ``(x, y, visibility)`` per keypoint.
            ``x`` and ``y`` are in sigmoid space (values in ``[0, 1]``).
            ``visibility`` is a raw logit.
        """
        raw = self.kpt_embed(hs)  # [..., Q, K*3]
        *leading, n_kpt3 = raw.shape
        kpts = raw.reshape(*leading, n_kpt3 // 3, 3)  # [..., Q, K, 3]
        # Sigmoid on x, y; raw logit for visibility
        kpts_xy = kpts[..., :2].sigmoid()
        kpts_vis = kpts[..., 2:3]  # raw logit
        import torch

        return torch.cat([kpts_xy, kpts_vis], dim=-1)
