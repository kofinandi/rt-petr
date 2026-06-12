# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pose estimation head for RF-DETR (YOLO-Pose-style per-query keypoint prediction)."""

from __future__ import annotations

import torch
import torch.nn as nn

from rfdetr.models.math import MLP


class PoseHead(nn.Module):
    """Per-query keypoint prediction head.

    Given decoder hidden states ``hs`` of shape ``[L, B, Q, D]``, predicts
    *K* keypoints (x, y in normalised ``[0, 1]`` image space) and a per-keypoint
    visibility logit for each of the *Q* queries at every decoder layer.

    Two separate light-weight heads are applied:

    * **kpt_embed** – 3-layer MLP → ``[L, B, Q, K*2]``, sigmoidised to ``[0, 1]``.
    * **kpt_vis_embed** – linear → ``[L, B, Q, K]`` (raw logits for BCE).

    Args:
        hidden_dim: Decoder hidden size.
        num_keypoints: Number of keypoints per instance (default 17 for COCO).
    """

    def __init__(self, hidden_dim: int, num_keypoints: int = 17) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kpt_embed = MLP(hidden_dim, hidden_dim, num_keypoints * 2, 3)
        self.kpt_vis_embed = nn.Linear(hidden_dim, num_keypoints)

        # Initialise last layer weights/biases to zero for stable early training.
        nn.init.constant_(self.kpt_embed.layers[-1].weight.data, 0.0)
        nn.init.constant_(self.kpt_embed.layers[-1].bias.data, 0.0)
        nn.init.constant_(self.kpt_vis_embed.weight.data, 0.0)
        nn.init.constant_(self.kpt_vis_embed.bias.data, 0.0)

    def forward(
        self, hs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict keypoints and visibility from decoder hidden states.

        Args:
            hs: Decoder hidden states of shape ``[L, B, Q, D]``.

        Returns:
            Tuple of:

            * **pred_kpts** – shape ``[L, B, Q, K, 2]``, normalised ``[0, 1]``
              (x, y) coordinates.
            * **pred_kpt_vis** – shape ``[L, B, Q, K]``, raw visibility logits.
        """
        K = self.num_keypoints
        kpt_logits = self.kpt_embed(hs)  # [L, B, Q, K*2]
        pred_kpts = kpt_logits.reshape(*kpt_logits.shape[:-1], K, 2).sigmoid()
        pred_kpt_vis = self.kpt_vis_embed(hs)  # [L, B, Q, K]
        return pred_kpts, pred_kpt_vis
