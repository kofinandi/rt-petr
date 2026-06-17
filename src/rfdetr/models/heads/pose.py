# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pose estimation head for predicting keypoints from decoder hidden states.

A lightweight MLP head applied to each decoder layer's hidden states that
directly predicts K keypoints in normalized image coordinates plus a per-keypoint
visibility logit.  Design mirrors the YOLO-Pose direct-regression style while
remaining compatible with the RF-DETR DETR-style decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from rfdetr.models.math import MLP


class PoseHead(nn.Module):
    """MLP head that predicts keypoints from decoder hidden states.

    For each decoder layer and each object query the head outputs K keypoints in
    normalized image space plus K visibility logits:

    - ``pred_keypoints[..., :2]`` – (x, y) in ``[0, 1]``, applied via sigmoid.
    - ``pred_keypoints[..., 2]``  – raw visibility logit (no activation; BCE loss).

    Args:
        hidden_dim: Transformer decoder hidden dimension.
        num_keypoints: Number of keypoints to predict (17 for COCO person).
        num_mlp_layers: Depth of the MLP head.  Defaults to ``3``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_keypoints: int,
        num_mlp_layers: int = 3,
    ) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.kpt_embed = MLP(hidden_dim, hidden_dim, num_keypoints * 3, num_mlp_layers)
        nn.init.constant_(self.kpt_embed.layers[-1].weight.data, 0.0)
        nn.init.constant_(self.kpt_embed.layers[-1].bias.data, 0.0)

    def forward(self, hs: torch.Tensor) -> list[torch.Tensor]:
        """Predict keypoints from stacked decoder hidden states.

        Args:
            hs: Decoder hidden states of shape ``[num_dec_layers, B, Q, C]``.

        Returns:
            List of length ``num_dec_layers``, each element a ``[B, Q, K, 3]``
            tensor with (x_normalized, y_normalized, vis_logit) per keypoint.
        """
        raw = self.kpt_embed(hs)             # [num_layers, batch, queries, K*3]
        num_layers, batch, queries, _ = raw.shape
        raw = raw.view(num_layers, batch, queries, self.num_keypoints, 3)
        xy = raw[..., :2].sigmoid()          # [num_layers, batch, queries, K, 2]  ∈ [0, 1]
        vis = raw[..., 2:3]                  # [num_layers, batch, queries, K, 1]  raw logit
        kpts = torch.cat([xy, vis], dim=-1)  # [num_layers, batch, queries, K, 3]
        return [kpts[i] for i in range(num_layers)]
