# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Extracted from lwdetr.py (Phase 10)
# Original copyrights: LW-DETR (Baidu), Conditional DETR (Microsoft),
# DETR (Facebook), Deformable DETR (SenseTime)
# ------------------------------------------------------------------------
"""Post-processing module for converting model outputs to COCO API format."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from rfdetr.utilities import box_ops


class PostProcess(nn.Module):
    """This module converts the model's output into the format expected by the coco api.

    Args:
        num_select: Maximum number of detections returned per image.
        pose_head: When ``True``, also gather and denormalise the top-K
            keypoint predictions emitted by the pose head.
    """

    def __init__(self, num_select: int = 300, pose_head: bool = False) -> None:
        super().__init__()
        self.num_select = num_select
        self.pose_head = pose_head

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation) For
                          visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        out_masks = outputs.get("pred_masks", None)
        out_keypoints = outputs.get("pred_keypoints", None)  # (B, Q, K, 3) or None

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Optionally gather masks corresponding to the same top-K queries and resize to original size
        results = []
        if out_masks is not None:
            for i in range(out_masks.shape[0]):
                res_i = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
                k_idx = topk_boxes[i]
                masks_i = torch.gather(
                    out_masks[i],
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).repeat(1, out_masks.shape[-2], out_masks.shape[-1]),
                )  # [K, Hm, Wm]
                h, w = target_sizes[i].tolist()
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(int(h), int(w)),
                    mode="bilinear",
                    align_corners=False,
                )  # [K,1,H,W]
                res_i["masks"] = masks_i > 0.0
                results.append(res_i)
        else:
            results = [
                {"scores": score, "labels": label, "boxes": box} for score, label, box in zip(scores, labels, boxes)
            ]

        # Gather and denormalise keypoints when pose_head is active
        if out_keypoints is not None:
            # out_keypoints: (B, Q, K, 3) with (x,y) in [0,1]
            for i, res_i in enumerate(results):
                k_idx = topk_boxes[i]  # (num_select,)
                # Gather the matching query keypoints
                kpts_i = out_keypoints[i]  # (Q, K, 3)
                k, kd = kpts_i.shape[1], kpts_i.shape[2]
                kpts_sel = torch.gather(
                    kpts_i,
                    0,
                    k_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, k, kd),
                )  # (num_select, K, 3)
                # Denormalise x, y to absolute pixel coordinates
                h_i, w_i = target_sizes[i][0].item(), target_sizes[i][1].item()
                kpts_sel = kpts_sel.clone()
                kpts_sel[:, :, 0] *= w_i
                kpts_sel[:, :, 1] *= h_i
                # Convert visibility logit → probability
                kpts_sel[:, :, 2] = kpts_sel[:, :, 2].sigmoid()
                res_i["keypoints"] = kpts_sel  # (num_select, K, 3): (x,y,vis_prob)

        return results
