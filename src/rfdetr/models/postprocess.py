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
    """This module converts the model's output into the format expected by the coco api."""

    def __init__(self, num_select=300) -> None:
        super().__init__()
        self.num_select = num_select

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

        return results


class PosePostProcess(nn.Module):
    """Post-process model outputs for COCO keypoint evaluation.

    Selects the top-*K* scoring person queries and converts their predicted
    bounding boxes and keypoints to absolute image coordinates.

    Args:
        num_select: Maximum number of detections to return per image.
        num_keypoints: Number of keypoints per instance (default 17 for COCO).
    """

    def __init__(self, num_select: int = 300, num_keypoints: int = 17) -> None:
        super().__init__()
        self.num_select = num_select
        self.num_keypoints = num_keypoints

    @torch.no_grad()
    def forward(self, outputs: dict, target_sizes: torch.Tensor) -> list[dict]:
        """Convert model outputs to COCO-eval-compatible detections.

        Args:
            outputs: Model output dict containing ``"pred_logits"``,
                ``"pred_boxes"``, ``"pred_keypoints"``, and ``"pred_kpt_vis"``.
            target_sizes: Original image sizes ``[B, 2]`` (height, width).

        Returns:
            List of per-image dicts, each with keys:

            * ``"scores"`` – top-K confidence scores, shape ``[K]``.
            * ``"labels"`` – predicted class indices, shape ``[K]``.
            * ``"boxes"`` – absolute xyxy boxes, shape ``[K, 4]``.
            * ``"keypoints"`` – absolute (x, y, vis) keypoints, shape ``[K, num_keypoints, 3]``.
        """
        out_logits = outputs["pred_logits"]
        out_bbox = outputs["pred_boxes"]
        out_kpts = outputs.get("pred_keypoints")   # [B, Q, K, 2]
        out_vis = outputs.get("pred_kpt_vis")      # [B, Q, K]

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), self.num_select, dim=1
        )
        scores = topk_values
        topk_boxes_idx = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes_idx.unsqueeze(-1).expand(-1, -1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = []
        for i in range(out_logits.shape[0]):
            h, w = target_sizes[i].tolist()
            res: dict = {"scores": scores[i], "labels": labels[i], "boxes": boxes[i]}
            if out_kpts is not None:
                kpts_i = torch.gather(
                    out_kpts[i],
                    0,
                    topk_boxes_idx[i].unsqueeze(-1).unsqueeze(-1).expand(-1, self.num_keypoints, 2),
                )  # [K_sel, K_kpt, 2]
                kpts_abs = kpts_i * kpts_i.new_tensor([w, h]).unsqueeze(0).unsqueeze(0)
                if out_vis is not None:
                    vis_i = torch.gather(
                        out_vis[i], 0, topk_boxes_idx[i].unsqueeze(-1).expand(-1, self.num_keypoints)
                    )  # [K_sel, K_kpt]
                    vis_score = vis_i.sigmoid()
                    kpts_out = torch.cat([kpts_abs, vis_score.unsqueeze(-1)], dim=-1)  # [K_sel, K_kpt, 3]
                else:
                    kpts_out = torch.cat(
                        [kpts_abs, kpts_abs.new_ones(*kpts_abs.shape[:-1], 1)], dim=-1
                    )
                res["keypoints"] = kpts_out
            results.append(res)

        return results
