# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Pose-evaluation callback: logs COCO keypoint AP using pycocotools."""

from __future__ import annotations

import json
import logging
import tempfile
from typing import TYPE_CHECKING, Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from rfdetr.utilities.distributed import get_rank, get_world_size, is_dist_avail_and_initialized
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger()


class PoseEvalCallback(Callback):
    """Compute COCO keypoint AP at the end of every validation/test epoch.

    Accumulates per-image predictions during validation, gathers them across
    distributed ranks, and evaluates against the ground-truth COCO API object
    using :class:`pycocotools.cocoeval.COCOeval` with ``iouType='keypoints'``.

    Logs (on the ``pl_module``):

    * ``val/kpt_mAP`` – AP at OKS 0.50:0.95 (primary metric for best-model tracking)
    * ``val/kpt_mAP50`` – AP at OKS 0.50
    * ``val/kpt_mAP75`` – AP at OKS 0.75
    * ``val/kpt_mAP_medium`` – AP for medium-scale persons
    * ``val/kpt_mAP_large`` – AP for large-scale persons

    Args:
        eval_interval: Evaluate every this many epochs (default 1).
        num_keypoints: Number of keypoints per person (default 17 for COCO).
    """

    def __init__(self, eval_interval: int = 1, num_keypoints: int = 17) -> None:
        super().__init__()
        self.eval_interval = eval_interval
        self.num_keypoints = num_keypoints
        self._preds: list[dict] = []
        self._coco_gt = None

    # ------------------------------------------------------------------
    # Setup — grab COCO GT from the DataModule's validation dataset
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        from rfdetr.datasets import get_coco_api_from_dataset

        dm = getattr(trainer, "datamodule", None)
        if dm is None:
            return
        if getattr(dm, "_dataset_val", None) is None:
            dm.setup("validate")
        self._coco_gt = get_coco_api_from_dataset(dm._dataset_val)

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        self._preds = []

    def on_validation_batch_end(  # noqa: D102
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        results = outputs.get("results", [])
        targets = outputs.get("targets", [])
        for res, tgt in zip(results, targets):
            image_id = int(tgt["image_id"].item())
            self._convert_and_accumulate(image_id, res)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        if trainer.current_epoch % self.eval_interval != 0:
            return
        self._run_eval(pl_module, stage="val")

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        self._preds = []

    def on_test_batch_end(  # noqa: D102
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if outputs is None:
            return
        results = outputs.get("results", [])
        targets = outputs.get("targets", [])
        for res, tgt in zip(results, targets):
            image_id = int(tgt["image_id"].item())
            self._convert_and_accumulate(image_id, res)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:  # noqa: D102
        self._run_eval(pl_module, stage="val")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _convert_and_accumulate(self, image_id: int, res: dict) -> None:
        """Convert a single image result to COCO keypoint format and accumulate."""
        scores = res.get("scores")
        kpts = res.get("keypoints")  # [K_sel, num_keypoints, 3] — abs (x, y, vis_score)
        if scores is None or kpts is None:
            return
        scores_np = scores.detach().cpu().float()
        kpts_np = kpts.detach().cpu().float()
        for i in range(len(scores_np)):
            score = float(scores_np[i].max())
            # COCO keypoint format: [x1, y1, v1, x2, y2, v2, ...]
            kpt_flat = kpts_np[i].reshape(-1).tolist()  # [num_keypoints*3]
            self._preds.append(
                {
                    "image_id": image_id,
                    "category_id": 1,  # person
                    "keypoints": kpt_flat,
                    "score": score,
                }
            )

    def _gather_preds(self) -> list[dict]:
        """Gather predictions from all distributed ranks using all_gather_object."""
        if not is_dist_avail_and_initialized() or get_world_size() == 1:
            return self._preds

        world_size = get_world_size()
        gathered: list[list[dict]] = [None] * world_size  # type: ignore[list-item]
        torch.distributed.all_gather_object(gathered, self._preds)

        all_preds: list[dict] = []
        for chunk in gathered:
            if chunk:
                all_preds.extend(chunk)
        return all_preds

    def _run_eval(self, pl_module: LightningModule, stage: str = "val") -> None:
        if self._coco_gt is None:
            logger.warning("PoseEvalCallback: COCO GT not available — skipping evaluation.")
            return

        from pycocotools.cocoeval import COCOeval

        all_preds = self._gather_preds()

        # Only rank-0 runs evaluation.
        if is_dist_avail_and_initialized() and get_rank() != 0:
            return

        if not all_preds:
            logger.warning("PoseEvalCallback: no predictions collected — skipping.")
            return

        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(all_preds, f)
                tmp_path = f.name

            coco_dt = self._coco_gt.loadRes(tmp_path)
            coco_eval = COCOeval(self._coco_gt, coco_dt, iouType="keypoints")
            coco_eval.evaluate()
            coco_eval.accumulate()

            # Suppress pycocotools print output.
            _old_level = logging.getLogger("pycocotools").level
            logging.getLogger("pycocotools").setLevel(logging.WARNING)
            coco_eval.summarize()
            logging.getLogger("pycocotools").setLevel(_old_level)

            stats = coco_eval.stats  # 10-value array
            # Indices: 0=AP, 1=AP50, 2=AP75, 3=AP_medium, 4=AP_large, ...
            metrics = {
                f"{stage}/kpt_mAP": float(stats[0]),
                f"{stage}/kpt_mAP50": float(stats[1]),
                f"{stage}/kpt_mAP75": float(stats[2]),
                f"{stage}/kpt_mAP_medium": float(stats[3]),
                f"{stage}/kpt_mAP_large": float(stats[4]),
            }
            pl_module.log_dict(metrics, prog_bar=True, sync_dist=False, rank_zero_only=True)
            logger.info(
                "COCO Keypoint AP: %.3f | AP50: %.3f | AP75: %.3f",
                stats[0],
                stats[1],
                stats[2],
            )
        except Exception as exc:
            logger.warning("PoseEvalCallback: evaluation failed — %s", exc)
