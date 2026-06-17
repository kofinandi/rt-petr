# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PoseEvalCallback — COCO keypoint mAP evaluation using pycocotools CocoEvaluator."""

from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
from pycocotools.cocoeval import COCOeval
from pytorch_lightning import Callback

from rfdetr.evaluation.coco_eval import CocoEvaluator
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class PoseEvalCallback(Callback):
    """Validation callback that computes COCO keypoint mAP using ``CocoEvaluator``.

    Accumulates pose predictions and original image sizes across all validation
    batches, then at epoch end calls :class:`~rfdetr.evaluation.coco_eval.CocoEvaluator`
    (``iou_types=["keypoints"]``) to produce the standard COCO pose metrics.

    Logged metrics (``val/`` prefix):
    - ``val/kpt_AP``, ``val/kpt_AP50``, ``val/kpt_AP75``
    - ``val/kpt_AR``

    Multi-GPU / DDP design:
        ``CocoEvaluator.synchronize_between_processes()`` internally calls
        ``dist.all_gather``, which is a collective operation — every rank must
        enter it at the same time.  To satisfy this requirement:

        * Every DDP rank creates its own ``CocoEvaluator`` and accumulates
          predictions for its own subset of val images.
        * All ranks call ``synchronize_between_processes()`` together so the
          all_gather completes.
        * Only rank 0 then calls ``accumulate()``, ``summarize()``, and logs.

        The deepcopy inside ``CocoEvaluator.__init__`` is performed in
        ``on_fit_start`` (before DataLoader workers exist) on every rank.

    Args:
        eval_interval: Run validation metrics every N epochs.  Defaults to 1.
    """

    def __init__(self, eval_interval: int = 1) -> None:
        super().__init__()
        self._eval_interval = max(1, int(eval_interval))
        self._evaluator: Optional[CocoEvaluator] = None

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        """Build the CocoEvaluator on every rank, before DataLoader workers are spawned.

        The ``copy.deepcopy`` inside ``CocoEvaluator.__init__`` is safe here
        because no DataLoader worker processes exist yet.  Subsequent epochs
        reuse the already-copied ``coco_gt`` via ``COCOeval`` directly.

        Every rank (not just rank 0) creates an evaluator so that all ranks
        can participate in the ``all_gather`` collective inside
        ``synchronize_between_processes()``.
        """
        dm = trainer.datamodule
        if dm is None:
            return
        val_ds = getattr(dm, "_dataset_val", None)
        coco_gt = getattr(val_ds, "coco", None) if val_ds is not None else None
        if coco_gt is not None:
            # max_dets=20 matches the COCO keypoints standard used by
            # _summarizeKps() in patched_pycocotools_summarize (which looks
            # for maxDets==20 when building the stats array).  Using the
            # default max_dets=100 causes the lookup to produce an empty
            # index slice, making all stats return -1.
            self._evaluator = CocoEvaluator(coco_gt, ["keypoints"], max_dets=20)

    def on_validation_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        """Reset per-epoch accumulation state on every rank without calling deepcopy.

        Reuses the ``coco_gt`` that was already deepcopied in ``on_fit_start``
        to construct fresh ``COCOeval`` objects.  ``COCOeval.__init__`` does not
        deepcopy, so this is safe even while DataLoader workers are running.
        """
        if self._evaluator is None:
            return
        self._evaluator.img_ids = []
        self._evaluator.eval_imgs = {k: [] for k in self._evaluator.iou_types}
        for iou_type in self._evaluator.iou_types:
            new_eval = COCOeval(self._evaluator.coco_gt, iouType=iou_type)
            new_eval.params.maxDets = [1, 10, self._evaluator.max_dets]
            self._evaluator.coco_eval[iou_type] = new_eval

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate predictions from each validation batch (every rank)."""
        if self._evaluator is None:
            return
        if outputs is None:
            return
        results = outputs.get("results", None)
        targets = outputs.get("targets", None)
        if results is None or targets is None:
            return
        image_ids = [
            int(t["image_id"].item() if torch.is_tensor(t["image_id"]) else t["image_id"])
            for t in targets
        ]
        self._accumulate(results, image_ids)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Synchronize across ranks, then report COCO keypoint mAP on rank 0.

        All ranks must call ``synchronize_between_processes()`` because it
        contains a ``dist.all_gather`` collective.  Only rank 0 then runs
        ``accumulate()`` / ``summarize()`` and logs the metrics.
        """
        current_epoch = trainer.current_epoch + 1
        if current_epoch % self._eval_interval != 0:
            return
        if self._evaluator is None:
            return

        # Collective — every rank must reach this point simultaneously.
        self._evaluator.synchronize_between_processes()

        # Only rank 0 computes and logs the final metrics.
        if not getattr(trainer, "is_global_zero", True):
            return

        self._evaluator.accumulate()
        with contextlib.redirect_stdout(None):
            self._evaluator.summarize()

        stats = self._evaluator.coco_eval.get("keypoints", None)
        if stats is not None and hasattr(stats, "stats"):
            ap, ap50, ap75 = stats.stats[0], stats.stats[1], stats.stats[2]
            ar = stats.stats[5]  # AR @ max=20
            pl_module.log("val/kpt_AP", ap, sync_dist=False, prog_bar=True, rank_zero_only=True)
            pl_module.log("val/kpt_AP50", ap50, sync_dist=False, rank_zero_only=True)
            pl_module.log("val/kpt_AP75", ap75, sync_dist=False, rank_zero_only=True)
            pl_module.log("val/kpt_AR", ar, sync_dist=False, rank_zero_only=True)
            logger.info(f"Keypoint mAP: AP={ap:.4f}  AP50={ap50:.4f}  AP75={ap75:.4f}  AR={ar:.4f}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accumulate(
        self,
        results: list[dict],
        image_ids: list[int],
        orig_sizes: Optional[torch.Tensor] = None,
    ) -> None:
        """Convert model results to COCO format and update the evaluator.

        Args:
            results: List of per-image output dicts (from PostProcess), each with
                ``scores``, ``labels``, ``boxes``, ``keypoints`` (optional).
            image_ids: List of COCO image IDs corresponding to the results.
            orig_sizes: Unused; reserved for future use.
        """
        if self._evaluator is None:
            return
        predictions: dict[int, dict] = {}
        for img_id, res in zip(image_ids, results):
            kpts = res.get("keypoints", None)
            if kpts is None:
                continue
            predictions[int(img_id)] = {
                "scores": res["scores"].cpu(),
                "labels": res["labels"].cpu(),
                "boxes": res["boxes"].cpu(),
                "keypoints": kpts.cpu(),
            }
        if predictions:
            self._evaluator.update(predictions)
