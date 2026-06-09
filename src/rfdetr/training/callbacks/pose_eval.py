# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCOPoseEvalCallback — COCO keypoint AP evaluation callback.

Computes COCO pose AP (OKS-based) metrics at the end of each validation
and test epoch using pycocotools.  Complements :class:`COCOEvalCallback`
which evaluates bounding-box mAP via torchmetrics.
"""

from __future__ import annotations

import contextlib
import copy
from typing import Any

import torch
from pytorch_lightning import Callback

from rfdetr.utilities.logger import get_logger

logger = get_logger()

# COCO person category id
_COCO_PERSON_CAT_ID = 1


class COCOPoseEvalCallback(Callback):
    """Evaluate COCO keypoint AP on the validation (and test) set.

    Requires the dataset to expose a ``.coco`` attribute (a
    ``pycocotools.coco.COCO`` object) so that ground-truth annotations are
    available for OKS evaluation.

    For each image the postprocessor must return a dict with keys:

    - ``"scores"`` – ``(K,)`` confidence scores.
    - ``"labels"`` – ``(K,)`` integer class labels.
    - ``"boxes"`` – ``(K, 4)`` absolute xyxy boxes.
    - ``"keypoints"`` – ``(K, 17, 3)`` keypoints with ``(x, y, vis_prob)``.

    Metrics logged (under ``val/`` or ``test/`` prefix):

    - ``kpt_AP``        – mean AP @IoU/OKS 0.50:0.95
    - ``kpt_AP_50``     – AP @OKS 0.50
    - ``kpt_AP_75``     – AP @OKS 0.75
    - ``kpt_AP_M``      – AP for medium persons
    - ``kpt_AP_L``      – AP for large persons
    - ``kpt_AR``        – mean AR @OKS 0.50:0.95 (maxDets=20)

    Args:
        eval_interval: Compute metrics every N epochs. Defaults to 1.
        max_dets: Maximum detections per image. Defaults to 20 (COCO default).
        score_threshold: Detections below this score are discarded. Defaults
            to 0.0 (keep all).
    """

    def __init__(
        self,
        eval_interval: int = 1,
        max_dets: int = 20,
        score_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self._eval_interval = max(1, int(eval_interval))
        self._max_dets = max_dets
        self._score_threshold = score_threshold
        self._val_predictions: dict[int, Any] = {}
        self._test_predictions: dict[int, Any] = {}
        self._coco_gt: Any = None

    # ------------------------------------------------------------------
    # PTL lifecycle hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Cache the ground-truth COCO object from the validation dataset.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            stage: Training stage.
        """
        dm = trainer.datamodule
        if dm is None:
            return
        for attr in ("_dataset_val", "_dataset_train"):
            dataset = getattr(dm, attr, None)
            if dataset is None:
                continue
            coco = getattr(dataset, "coco", None)
            if coco is not None:
                self._coco_gt = copy.deepcopy(coco)
                logger.info(
                    "COCOPoseEvalCallback: loaded GT from %s (%d images, %d annotations)",
                    attr,
                    len(coco.imgs),
                    len(coco.anns),
                )
                break

    def on_validation_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Accumulate per-image keypoint predictions.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``validation_step``.
            batch: The device-transferred batch.
            batch_idx: Batch index within the validation epoch.
        """
        self._accumulate(outputs, self._val_predictions)

    def on_test_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate per-image keypoint predictions (test split).

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            outputs: Return value of ``test_step``.
            batch: Raw batch.
            batch_idx: Batch index.
            dataloader_idx: Index of the test dataloader.
        """
        self._accumulate(outputs, self._test_predictions)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log keypoint AP at the end of validation.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        current_epoch = int(getattr(trainer, "current_epoch", 0)) + 1
        max_epochs = getattr(trainer, "max_epochs", None)
        is_last = isinstance(max_epochs, int) and max_epochs > 0 and current_epoch >= max_epochs
        if current_epoch % self._eval_interval != 0 and not is_last:
            self._val_predictions.clear()
            return
        self._compute_and_log(trainer, pl_module, self._val_predictions, split="val")
        self._val_predictions.clear()

    def on_test_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        """Compute and log keypoint AP at the end of test.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
        """
        self._compute_and_log(trainer, pl_module, self._test_predictions, split="test")
        self._test_predictions.clear()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _accumulate(self, outputs: dict[str, Any], store: dict[int, Any]) -> None:
        """Append PostProcess results keyed by image_id into *store*.

        Args:
            outputs: The step output dict with keys ``"results"`` and
                ``"targets"``.
            store: Mutable dict to accumulate into (keyed by image_id).
        """
        results = outputs.get("results", [])
        targets = outputs.get("targets", [])
        for res, tgt in zip(results, targets):
            image_id = int(tgt["image_id"].item() if torch.is_tensor(tgt["image_id"]) else tgt["image_id"])
            store[image_id] = res

    def _compute_and_log(
        self,
        trainer: Any,
        pl_module: Any,
        predictions: dict[int, Any],
        split: str,
    ) -> None:
        """Run pycocotools keypoint evaluation and log metrics.

        Args:
            trainer: The PTL Trainer.
            pl_module: The LightningModule.
            predictions: Per-image prediction dicts keyed by image_id.
            split: ``"val"`` or ``"test"``.
        """
        if self._coco_gt is None:
            logger.warning("COCOPoseEvalCallback: no COCO GT object available; skipping keypoint eval.")
            return

        try:
            from pycocotools.cocoeval import COCOeval
        except ImportError:
            logger.warning("COCOPoseEvalCallback: pycocotools not installed; skipping keypoint eval.")
            return

        results = self._format_predictions(predictions)
        if not results:
            logger.warning("COCOPoseEvalCallback: no predictions to evaluate.")
            return

        with contextlib.redirect_stdout(None):
            coco_dt = self._coco_gt.loadRes(results)
            coco_eval = COCOeval(self._coco_gt, coco_dt, "keypoints")
            coco_eval.params.maxDets = [self._max_dets]
            coco_eval.evaluate()
            coco_eval.accumulate()

        # Suppress the printout from summarize
        import io
        import sys

        _stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            coco_eval.summarize()
        finally:
            sys.stdout = _stdout

        stats = coco_eval.stats  # 10 values (see pycocotools docs)
        metrics = {
            f"{split}/kpt_AP": float(stats[0]),
            f"{split}/kpt_AP_50": float(stats[1]),
            f"{split}/kpt_AP_75": float(stats[2]),
            f"{split}/kpt_AP_M": float(stats[3]),
            f"{split}/kpt_AP_L": float(stats[4]),
            f"{split}/kpt_AR": float(stats[5]),
        }

        for k, v in metrics.items():
            pl_module.log(k, v, prog_bar=(k == f"{split}/kpt_AP"), sync_dist=True)
            trainer.callback_metrics[k] = torch.tensor(v)

        if getattr(trainer, "is_global_zero", True):
            header = f"\n{'─' * 40}\n  Pose AP ({split})\n{'─' * 40}"
            rows = "\n".join(f"  {k.split('/')[-1]:20s} {v:.4f}" for k, v in metrics.items())
            logger.info(f"{header}\n{rows}\n{'─' * 40}")

    def _format_predictions(self, predictions: dict[int, Any]) -> list[dict[str, Any]]:
        """Convert PostProcess outputs to COCO keypoint result format.

        Args:
            predictions: Per-image result dicts (keyed by image_id), each
                containing ``"scores"``, ``"labels"``, ``"boxes"``, and
                ``"keypoints"``.

        Returns:
            List of COCO-API result dicts with keys ``image_id``,
            ``category_id``, ``keypoints`` (flat list of K*3 values),
            ``score``.
        """
        coco_results = []
        for image_id, pred in predictions.items():
            if not pred or "keypoints" not in pred:
                continue
            scores = pred["scores"].cpu()
            keypoints = pred["keypoints"].cpu()  # (K, 17, 3)
            boxes = pred["boxes"].cpu()  # (K, 4) xyxy absolute

            for k in range(len(scores)):
                score = float(scores[k])
                if score < self._score_threshold:
                    continue
                kpt_k = keypoints[k]  # (17, 3): (x, y, vis_prob)
                # COCO expects [x1,y1,v1, x2,y2,v2, ...] with v in {0,1,2}
                flat_kpts = []
                for kp in kpt_k:
                    x, y, v = float(kp[0]), float(kp[1]), float(kp[2])
                    vis_flag = 2 if v > 0.5 else 0  # binarise visibility
                    flat_kpts.extend([x, y, vis_flag])
                # Bounding-box area for score weighting (COCO keypoint eval uses
                # the predicted box area as part of the confidence score)
                x1, y1, x2, y2 = boxes[k].tolist()
                area = max(0.0, (x2 - x1) * (y2 - y1))
                # Use person category (id=1) regardless of predicted label,
                # since this is a single-class pose model
                coco_results.append(
                    {
                        "image_id": image_id,
                        "category_id": _COCO_PERSON_CAT_ID,
                        "keypoints": flat_kpts,
                        "score": score,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": area,
                    }
                )
        return coco_results
