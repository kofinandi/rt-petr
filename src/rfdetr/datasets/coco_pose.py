# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""COCO Keypoints / Human Pose dataset for RF-DETR pose estimation training.

Loads person instances with their 17 COCO keypoints and bounding boxes.
Only person-category (``category_id == 1``) annotations are used; crowd
annotations and instances without any keypoint are filtered out.

Target dict keys (after full transform pipeline):

- ``"boxes"``      – ``(N, 4)`` float32, ``cxcywh`` normalised to ``[0, 1]``
- ``"labels"``     – ``(N,)`` int64, always 0 (single person class)
- ``"keypoints"``  – ``(N, K, 3)`` float32, ``(x, y, vis)`` with x/y
  normalised to ``[0, 1]``
- ``"image_id"``   – scalar int64
- ``"area"``       – ``(N,)`` float32
- ``"iscrowd"``    – ``(N,)`` int64
- ``"orig_size"``  – ``(2,)`` int64, ``[H, W]`` before padding
- ``"size"``       – ``(2,)`` int64, current H/W
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.transforms.v2 import Compose, ToDtype, ToImage

from rfdetr.datasets.aug_config import AUG_CONFIG
from rfdetr.datasets.coco import (
    _resolve_runtime_augmentation_backend,
    compute_multi_scale_scales,
)
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from rfdetr.utilities.logger import get_logger

logger = get_logger()

# COCO person category id
_COCO_PERSON_CATEGORY_ID = 1

# Number of keypoints in COCO person annotations
COCO_NUM_KEYPOINTS = 17

# Per-keypoint sigma values from the COCO evaluation paper
# (used for OKS computation)
COCO_KPT_SIGMAS = np.array(
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
    dtype=np.float32,
)

# COCO keypoint names (for reference / visualisation)
COCO_KPT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class CocoPoseDetection(torchvision.datasets.CocoDetection):
    """COCO keypoints dataset for human pose estimation.

    Wraps ``torchvision.datasets.CocoDetection`` to also load keypoint
    annotations alongside bounding boxes.  Only person-category annotations
    are kept; crowd and zero-keypoint annotations are filtered in
    :class:`ConvertCocoPose`.

    Args:
        img_folder: Path to the image directory.
        ann_file: Path to the COCO-format JSON annotation file
            (e.g. ``person_keypoints_train2017.json``).
        transforms: Transform pipeline applied after annotation conversion.
    """

    def __init__(
        self,
        img_folder: Path,
        ann_file: Path,
        transforms: Optional[Any] = None,
    ) -> None:
        super().__init__(str(img_folder), str(ann_file))
        self._transforms = transforms
        self.prepare = ConvertCocoPose()

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class ConvertCocoPose:
    """Convert raw COCO keypoint annotations into model-ready tensors.

    Processes person-category annotations only. Crowd annotations and
    instances with no labelled keypoints (all visibilities == 0) are removed.

    Returns:
        target dict with keys described in the module docstring.
    """

    def __call__(self, image: Image.Image, target: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
        """Convert annotation dict to tensor target.

        Args:
            image: PIL image.
            target: Raw COCO annotation dict with keys ``"image_id"`` and
                ``"annotations"``.

        Returns:
            Tuple of (image, target dict).
        """
        w, h = image.size
        image_id = torch.tensor([target["image_id"]])
        anno = target["annotations"]

        # Keep only non-crowd person annotations that have keypoints
        anno = [
            obj
            for obj in anno
            if obj.get("iscrowd", 0) == 0 and obj.get("category_id") == _COCO_PERSON_CATEGORY_ID and "keypoints" in obj
        ]

        # Filter out annotations where every keypoint is unlabelled (v==0)
        anno = [obj for obj in anno if sum(obj["keypoints"][2::3]) > 0]

        # --- Bounding boxes ------------------------------------------------
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # COCO xywh -> xyxy absolute
        if boxes.numel() > 0:
            boxes[:, 2:] += boxes[:, :2]
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)

        # Filter degenerate boxes (zero area)
        keep = torch.ones(len(anno), dtype=torch.bool)
        if boxes.numel() > 0:
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        # --- Labels (always 0 = person) ------------------------------------
        labels = torch.zeros(boxes.shape[0], dtype=torch.int64)

        # --- Keypoints -----------------------------------------------------
        # COCO format: [x1, y1, v1, x2, y2, v2, ...] (flat list, length K*3)
        kpts_list: List[List[float]] = [obj["keypoints"] for obj in anno]
        kpts_list = [kpts_list[i] for i in range(len(kpts_list)) if keep[i]]

        if len(kpts_list) > 0:
            kpts = torch.tensor(kpts_list, dtype=torch.float32)  # (N, K*3)
            kpts = kpts.reshape(-1, COCO_NUM_KEYPOINTS, 3)  # (N, K, 3)
        else:
            kpts = torch.zeros((0, COCO_NUM_KEYPOINTS, 3), dtype=torch.float32)

        # Clamp keypoint coordinates to image boundaries
        if kpts.numel() > 0:
            kpts[:, :, 0].clamp_(min=0, max=w)
            kpts[:, :, 1].clamp_(min=0, max=h)

        # --- Area / iscrowd ------------------------------------------------
        area = torch.tensor([obj.get("area", 0.0) for obj in anno], dtype=torch.float32)
        iscrowd = torch.tensor([obj.get("iscrowd", 0) for obj in anno], dtype=torch.int64)
        keep_idx = torch.where(keep)[0]
        area = area[keep_idx]
        iscrowd = iscrowd[keep_idx]

        target_out = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": kpts,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)]),
        }
        return image, target_out


class NormalizePose(Normalize):
    """Normalise images and convert pose annotations to model-ready format.

    Extends :class:`~rfdetr.datasets.transforms.Normalize` to also normalise
    keypoint coordinates to ``[0, 1]`` by dividing ``x`` by image width and
    ``y`` by image height.  Visibility values are left unchanged.
    """

    def __call__(
        self,
        image: torch.Tensor,
        target: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Normalise pixel values and keypoint/box coordinates.

        Args:
            image: Float tensor ``[C, H, W]``.
            target: Target dict.  Modified in-place on a copy.

        Returns:
            Tuple of (normalised image, updated target).
        """
        image, target = super().__call__(image, target)
        if target is None:
            return image, None
        h, w = image.shape[-2:]
        if "keypoints" in target:
            kpts = target["keypoints"].clone()  # (N, K, 3)
            kpts[:, :, 0] /= w  # x
            kpts[:, :, 1] /= h  # y
            # visibility stays unchanged
            target["keypoints"] = kpts
        return image, target


class AlbumentationsWrapperPose(AlbumentationsWrapper):
    """Albumentations wrapper extended to transform keypoints.

    Inherits all bbox handling from :class:`AlbumentationsWrapper` and adds
    keypoint co-transformation so keypoints stay aligned with boxes after
    geometric augmentations.
    """

    def _apply_geometric_transform(
        self,
        image_np: np.ndarray,
        target: Dict[str, Any],
        labels: List[int],
    ):
        """Apply geometric transform to image, boxes, and keypoints.

        Overrides the parent implementation to pass keypoints through the
        Albumentations pipeline when they are present in the target dict.

        Args:
            image_np: HWC uint8 numpy image.
            target: Target dict (may contain ``"keypoints"``).
            labels: Category-id list, one per box.

        Returns:
            Tuple of (transformed PIL Image, updated target dict).
        """
        # Let parent handle the spatial transform (boxes etc.)
        image_out, target_out = super()._apply_geometric_transform(image_np, target, labels)

        # Re-apply the same filter to keypoints using the kept box indices
        if "keypoints" in target and "keypoints" in target_out:
            # The parent stores kept_idxs but doesn't return them; we need to
            # recompute them by comparing the original and output labels.
            # However, AlbumentationsWrapper already filters per-instance fields
            # generically — it will handle 'keypoints' if it has the same leading
            # dimension as 'boxes'.  We only need to override the normalisation
            # step to handle the (N, K, 3) shape properly.
            pass  # handled by _filter_per_instance_fields in parent

        return image_out, target_out

    def __call__(
        self,
        image: Any,
        target: Optional[Dict[str, Any]],
    ):
        """Apply transform, ensuring keypoints are aligned with boxes.

        Flattens keypoints to ``(N, K*3)`` so
        :meth:`AlbumentationsWrapper._filter_per_instance_fields` can handle
        them via the standard per-instance field pathway, then reshapes back.

        Args:
            image: PIL Image.
            target: Target dict.

        Returns:
            Tuple of (augmented image, updated target).
        """
        if target is not None and "keypoints" in target:
            kpts = target["keypoints"]  # (N, K, 3)
            n = kpts.shape[0]
            # Temporarily flatten to (N, K*3) so parent filtering works
            target = dict(target)
            target["keypoints"] = kpts.reshape(n, -1)
        image_out, target_out = super().__call__(image, target)
        # Reshape keypoints back
        if target_out is not None and "keypoints" in target_out:
            flat = target_out["keypoints"]  # (N', K*3)
            n_out = flat.shape[0]
            target_out = dict(target_out)
            target_out["keypoints"] = flat.reshape(n_out, COCO_NUM_KEYPOINTS, 3)
        return image_out, target_out


def make_pose_transforms(
    image_set: str,
    resolution: int,
    multi_scale: bool = False,
    expanded_scales: bool = False,
    skip_random_resize: bool = False,
    patch_size: int = 16,
    num_windows: int = 4,
    aug_config: Optional[Dict[str, Any]] = None,
    gpu_postprocess: bool = False,
    square: bool = True,
) -> Compose:
    """Build the transform pipeline for COCO pose estimation.

    Mirrors :func:`~rfdetr.datasets.coco.make_coco_transforms_square_div_64`
    but replaces :class:`~rfdetr.datasets.transforms.Normalize` with
    :class:`NormalizePose` and replaces :class:`AlbumentationsWrapper` with
    :class:`AlbumentationsWrapperPose` so that keypoints are co-transformed.

    Args:
        image_set: ``"train"``, ``"val"``, or ``"test"``.
        resolution: Target resolution in pixels.
        multi_scale: Enable multi-scale training.
        expanded_scales: Use a wider scale range.
        skip_random_resize: Skip random resize even when multi_scale=True.
        patch_size: Patch size for divisibility checks.
        num_windows: Number of attention windows.
        aug_config: Albumentations augmentation config dict.
        gpu_postprocess: Skip CPU-side augmentation / normalisation.
        square: Use square resize.

    Returns:
        :class:`torchvision.transforms.v2.Compose` pipeline.
    """
    from rfdetr.datasets.coco import _build_train_resize_config

    to_image = ToImage()
    to_float = ToDtype(torch.float32, scale=True)
    normalize = NormalizePose()

    scales = [resolution]
    if multi_scale:
        scales = compute_multi_scale_scales(resolution, expanded_scales, patch_size, num_windows)
        if skip_random_resize:
            scales = [scales[-1]]
        logger.info(f"Pose dataset: multi-scale training with scales {scales}")

    if image_set == "train":
        resolved_aug_config = aug_config if aug_config is not None else AUG_CONFIG
        resize_wrappers_raw = AlbumentationsWrapper.from_config(
            _build_train_resize_config(scales, square=square, max_size=None if square else 1333)
        )
        # Replace each AlbumentationsWrapper with AlbumentationsWrapperPose so
        # that keypoints are co-transformed during geometric augmentations.
        resize_wrappers = [AlbumentationsWrapperPose(w.transform.transforms[0]) for w in resize_wrappers_raw]
        pipeline = [*resize_wrappers]
        if not gpu_postprocess:
            aug_wrappers_raw = AlbumentationsWrapper.from_config(resolved_aug_config)
            aug_wrappers = [AlbumentationsWrapperPose(w.transform.transforms[0]) for w in aug_wrappers_raw]
            pipeline += [*aug_wrappers]
        pipeline += [to_image, to_float]
        if not gpu_postprocess:
            pipeline += [normalize]
        return Compose(pipeline)

    if image_set in ("val", "test"):
        if square:
            resize_wrappers_raw = AlbumentationsWrapper.from_config(
                [{"Resize": {"height": resolution, "width": resolution}}]
            )
        else:
            resize_wrappers_raw = AlbumentationsWrapper.from_config(
                [
                    {"SmallestMaxSize": {"max_size": resolution}},
                    {"LongestMaxSize": {"max_size": 1333}},
                ]
            )
        resize_wrappers = [AlbumentationsWrapperPose(w.transform.transforms[0]) for w in resize_wrappers_raw]
        return Compose([*resize_wrappers, to_image, to_float, normalize])

    raise ValueError(f"Unknown image_set: {image_set!r}")


def build_coco_pose(image_set: str, args: Any, resolution: int) -> CocoPoseDetection:
    """Build the COCO keypoints dataset for pose estimation.

    Uses the standard COCO 2017 layout::

        <coco_path>/
          train2017/            # images
          val2017/
          annotations/
            person_keypoints_train2017.json
            person_keypoints_val2017.json

    Args:
        image_set: ``"train"`` or ``"val"``.
        args: Namespace with at least ``dataset_dir`` (or ``coco_path``).
        resolution: Target image resolution in pixels.

    Returns:
        :class:`CocoPoseDetection` dataset ready for use with
        :class:`~rfdetr.training.module_data.RFDETRDataModule`.

    Raises:
        FileNotFoundError: If the dataset root or annotation file does not exist.
    """
    root = Path(getattr(args, "dataset_dir", None) or args.coco_path)
    if not root.exists():
        raise FileNotFoundError(f"COCO dataset root not found: {root}")

    PATHS = {  # noqa: N806
        "train": (
            root / "train2017",
            root / "annotations" / "person_keypoints_train2017.json",
        ),
        "val": (
            root / "val2017",
            root / "annotations" / "person_keypoints_val2017.json",
        ),
    }

    split = image_set.split("_")[0]
    if split not in PATHS:
        raise ValueError(f"Unknown split {image_set!r}; expected 'train' or 'val'")

    img_folder, ann_file = PATHS[split]
    if not ann_file.exists():
        raise FileNotFoundError(f"COCO keypoints annotation file not found: {ann_file}")

    square_resize = getattr(args, "square_resize_div_64", True)
    multi_scale = getattr(args, "multi_scale", False)
    expanded_scales = getattr(args, "expanded_scales", False)
    skip_random_resize = not getattr(args, "do_random_resize_via_padding", False)
    patch_size = getattr(args, "patch_size", 16)
    num_windows = getattr(args, "num_windows", 4)
    aug_config = getattr(args, "aug_config", None)
    augmentation_backend = getattr(args, "augmentation_backend", "cpu")
    resolved_backend = _resolve_runtime_augmentation_backend(augmentation_backend)
    gpu_postprocess = resolved_backend != "cpu"

    transforms = make_pose_transforms(
        image_set=image_set,
        resolution=resolution,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        skip_random_resize=skip_random_resize,
        patch_size=patch_size,
        num_windows=num_windows,
        aug_config=aug_config,
        gpu_postprocess=gpu_postprocess,
        square=square_resize,
    )

    logger.info(f"Building COCO pose {image_set} dataset from {img_folder}")
    return CocoPoseDetection(img_folder, ann_file, transforms=transforms)
