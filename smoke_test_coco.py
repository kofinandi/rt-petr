# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Quick smoke test: 5 training steps on COCO 2017 (DINOv2 backbone only).

Runs fast_dev_run=5, so only 5 batches are executed.
Confirms: DINOv2 HF weights load, COCO dataset loads, CUDA forward+backward works.

Usage:
    CUDA_VISIBLE_DEVICES=1 python smoke_test_coco.py
"""

from rfdetr import RFDETRBase
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

COCO_DIR = "/data/nkoefarago/data/coco"

model = RFDETRBase(pretrain_weights=None)

config = model.get_train_config(
    dataset_dir=COCO_DIR,
    dataset_file="coco",
    output_dir="output/coco_smoke",
    batch_size=4,
    grad_accum_steps=1,
    num_workers=4,
    epochs=1,
)

model._align_num_classes_from_dataset(COCO_DIR)
model.model_config.model_name = type(model).__name__

module = RFDETRModelModule(model.model_config, config)
datamodule = RFDETRDataModule(model.model_config, config)

trainer = build_trainer(config, model.model_config, fast_dev_run=5)
trainer.fit(module, datamodule)

print("\nSmoke test PASSED — 5 training steps completed successfully.")
