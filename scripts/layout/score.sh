#!/bin/bash
python score.py \
    --gt_dir /workspace/datasets/layout/dataelem_layout/mrcnn_merge_all/val_txts \
    --pred_dir workspace/predict/yolo11l_dataelem_layout_epoch40_imgsz640_bs256 \
    --multi_class \
    --iou 0.5 \
    --dataset_name dataelem_layout
