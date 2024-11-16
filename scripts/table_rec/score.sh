#!/bin/bash

# score table cell det
python score.py \
    --gt_dir /workspace/datasets/table_rec/general_table_structure_cell_pad50_with_hsbc_ablation_line_exp11/txts_val \
    --pred_dir workspace/predict/yolo11l_table_cell_det_epoch50_imgsz1024_bs16 \
    --multi_class \
    --iou 0.8 \
    --dataset_name table_cell_det
