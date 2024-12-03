#!/bin/bash

# score table cell det
# python score.py \
#     --gt_dir /workspace/datasets/table_rec/general_table_structure_cell_pad50_with_hsbc_ablation_line_exp11/txts_val \
#     --pred_dir workspace/predict/yolo11l_table_cell_det_epoch50_imgsz1024_bs16 \
#     --multi_class \
#     --iou 0.8 \
#     --dataset_name table_cell_det

# python score.py \
#     --gt_dir /workspace/datasets/table_rec/general_table_structure_row_col_pad50_with_hsbc/txts_val \
#     --pred_dir workspace/predict/yolo11l_table_rowcol_det_epoch50_imgsz1024_bs64 \
#     --multi_class \
#     --iou 0.8 \
#     --dataset_name table_row_col


python score.py \
    --gt_dir /workspace/datasets/text_det/general_text_det_dataset_v2.0/txts_val \
    --pred_dir workspace/predict/yolo11x-obb_text_det_epoch50_imgsz1024_bs32 \
    --multi_class \
    --iou 0.8 \
    --dataset_name text_det

# # Create txts_val directory if it doesn't exist
# mkdir -p /workspace/datasets/text_det/general_text_det_dataset_v2.0/txts_val

# # Copy annotation files from txts that correspond to files in val
# for img in /workspace/datasets/text_det/general_text_det_dataset_v2.0/val/*; do
#     basename=$(basename "$img")
#     filename="${basename%.*}"
#     if [ -f "/workspace/datasets/text_det/general_text_det_dataset_v2.0/txts/${filename}.txt" ]; then
#         cp "/workspace/datasets/text_det/general_text_det_dataset_v2.0/txts/${filename}.txt" \
#            "/workspace/datasets/text_det/general_text_det_dataset_v2.0/txts_val/"
#     fi
# done

# 遍历 txts_val 目录下的所有 txt 文件
# for file in /workspace/datasets/table_rec/table_det_cls/txts_val/*.txt; do
#     # 使用 sed 命令进行替换，-i 表示直接修改原文件
#     sed -i 's/有线表/wired_table/g' "$file"
#     sed -i 's/少线表/lineless_table/g' "$file"
# done