python score.py \
    --gt_dir /workspace/datasets/table_rec/general_table_structure_cell_pad50_with_hsbc_ablation_line_exp11/txts_val \
    --pred_dir /workspace/youjiachen/workspace/ultralytics/table_cell_det_yolo11x/yolo11x_table_cell_det_epoch500_imgsz1280_bs8/predict \
    --multi_class \
    --iou 0.5 \
    --task table_cell_det
