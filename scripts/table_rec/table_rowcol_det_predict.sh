#!/bin/bash

# 设置基础路径
MODEL_PATH=table_rowcol_det_yolo11l_reg64/yolo11l_table_rowcol_det_epoch60_imgsz640_bs128/weights/best.pt
SAVE_DIR=workspace/table_rowcol_det_yolo11l_reg64/yolo11l_table_rowcol_det_epoch60_imgsz640_bs128

# val路径
DATA_DIR=/workspace/datasets/table_rec/table_row_col_det_yolo_format
# DATA_DIR=/workspace/datasets/layout/DocLayout-YOLO/layout_data/doclaynet/
# DATA_DIR=/workspace/datasets/layout/dataelem_layout/yolo_format_merge_all # dataelem layout

DEVICE="2"

# 预测参数
BATCH_SIZE=16
CONF=0.3
MAX_DET=1000
IMGSZ=640
TASK='seg'

# 运行预测
python predict.py \
    --model-path ${MODEL_PATH} \
    --data-dir ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --task ${TASK} \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --conf ${CONF} \
    --max-det ${MAX_DET} \
    --imgsz ${IMGSZ}