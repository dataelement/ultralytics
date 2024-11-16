#!/bin/bash

# 设置基础路径
MODEL_PATH=table_cell_det_yolo11l/yolo11l_table_cell_det_epoch50_imgsz1024_bs16/weights/best.pt
SAVE_DIR=workspace/predict/yolo11l_table_cell_det_epoch50_imgsz1024_bs16

# val路径
DATA_DIR=/workspace/datasets/table_rec/table_cell_det_yolo_format
# DATA_DIR=/workspace/datasets/layout/DocLayout-YOLO/layout_data/doclaynet/
# DATA_DIR=/workspace/datasets/layout/dataelem_layout/yolo_format_merge_all # dataelem layout

DEVICE="2"

# 预测参数
BATCH_SIZE=16
CONF=0.4
MAX_DET=1000
IMGSZ=1024

# 运行预测
python predict.py \
    --model-path ${MODEL_PATH} \
    --data-dir ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --conf ${CONF} \
    --max-det ${MAX_DET} \
    --imgsz ${IMGSZ}