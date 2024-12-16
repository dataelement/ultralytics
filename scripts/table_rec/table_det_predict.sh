#!/bin/bash

# 设置基础路径
MODEL_PATH=table_det_yolo11x_obb/yolo11l-obb_table_det_epoch50_imgsz640_bs128/weights/best.pt
SAVE_DIR=workspace/table_det_yolo11x_obb/yolo11l-obb_table_det_epoch50_imgsz640_bs128

# val路径
DATA_DIR=/workspace/datasets/table_rec/table_det_cls_yolo_format
# DATA_DIR=/workspace/datasets/layout/DocLayout-YOLO/layout_data/doclaynet/
# DATA_DIR=/workspace/datasets/layout/dataelem_layout/yolo_format_merge_all # dataelem layout

DEVICE="2"

# 预测参数
BATCH_SIZE=32
CONF=0.4
MAX_DET=1000
IMGSZ=640
TASK=obb

# 运行预测
python predict.py \
    --model-path ${MODEL_PATH} \
    --task ${TASK} \
    --data-dir ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --conf ${CONF} \
    --max-det ${MAX_DET} \
    --imgsz ${IMGSZ}