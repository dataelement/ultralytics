#!/bin/bash

# 设置基础路径
MODEL_PATH=/workspace/models/hantian/yolo-doclaynet/yolov10b-doclaynet.onnx
# DATA_DIR=/workspace/datasets/table_rec/table_cell_det_yolo_format
DATA_DIR=/workspace/datasets/layout/DocLayout-YOLO/layout_data/doclaynet/
SAVE_DIR=workspace/predict/doclaynet_onnx_yolov10b

DEVICE="0"

# 预测参数
BATCH_SIZE=24
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