#!/bin/bash

# 设置基础路径
MODEL_PATH=text_det_yolo11x/yolo11x-obb_text_det_epoch50_imgsz1024_bs32/weights/best.pt
SAVE_DIR=workspace/predict/yolo11x-obb_text_det_epoch50_imgsz1024_bs32

# val路径
DATA_DIR=/workspace/datasets/text_det/general_text_det_dataset_v2.0_yolo_format

DEVICE="2"

# 预测参数
BATCH_SIZE=8
CONF=0.4
MAX_DET=1000
IMGSZ=1024

TASK="obb"
# 运行预测
python predict.py \
    --model-path ${MODEL_PATH} \
    --data-dir ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --conf ${CONF} \
    --task ${TASK} \
    --max-det ${MAX_DET} \
    --imgsz ${IMGSZ}