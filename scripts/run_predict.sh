#!/bin/bash

# 设置基础路径
MODEL_PATH=/workspace/youjiachen/workspace/ultralytics/table_cell_det_yolo11x/yolo11x_table_cell_det_epoch500_imgsz1280_bs8/weights/best.pt
DATA_DIR=/workspace/datasets/table_rec/table_cell_det_yolo_format
SAVE_DIR="/workspace/youjiachen/workspace/ultralytics/table_cell_det_yolo11x/yolo11x_table_cell_det_epoch500_imgsz1280_bs8/predict"

DEVICE="0"

# 预测参数
BATCH_SIZE=24
CONF=0.4
MAX_DET=1000

# 运行预测
python predict.py \
    --model-path ${MODEL_PATH} \
    --data-dir ${DATA_DIR} \
    --save-dir ${SAVE_DIR} \
    --batch-size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --conf ${CONF} \
    --max-det ${MAX_DET}