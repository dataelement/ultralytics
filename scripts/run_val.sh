#!/bin/bash

# MODEL_PATH=/workspace/models/hantian/yolo-doclaynet/yolov10m-doclaynet.pt
# MODEL_PATH=/workspace/models/hantian/yolo-doclaynet/yolov10b-doclaynet.pt
MODEL_PATH=layout_yolo11l_doclaynet/yolo11l_doclaynet_epoch500_imgsz1024_bs32/weights/best.pt
DATA_NAME=doclaynet_2
DEVICE=2
BATCH_SIZE=16
IMGSZ=1024

python val.py \
--data ${DATA_NAME} \
--model ${MODEL_PATH} \
--device ${DEVICE} \
--batch-size ${BATCH_SIZE} \
--imgsz ${IMGSZ} \
--split test

