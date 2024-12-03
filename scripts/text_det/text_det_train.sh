# finetune yolo11n det on table_det
# python train.py \
#     --data table_det \
#     --model_path /workspace/models/YOLO11/yolo11s.pt \
#     --epoch 50 \
#     --close_mosaic 10 \
#     --image-size 640 \
#     --batch-size 512 \
#     --project table_det_yolo11s\
#     --plot 1 \
#     --device "0,1,2,3,4,5,6,7"

# python train.py \
#     --data text_det \
#     --model_path /workspace/models/YOLO11/yolo11l-seg.pt \
#     --model_type yolo \
#     --epoch 50 \
#     --close_mosaic 10 \
#     --image-size 640 \
#     --batch-size 64 \
#     --project text_det_yolo11l-seg \
#     --plot 1 \
#     --device "0,1,2,3,4,5,6,7"

python train.py \
    --data text_det \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --model_type yolo \
    --epoch 50 \
    --close_mosaic 10 \
    --image-size 1024 \
    --batch-size 32 \
    --project text_det_yolo11l\
    --plot 1 \
    --device "0,1,2,3,4,5,6,7"