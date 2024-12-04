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

python train.py \
    --data table_det \
    --model_path yolo11s-obb.pt \
    --epoch 50 \
    --cfg ultralytics/cfg/obb.yaml \
    --image-size 1024 \
    --batch-size 256 \
    --project table_det_yolo11s_test\
    --plot 1 \
    --device "0,1,2,3,4,5,6,7"