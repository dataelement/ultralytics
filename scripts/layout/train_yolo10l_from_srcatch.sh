# tarin on doclaynet
# python train.py \
#     --data doclaynet_2 \
#     --model_path /workspace/models/YOLO11/yolo11l.pt \
#     --epoch 50 \
#     --image-size 1024 \
#     --batch-size 64 \
#     --project layout_yolo11l_doclaynet_2 \
#     --plot 1 \
#     --close_mosaic 20 \
#     --device "4,5,6,7" 

# tarin on dataelem_layout
python train.py \
    --data dataelem_layout \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 40 \
    --image-size 640 \
    --batch-size 256 \
    --project layout_yolo10l_dataelem_layout \
    --plot 1 \
    --close_mosaic 10 \
    --device "0,1,2,3,4,5,6,7" 