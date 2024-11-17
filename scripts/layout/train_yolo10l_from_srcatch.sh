

python train.py \
    --data doclaynet_2 \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 50 \
    --image-size 1024\
    --batch-size 64 \
    --project layout_yolo11l_doclaynet_2 \
    --plot 1 \
    --close_mosaic 20 \
    --device "4,5,6,7" 