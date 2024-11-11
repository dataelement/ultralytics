# finetune yolo11x seg on dataelem_layout
# python train.py \
#     --data dataelem_layout \
#     --model_path /workspace/models/YOLO11/yolo11x-seg.pt \
#     --epoch 500 \
#     --image-size 1280 \
#     --batch-size 24 \
#     --project layout_yolo11x_seg \
#     --plot 1 \
#     --optimizer SGD \
#     --lr0 0.005 \
#     --patience 50 \
#     --device "2,3,4,5,6,7" 

# finetune yolo11x det on dataelem_layout
# python train.py \
#     --data dataelem_layout \
#     --model_path /workspace/models/YOLO11/yolo11x.pt \
#     --epoch 500 \
#     --image-size 1280 \
#     --batch-size 24 \
#     --project layout_yolo11x\
#     --plot 1 \
#     --optimizer SGD \
#     --lr0 0.005 \
#     --patience 50 \
#     --device "2,3,4,5,6,7" 

# finetune yolo11x det on dataelem_layout multi scale
# python train.py \
#     --data dataelem_layout \
#     --model_path /workspace/models/YOLO11/yolo11x.pt \
#     --epoch 500 \
#     --image-size 1280 \
#     --batch-size 16 \
#     --project layout_yolo11x_multi_scale\
#     --plot 1 \
#     --optimizer SGD \
#     --lr0 0.005 \
#     --patience 50 \
#     --device "0,1,2,3,4,5,6,7" 

# finetune yolo11x det on d4la
# python train.py \
#     --data d4la \
#     --model_path /workspace/models/YOLO11/yolo11x.pt \
#     --epoch 500 \
#     --image-size 1280 \
#     --batch-size 32 \
#     --project layout_yolo11x_d4la \
#     --plot 1 \
#     --optimizer SGD \
#     --lr0 0.005 \
#     --patience 50 \
#     --device "0,1,2,3,4,5,6,7" 

python train.py \
    --data doclaynet \
    --model_path /workspace/models/YOLO11/yolo11x.pt \
    --epoch 500 \
    --image-size 1280 \
    --batch-size 30 \
    --project layout_yolo11x_doclaynet \
    --plot 1 \
    --optimizer SGD \
    --lr0 0.005 \
    --patience 50 \
    --device "2,3,4,5,6,7" 
