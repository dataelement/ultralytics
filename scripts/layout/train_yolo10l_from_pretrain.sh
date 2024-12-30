# tarin on doclaynet
# python train.py \
#     --data doclaynet_2 \
#     --model_path /workspace/datasets/layout/yolo_layout_pretrain_model/yolo11l_layout_pretrain.pt \
#     --epoch 40 \
#     --image-size 1024\
#     --batch-size 16 \
#     --project layout_yolo11l_doclaynet_2_from_pretrain \
#     --plot 1 \
#     --close_mosaic 10 \
#     --device "6,7"


# tarin on dataelem_layout
# python train.py \
#     --data dataelem_layout \
#     --model_path /workspace/datasets/layout/yolo_layout_pretrain_model/yolo11l_layout_pretrain.pt \
#     --epoch 40 \
#     --image-size 1024\
#     --batch-size 64 \
#     --project layout_yolo10l_dataelem_layout_from_pretrain \
#     --plot 1 \
#     --close_mosaic 10 \
#     --device "0,1,2,3,4,5,6,7"

# tarin on unsv2
python train.py \
    --data /workspace/datasets/layout/unsv2_layout_yolo_data/data.yaml \
    --model_path /workspace/models/yolo_layout_pretrain_model/yolo11l_layout_pretrain.pt \
    --epoch 40 \
    --image-size 1024\
    --batch-size 64 \
    --project layout_yolo10l_dataelem_layout_from_pretrain \
    --plot 1 \
    --close_mosaic 10 \
    --device "0,1,2,3,4,5,6,7"