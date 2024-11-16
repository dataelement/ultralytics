# finetune yolo11l det on table_cell_det
python train.py \
    --data table_cell_det \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 50 \
    --close_mosaic 20 \
    --image-size 1024 \
    --batch-size 16 \
    --project table_cell_det_yolo11l\
    --plot 1 \
    --device "4,5,6,7" 


# finetune yolo11x det on table_cell_det
# python train.py \
#     --data table_cell_det \
#     --model_path /workspace/models/YOLO11/yolo11x.pt \
#     --epoch 40 \
#     --image-size 1024 \
#     --batch-size 16 \
#     --project table_cell_det_yolo11x\
#     --plot 1 \
#     --device "4,5,6,7" 