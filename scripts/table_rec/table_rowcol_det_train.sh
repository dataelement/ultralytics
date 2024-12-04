# finetune yolo11n det on table_det
# python train.py \
#     --data table_rowcol_det \
#     --model_path /workspace/models/YOLO11/yolo11x.pt \
#     --epoch 50 \
#     --image-size 640 \
#     --batch-size 128 \
#     --project table_rowcol_det_yolo11x \
#     --plot 1 \
#     --device "0,1,2,3,4,5,6,7"

# seg
# python train.py \
#     --data table_rowcol_det \
#     --model_path /workspace/models/YOLO11/yolo11l.pt \
#     --epoch 50 \
#     --image-size 1600 \
#     --batch-size 16 \
#     --project table_rowcol_det_yolo11l \
#     --plot 1 \
#     --device "0,1,2,3,4,5,6,7"

python train.py \
    --data table_rowcol_det \
    --model_path /workspace/models/YOLO11/yolo11l-seg.pt \
    --model_type yolo \
    --epoch 50 \
    --image-size 640 \
    --batch-size 128 \
    --project table_rowcol_det_yolo11l-seg_change_loss \
    --plot 1 \
    --device "0,1,2,3,4,5,6,7"