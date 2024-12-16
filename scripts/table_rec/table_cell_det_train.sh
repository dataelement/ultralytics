# finetune yolo11l det on table_cell_det
python train.py \
    --data table_cell_det \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 100 \
    --mosaic 1 \
    --close_mosaic 10 \
    --image-size 1024 \
    --reg_max 16 \
    --batch-size 32 \
    --project table_cell_det_yolo11l \
    --plot 1 \
    --device "0,1,2,3,4,5,6,7" 