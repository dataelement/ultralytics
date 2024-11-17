# finetune yolo11n det on table_det
python train.py \
    --data table_rowcol_det \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 50 \
    --mosaic 0 \
    --image-size 1024 \
    --batch-size 64 \
    --project table_rowcol_det_yolo11l\
    --plot 1 \
    --device "0,1,2,3,4,5,6,7"
