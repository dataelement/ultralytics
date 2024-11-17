# finetune yolo11n det on table_det
python train.py \
    --data table_rowcol_det \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 50 \
    --close_mosaic 10 \
    --image-size 640 \
    --batch-size 256 \
    --project table_rowcol_det_yolo11l\
    --plot 1 \
    --device "0,1,2,3,4,5,6,7"
