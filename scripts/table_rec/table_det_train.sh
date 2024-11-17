# finetune yolo11n det on table_det
python train.py \
    --data table_det \
    --model_path /workspace/models/YOLO11/yolo11n.pt \
    --epoch 50 \
    --close_mosaic 10 \
    --image-size 1280 \
    --batch-size 256 \
    --project table_det_yolo11n\
    --plot 1 \
    --device "0,1,2,3,4,5,6,7"
