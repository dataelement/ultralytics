# finetune yolo11x det on dataelem_layout
python train.py \
    --data table_cell_det \
    --model_path /workspace/models/YOLO11/yolo11x.pt \
    --epoch 500 \
    --image-size 1280 \
    --batch-size 8 \
    --project table_cell_det_yolo11x\
    --plot 1 \
    --optimizer SGD \
    --lr0 0.005 \
    --patience 50 \
    --device "4,5,6,7" 
