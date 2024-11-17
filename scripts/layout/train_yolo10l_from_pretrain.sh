# tarin on doclaynet
python train.py \
    --data doclaynet_2 \
    --model_path /workspace/youjiachen/workspace/ultralytics/layout_yolo11l_docsynth300k/yolo11l_docsynth300k_epoch30_imgsz1024_bs32/weights/last.pt \
    --epoch 50 \
    --image-size 1024\
    --batch-size 16 \
    --project layout_yolo11l_doclaynet_2_from_pretrain \
    --plot 1 \
    --close_mosaic 10 \
    --device "6,7"


