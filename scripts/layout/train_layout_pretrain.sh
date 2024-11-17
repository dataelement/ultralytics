# train yolo11l layout pretrain model 
python train.py \
    --data docsynth300k \
    --model_path /workspace/models/YOLO11/yolo11l.pt \
    --epoch 30 \
    --val 0 \
    --image-size 1024\
    --batch-size 32 \
    --project layout_yolo11l_docsynth300k \
    --plot 1 \
    --device "4,5,6,7" 