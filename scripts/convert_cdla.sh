CDLA_dir=/workspace/datasets/layout/CDLA_DATASET
OUTPUT_dir=/workspace/datasets/layout/CDLA_DATASET_COCO_VAL
# train
# python3 labelme2coco.py $CDLA_dir/train $OUTPUT_dir --labels $CDLA_dir/labels.txt --novi

# val
python3 labelme2coco.py $CDLA_dir/val $OUTPUT_dir --labels $CDLA_dir/labels.txt --novi
