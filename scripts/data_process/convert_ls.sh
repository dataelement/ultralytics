python label_studio.py \
    --data_dir /workspace/datasets/layout/unsv2_layout/CDLA \
    --coco_json_file /workspace/datasets/layout/unsv2_layout/CDLA/train.json \
    --output_file /workspace/datasets/layout/unsv2_layout/CDLA/cdla_train.json \
    --url_prefix http://192.168.106.8/datasets/unsv2/layout/CDLA/images

python label_studio.py \
    --data_dir /workspace/datasets/layout/unsv2_layout/CDLA \
    --coco_json_file /workspace/datasets/layout/unsv2_layout/CDLA/val.json \
    --output_file /workspace/datasets/layout/unsv2_layout/CDLA/cdla_val.json \
    --url_prefix http://192.168.106.8/datasets/unsv2/layout/CDLA/images