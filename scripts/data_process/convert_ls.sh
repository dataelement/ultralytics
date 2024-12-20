# 定义数据集名称数组
datasets=("CDLA" "D4LA" "DocLayNet" "ElemLayout" "M6Doc")

# 遍历每个数据集
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    # 转换为小写用于输出文件名
    dataset_lower=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')
    
    # 处理训练集
    python label_studio.py \
        --data_dir "/workspace/datasets/layout/unsv2_layout/${dataset}" \
        --coco_json_file "/workspace/datasets/layout/unsv2_layout/${dataset}/train.json" \
        --output_file "/workspace/datasets/layout/unsv2_layout/${dataset}/${dataset_lower}_train.json" \
        --url_prefix "http://192.168.106.8/datasets/unsv2/layout/${dataset}/images"
    
    # 处理验证集
    python label_studio.py \
        --data_dir "/workspace/datasets/layout/unsv2_layout/${dataset}" \
        --coco_json_file "/workspace/datasets/layout/unsv2_layout/${dataset}/val.json" \
        --output_file "/workspace/datasets/layout/unsv2_layout/${dataset}/${dataset_lower}_val.json" \
        --url_prefix "http://192.168.106.8/datasets/unsv2/layout/${dataset}/images"
done